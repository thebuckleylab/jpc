import os
import argparse
import numpy as np

import jax.random as jr
import jax.numpy as jnp
from jax import vmap

import equinox as eqx
import equinox.nn as nn
import optax
import jpc

from experiments.datasets import get_dataloaders
from utils import (
    set_seed,
    init_weights,
    compute_param_l2_norms,
    compute_param_spectral_norms
)


class MLP(eqx.Module):
    D: int
    N: int
    L: int
    param_type: str
    use_skips: bool
    layers: list

    def __init__(
            self,
            key,
            d_in,
            N,
            L,
            d_out,
            act_fn,
            param_type,
            use_bias=False,
            use_skips=False
    ):
        self.D = d_in
        self.N = N
        self.L = L
        self.param_type = param_type
        self.use_skips = use_skips

        keys = jr.split(key, L)
        self.layers = []
        for i in range(L):
            act_fn_l = nn.Identity() if i == 0 else jpc.get_act_fn(act_fn)
            _in = d_in if i == 0 else N
            _out = d_out if (i + 1) == L else N
            layer = nn.Sequential(
                [
                    nn.Lambda(act_fn_l),
                    nn.Linear(
                        _in,
                        _out,
                        use_bias=use_bias,
                        key=keys[i]
                    )
                ]
            )
            self.layers.append(layer)

    def __call__(self, x):
        pre_activs = []

        if self.param_type == "depth_mup":
            for i, f in enumerate(self.layers):
                if (i + 1) == 1:
                    x = f(x) / jnp.sqrt(self.D)
                elif 1 < (i + 1) < self.L:
                    residual = x if self.use_skips else 0
                    rescaling = jnp.sqrt(
                        self.N * self.L
                    ) if self.use_skips else jnp.sqrt(self.N)
                    x = (f(x) / rescaling) + residual
                elif (i + 1) == self.L:
                    x = f(x) / self.N

                pre_activs.append(x)

        else:
            for i, f in enumerate(self.layers):
                residual = x if self.use_skips and (1 < (i + 1) < self.L) else 0

                x = f(x) + residual

                pre_activs.append(x)

        return pre_activs


def mse_loss(model, x, y):
    y_pred = vmap(model)(x)[-1]
    return jnp.mean((y - y_pred) ** 2)


@eqx.filter_jit
def make_step(model, optim, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(mse_loss)(model, x, y)
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def test_fwd_pass(
        seed,
        dataset,
        width,
        depth,
        act_fn,
        optim_id,
        param_type,
        use_skips,
        lr,
        batch_size,
        n_checks
):
    set_seed(seed)

    key = jr.PRNGKey(seed)
    keys = jr.split(key, 2)
    model = MLP(
        key=keys[0],
        d_in=784,
        N=width,
        L=depth,
        d_out=10,
        act_fn=act_fn,
        param_type=param_type,
        use_bias=False,
        use_skips=use_skips
    )

    if param_type == "depth_mup":
        model = init_weights(
            model=model,
            init_fn_id="standard_gauss",
            key=keys[1]
        )
    elif param_type == "orthogonal":
        model = init_weights(
            model=model,
            init_fn_id="orthogonal",
            key=keys[1],
            gain=1.05 if act_fn == "tanh" else 1
        )

    optim = optax.sgd(lr) if optim_id == "sgd" else optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    layer_idxs = [0, int(depth/4)-1, int(depth/2)-1, int(depth*3/4)-1, depth-1]
    avg_activity_l1 = np.zeros((len(layer_idxs), n_checks))
    avg_activity_l2 = np.zeros_like(avg_activity_l1)
    param_l2_norms = np.zeros_like(avg_activity_l1)
    param_spectral_norms = np.zeros_like(avg_activity_l1)

    train_loader, _ = get_dataloaders(dataset, batch_size)
    for t, (img_batch, label_batch) in enumerate(train_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        pre_activities = vmap(model)(img_batch)
        i = 0
        for l, pre_act in enumerate(pre_activities):
            if l in layer_idxs:
                avg_activity_l1[i, t] = jnp.abs(pre_act).mean()
                avg_activity_l2[i, t] = jnp.sqrt(jnp.mean(pre_act**2))
                i += 1

        param_l2_norms[:, t] = compute_param_l2_norms(
            model=model.layers,
            act_fn=act_fn,
            layer_idxs=layer_idxs
        )
        param_spectral_norms[:, t] = compute_param_spectral_norms(
            model=model.layers,
            act_fn=act_fn,
            layer_idxs=layer_idxs
        )
        model, opt_state, _ = make_step(
            model=model,
            optim=optim,
            opt_state=opt_state,
            x=img_batch,
            y=label_batch
        )
        if t >= (n_checks - 1):
            break

    return (
        avg_activity_l1,
        avg_activity_l2,
        param_l2_norms,
        param_spectral_norms
    )


if __name__ == "__main__":
    RESULTS_DIR = "mlp_fwd_pass_results"
    DATASET = "MNIST"
    WIDTHS = [2 ** i for i in range(7, 11)]
    DEPTHS = [2 ** i for i in range(4, 10)]
    LR = 1e-3
    BATCH_SIZE = 64
    N_RECORDED_LAYERS = 5
    N_CHECKS = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--act_fns", type=str, nargs='+', default=["linear", "tanh", "relu"])
    parser.add_argument("--optim_ids", type=str, nargs='+', default=["sgd", "adam"])
    parser.add_argument("--param_types", type=str, nargs='+', default=["sp", "depth_mup", "orthogonal"])
    parser.add_argument("--seed", type=int, default=54638)
    args = parser.parse_args()

    for act_fn in args.act_fns:
        print(f"\nact_fn: {act_fn}")

        for optim_id in args.optim_ids:
            print(f"\n\toptim: {optim_id}")

            for param_type in args.param_types:
                print(f"\n\t\tparam_type: {param_type}")
                
                skip_uses = [False, True] if param_type != "orthogonal" else [False]
                for use_skips in skip_uses:
                    print(f"\n\t\t\tuse_skips: {use_skips}")
                    
                    save_dir = os.path.join(
                        RESULTS_DIR,
                        act_fn,
                        optim_id,
                        param_type,
                        "skips" if use_skips else "no_skips",
                        str(args.seed)
                    )
                    os.makedirs(save_dir, exist_ok=True)

                    avg_activity_l1_per_N_L = np.zeros((N_RECORDED_LAYERS, N_CHECKS, len(WIDTHS), len(DEPTHS)))
                    avg_activity_l2_per_N_L = np.zeros_like(avg_activity_l1_per_N_L)
                    param_l2_norms_per_N_L = np.zeros_like(avg_activity_l1_per_N_L)
                    param_spectral_norms_per_N_L = np.zeros_like(avg_activity_l1_per_N_L)

                    for w, width in enumerate(WIDTHS):
                        print(f"\n\t\t\t\tN = {width}\n")
                        for d, depth in enumerate(DEPTHS):
                            print(f"\t\t\t\t\tL = {depth}")

                            avg_activity_l1, avg_activity_l2, param_l2_norms, param_spectral_norms = test_fwd_pass(
                                seed=args.seed,
                                dataset=DATASET,
                                width=width,
                                depth=depth,
                                act_fn=act_fn,
                                optim_id=optim_id,
                                param_type=param_type,
                                use_skips=use_skips,
                                lr=LR,
                                batch_size=BATCH_SIZE,
                                n_checks=N_CHECKS
                            )
                            avg_activity_l1_per_N_L[:, :, w, d] = avg_activity_l1
                            avg_activity_l2_per_N_L[:, :, w, d] = avg_activity_l2
                            param_l2_norms_per_N_L[:, :, w, d] = param_l2_norms
                            param_spectral_norms_per_N_L[:, :, w, d] = param_spectral_norms

                    np.save(f"{save_dir}/avg_activity_l1_per_N_L.npy", avg_activity_l1_per_N_L)
                    np.save(f"{save_dir}/avg_activity_l2_per_N_L.npy", avg_activity_l2_per_N_L)
                    np.save(f"{save_dir}/param_l2_norms_per_N_L.npy", param_l2_norms_per_N_L)
                    np.save(f"{save_dir}/param_spectral_norms_per_N_L.npy", param_spectral_norms_per_N_L)
