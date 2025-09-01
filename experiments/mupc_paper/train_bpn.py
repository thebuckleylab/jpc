import os
import argparse
import numpy as np

import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from jax.nn import log_softmax

import equinox as eqx
import equinox.nn as nn
import optax
import jpc

from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import set_seed, init_weights


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

        else:
            for i, f in enumerate(self.layers):
                residual = x if self.use_skips and (1 < (i + 1) < self.L) else 0

                x = f(x) + residual

        return x
    

def evaluate(model, testloader, loss_id):
    loss_fn = get_loss_fn(loss_id)
    avg_test_loss, avg_test_acc = 0, 0
    for x, y in testloader:
        x, y = x.numpy(), y.numpy()
        avg_test_loss += loss_fn(model, x, y)
        avg_test_acc += compute_accuracy(model, x, y)
    return avg_test_loss / len(testloader), avg_test_acc / len(testloader)


@eqx.filter_jit
def mse_loss(model, x, y):
    y_pred = vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)


@eqx.filter_jit
def cross_entropy_loss(model, x, y):
    logits = vmap(model)(x)
    log_probs = log_softmax(logits)
    return - jnp.mean(jnp.sum(y * log_probs, axis=-1))


def get_loss_fn(loss_id):
    if loss_id == "mse":
        return mse_loss
    elif loss_id == "ce":
        return cross_entropy_loss


@eqx.filter_jit
def compute_accuracy(model, x, y):
    pred_y = vmap(model)(x)
    return jnp.mean(
        jnp.argmax(y, axis=1) == jnp.argmax(pred_y, axis=1)
    ) * 100


@eqx.filter_jit
def make_step(model, optim, opt_state, x, y, loss_id="mse"):
    loss_fn = get_loss_fn(loss_id)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_mlp(
        seed,
        dataset,
        loss_id,
        width,
        n_hidden,
        act_fn,
        param_type,
        optim_id,
        lr,
        batch_size,
        max_epochs,
        test_every,
        save_dir
):
    set_seed(seed)
    key = jr.PRNGKey(seed)
    model_key, init_key = jr.split(key, 2)
    os.makedirs(save_dir, exist_ok=True)

    model = MLP(
        key=model_key,
        d_in=3072,  #784, 3072
        N=width,
        L=n_hidden+1,
        d_out=10,
        act_fn=act_fn,
        param_type=param_type,
        use_bias=False,
        use_skips=True if param_type == "depth_mup" else False
    )
    if param_type == "depth_mup":
        model = init_weights(
            model=model,
            init_fn_id="standard_gauss",
            key=init_key
        )

    optim = optax.sgd(lr) if optim_id == "sgd" else optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # data
    train_loader, test_loader = get_dataloaders(dataset, batch_size)

    # key metrics
    train_losses = []
    test_losses, test_accs = [], []

    diverged = no_learning = False
    global_batch_id = 0
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}\n-------------------------------")

        for train_iter, (img_batch, label_batch) in enumerate(train_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

            model, opt_state, train_loss = make_step(
                model=model,
                optim=optim,
                opt_state=opt_state,
                x=img_batch,
                y=label_batch,
                loss_id=loss_id
            )
            train_losses.append(train_loss)
            global_batch_id += 1

            if global_batch_id % test_every == 0:
                print(
                    f"Train loss: {train_loss:.7f} [{train_iter * len(img_batch)}/{len(train_loader.dataset)}]"
                )
                avg_test_loss, avg_test_acc = evaluate(
                    model, 
                    test_loader, 
                    loss_id
                )
                test_losses.append(avg_test_loss)
                test_accs.append(avg_test_acc)
                print(f"Avg test accuracy: {avg_test_acc:.4f}\n")

            if np.isinf(train_loss) or np.isnan(train_loss):
                diverged = True
                break
            
            if global_batch_id >= test_every and avg_test_acc < 15:
                no_learning = True
                break
        
        if diverged:
            print(
                f"Stopping training because of diverging loss: {train_loss}"
            )
            break
        
        if no_learning:
            print(
                f"Stopping training because of close-to-chance accuracy (no learning): {avg_test_acc}"
            )
            break
    
    np.save(f"{save_dir}/train_losses.npy", train_losses)
    np.save(f"{save_dir}/test_losses.npy", test_losses)
    np.save(f"{save_dir}/test_accs.npy", test_accs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="bp_results")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--loss_id", type=str, default="ce")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--n_hidden", type=int, default=8)
    parser.add_argument("--act_fns", type=str, nargs='+', default=["relu"])
    parser.add_argument("--param_type", type=str, default="depth_mup") 
    parser.add_argument("--optim_id", type=str, default="adam")
    parser.add_argument("--lrs", type=float, nargs='+', default=[1e-2])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--test_every", type=int, default=780)
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    for act_fn in args.act_fns:
        for lr in args.lrs:
            for seed in range(args.n_seeds):
                save_dir = os.path.join(
                    args.results_dir,
                    args.dataset,
                    f"{args.loss_id}_loss",
                    f"width_{args.width}",
                    f"{args.n_hidden}_n_hidden",
                    act_fn,
                    f"{args.param_type}_param",
                    args.optim_id,
                    f"lr_{lr}",
                    f"batch_size_{args.batch_size}",
                    f"{args.max_epochs}_epochs",
                    str(seed)
                )
                print(f"Starting training with config: {save_dir} with seed: {seed}")
                train_mlp(
                    seed=seed,
                    dataset=args.dataset,
                    loss_id=args.loss_id,
                    width=args.width,
                    n_hidden=args.n_hidden,
                    act_fn=act_fn,
                    param_type=args.param_type,
                    optim_id=args.optim_id,
                    lr=lr,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    test_every=args.test_every,
                    save_dir=save_dir
                )
