"""Origin-saddle toy dynamics for energy-scaled µPC vs BP.

Gaussian inputs with target ``y = -x``. Finite-size µPC puts width, depth,
and ``γ`` in the energy rather than the PC learning rate:

* output precision ``λ = γ² N L``
* hidden precision ``κ = L``

BP GD still bakes ``γ² N`` into the optimiser. Both nets share the same
near-origin Gaussian weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import jpc
from experiments.dmft.src.utils import (
    MLP,
    bp_gd_style_lr,
    get_hidden_energy_scaling,
    get_output_energy_scaling,
    make_sgd_param_optim,
)
from experiments.mupc_paper.utils import set_seed


def energy_scalings(param_type: str, gamma_0: float, width: int, depth: int):
    """Return ``(λ, κ)`` for the given parameterisation."""
    return (
        get_output_energy_scaling(param_type, gamma_0, width, depth),
        get_hidden_energy_scaling(param_type, depth),
    )


def apply_origin_init(pc_model, key, init_std: float):
    """Replace every linear map with ``N(0, init_std²)`` (origin saddle)."""
    keys = jr.split(key, len(pc_model))
    layers = []
    for i, layer in enumerate(pc_model):
        linear = layer[1]
        weight = init_std * jr.normal(keys[i], linear.weight.shape)
        linear = eqx.tree_at(lambda lin: lin.weight, linear, weight)
        layers.append(eqx.tree_at(lambda seq: seq[1], layer, linear))
    return layers


def copy_pc_weights_to_bp(pc_model, bp_model):
    """Copy PC linear weights onto the BP MLP so both start identical."""
    for i in range(len(pc_model)):
        bp_model = eqx.tree_at(
            lambda m, i=i: m.layers[i][1].weight,
            bp_model,
            jnp.copy(pc_model[i][1].weight),
        )
    return bp_model


def max_weight_abs_diff(pc_model, bp_model):
    diffs = [
        jnp.max(jnp.abs(pc_model[i][1].weight - bp_model.layers[i][1].weight))
        for i in range(len(pc_model))
    ]
    return jnp.max(jnp.stack(diffs))


def make_origin_models(
    key,
    *,
    input_dim: int,
    width: int,
    depth: int,
    output_dim: int,
    act_fn: str,
    param_type: str,
    gamma: float,
    init_std: float,
    use_skips: bool = False,
):
    """PC ``jpc.make_mlp`` and BP ``MLP`` with shared origin-scale weights."""
    model_key, init_key = jr.split(key)
    pc_model = jpc.make_mlp(
        model_key,
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn,
        use_bias=False,
        param_type=param_type,
    )
    pc_model = apply_origin_init(pc_model, init_key, init_std)
    bp_model = MLP(
        key=model_key,
        d_in=input_dim,
        N=width,
        L=depth,
        d_out=output_dim,
        act_fn=act_fn,
        param_type=param_type,
        gamma=gamma,
        use_bias=False,
        use_skips=use_skips,
    )
    bp_model = copy_pc_weights_to_bp(pc_model, bp_model)
    skip_model = jpc.make_skip_model(depth) if use_skips else None
    return pc_model, bp_model, skip_model


def make_pc_param_optim(param_lr: float):
    """PC GD: plain ``param_lr``; ``λ`` / ``κ`` live in the energy."""
    return make_sgd_param_optim(param_lr, "gd")


def make_bp_param_optim(param_lr: float, param_type: str, gamma_0: float, width: int):
    """BP GD: µP bakes ``γ² N`` into the optimiser."""
    return make_sgd_param_optim(
        bp_gd_style_lr(param_lr, param_type, gamma_0, width),
        "gd",
    )


def init_pc_opt_state(pc_model, skip_model, optim):
    return optim.init((eqx.filter(pc_model, eqx.is_array), skip_model))


def init_bp_opt_state(bp_model, optim):
    return optim.init(eqx.filter(bp_model, eqx.is_array))


@eqx.filter_jit
def pc_ffwd_loss(model, x, y, param_type, gamma, skip_model=None):
    preds = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        skip_model=skip_model,
        param_type=param_type,
        gamma=gamma,
    )[-1]
    return jpc.mse_loss(preds, y)


@eqx.filter_jit
def bp_ffwd_loss(model, x, y):
    preds = jax.vmap(model)(x)
    return 0.5 * jnp.mean(jnp.sum((y - preds) ** 2, axis=1))


@eqx.filter_jit
def pc_step(
    model,
    skip_model,
    x,
    y,
    param_optim,
    param_opt_state,
    n_infer_iters: int,
    activity_lr: float,
    param_type: str,
    gamma: float,
    output_energy_scaling: float,
    hidden_energy_scaling: float,
):
    """One inference sweep + PC parameter update (µPC energy scalings)."""
    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        skip_model=skip_model,
        param_type=param_type,
        gamma=gamma,
    )
    activity_optim = optax.sgd(activity_lr * x.shape[0])
    activity_opt_state = activity_optim.init(activities)

    def body(carry, _):
        acts, st = carry
        out = jpc.update_pc_activities(
            params=(model, skip_model),
            activities=acts,
            optim=activity_optim,
            opt_state=st,
            output=y,
            input=x,
            param_type=param_type,
            gamma=gamma,
            loss_id="mse",
            output_energy_scaling=output_energy_scaling,
            hidden_energy_scaling=hidden_energy_scaling,
        )
        return (out["activities"], out["opt_state"]), out["energy"]

    (activities, _), energies = jax.lax.scan(
        body, (activities, activity_opt_state), xs=None, length=n_infer_iters
    )
    result = jpc.update_pc_params(
        params=(model, skip_model),
        activities=activities,
        optim=param_optim,
        opt_state=param_opt_state,
        output=y,
        input=x,
        param_type=param_type,
        gamma=gamma,
        loss_id="mse",
        output_energy_scaling=output_energy_scaling,
        hidden_energy_scaling=hidden_energy_scaling,
    )
    return result["model"], result["skip_model"], result["opt_state"], energies[-1]


@eqx.filter_jit
def bp_step(model, x, y, optim, opt_state):
    def loss_fn(model, x, y):
        preds = jax.vmap(model)(x)
        return 0.5 * jnp.mean(jnp.sum((y - preds) ** 2, axis=1))

    _, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state


def create_pc_saddles_dataset(key, d: int, n_samples: int, mean: float, std: float):
    """Gaussian inputs with target ``y = -x``."""
    x = mean + std * jr.normal(key, (n_samples, d))
    y = -x
    return x, y


def parse_args():
    p = argparse.ArgumentParser(
        description="Origin-saddle toy: energy-scaled µPC vs BP"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input-dim", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--data-mean", type=float, default=1.0)
    p.add_argument("--data-std", type=float, default=0.1)
    p.add_argument("--n-steps", type=int, default=100000)
    p.add_argument(
        "--param-lr",
        type=float,
        default=0.025,
        help="SGD step size. PC keeps this raw; BP µP multiplies by γ² N.",
    )
    p.add_argument("--widths", type=int, nargs="+", default=[1])
    p.add_argument(
        "--n-layers",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Number of layers L (weight maps). Hidden-layer count is L-1.",
    )
    p.add_argument(
        "--act-fn",
        type=str,
        default="tanh",
        choices=["tanh", "relu", "linear"],
    )
    p.add_argument("--param-type", type=str, default="mupc", choices=["sp", "mupc"])
    p.add_argument("--gamma-0", type=float, default=1.0)
    p.add_argument(
        "--init-std",
        type=float,
        default=None,
        help="Weight init std near the origin.",
    )
    p.add_argument(
        "--n-infer-iters",
        type=int,
        default=50,
        help="PC inference steps.",
    )
    p.add_argument(
        "--activity-lr",
        type=float,
        default=0.05,
        help="Inference Euler step.",
    )
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


def default_save_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def run_dir(base: Path, width: int, n_layers: int, seed: int) -> Path:
    return base / f"N{width}" / f"L{n_layers}" / f"seed_{seed}"


def hparams_for_width(width: int, args):
    param_lr = args.param_lr if args.param_lr is not None else (
        0.4 if width == 1 else 1e-3
    )
    init_std = args.init_std if args.init_std is not None else (
        5e-2 if width == 1 else 1e-1
    )
    n_infer = args.n_infer_iters if args.n_infer_iters is not None else (
        20 if width == 1 else 50
    )
    return param_lr, init_std, n_infer


def train_one(key, x, y, width: int, n_layers: int, args) -> dict:
    n_hidden = n_layers - 1
    if n_hidden < 1:
        raise ValueError("Need at least two layers (one hidden layer).")
    param_lr, init_std, n_infer = hparams_for_width(width, args)
    output_dim = x.shape[-1]
    lam, kappa = energy_scalings(args.param_type, args.gamma_0, width, n_layers)
    pc_model, bp_model, skip_model = make_origin_models(
        key,
        input_dim=args.input_dim,
        width=width,
        depth=n_layers,
        output_dim=output_dim,
        act_fn=args.act_fn,
        param_type=args.param_type,
        gamma=args.gamma_0,
        init_std=init_std,
    )
    pc_optim = make_pc_param_optim(param_lr)
    bp_optim = make_bp_param_optim(param_lr, args.param_type, args.gamma_0, width)
    pc_opt_state = init_pc_opt_state(pc_model, skip_model, pc_optim)
    bp_opt_state = init_bp_opt_state(bp_model, bp_optim)
    bp_lr = float(
        param_lr if args.param_type == "sp" else param_lr * (args.gamma_0 ** 2) * width
    )
    wdiff = float(jax.device_get(max_weight_abs_diff(pc_model, bp_model)))
    print(
        f"  N={width}, H={n_hidden}, L={n_layers}: η={param_lr:g} "
        f"(PC raw, BP scaled={bp_lr:g}), σ={init_std:g}, T={n_infer}, "
        f"dt={args.activity_lr}, λ={lam:g}, κ={kappa:g}, "
        f"max|W_PC-W_BP|={wdiff:.2e}"
    )

    history = {"bp": [], "pc": []}

    def eval_losses(pc, bp):
        pc_loss = pc_ffwd_loss(
            pc, x, y, args.param_type, args.gamma_0, skip_model
        )
        bp_loss = bp_ffwd_loss(bp, x, y)
        return pc_loss, bp_loss

    def record(pc_loss, bp_loss, step: int):
        history["pc"].append(pc_loss)
        history["bp"].append(bp_loss)
        if step % args.log_every == 0:
            pc_v, bp_v = jax.device_get((pc_loss, bp_loss))
            print(
                f"    step {step:4d}  PC={float(pc_v):.4e}  BP={float(bp_v):.4e}"
            )

    record(*eval_losses(pc_model, bp_model), step=0)
    for step in range(args.n_steps):
        pc_model, skip_model, pc_opt_state, _ = pc_step(
            pc_model,
            skip_model,
            x,
            y,
            pc_optim,
            pc_opt_state,
            n_infer,
            args.activity_lr,
            args.param_type,
            args.gamma_0,
            lam,
            kappa,
        )
        bp_model, bp_opt_state = bp_step(
            bp_model, x, y, bp_optim, bp_opt_state
        )
        record(*eval_losses(pc_model, bp_model), step=step + 1)

    return {
        name: np.asarray(jax.device_get(jnp.stack(vals)))
        for name, vals in history.items()
    }


def save_history(path: Path, history: dict, width: int, n_layers: int, args) -> None:
    param_lr, init_std, n_infer = hparams_for_width(width, args)
    lam, kappa = energy_scalings(args.param_type, args.gamma_0, width, n_layers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        bp=history["bp"],
        pc=history["pc"],
        width=np.asarray(width),
        n_layers=np.asarray(n_layers),
        param_lr=np.asarray(param_lr),
        init_std=np.asarray(init_std),
        n_infer_iters=np.asarray(n_infer),
        n_steps=np.asarray(args.n_steps),
        gamma_0=np.asarray(args.gamma_0),
        output_energy_scaling=np.asarray(lam),
        hidden_energy_scaling=np.asarray(kappa),
        param_type=np.asarray(args.param_type),
    )


def main():
    args = parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else default_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    key = jr.PRNGKey(args.seed)
    data_key, key = jr.split(key)
    x, y = create_pc_saddles_dataset(
        data_key, args.input_dim, args.n_samples, args.data_mean, args.data_std
    )
    print(
        f"pc-saddles toy: y=-x, x~N({args.data_mean},{args.data_std}), "
        f"P={args.n_samples}, D={args.input_dim}, {args.n_steps} full-batch steps, "
        f"act={args.act_fn}, param_type={args.param_type}"
    )
    for width in args.widths:
        for n_layers in args.n_layers:
            model_key, key = jr.split(key)
            out_dir = run_dir(save_dir, width, n_layers, args.seed)
            history = train_one(model_key, x, y, width, n_layers, args)
            save_history(out_dir / "history.npz", history, width, n_layers, args)
    print("Done.")


if __name__ == "__main__":
    main()
