"""Origin-saddle dynamics."""

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
import numpy as np

from experiments.bregman_pc.bp import update_bp
from experiments.bregman_pc.evaluate import evaluate_batch, evaluate_jpc_batch
from experiments.bregman_pc.model import BregmanMLP, scaled_param_lr
from experiments.bregman_pc.steps import (
    bregman_mlp_to_jpc,
    bregman_pc_step,
    clone_eqx,
    init_jpc_opt_state,
    jpc_loss_id,
    standard_pc_step,
)
from experiments.bregman_pc.train import make_param_optim, set_seed


def create_pc_saddles_dataset(key, d: int, n_samples: int, mean: float, std: float):
    """Gaussian inputs with target ``y = -x``, as in pc-saddles."""
    x = mean + std * jax.random.normal(key, (n_samples, d))
    y = -x
    return x, y


def parse_args():
    p = argparse.ArgumentParser(
        description="pc-saddles origin-saddle toy: BP vs standard PC vs Bregman PC"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input-dim", type=int, default=3)
    p.add_argument(
        "--n-samples",
        type=int,
        default=64,
    )
    p.add_argument("--data-mean", type=float, default=1.0)
    p.add_argument("--data-std", type=float, default=0.1)
    p.add_argument("--n-steps", type=int, default=100000)
    p.add_argument(
        "--param-lr",
        type=float,
        default=0.025,
        help="SGD step size..",
    )
    p.add_argument("--widths", type=int, nargs="+", default=[1])
    p.add_argument(
        "--n-layers",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Number of layers L (weight maps). Hidden-layer count is L-1.",
    )
    p.add_argument("--act-fn", type=str, default="tanh", choices=["tanh", "sigmoid"])
    p.add_argument("--param-type", type=str, default="sp", choices=["sp", "mupc"])
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
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


def default_save_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / "origin_saddle"


def run_dir(base: Path, width: int, n_layers: int, seed: int) -> Path:
    return base / f"N{width}" / f"L{n_layers}" / f"seed_{seed}"


def hparams_for_width(width: int, args):
    param_lr = args.param_lr if args.param_lr is not None else (0.4 if width == 1 else 1e-3)
    init_std = args.init_std if args.init_std is not None else (5e-2 if width == 1 else 1e-1)
    n_infer = args.n_infer_iters if args.n_infer_iters is not None else (20 if width == 1 else 50)
    return param_lr, init_std, n_infer


def train_one(key, x, y, width: int, n_layers: int, args) -> dict:
    n_hidden = n_layers - 1
    if n_hidden < 1:
        raise ValueError("Need at least two layers (one hidden layer).")
    param_lr, init_std, n_infer = hparams_for_width(width, args)
    output_dim = x.shape[-1]
    layer_sizes = [args.input_dim] + [width] * n_hidden + [output_dim]
    init_model = BregmanMLP(
        key=key,
        layer_sizes=layer_sizes,
        act_fn=args.act_fn,
        output_loss="mse",
        init_scale=init_std**2,
        param_type=args.param_type,
        gamma=args.gamma_0,
    )
    bregman_model = clone_eqx(init_model)
    std_pc_model = bregman_mlp_to_jpc(init_model)
    bp_model = clone_eqx(init_model)
    std_pc_loss = jpc_loss_id("mse")
    lr = scaled_param_lr(
        args.param_type, "sgd", param_lr, width, n_layers, args.gamma_0
    )
    bregman_optim = make_param_optim("sgd", lr)
    std_pc_optim = make_param_optim("sgd", lr)
    bp_optim = make_param_optim("sgd", lr)
    params0 = eqx.filter(bp_model, eqx.is_array)
    bregman_opt_state = init_jpc_opt_state(bregman_model.layers, bregman_optim)
    std_pc_opt_state = init_jpc_opt_state(std_pc_model, std_pc_optim)
    bp_opt_state = bp_optim.init(params0)

    history = {"bp": [], "std_pc": [], "bregman": []}
    print(
        f"  N={width}, H={n_hidden}, L={n_layers}: η={param_lr:g} (scaled={lr:g}), "
        f"σ={init_std:g}, T={n_infer}, dt={args.activity_lr}"
    )

    def eval_losses(bregman, std_pc, bp):
        bregman_loss, _ = evaluate_batch(bregman, x, y, task="classify")
        std_loss, _ = evaluate_jpc_batch(std_pc, x, y, std_pc_loss, task="classify")
        bp_loss, _ = evaluate_batch(bp, x, y, task="classify")
        return bregman_loss, std_loss, bp_loss

    def record(bregman_loss, std_loss, bp_loss, step: int):
        history["bregman"].append(bregman_loss)
        history["std_pc"].append(std_loss)
        history["bp"].append(bp_loss)
        if step % 100 == 0:
            bregman_v, std_v, bp_v = jax.device_get(
                (bregman_loss, std_loss, bp_loss)
            )
            print(
                f"    step {step:4d}  Bregman={float(bregman_v):.4e}  "
                f"StdPC={float(std_v):.4e}  BP={float(bp_v):.4e}"
            )

    record(*eval_losses(bregman_model, std_pc_model, bp_model), step=0)
    for step in range(args.n_steps):
        bregman_model, bregman_opt_state, _ = bregman_pc_step(
            bregman_model,
            x,
            y,
            bregman_optim,
            bregman_opt_state,
            n_infer,
            args.activity_lr,
        )
        std_pc_model, std_pc_opt_state, _ = standard_pc_step(
            std_pc_model,
            x,
            y,
            std_pc_optim,
            std_pc_opt_state,
            n_infer,
            args.activity_lr,
            std_pc_loss,
        )
        bp_model, bp_opt_state, _, _ = update_bp(bp_model, x, y, bp_optim, bp_opt_state)
        record(
            *eval_losses(bregman_model, std_pc_model, bp_model),
            step=step + 1,
        )

    return {
        name: np.asarray(jax.device_get(jnp.stack(vals)))
        for name, vals in history.items()
    }


def save_history(path: Path, history: dict, width: int, n_layers: int, args) -> None:
    param_lr, init_std, n_infer = hparams_for_width(width, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        bp=history["bp"],
        std_pc=history["std_pc"],
        bregman=history["bregman"],
        width=np.asarray(width),
        n_layers=np.asarray(n_layers),
        param_lr=np.asarray(param_lr),
        init_std=np.asarray(init_std),
        n_infer_iters=np.asarray(n_infer),
        n_steps=np.asarray(args.n_steps),
    )


def main():
    args = parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else default_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    n_layers_list = args.n_layers

    key = jax.random.PRNGKey(args.seed)
    data_key, key = jax.random.split(key)
    x, y = create_pc_saddles_dataset(
        data_key, args.input_dim, args.n_samples, args.data_mean, args.data_std
    )
    print(
        f"pc-saddles toy: y=-x, x~N({args.data_mean},{args.data_std}), "
        f"P={args.n_samples}, D={args.input_dim}, {args.n_steps} full-batch steps, "
        f"act={args.act_fn}"
    )
    for width in args.widths:
        for n_layers in n_layers_list:
            model_key, key = jax.random.split(key)
            out_dir = run_dir(save_dir, width, n_layers, args.seed)
            history = train_one(model_key, x, y, width, n_layers, args)
            save_history(out_dir / "history.npz", history, width, n_layers, args)
    print("Done.")


if __name__ == "__main__":
    main()
