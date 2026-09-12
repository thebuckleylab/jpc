"""MNIST origin-saddle experiment from Innocenti et al. (2024, Figure 5)."""

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
from torch import Generator
from torch.utils.data import DataLoader

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

import experiments.datasets as datasets

_INPUT_DIMS = {"MNIST": 784, "Fashion-MNIST": 784}
_OUTPUT_DIM = 10
_DATA_DIR = str(_REPO_ROOT / "experiments" / "datasets")


def parse_args():
    p = argparse.ArgumentParser(
        description="pc-saddles MNIST origin saddle: BP vs standard PC vs Bregman PC"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", type=str, default="MNIST", choices=list(_INPUT_DIMS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap the number of weight updates (default: full epochs).",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--param-lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=500)
    p.add_argument(
        "--n-hidden",
        type=int,
        default=4,
        help="Hidden layers. Paper uses 5 fully connected layers, so 4 hidden.",
    )
    p.add_argument("--act-fn", type=str, default="tanh", choices=["tanh", "sigmoid"])
    p.add_argument("--param-type", type=str, default="sp", choices=["sp", "mupc"])
    p.add_argument("--gamma-0", type=float, default=1.0)
    p.add_argument(
        "--init-std",
        type=float,
        default=5e-3,
        help="Weight init std near the origin.",
    )
    p.add_argument("--n-infer-iters", type=int, default=50)
    p.add_argument("--activity-lr", type=float, default=0.05)
    p.add_argument(
        "--loss-tol",
        type=float,
        default=None,
        help=(
            "If set, stop when standard-PC epoch train-loss drop falls below "
            "this. Default: run all epochs (paper origin-init)."
        ),
    )
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


def get_loaders(dataset: str, batch_size: int, seed: int):
    if dataset == "MNIST":
        train_data = datasets.MNIST(train=True, save_dir=_DATA_DIR)
        test_data = datasets.MNIST(train=False, save_dir=_DATA_DIR)
    else:
        train_data = datasets.FashionMNIST(train=True, save_dir=_DATA_DIR)
        test_data = datasets.FashionMNIST(train=False, save_dir=_DATA_DIR)
    generator = Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, test_loader


def default_save_dir(dataset: str) -> Path:
    slug = "mnist" if dataset == "MNIST" else "fashion_mnist"
    return Path(__file__).resolve().parent / "results" / f"{slug}_origin_saddle"


def train(args) -> dict:
    set_seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    n_layers = args.n_hidden + 1
    input_dim = _INPUT_DIMS[args.dataset]
    layer_sizes = [input_dim] + [args.width] * args.n_hidden + [_OUTPUT_DIM]
    init_model = BregmanMLP(
        key=key,
        layer_sizes=layer_sizes,
        act_fn=args.act_fn,
        output_loss="mse",
        init_scale=args.init_std**2,
        param_type=args.param_type,
        gamma=args.gamma_0,
    )
    bregman_model = clone_eqx(init_model)
    std_pc_model = bregman_mlp_to_jpc(init_model)
    bp_model = clone_eqx(init_model)
    std_pc_loss = jpc_loss_id("mse")
    lr = scaled_param_lr(
        args.param_type, "sgd", args.param_lr, args.width, n_layers, args.gamma_0
    )
    # jpc energies are batch-mean; the paper's PC `set_grads` is a sum over the
    # batch, so the same η gives PC weight steps larger by B. Match that here so
    # PC can leave the origin while BP (mean MSE) stays on the plateau.
    pc_lr = lr * args.batch_size
    bregman_optim = make_param_optim("sgd", pc_lr)
    std_pc_optim = make_param_optim("sgd", pc_lr)
    bp_optim = make_param_optim("sgd", lr)
    params0 = eqx.filter(bp_model, eqx.is_array)
    bregman_opt_state = init_jpc_opt_state(bregman_model.layers, bregman_optim)
    std_pc_opt_state = init_jpc_opt_state(std_pc_model, std_pc_optim)
    bp_opt_state = bp_optim.init(params0)

    train_loader, test_loader = get_loaders(args.dataset, args.batch_size, args.seed)
    w_bp = bp_model.layers[0].linear.weight
    w_bregman = bregman_model.layers[0].linear.weight
    w_std = std_pc_model[0].layers[0].linear.weight
    print(
        "  init max|W_Bregman-W_BP|="
        f"{float(jax.device_get(jnp.max(jnp.abs(w_bregman - w_bp)))):.2e}  "
        "max|W_StdPC-W_BP|="
        f"{float(jax.device_get(jnp.max(jnp.abs(w_std - w_bp)))):.2e}"
    )

    history = {"bp": [], "std_pc": [], "bregman": []}
    print(
        f"{args.dataset} origin saddle: L={n_layers}, N={args.width}, "
        f"η={args.param_lr:g} (BP scaled={lr:g}, PC scaled={pc_lr:g}), σ={args.init_std:g}, "
        f"T={args.n_infer_iters}, dt={args.activity_lr}, batch={args.batch_size}"
    )
    prev_std_epoch_loss = None
    step = 0
    stop = False
    recorded_init = False
    for epoch in range(1, args.epochs + 1):
        epoch_std_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = jnp.asarray(x.numpy())
            y = jnp.asarray(y.numpy())
            if not recorded_init:
                bregman_loss, bregman_acc = evaluate_batch(
                    bregman_model, x, y, task="classify"
                )
                std_loss, std_acc = evaluate_jpc_batch(
                    std_pc_model, x, y, std_pc_loss, task="classify"
                )
                bp_loss, bp_acc = evaluate_batch(bp_model, x, y, task="classify")
                history["bregman"].append(bregman_loss)
                history["std_pc"].append(std_loss)
                history["bp"].append(bp_loss)
                bregman_v, std_v, bp_v, bregman_a, std_a, bp_a = jax.device_get(
                    (bregman_loss, std_loss, bp_loss, bregman_acc, std_acc, bp_acc)
                )
                print(
                    f"    step {0:6d}  Bregman={float(bregman_v):.4e} "
                    f"({100 * float(bregman_a):.1f}%)  "
                    f"StdPC={float(std_v):.4e} ({100 * float(std_a):.1f}%)  "
                    f"BP={float(bp_v):.4e} ({100 * float(bp_a):.1f}%)"
                )
                recorded_init = True
            bregman_model, bregman_opt_state, _ = bregman_pc_step(
                bregman_model,
                x,
                y,
                bregman_optim,
                bregman_opt_state,
                args.n_infer_iters,
                args.activity_lr,
            )
            std_pc_model, std_pc_opt_state, _ = standard_pc_step(
                std_pc_model,
                x,
                y,
                std_pc_optim,
                std_pc_opt_state,
                args.n_infer_iters,
                args.activity_lr,
                std_pc_loss,
            )
            bp_model, bp_opt_state, _, _ = update_bp(
                bp_model, x, y, bp_optim, bp_opt_state
            )
            bregman_loss, bregman_acc = evaluate_batch(
                bregman_model, x, y, task="classify"
            )
            std_loss, std_acc = evaluate_jpc_batch(
                std_pc_model, x, y, std_pc_loss, task="classify"
            )
            bp_loss, bp_acc = evaluate_batch(bp_model, x, y, task="classify")
            history["bregman"].append(bregman_loss)
            history["std_pc"].append(std_loss)
            history["bp"].append(bp_loss)
            epoch_std_loss += float(jax.device_get(std_loss))
            n_batches += 1
            step += 1
            if step % args.log_every == 0 or step == 1:
                bregman_v, std_v, bp_v, bregman_a, std_a, bp_a = jax.device_get(
                    (bregman_loss, std_loss, bp_loss, bregman_acc, std_acc, bp_acc)
                )
                print(
                    f"    step {step:6d}  Bregman={float(bregman_v):.4e} "
                    f"({100 * float(bregman_a):.1f}%)  "
                    f"StdPC={float(std_v):.4e} ({100 * float(std_a):.1f}%)  "
                    f"BP={float(bp_v):.4e} ({100 * float(bp_a):.1f}%)"
                )
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        mean_std_epoch = epoch_std_loss / max(n_batches, 1)
        print(f"  epoch {epoch}: std PC mean train MSE={mean_std_epoch:.4e}")
        if (
            args.loss_tol is not None
            and prev_std_epoch_loss is not None
            and (prev_std_epoch_loss - mean_std_epoch) < args.loss_tol
        ):
            print(
                f"  stopping: std PC epoch drop "
                f"{prev_std_epoch_loss - mean_std_epoch:.4e} < {args.loss_tol:g}"
            )
            stop = True
        prev_std_epoch_loss = mean_std_epoch
        if stop:
            break

    n_test = 0
    test = {"bp": 0.0, "std_pc": 0.0, "bregman": 0.0}
    test_acc = {"bp": 0.0, "std_pc": 0.0, "bregman": 0.0}
    for x, y in test_loader:
        x = jnp.asarray(x.numpy())
        y = jnp.asarray(y.numpy())
        bregman_loss, bregman_acc = evaluate_batch(bregman_model, x, y, task="classify")
        std_loss, std_acc = evaluate_jpc_batch(
            std_pc_model, x, y, std_pc_loss, task="classify"
        )
        bp_loss, bp_acc = evaluate_batch(bp_model, x, y, task="classify")
        bregman_v, std_v, bp_v, bregman_a, std_a, bp_a = jax.device_get(
            (bregman_loss, std_loss, bp_loss, bregman_acc, std_acc, bp_acc)
        )
        test["bregman"] += float(bregman_v)
        test["std_pc"] += float(std_v)
        test["bp"] += float(bp_v)
        test_acc["bregman"] += float(bregman_a)
        test_acc["std_pc"] += float(std_a)
        test_acc["bp"] += float(bp_a)
        n_test += 1
    if n_test:
        print(
            "  test  "
            + "  ".join(
                f"{name}={test[name] / n_test:.4e} "
                f"({100 * test_acc[name] / n_test:.1f}%)"
                for name in ("bregman", "std_pc", "bp")
            )
        )

    return {
        name: np.asarray(jax.device_get(jnp.stack(vals)))
        for name, vals in history.items()
    }


def main():
    args = parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else default_save_dir(args.dataset)
    save_dir.mkdir(parents=True, exist_ok=True)
    history = train(args)
    np.savez(
        save_dir / "history.npz",
        bp=history["bp"],
        std_pc=history["std_pc"],
        bregman=history["bregman"],
        width=np.asarray(args.width),
        n_hidden=np.asarray(args.n_hidden),
        param_lr=np.asarray(args.param_lr),
        init_std=np.asarray(args.init_std),
        n_infer_iters=np.asarray(args.n_infer_iters),
        activity_lr=np.asarray(args.activity_lr),
    )
    print("Done.")


if __name__ == "__main__":
    main()
