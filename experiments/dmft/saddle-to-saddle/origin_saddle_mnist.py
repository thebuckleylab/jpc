"""MNIST origin-saddle experiment with energy-scaled µPC vs BP.

Same task as the Bregman origin-saddle MNIST run (Innocenti et al. 2024,
Figure 5): a near-origin MLP trained on MNIST or Fashion-MNIST. Finite-size
µPC puts width, depth, and ``γ`` in the energy rather than the PC learning
rate, matching ``origin_saddle_toy.py``:

* output precision ``λ = γ² N L``
* hidden precision ``κ = L``

BP GD still bakes ``γ² N`` into the optimiser. Both nets share the same
near-origin Gaussian weights and an MSE readout on one-hot labels.

The Bregman origin-saddle script is standard parameterisation with
``σ = 5e-3`` and ``η = 1e-3``. Those values underflow once µPC's
``1/√N`` and ``1/N`` forward scales are included. Defaults here are
``σ = 0.1`` and ``η = 0.1``. The smaller init lengthens the origin
plateau; PC then leaves it while BP stays.
Training runs for 20 epochs (937 steps each on MNIST at batch size 64).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SADDLE_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SADDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_SADDLE_DIR))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from torch import Generator
from torch.utils.data import DataLoader

import jpc
import origin_saddle_toy as saddle
from experiments.mupc_paper.utils import set_seed

import experiments.datasets as datasets

_INPUT_DIMS = {"MNIST": 784, "Fashion-MNIST": 784}
_OUTPUT_DIM = 10
_DATA_DIR = str(_REPO_ROOT / "experiments" / "datasets")


def parse_args():
    p = argparse.ArgumentParser(
        description="Origin-saddle MNIST: energy-scaled µPC vs BP"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", type=str, default="MNIST", choices=list(_INPUT_DIMS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap the number of weight updates (default: full epochs).",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--param-lr",
        type=float,
        default=0.1,
        help=(
            "SGD step size. PC keeps this raw; BP µP multiplies by γ² N."
        ),
    )
    p.add_argument("--width", type=int, default=500)
    p.add_argument(
        "--n-hidden",
        type=int,
        default=4,
        help="Hidden layers. Paper uses 5 fully connected layers, so 4 hidden.",
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
        default=0.1,
        help=(
            "Weight init std near the origin. µPC applies 1/√N and 1/N "
            "in the forward pass, so the Bregman SP value 5e-3 underflows. "
            "0.1 lengthens the origin plateau and still trains."
        ),
    )
    p.add_argument("--n-infer-iters", type=int, default=50)
    p.add_argument("--activity-lr", type=float, default=0.05)
    p.add_argument(
        "--loss-tol",
        type=float,
        default=None,
        help=(
            "If set, stop when PC epoch train-loss drop falls below this. "
            "Default: run all epochs (paper origin-init)."
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


def run_dir(base: Path, width: int, n_layers: int, seed: int) -> Path:
    return base / f"N{width}" / f"L{n_layers}" / f"seed_{seed}"


@eqx.filter_jit
def pc_batch_metrics(model, x, y, param_type, gamma, skip_model):
    preds = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        skip_model=skip_model,
        param_type=param_type,
        gamma=gamma,
    )[-1]
    loss = jpc.mse_loss(preds, y)
    acc = jpc.compute_accuracy(y, preds) / 100.0
    return loss, acc


@eqx.filter_jit
def bp_batch_metrics(model, x, y):
    preds = jax.vmap(model)(x)
    loss = jpc.mse_loss(preds, y)
    acc = jpc.compute_accuracy(y, preds) / 100.0
    return loss, acc


def _log_step(step, pc_loss, bp_loss, pc_acc, bp_acc):
    pc_v, bp_v, pc_a, bp_a = jax.device_get((pc_loss, bp_loss, pc_acc, bp_acc))
    print(
        f"    step {step:6d}  PC={float(pc_v):.4e} ({100 * float(pc_a):.1f}%)  "
        f"BP={float(bp_v):.4e} ({100 * float(bp_a):.1f}%)"
    )


def train(args) -> dict:
    set_seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    n_layers = args.n_hidden + 1
    if args.n_hidden < 1:
        raise ValueError("Need at least one hidden layer.")
    input_dim = _INPUT_DIMS[args.dataset]
    lam, kappa = saddle.energy_scalings(
        args.param_type, args.gamma_0, args.width, n_layers
    )
    pc_model, bp_model, skip_model = saddle.make_origin_models(
        key,
        input_dim=input_dim,
        width=args.width,
        depth=n_layers,
        output_dim=_OUTPUT_DIM,
        act_fn=args.act_fn,
        param_type=args.param_type,
        gamma=args.gamma_0,
        init_std=args.init_std,
    )
    pc_optim = saddle.make_pc_param_optim(args.param_lr)
    bp_optim = saddle.make_bp_param_optim(
        args.param_lr, args.param_type, args.gamma_0, args.width
    )
    pc_opt_state = saddle.init_pc_opt_state(pc_model, skip_model, pc_optim)
    bp_opt_state = saddle.init_bp_opt_state(bp_model, bp_optim)
    bp_lr = float(
        args.param_lr
        if args.param_type == "sp"
        else args.param_lr * (args.gamma_0 ** 2) * args.width
    )
    wdiff = float(jax.device_get(saddle.max_weight_abs_diff(pc_model, bp_model)))
    train_loader, test_loader = get_loaders(args.dataset, args.batch_size, args.seed)
    print(
        f"{args.dataset} origin saddle: L={n_layers}, N={args.width}, "
        f"η={args.param_lr:g} (PC raw, BP scaled={bp_lr:g}), σ={args.init_std:g}, "
        f"T={args.n_infer_iters}, dt={args.activity_lr}, λ={lam:g}, κ={kappa:g}, "
        f"batch={args.batch_size}, max|W_PC-W_BP|={wdiff:.2e}"
    )

    history = {"bp": [], "pc": [], "bp_acc": [], "pc_acc": []}
    prev_pc_epoch_loss = None
    step = 0
    stop = False
    recorded_init = False

    def record(pc_loss, bp_loss, pc_acc, bp_acc, step_idx: int, *, log: bool):
        history["pc"].append(pc_loss)
        history["bp"].append(bp_loss)
        history["pc_acc"].append(pc_acc)
        history["bp_acc"].append(bp_acc)
        if log:
            _log_step(step_idx, pc_loss, bp_loss, pc_acc, bp_acc)

    for epoch in range(1, args.epochs + 1):
        epoch_pc_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = jnp.asarray(x.numpy())
            y = jnp.asarray(y.numpy())
            if not recorded_init:
                pc_loss, pc_acc = pc_batch_metrics(
                    pc_model, x, y, args.param_type, args.gamma_0, skip_model
                )
                bp_loss, bp_acc = bp_batch_metrics(bp_model, x, y)
                record(pc_loss, bp_loss, pc_acc, bp_acc, 0, log=True)
                recorded_init = True
            pc_model, skip_model, pc_opt_state, _ = saddle.pc_step(
                pc_model,
                skip_model,
                x,
                y,
                pc_optim,
                pc_opt_state,
                args.n_infer_iters,
                args.activity_lr,
                args.param_type,
                args.gamma_0,
                lam,
                kappa,
            )
            bp_model, bp_opt_state = saddle.bp_step(
                bp_model, x, y, bp_optim, bp_opt_state
            )
            pc_loss, pc_acc = pc_batch_metrics(
                pc_model, x, y, args.param_type, args.gamma_0, skip_model
            )
            bp_loss, bp_acc = bp_batch_metrics(bp_model, x, y)
            record(
                pc_loss,
                bp_loss,
                pc_acc,
                bp_acc,
                step + 1,
                log=(step + 1) % args.log_every == 0 or step + 1 == 1,
            )
            epoch_pc_loss += pc_loss
            n_batches += 1
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        mean_pc_epoch = float(jax.device_get(epoch_pc_loss / max(n_batches, 1)))
        print(f"  epoch {epoch}: PC mean train MSE={mean_pc_epoch:.4e}")
        if (
            args.loss_tol is not None
            and prev_pc_epoch_loss is not None
            and (prev_pc_epoch_loss - mean_pc_epoch) < args.loss_tol
        ):
            print(
                f"  stopping: PC epoch drop "
                f"{prev_pc_epoch_loss - mean_pc_epoch:.4e} < {args.loss_tol:g}"
            )
            stop = True
        prev_pc_epoch_loss = mean_pc_epoch
        if stop:
            break

    n_test = 0
    test = {"bp": 0.0, "pc": 0.0}
    test_acc = {"bp": 0.0, "pc": 0.0}
    for x, y in test_loader:
        x = jnp.asarray(x.numpy())
        y = jnp.asarray(y.numpy())
        pc_loss, pc_acc = pc_batch_metrics(
            pc_model, x, y, args.param_type, args.gamma_0, skip_model
        )
        bp_loss, bp_acc = bp_batch_metrics(bp_model, x, y)
        pc_v, bp_v, pc_a, bp_a = jax.device_get((pc_loss, bp_loss, pc_acc, bp_acc))
        test["pc"] += float(pc_v)
        test["bp"] += float(bp_v)
        test_acc["pc"] += float(pc_a)
        test_acc["bp"] += float(bp_a)
        n_test += 1
    if n_test:
        print(
            "  test  "
            + "  ".join(
                f"{name}={test[name] / n_test:.4e} "
                f"({100 * test_acc[name] / n_test:.1f}%)"
                for name in ("pc", "bp")
            )
        )

    return {
        name: np.asarray(jax.device_get(jnp.stack(vals)))
        for name, vals in history.items()
    }


def save_history(path: Path, history: dict, n_layers: int, args, lam: float, kappa: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        bp=history["bp"],
        pc=history["pc"],
        bp_acc=history["bp_acc"],
        pc_acc=history["pc_acc"],
        width=np.asarray(args.width),
        n_hidden=np.asarray(args.n_hidden),
        n_layers=np.asarray(n_layers),
        param_lr=np.asarray(args.param_lr),
        init_std=np.asarray(args.init_std),
        n_infer_iters=np.asarray(args.n_infer_iters),
        activity_lr=np.asarray(args.activity_lr),
        gamma_0=np.asarray(args.gamma_0),
        output_energy_scaling=np.asarray(lam),
        hidden_energy_scaling=np.asarray(kappa),
        param_type=np.asarray(args.param_type),
    )


def main():
    args = parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else default_save_dir(args.dataset)
    n_layers = args.n_hidden + 1
    lam, kappa = saddle.energy_scalings(
        args.param_type, args.gamma_0, args.width, n_layers
    )
    history = train(args)
    flat_path = save_dir / "history.npz"
    nested_path = run_dir(save_dir, args.width, n_layers, args.seed) / "history.npz"
    save_history(flat_path, history, n_layers, args, lam, kappa)
    save_history(nested_path, history, n_layers, args, lam, kappa)
    print(f"Saved {flat_path}")
    print("Done.")


if __name__ == "__main__":
    main()
