"""Train activation-matched Bregman PC with a matched BP baseline."""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch import Generator, manual_seed

from experiments.bregman_pc.bp import update_bp
from experiments.bregman_pc.evaluate import (
    evaluate_batch,
    evaluate_jpc_batch,
    evaluate_models,
    feedforward_loss,
)
from experiments.datasets import get_dataloaders
from experiments.bregman_pc.model import BregmanMLP, scaled_param_lr
from experiments.bregman_pc.plot import plot_metrics, plot_sweep
from experiments.bregman_pc.steps import (
    bregman_mlp_to_jpc,
    bregman_pc_bp_grad_cosine,
    bregman_pc_energy,
    bregman_pc_step,
    init_jpc_opt_state,
    jpc_loss_id,
    standard_pc_bp_grad_cosine,
    standard_pc_energy,
    standard_pc_step,
)


_DATASETS = ("MNIST", "Fashion-MNIST", "CIFAR10")
_INPUT_DIMS = {"MNIST": 784, "Fashion-MNIST": 784, "CIFAR10": 3072}
_N_CLASSES = 10
_SAVE_SLUGS = {
    "MNIST": "mnist",
    "Fashion-MNIST": "fashion_mnist",
    "CIFAR10": "cifar10",
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    manual_seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Activation-matched Bregman PC")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument(
        "--dataset",
        type=str,
        default="Fashion-MNIST",
        choices=list(_DATASETS)
    )
    p.add_argument("--width", type=int, nargs="+", default=[256])
    p.add_argument("--n-hidden", type=int, default=3)
    p.add_argument("--act-fn", type=str, default="tanh", choices=["tanh", "sigmoid"])
    p.add_argument("--output-loss", type=str, default="mse", choices=["ce", "mse", "bregman"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--n-infer-iters", type=int, nargs="+", default=[5, 10, 20, 50])   
    p.add_argument("--activity-lr", type=float, nargs="+", default=[5e-2, 1e-1,  5e-1])#1e-3, 5e-3, 1e-2]) # 5e-2, 1e-1,  5e-1
    p.add_argument("--param-lr", type=float, nargs="+", default=[1e-3, 5e-3, 1e-2])  #5e-4, 
    p.add_argument("--param-optim", type=str, default="adam", choices=["sgd", "adam"])
    p.add_argument(
        "--param-type",
        type=str,
        nargs="+",
        default=["sp"],
        choices=["sp", "mupc"],
        help="Weight parameterisation. μPC uses width scalings on the layer maps (default: sp).",
    )
    p.add_argument(
        "--gamma-0",
        type=float,
        nargs="+",
        default=[1.0],
        help="Output scale γ₀; last-layer map is divided by γ₀ (default: 1).",
    )
    p.add_argument("--init-scale", type=float, default=None)
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument(
        "--log-every",
        type=str,
        default="step",
        choices=["epoch", "step"],
        help="Record train energy/loss after each epoch or after each weight update. Test metrics are always logged per epoch (default: epoch).",
    )
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


def make_param_optim(name: str, lr: float):
    if name == "sgd":
        return optax.sgd(lr)
    if name == "adam":
        return optax.adam(lr)
    raise ValueError(f"Unknown param optim {name!r}. Options are 'sgd' and 'adam'.")


def empty_history():
    return {
        "t": [],
        "epoch": [],
        "bregman_train_energy": [],
        "std_pc_train_energy": [],
        "bregman_train_loss": [],
        "std_pc_train_loss": [],
        "bp_train_loss": [],
        "bregman_train_acc": [],
        "std_pc_train_acc": [],
        "bp_train_acc": [],
        "bregman_pc_bp_cos": [],
        "std_pc_bp_cos": [],
        "bregman_test_loss": [],
        "std_pc_test_loss": [],
        "bp_test_loss": [],
        "bregman_test_acc": [],
        "std_pc_test_acc": [],
        "bp_test_acc": [],
    }


def default_save_dir(dataset: str) -> Path:
    return Path(__file__).resolve().parent / "results" / _SAVE_SLUGS[dataset]


def run_save_dir(
    base: Path,
    param_types: list[str],
    param_type: str,
    gamma_0s: list[float],
    gamma_0: float,
    widths: list[int],
    width: int,
    param_lrs: list[float],
    param_lr: float,
    activity_lrs: list[float],
    activity_lr: float,
    n_infer_iters_list: list[int],
    n_infer_iters: int,
    n_seeds: int,
    seed: int,
) -> str:
    path = base
    if len(param_types) > 1:
        path = path / f"{param_type}_param_type"
    if len(gamma_0s) > 1:
        path = path / f"gamma_0_{gamma_0:g}"
    if len(widths) > 1:
        path = path / f"width_{width}"
    if len(param_lrs) > 1:
        path = path / f"param_lr_{param_lr:g}"
    if len(activity_lrs) > 1:
        path = path / f"activity_lr_{activity_lr:g}"
    if len(n_infer_iters_list) > 1:
        path = path / f"n_infer_iters_{n_infer_iters}"
    if n_seeds > 1:
        path = path / f"seed_{seed}"
    return str(path)


def _host_history(history: dict) -> dict:
    out = {}
    for key, vals in history.items():
        if not vals:
            out[key] = np.asarray(vals)
            continue
        if hasattr(vals[0], "shape"):
            out[key] = np.asarray(jax.device_get(jnp.stack(vals)))
        else:
            out[key] = np.asarray(vals)
    return out


def train(
    args,
    seed: int,
    width: int,
    param_type: str,
    gamma_0: float,
    param_lr: float,
    activity_lr: float,
    n_infer_iters: int,
    save_dir: str,
) -> dict:
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    layer_sizes = [_INPUT_DIMS[args.dataset]] + [width] * args.n_hidden + [_N_CLASSES]
    init_model = BregmanMLP(
        key=key,
        layer_sizes=layer_sizes,
        act_fn=args.act_fn,
        output_loss=args.output_loss,
        init_scale=args.init_scale,
        param_type=param_type,
        gamma=gamma_0,
    )
    # Same initial weights; later updates return new pytrees.
    bregman_model = init_model
    std_pc_model = bregman_mlp_to_jpc(init_model)
    bp_model = init_model
    std_pc_loss = jpc_loss_id(args.output_loss)
    depth = args.n_hidden + 1
    lr = scaled_param_lr(
        param_type, args.param_optim, param_lr, width, depth, gamma_0
    )
    bregman_optim = make_param_optim(args.param_optim, lr)
    std_pc_optim = make_param_optim(args.param_optim, lr)
    bp_optim = make_param_optim(args.param_optim, lr)
    params0 = eqx.filter(init_model, eqx.is_array)
    bregman_opt_state = init_jpc_opt_state(init_model.layers, bregman_optim)
    std_pc_opt_state = init_jpc_opt_state(std_pc_model, std_pc_optim)
    bp_opt_state = bp_optim.init(params0)

    generator = Generator()
    generator.manual_seed(seed)
    train_loader, test_loader = get_dataloaders(
        args.dataset, args.batch_size, flatten=True, generator=generator
    )
    history = empty_history()
    step = 0
    print(
        f"Bregman PC vs Std PC vs BP {args.dataset}: seed={seed}, "
        f"width={width}, n_hidden={args.n_hidden}, act={args.act_fn}, "
        f"output={args.output_loss}, init_scale={args.init_scale}, "
        f"param_type={param_type}, gamma_0={gamma_0}, "
        f"param_optim={args.param_optim}, param_lr={param_lr} (scaled={lr:g}), "
        f"activity_lr={activity_lr}, n_infer_iters={n_infer_iters}, "
        f"log_every={args.log_every}"
    )

    def current_models():
        return {
            "bregman": bregman_model,
            "std_pc": std_pc_model,
            "bp": bp_model,
        }

    def log_train(t, x, y, bregman_energy, std_energy, bp_loss):
        bregman_loss, bregman_acc = evaluate_batch(bregman_model, x, y)
        std_loss, std_acc = evaluate_jpc_batch(std_pc_model, x, y, std_pc_loss)
        bp_ff_loss, bp_acc = evaluate_batch(bp_model, x, y)
        history["t"].append(t)
        history["bregman_train_energy"].append(bregman_energy)
        history["std_pc_train_energy"].append(std_energy)
        history["bregman_train_loss"].append(bregman_loss)
        history["std_pc_train_loss"].append(std_loss)
        history["bp_train_loss"].append(bp_ff_loss)
        history["bregman_train_acc"].append(bregman_acc)
        history["std_pc_train_acc"].append(std_acc)
        history["bp_train_acc"].append(bp_acc)
        history["bregman_pc_bp_cos"].append(
            bregman_pc_bp_grad_cosine(
                bregman_model, x, y, n_infer_iters, activity_lr
            )
        )
        history["std_pc_bp_cos"].append(
            standard_pc_bp_grad_cosine(
                std_pc_model, x, y, n_infer_iters, activity_lr, std_pc_loss
            )
        )

    def log_eval(epoch, x, y, bregman_energy, std_energy, bp_loss):
        metrics = evaluate_models(
            current_models(),
            test_loader,
            max_batches=args.eval_batches,
            jpc_loss=std_pc_loss,
        )
        history["epoch"].append(epoch)
        for name in ("bregman", "std_pc", "bp"):
            history[f"{name}_test_loss"].append(metrics[name][0])
            history[f"{name}_test_acc"].append(metrics[name][1])
        bregman_acc = evaluate_batch(bregman_model, x, y)[1]
        std_acc = evaluate_jpc_batch(std_pc_model, x, y, std_pc_loss)[1]
        bp_acc = evaluate_batch(bp_model, x, y)[1]
        bregman_energy, std_energy, bp_loss, bregman_acc, std_acc, bp_acc = jax.device_get(
            (bregman_energy, std_energy, bp_loss, bregman_acc, std_acc, bp_acc)
        )
        print(
            f"epoch {epoch:3d}  step {step:5d}  "
            f"Bregman E={float(bregman_energy):.4f} acc={float(bregman_acc):.3f} "
            f"test={metrics['bregman'][1]:.3f}  |  "
            f"StdPC E={float(std_energy):.4f} acc={float(std_acc):.3f} "
            f"test={metrics['std_pc'][1]:.3f}  |  "
            f"BP L={float(bp_loss):.4f} acc={float(bp_acc):.3f} "
            f"test={metrics['bp'][1]:.3f}"
        )
        return metrics

    x0, y0 = next(iter(train_loader))
    x0 = jnp.asarray(x0.numpy())
    y0 = jnp.asarray(y0.numpy())
    init_bregman = bregman_pc_energy(
        bregman_model, x0, y0, n_infer_iters, activity_lr
    )
    init_std = standard_pc_energy(
        std_pc_model, x0, y0, n_infer_iters, activity_lr, std_pc_loss
    )
    init_bp = feedforward_loss(bp_model, x0, y0)
    log_train(0, x0, y0, init_bregman, init_std, init_bp)
    log_eval(0, x0, y0, init_bregman, init_std, init_bp)

    log_every_step = args.log_every == "step"
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x = jnp.asarray(x.numpy())
            y = jnp.asarray(y.numpy())

            bregman_model, bregman_opt_state, bregman_energy = bregman_pc_step(
                bregman_model,
                x,
                y,
                bregman_optim,
                bregman_opt_state,
                n_infer_iters,
                activity_lr,
            )

            std_pc_model, std_pc_opt_state, std_energy = standard_pc_step(
                std_pc_model,
                x,
                y,
                std_pc_optim,
                std_pc_opt_state,
                n_infer_iters,
                activity_lr,
                std_pc_loss,
            )

            bp_model, bp_opt_state, _, bp_loss = update_bp(
                bp_model, x, y, bp_optim, bp_opt_state
            )

            step += 1
            if log_every_step:
                log_train(step, x, y, bregman_energy, std_energy, bp_loss)
            if args.max_steps is not None and step >= args.max_steps:
                break

        if not log_every_step:
            log_train(epoch + 1, x, y, bregman_energy, std_energy, bp_loss)
        log_eval(epoch + 1, x, y, bregman_energy, std_energy, bp_loss)

        if args.max_steps is not None and step >= args.max_steps:
            break

    metrics = evaluate_models(
        current_models(),
        test_loader,
        max_batches=None,
        jpc_loss=std_pc_loss,
    )
    print(
        f"final  Bregman test={metrics['bregman'][0]:.4f} acc={metrics['bregman'][1]:.3f}  |  "
        f"StdPC test={metrics['std_pc'][0]:.4f} acc={metrics['std_pc'][1]:.3f}  |  "
        f"BP test={metrics['bp'][0]:.4f} acc={metrics['bp'][1]:.3f}"
    )

    history = _host_history(history)
    history["log_every"] = np.asarray(args.log_every)

    with open(os.path.join(save_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "seed": seed,
                "dataset": args.dataset,
                "n_steps": step,
                "width": width,
                "n_hidden": args.n_hidden,
                "act_fn": args.act_fn,
                "output_loss": args.output_loss,
                "init_scale": args.init_scale,
                "param_type": param_type,
                "gamma_0": gamma_0,
                "param_lr": param_lr,
                "scaled_param_lr": lr,
                "activity_lr": activity_lr,
                "n_infer_iters": n_infer_iters,
                "param_optim": args.param_optim,
                "log_every": args.log_every,
                "bregman_final_test_loss": metrics["bregman"][0],
                "bregman_final_test_acc": metrics["bregman"][1],
                "std_pc_final_test_loss": metrics["std_pc"][0],
                "std_pc_final_test_acc": metrics["std_pc"][1],
                "bp_final_test_loss": metrics["bp"][0],
                "bp_final_test_acc": metrics["bp"][1],
            },
            f,
            indent=2,
        )
    plot_metrics(history, save_dir)
    np.savez(
        os.path.join(save_dir, "history.npz"),
        **{k: np.asarray(v) for k, v in history.items()},
    )
    return history


if __name__ == "__main__":
    args = parse_args()
    base = Path(args.save_dir) if args.save_dir else default_save_dir(args.dataset)
    for param_type in args.param_type:
        for gamma_0 in args.gamma_0:
            for width in args.width:
                for param_lr in args.param_lr:
                    for activity_lr in args.activity_lr:
                        for n_infer_iters in args.n_infer_iters:
                            for seed in range(args.n_seeds):
                                train(
                                    args,
                                    seed,
                                    width,
                                    param_type,
                                    gamma_0,
                                    param_lr,
                                    activity_lr,
                                    n_infer_iters,
                                    run_save_dir(
                                        base,
                                        args.param_type,
                                        param_type,
                                        args.gamma_0,
                                        gamma_0,
                                        args.width,
                                        width,
                                        args.param_lr,
                                        param_lr,
                                        args.activity_lr,
                                        activity_lr,
                                        args.n_infer_iters,
                                        n_infer_iters,
                                        args.n_seeds,
                                        seed,
                                    ),
                                )
    plot_sweep(base)
