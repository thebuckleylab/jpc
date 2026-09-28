"""Width×depth sweep: residual μPC Bregman PC → BP as N ≫ L."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
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
    bregman_pc_step_with_bp_cosine,
    clone_eqx,
    init_jpc_opt_state,
    jpc_loss_id,
    standard_pc_step_with_bp_cosine,
)
from experiments.bregman_pc.train import create_toy_dataset, make_param_optim, set_seed
from experiments.datasets import get_dataloaders


_ARRAY_NAMES = (
    "bregman_grad_cosine_similarities",
    "std_pc_grad_cosine_similarities",
    "bregman_energies",
    "std_pc_energies",
    "bregman_train_losses",
    "std_pc_train_losses",
    "bp_losses",
)


def _optim_name(param_optim_id: str) -> str:
    if param_optim_id in ("gd", "sgd"):
        return "sgd"
    if param_optim_id == "adam":
        return "adam"
    raise ValueError(f"Unknown param optim {param_optim_id!r}.")


def setup_run_dir(
    results_dir,
    input_dim,
    n_samples,
    n_hidden,
    use_skips,
    act_fn,
    param_type,
    param_optim_id,
    param_lr,
    gamma_0,
    n_train_iters,
    n_infer_iters,
    activity_lr,
    width,
    loss_id,
    seed,
) -> Path:
    return Path(
        results_dir,
        f"{input_dim}_input_dim",
        f"{n_samples}_n_samples",
        f"{n_hidden}_n_hidden",
        f"{use_skips}_use_skips",
        f"{act_fn}_act_fn",
        f"{param_type}_param_type",
        f"{param_optim_id}_param_optim",
        f"{param_lr}_param_lr",
        f"{gamma_0}_gamma_0",
        f"{n_train_iters}_n_train_iters",
        f"{n_infer_iters}_n_infer_iters",
        f"{activity_lr}_activity_lr",
        f"{width}_width",
        f"{loss_id}_loss_id",
        str(seed),
    )


def run_is_complete(save_dir: Path) -> bool:
    needed = ("metrics.json", "grad_cosine_similarities.npy", "energies.npy", "bp_losses.npy")
    return all((save_dir / name).is_file() for name in needed)


def train_width(
    key,
    x,
    y,
    width: int,
    n_hidden: int,
    act_fn: str,
    output_loss: str,
    param_type: str,
    gamma_0: float,
    param_optim_id: str,
    param_lr: float,
    activity_lr: float,
    n_infer_iters: int,
    n_train_iters: int,
    save_dir,
    log_every: int = 1,
    use_skips: bool = True,
    init_scale: float | None = None,
) -> dict:
    """Full-batch residual μPC: Bregman PC, standard PC, and BP from one init."""
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    input_dim = int(x.shape[-1])
    output_dim = int(y.shape[-1])
    layer_sizes = [input_dim] + [width] * n_hidden + [output_dim]
    depth = n_hidden + 1
    init_model = BregmanMLP(
        key=key,
        layer_sizes=layer_sizes,
        act_fn=act_fn,
        output_loss=output_loss,
        init_scale=init_scale,
        param_type=param_type,
        gamma=gamma_0,
        use_skips=use_skips,
    )
    bregman_model = clone_eqx(init_model)
    std_pc_model = bregman_mlp_to_jpc(init_model)
    bp_model = clone_eqx(init_model)
    std_pc_loss = jpc_loss_id(output_loss)
    optim_id = _optim_name(param_optim_id)
    lr = scaled_param_lr(
        param_type,
        optim_id,
        param_lr,
        width,
        depth,
        gamma_0,
        use_skips=use_skips,
    )
    bregman_optim = make_param_optim(optim_id, lr)
    std_pc_optim = make_param_optim(optim_id, lr)
    bp_optim = make_param_optim(optim_id, lr)
    bregman_opt_state = init_jpc_opt_state(bregman_model.layers, bregman_optim)
    std_pc_opt_state = init_jpc_opt_state(std_pc_model, std_pc_optim)
    bp_opt_state = bp_optim.init(eqx.filter(bp_model, eqx.is_array))

    logs = {name: [] for name in _ARRAY_NAMES}
    print(
        f"  N={width}  H={n_hidden}  L={depth}  skips={use_skips}  "
        f"param_type={param_type}  lr={param_lr:g} (scaled={lr:g})  "
        f"n_infer={n_infer_iters}  steps={n_train_iters}"
    )

    for t in range(n_train_iters):
        bregman_model, bregman_opt_state, bregman_energy, bregman_cos = (
            bregman_pc_step_with_bp_cosine(
                bregman_model,
                x,
                y,
                bregman_optim,
                bregman_opt_state,
                n_infer_iters,
                activity_lr,
            )
        )
        std_pc_model, std_pc_opt_state, std_energy, std_cos = (
            standard_pc_step_with_bp_cosine(
                std_pc_model,
                x,
                y,
                std_pc_optim,
                std_pc_opt_state,
                n_infer_iters,
                activity_lr,
                std_pc_loss,
            )
        )
        bp_model, bp_opt_state, _, bp_loss = update_bp(
            bp_model, x, y, bp_optim, bp_opt_state
        )
        bregman_loss, _ = evaluate_batch(bregman_model, x, y)
        std_loss, _ = evaluate_jpc_batch(std_pc_model, x, y, std_pc_loss)
        logs["bregman_grad_cosine_similarities"].append(bregman_cos)
        logs["std_pc_grad_cosine_similarities"].append(std_cos)
        logs["bregman_energies"].append(bregman_energy)
        logs["std_pc_energies"].append(std_energy)
        logs["bregman_train_losses"].append(bregman_loss)
        logs["std_pc_train_losses"].append(std_loss)
        logs["bp_losses"].append(bp_loss)
        if log_every > 0 and (t % log_every == 0 or t + 1 == n_train_iters):
            print(
                f"    t={t:4d}  cos_B={float(bregman_cos):.4f}  "
                f"cos_S={float(std_cos):.4f}  "
                f"E_B={float(bregman_energy):.4f}  L_BP={float(bp_loss):.4f}"
            )

    result = {
        name: np.asarray(jax.device_get(jnp.stack(vals))) for name, vals in logs.items()
    }
    np.save(save_dir / "grad_cosine_similarities.npy", result["bregman_grad_cosine_similarities"])
    np.save(save_dir / "std_pc_grad_cosine_similarities.npy", result["std_pc_grad_cosine_similarities"])
    np.save(save_dir / "energies.npy", result["bregman_energies"])
    np.save(save_dir / "std_pc_energies.npy", result["std_pc_energies"])
    np.save(save_dir / "train_losses.npy", result["bregman_train_losses"])
    np.save(save_dir / "std_pc_train_losses.npy", result["std_pc_train_losses"])
    np.save(save_dir / "bp_losses.npy", result["bp_losses"])
    np.save(save_dir / "losses.npy", result["bp_losses"])
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "width": width,
                "n_hidden": n_hidden,
                "depth": depth,
                "use_skips": use_skips,
                "act_fn": act_fn,
                "output_loss": output_loss,
                "param_type": param_type,
                "gamma_0": gamma_0,
                "param_optim": param_optim_id,
                "param_lr": param_lr,
                "scaled_param_lr": lr,
                "activity_lr": activity_lr,
                "n_infer_iters": n_infer_iters,
                "n_train_iters": n_train_iters,
                "bregman_cos_t0": float(result["bregman_grad_cosine_similarities"][0]),
                "std_pc_cos_t0": float(result["std_pc_grad_cosine_similarities"][0]),
                "bregman_cos_final": float(result["bregman_grad_cosine_similarities"][-1]),
                "std_pc_cos_final": float(result["std_pc_grad_cosine_similarities"][-1]),
                "bp_loss_t0": float(result["bp_losses"][0]),
                "bregman_energy_t0": float(result["bregman_energies"][0]),
            },
            f,
            indent=2,
        )
    return result


def load_dataset(args, key):
    if args.dataset == "toy":
        x, y = create_toy_dataset(key, args.input_dim, args.n_samples)
        x = jnp.asarray(x.T)
        y = y[:, None]
        return x, y, args.input_dim, 1
    train_loader, _ = get_dataloaders(args.dataset, args.n_samples, flatten=True)
    img_batch, label_batch = next(iter(train_loader))
    x = jnp.asarray(img_batch.numpy())
    y = jnp.asarray(label_batch.numpy())
    return x, y, int(x.shape[-1]), int(y.shape[-1])


def parse_args():
    p = argparse.ArgumentParser(
        description="Residual μPC Bregman PC → BP width×depth sweep"
    )
    p.add_argument("--results-dir", type=str, default="results/mupc_bp_convergence")
    p.add_argument("--plot-dir", type=str, default="width_vs_depth_plots")
    p.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["toy", "MNIST", "Fashion-MNIST", "CIFAR10"],
    )
    p.add_argument("--input-dim", type=int, default=40)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--act-fn", type=str, default="tanh", choices=["tanh", "sigmoid"])
    p.add_argument(
        "--param-types",
        type=str,
        nargs="+",
        default=["mupc"],
        choices=["mupc", "sp"],
    )
    p.add_argument("--use-skips", type=int, nargs="+", default=[1], choices=[0, 1])
    p.add_argument("--param-optim", type=str, default="adam", choices=["gd", "sgd", "adam"])
    p.add_argument("--param-lr", type=float, default=1e-3)
    p.add_argument("--gamma-0s", type=float, nargs="+", default=[1.0])
    p.add_argument("--n-train-iters", type=int, default=100)
    p.add_argument(
        "--output-loss",
        type=str,
        default="mse",
        choices=["mse", "ce", "bregman"],
    )
    p.add_argument(
        "--n-infer-iters",
        type=int,
        default=None,
        help="Inference steps. Default: 100 × n_hidden (limits-paper optim mode).",
    )
    p.add_argument("--activity-lrs", type=float, nargs="+", default=[0.3])
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--n-hiddens", type=int, nargs="+", default=[1, 3, 7, 15, 31])
    p.add_argument("--widths", type=int, nargs="+", default=[2, 8, 32, 128, 512, 2048])
    p.add_argument("--init-scale", type=float, default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training and plot whatever is already in --results-dir.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plotting (limits_paper/plot_width_vs_depth_results.py style)
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    import matplotlib.pyplot as plt

    usetex = shutil.which("latex") is not None
    rc = {
        "text.usetex": usetex,
        "font.family": "serif",
        "axes.unicode_minus": False,
    }
    if usetex:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    else:
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)
    return plt


def _load_npy_safe(path):
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return np.array(
            [x.item() if hasattr(x, "item") else float(x) for x in arr]
        )
    return np.asarray(arr).reshape(-1)


def discover_configs(results_dir, seed=None, param_type=None, use_skips=None):
    """Map (width, n_hidden) → directory for matching cosine result files."""
    configs = {}
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    for root, _, files in os.walk(results_dir):
        if "grad_cosine_similarities.npy" not in files:
            continue
        if seed is not None and root.split(os.sep)[-1] != str(seed):
            continue
        if param_type_str is not None and param_type_str not in root:
            continue
        if use_skips_str is not None and use_skips_str not in root:
            continue
        width = n_hidden = None
        for part in root.split(os.sep):
            if part.endswith("_width"):
                try:
                    width = int(part.replace("_width", ""))
                except ValueError:
                    pass
            elif part.endswith("_n_hidden"):
                try:
                    n_hidden = int(part.replace("_n_hidden", ""))
                except ValueError:
                    pass
        if width is not None:
            configs[(width, n_hidden)] = root
    return configs


def _collect_metric(results_dir, filename, seeds, param_type, use_skips, time_step=0):
    """Average ``filename[time_step]`` over seeds for each (width, n_hidden)."""
    points = []
    if not seeds:
        return points
    base = discover_configs(
        results_dir, seed=seeds[0], param_type=param_type, use_skips=use_skips
    )
    for (width, n_hidden) in base:
        vals = []
        for seed in seeds:
            cfgs = discover_configs(
                results_dir, seed=seed, param_type=param_type, use_skips=use_skips
            )
            if (width, n_hidden) not in cfgs:
                continue
            arr = _load_npy_safe(os.path.join(cfgs[(width, n_hidden)], filename))
            if arr is None or len(arr) == 0:
                continue
            idx = min(time_step, len(arr) - 1)
            vals.append(float(arr[idx]))
        if vals:
            L = (n_hidden + 1) if n_hidden is not None else 1
            points.append((width, L, float(np.mean(vals))))
    return points


def _available_seeds(results_dir, param_type=None, use_skips=None):
    seeds = set()
    param_type_str = f"{param_type}_param_type" if param_type else None
    use_skips_str = f"{use_skips}_use_skips" if use_skips is not None else None
    for root, _, files in os.walk(results_dir):
        if "grad_cosine_similarities.npy" not in files:
            continue
        if param_type_str is not None and param_type_str not in root:
            continue
        if use_skips_str is not None and use_skips_str not in root:
            continue
        try:
            seeds.add(int(root.split(os.sep)[-1]))
        except ValueError:
            pass
    return sorted(seeds)


def _heatmap(ax, plt, points, vmin, vmax, cmap, cbar_label, title=None):
    if not points:
        return
    widths = np.array([p[0] for p in points])
    depths = np.array([p[1] for p in points])
    values = np.array([p[2] for p in points])
    unique_w = np.sort(np.unique(widths))
    unique_d = np.sort(np.unique(depths))
    grid = np.full((len(unique_w), len(unique_d)), np.nan)
    for w, d, v in zip(widths, depths, values):
        grid[np.where(unique_w == w)[0][0], np.where(unique_d == d)[0][0]] = v
    im = ax.imshow(
        np.nan_to_num(grid, nan=0.0).T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(unique_w)))
    ax.set_xticklabels([f"${w}$" for w in unique_w], fontsize=18)
    ax.set_yticks(np.arange(len(unique_d)))
    ax.set_yticklabels([f"${d}$" for d in unique_d], fontsize=18)
    ax.set_xlabel("Width $N$", fontsize=22, labelpad=12)
    ax.set_ylabel("Depth $L$", fontsize=22, labelpad=12)
    if title:
        ax.set_title(title, fontsize=22, pad=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=20, rotation=270, labelpad=28)
    cbar.ax.tick_params(labelsize=16)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scatter3d(ax, points, zlabel, zlim=None):
    widths = np.array([p[0] for p in points])
    depths = np.array([p[1] for p in points])
    values = np.array([p[2] for p in points])
    ax.scatter(
        np.log2(widths),
        np.log2(depths),
        values,
        color="#1E88E5",
        s=160,
        alpha=0.9,
        depthshade=True,
        edgecolors="darkblue",
        linewidths=1,
    )
    ax.set_xlabel("$N$", fontsize=20, labelpad=12)
    ax.set_ylabel("$L$", fontsize=20, labelpad=12)
    ax.set_zlabel(zlabel, fontsize=16, labelpad=12)
    w_ticks = np.unique(np.round(np.log2(widths)).astype(int))
    ax.set_xticks(w_ticks)
    ax.set_xticklabels([f"$2^{{{p}}}$" for p in w_ticks])
    d_ticks = np.unique(np.round(np.log2(depths)).astype(int))
    ax.set_yticks(d_ticks)
    ax.set_yticklabels([f"$2^{{{p}}}$" for p in d_ticks])
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.tick_params(axis="both", labelsize=14)
    ax.invert_xaxis()


def plot_width_depth_results(results_dir, plot_dir, param_type=None, use_skips=True):
    """Cosine heatmaps/3D scatter and PC-energy vs BP-loss surfaces vs N and L."""
    plt = _setup_matplotlib()
    os.makedirs(plot_dir, exist_ok=True)
    seeds = _available_seeds(results_dir, param_type=param_type, use_skips=use_skips)
    if not seeds:
        print(f"  No cosine results under {results_dir}")
        return
    print(f"  Plotting seeds {seeds}  param_type={param_type}  use_skips={use_skips}")

    for time_step in (0,):
        bregman_cos = _collect_metric(
            results_dir,
            "grad_cosine_similarities.npy",
            seeds,
            param_type,
            use_skips,
            time_step,
        )
        std_cos = _collect_metric(
            results_dir,
            "std_pc_grad_cosine_similarities.npy",
            seeds,
            param_type,
            use_skips,
            time_step,
        )
        if bregman_cos:
            fig, ax = plt.subplots(figsize=(8, 6))
            _heatmap(
                ax,
                plt,
                bregman_cos,
                vmin=0,
                vmax=1,
                cmap="viridis_r",
                cbar_label=r"$\cos(\nabla\mathcal{F}_{\mathrm{B}}^*, \nabla\mathcal{L})$",
                title=f"Bregman PC  $t={time_step}$",
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(plot_dir, f"bregman_cosine_heatmap_t{time_step}.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            _scatter3d(
                ax,
                bregman_cos,
                r"$\cos(\nabla\mathcal{F}_{\mathrm{B}}^*, \nabla\mathcal{L})$",
                zlim=(0, 1),
            )
            fig.savefig(
                os.path.join(plot_dir, f"bregman_cosine_vs_width_depth_t{time_step}.pdf"),
                bbox_inches="tight",
                pad_inches=0.6,
            )
            plt.close(fig)
            print(f"  Bregman cosine range: {min(p[2] for p in bregman_cos):.4f}–{max(p[2] for p in bregman_cos):.4f}")

        if std_cos:
            fig, ax = plt.subplots(figsize=(8, 6))
            _heatmap(
                ax,
                plt,
                std_cos,
                vmin=0,
                vmax=1,
                cmap="viridis_r",
                cbar_label=r"$\cos(\nabla\mathcal{F}^*, \nabla\mathcal{L})$",
                title=f"Standard PC  $t={time_step}$",
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(plot_dir, f"std_pc_cosine_heatmap_t{time_step}.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

    energies = _collect_metric(
        results_dir, "energies.npy", seeds, param_type, use_skips, 0
    )
    bp_losses = _collect_metric(
        results_dir, "bp_losses.npy", seeds, param_type, use_skips, 0
    )
    if energies and bp_losses:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        e_w = np.array([p[0] for p in energies])
        e_l = np.array([p[1] for p in energies])
        ax.scatter(
            np.log2(e_w),
            np.log2(e_l),
            [p[2] for p in energies],
            color="blue",
            s=120,
            alpha=0.8,
            label=r"$\mathcal{F}_{\mathrm{B}}^*$",
        )
        b_w = np.array([p[0] for p in bp_losses])
        b_l = np.array([p[1] for p in bp_losses])
        ax.scatter(
            np.log2(b_w),
            np.log2(b_l),
            [p[2] for p in bp_losses],
            color="red",
            s=120,
            alpha=0.8,
            marker="^",
            label=r"$\mathcal{L}$ (BP)",
        )
        ax.set_xlabel("$N$", fontsize=20, labelpad=12)
        ax.set_ylabel("$L$", fontsize=20, labelpad=12)
        ax.set_zlabel(r"$l(\boldsymbol{\theta})$", fontsize=18, labelpad=12)
        ax.legend(fontsize=14)
        ax.invert_xaxis()
        fig.savefig(
            os.path.join(plot_dir, "energy_vs_bp_loss.pdf"),
            bbox_inches="tight",
            pad_inches=0.6,
        )
        plt.close(fig)

    print(f"  Saved plots to {plot_dir}")


def main():
    args = parse_args()
    # f64 for large N, L (also when a submit script shards to one (width, depth) per process)
    jax.config.update("jax_enable_x64", True)

    results_dir = Path(args.results_dir)
    if not args.plot_only:
        os.makedirs(results_dir, exist_ok=True)
        for seed in range(args.n_seeds):
            print(f"\nseed {seed}")
            set_seed(seed)
            key = jax.random.PRNGKey(seed)
            data_key, model_key = jax.random.split(key)
            x, y, input_dim, _ = load_dataset(args, data_key)
            print(f"  dataset={args.dataset}  P={x.shape[0]}  D={input_dim}")

            for n_hidden in args.n_hiddens:
                n_infer = (
                    args.n_infer_iters
                    if args.n_infer_iters is not None
                    else n_hidden * 100
                )
                for use_skips in (bool(v) for v in args.use_skips):
                    for gamma_0 in args.gamma_0s:
                        for param_type in args.param_types:
                            for activity_lr in args.activity_lrs:
                                for width in args.widths:
                                    save_dir = setup_run_dir(
                                        results_dir,
                                        input_dim,
                                        args.n_samples,
                                        n_hidden,
                                        use_skips,
                                        args.act_fn,
                                        param_type,
                                        args.param_optim,
                                        args.param_lr,
                                        gamma_0,
                                        args.n_train_iters,
                                        n_infer,
                                        activity_lr,
                                        width,
                                        args.output_loss,
                                        seed,
                                    )
                                    if args.resume and run_is_complete(save_dir):
                                        print(f"  skip existing {save_dir}")
                                        continue
                                    width_key = jax.random.fold_in(model_key, width + 1000 * n_hidden)
                                    train_width(
                                        width_key,
                                        x,
                                        y,
                                        width=width,
                                        n_hidden=n_hidden,
                                        act_fn=args.act_fn,
                                        output_loss=args.output_loss,
                                        param_type=param_type,
                                        gamma_0=gamma_0,
                                        param_optim_id=args.param_optim,
                                        param_lr=args.param_lr,
                                        activity_lr=activity_lr,
                                        n_infer_iters=n_infer,
                                        n_train_iters=args.n_train_iters,
                                        save_dir=save_dir,
                                        log_every=args.log_every,
                                        use_skips=use_skips,
                                        init_scale=args.init_scale,
                                    )

    if not args.no_plot:
        for param_type in args.param_types:
            for use_skips in (bool(v) for v in args.use_skips):
                plot_dir = (
                    Path(args.plot_dir)
                    / args.dataset
                    / f"{use_skips}_use_skips"
                    / f"{param_type}_param_type"
                )
                plot_width_depth_results(
                    results_dir,
                    plot_dir,
                    param_type=param_type,
                    use_skips=use_skips,
                )


if __name__ == "__main__":
    main()
