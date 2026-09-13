"""Plot toy µPC vs BP results saved by ``train_toy.py``.

    python plot_toy.py
    python plot_toy.py --plot_widths 256 512 1024
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from experiments.limits_paper import plot_gamma0_sweep as gamma_plots
from experiments.limits_paper import plot_toy_results as toy_plots
from experiments.limits_paper.utils import setup_bp_experiment, setup_pc_experiment


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lower = str(value).lower()
    if lower in ("true", "1", "yes"):
        return True
    if lower in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean {value!r}; use True/False."
    )


def add_common_args(parser):
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/toy_energy_scaled",
    )
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument(
        "--act_fn",
        type=str,
        default="linear",
        choices=["linear", "tanh", "relu"],
    )
    parser.add_argument(
        "--param_types",
        type=str,
        nargs="+",
        default=["mupc"],
        choices=["mupc", "sp"],
    )
    parser.add_argument(
        "--use_skips",
        type=parse_bool,
        nargs="+",
        default=[False],
    )
    parser.add_argument(
        "--param_optim",
        type=str,
        default="gd",
        choices=["gd", "adam", "sgd_momentum"],
    )
    parser.add_argument("--param_lr", type=float, default=0.05)
    parser.add_argument("--gamma_0s", type=float, nargs="+", default=[1.0])
    parser.add_argument("--n_train_iters", type=int, default=100)
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="closed_form",
        choices=["optim", "closed_form"],
    )
    parser.add_argument("--n_infer_iters", type=int, default=20)
    parser.add_argument("--activity_lrs", type=float, nargs="+", default=[5e-1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_hiddens", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--log_x_scale", action="store_true", default=False)
    return parser


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot toy µPC vs BP results from train_toy.py."
    )
    add_common_args(parser)
    parser.add_argument(
        "--plot_widths",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Subset of --widths for losses.pdf, losses_and_energies*.pdf, "
            "and grads_cosine_similarities.pdf (default: all). "
            "Rescaling plots keep every width."
        ),
    )
    return parser.parse_args()


def bp_dmft_loss_path(results_dir, gamma_0, n_hidden, seed):
    return os.path.join(
        results_dir,
        f"dmft_loss_{gamma_0}_gamma_0_{n_hidden}_n_hidden_seed_{seed}.npy",
    )


def load_bp_dmft_loss(results_dir, gamma_0, n_hidden, seed):
    path = bp_dmft_loss_path(results_dir, gamma_0, n_hidden, seed)
    return toy_plots._load_npy_safe(path)


def mup_loss_scale(param_type, gamma_0, width):
    """µP factor ``γ² N`` that BP GD puts in the LR (1 for SP)."""
    if param_type == "sp":
        return 1.0
    return float(gamma_0) ** 2 * float(width)


def pc_energies_on_mse_scale(data, param_type, *, sweep):
    """Copy of ``data`` with ``F*`` divided by ``γ² N`` for MSE-scale overlays."""
    data = dict(data)
    energies = data.get("pc_energies") or {}
    scaled = {}
    if sweep == "width":
        gamma_0 = data["gamma_0"]
        for width, arr in energies.items():
            scaled[width] = (
                np.asarray(arr, dtype=float)
                / mup_loss_scale(param_type, gamma_0, width)
            )
    else:
        width = data["width"]
        for gamma_0, arr in energies.items():
            scaled[gamma_0] = (
                np.asarray(arr, dtype=float)
                / mup_loss_scale(param_type, gamma_0, width)
            )
    data["pc_energies"] = scaled
    return data


def data_for_widths(data, widths):
    """Copy of width-sweep ``data`` containing only ``widths``."""
    if not widths:
        return data
    width_set = set(widths)
    data = dict(data)
    data["widths"] = [w for w in data["widths"] if w in width_set]
    for key in (
        "pc_energies",
        "pc_train_losses",
        "pc_rescalings",
        "bp_losses",
        "pc_grads",
        "bp_grads",
        "grad_cosine_similarities",
    ):
        mapping = data.get(key)
        if isinstance(mapping, dict):
            data[key] = {w: v for w, v in mapping.items() if w in width_set}
    return data


def experiment_dirs(
    results_dir,
    *,
    input_dim,
    n_samples,
    n_hidden,
    use_skips,
    act_fn,
    param_type,
    param_optim,
    param_lr,
    gamma_0,
    n_train_iters,
    infer_mode,
    n_infer_iters,
    activity_lr,
    width,
    loss_id,
    seed,
):
    pc_dir = setup_pc_experiment(
        results_dir=results_dir,
        input_dim=input_dim,
        n_samples=n_samples,
        n_hidden=n_hidden,
        use_skips=use_skips,
        act_fn=act_fn,
        param_type=param_type,
        param_lr=param_lr,
        gamma_0=gamma_0,
        param_optim_id=param_optim,
        n_train_iters=n_train_iters,
        infer_mode=infer_mode,
        n_infer_iters=n_infer_iters,
        activity_lr=activity_lr,
        width=width,
        loss_id=loss_id,
        seed=seed,
    )
    bp_dir = setup_bp_experiment(
        results_dir=results_dir,
        input_dim=input_dim,
        n_samples=n_samples,
        n_hidden=n_hidden,
        use_skips=use_skips,
        act_fn=act_fn,
        param_type=param_type,
        optim_id=param_optim,
        param_lr=param_lr,
        gamma_0=gamma_0,
        n_train_iters=n_train_iters,
        width=width,
        loss_id=loss_id,
        seed=seed,
    )
    return pc_dir, bp_dir


def load_run_metrics(pc_dir, bp_dir):
    """Load PC/BP arrays for one (width, gamma) run from known save dirs."""
    metrics = {}
    mapping = {
        "pc_energies": (pc_dir, "energies.npy", True),
        "pc_train_losses": (pc_dir, "train_losses.npy", True),
        "pc_rescalings": (pc_dir, "loss_rescalings.npy", True),
        "grad_cosine_similarities": (
            pc_dir, "grad_cosine_similarities.npy", True
        ),
        "bp_losses": (bp_dir, "losses.npy", True),
    }
    for key, (directory, filename, flatten) in mapping.items():
        arr = toy_plots._load_npy_safe(
            os.path.join(directory, filename), flatten=flatten
        )
        if arr is not None:
            metrics[key] = arr
    return metrics


def load_width_data(args, *, seed, n_hidden, use_skips, param_type, activity_lr, gamma_0):
    data = {
        "widths": list(args.widths),
        "pc_rescalings": {},
        "pc_energies": {},
        "pc_train_losses": {},
        "pc_grads": {},
        "bp_losses": {},
        "bp_grads": {},
        "grad_cosine_similarities": {},
        "dmft_loss": None,
        "gamma_0": gamma_0,
    }
    for width in args.widths:
        pc_dir, bp_dir = experiment_dirs(
            args.results_dir,
            input_dim=args.input_dim,
            n_samples=args.n_samples,
            n_hidden=n_hidden,
            use_skips=use_skips,
            act_fn=args.act_fn,
            param_type=param_type,
            param_optim=args.param_optim,
            param_lr=args.param_lr,
            gamma_0=gamma_0,
            n_train_iters=args.n_train_iters,
            infer_mode=args.infer_mode,
            n_infer_iters=args.n_infer_iters,
            activity_lr=activity_lr,
            width=width,
            loss_id="mse",
            seed=seed,
        )
        metrics = load_run_metrics(pc_dir, bp_dir)
        for key in (
            "pc_energies",
            "pc_train_losses",
            "pc_rescalings",
            "bp_losses",
            "grad_cosine_similarities",
        ):
            if key in metrics:
                data[key][width] = metrics[key]
    if not use_skips:
        data["dmft_loss"] = load_bp_dmft_loss(
            args.results_dir, gamma_0, n_hidden, seed
        )
    return data


def load_gamma_data(args, *, seed, n_hidden, use_skips, param_type, activity_lr, width):
    data = {
        "gamma_0s": list(args.gamma_0s),
        "width": width,
        "pc_rescalings": {},
        "pc_energies": {},
        "pc_train_losses": {},
        "pc_grads": {},
        "bp_losses": {},
        "bp_grads": {},
        "grad_cosine_similarities": {},
        "dmft_loss": None,
        "dmft_losses": {},
    }
    for gamma_0 in args.gamma_0s:
        pc_dir, bp_dir = experiment_dirs(
            args.results_dir,
            input_dim=args.input_dim,
            n_samples=args.n_samples,
            n_hidden=n_hidden,
            use_skips=use_skips,
            act_fn=args.act_fn,
            param_type=param_type,
            param_optim=args.param_optim,
            param_lr=args.param_lr,
            gamma_0=gamma_0,
            n_train_iters=args.n_train_iters,
            infer_mode=args.infer_mode,
            n_infer_iters=args.n_infer_iters,
            activity_lr=activity_lr,
            width=width,
            loss_id="mse",
            seed=seed,
        )
        metrics = load_run_metrics(pc_dir, bp_dir)
        for key in (
            "pc_energies",
            "pc_train_losses",
            "pc_rescalings",
            "bp_losses",
            "grad_cosine_similarities",
        ):
            if key in metrics:
                data[key][gamma_0] = metrics[key]
    if not use_skips:
        for gamma_0 in args.gamma_0s:
            dmft = load_bp_dmft_loss(
                args.results_dir, gamma_0, n_hidden, seed
            )
            if dmft is not None:
                data["dmft_losses"][gamma_0] = dmft
                if data["dmft_loss"] is None:
                    data["dmft_loss"] = dmft
    return data


def plot_inv_rescaling_vs_width(data, plot_dir, gamma_0, param_type):
    """Plot ``1/s(θ_0)`` vs width, with infinite-width theory ``γ² N``."""
    rescalings = data.get("pc_rescalings") or {}
    widths = sorted(w for w in data["widths"] if w in rescalings)
    first = []
    valid = []
    for width in widths:
        values = np.asarray(rescalings[width]).flatten()
        if values.size == 0:
            continue
        s0 = float(values[0])
        if s0 == 0.0 or not np.isfinite(s0):
            continue
        first.append(1.0 / s0)
        valid.append(width)
    if not valid:
        print("  Skipping 1/s vs N: no rescaling data (need closed_form).")
        return

    plt.figure(figsize=toy_plots.FIG_SIZE)
    plt.scatter(
        valid,
        first,
        s=300,
        alpha=toy_plots.ALPHA,
        color="#1E88E5",
        label="Data",
        zorder=3,
    )
    if param_type != "sp":
        theory_widths = np.logspace(np.log10(min(valid)), np.log10(max(valid)), 100)
        theory = (gamma_0 ** 2) * theory_widths
        plt.plot(
            theory_widths,
            theory,
            "--",
            color="black",
            linewidth=toy_plots.LINE_WIDTH,
            label=r"Theory: $\Theta(N)$",
            alpha=0.8,
            zorder=2,
        )
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("$N$", fontsize=toy_plots.FONT_SIZES["label"], labelpad=toy_plots.LABEL_PAD)
    plt.ylabel(
        r"$1/s(\boldsymbol{\theta}_0)$",
        fontsize=toy_plots.FONT_SIZES["label"],
        labelpad=toy_plots.LABEL_PAD,
    )
    plt.legend(fontsize=toy_plots.FONT_SIZES["legend"])
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis="both", labelsize=toy_plots.FONT_SIZES["tick"])
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    toy_plots.save_plot(plot_dir, "inv_rescaling_vs_width.pdf", None)


def plot_losses_and_energies_logy(data, plot_dir, *, sweep, log_x_scale, n_hidden=None):
    """PC energies and BP losses on a log-y axis (same series as the linear overlay).

    Callers should pass PC energies already divided by ``γ² N``.
    """
    if sweep == "width":
        plt.figure(figsize=(12.5, 6))
        max_width = max(data["widths"])
        pc_legend_color = "#4A90E2"
        bp_color = "#DC143C"
        widths_list = sorted([w for w in data["widths"] if w in data["pc_energies"]])
        if widths_list:
            blues_cmap = plt.get_cmap("Blues")
            n_widths = len(widths_list)
            for idx, width in enumerate(widths_list):
                energies = np.array(data["pc_energies"][width]).flatten()
                iterations = np.arange(1, len(energies) + 1)
                color = blues_cmap(toy_plots.get_color_val(idx, n_widths, "Blues"))
                plt.plot(
                    iterations,
                    energies,
                    "-",
                    alpha=toy_plots.ALPHA,
                    linewidth=toy_plots.LINE_WIDTH,
                    color=color,
                )
        if max_width in data["bp_losses"]:
            bp_loss = np.array(data["bp_losses"][max_width]).flatten()
            iterations = np.arange(1, len(bp_loss) + 1)
            plt.plot(
                iterations,
                bp_loss,
                "-",
                color=bp_color,
                linewidth=toy_plots.LINE_WIDTH,
                alpha=toy_plots.ALPHA,
            )
        legend_handles = []
        legend_labels = []
        if widths_list:
            legend_handles.append(
                plt.Line2D(
                    [0], [0], color=pc_legend_color,
                    linewidth=toy_plots.LINE_WIDTH, alpha=toy_plots.ALPHA,
                )
            )
            legend_labels.append(
                r"$\mathcal{F}^*(\boldsymbol{\theta})/N$ (PC)"
            )
        if max_width in data["bp_losses"]:
            legend_handles.append(
                plt.Line2D(
                    [0], [0], color=bp_color,
                    linewidth=toy_plots.LINE_WIDTH, alpha=toy_plots.ALPHA,
                )
            )
            legend_labels.append(r"$\mathcal{L}(\boldsymbol{\theta})$ (BP)")
        gray_cmap = plt.get_cmap("Greys")
        all_widths = sorted(
            set(widths_list + ([max_width] if max_width in data["bp_losses"] else []))
        )
        n_all = len(all_widths)
        for idx, width in enumerate(all_widths):
            gray_val = 0.3 + (idx / max(n_all - 1, 1)) * 0.5 if n_all > 1 else 0.5
            legend_handles.append(
                plt.Line2D([0], [0], color=gray_cmap(gray_val), linewidth=toy_plots.LINE_WIDTH)
            )
            legend_labels.append(f"$N = {width}$")
        if data.get("dmft_loss") is not None:
            dmft_loss = np.array(data["dmft_loss"]).flatten()
            iterations = np.arange(1, len(dmft_loss) + 1)
            plt.plot(
                iterations,
                dmft_loss,
                "--",
                color="black",
                linewidth=4,
                alpha=0.8,
            )
            legend_handles.append(
                plt.Line2D(
                    [0], [0], color="black", linewidth=4,
                    linestyle="--", alpha=0.8,
                )
            )
            legend_labels.append(r"BP theory ($N \rightarrow \infty$)")
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xlabel("$t$", fontsize=toy_plots.FONT_SIZES["label"], labelpad=toy_plots.LABEL_PAD)
        plt.ylabel(r"$l(\boldsymbol{\theta}_t)$", fontsize=toy_plots.FONT_SIZES["label"], labelpad=toy_plots.LABEL_PAD)
        if log_x_scale:
            plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        if legend_handles:
            plt.legend(
                handles=legend_handles,
                labels=legend_labels,
                fontsize=toy_plots.FONT_SIZES["legend"],
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.tick_params(axis="both", labelsize=toy_plots.FONT_SIZES["tick"])
        toy_plots.save_plot(plot_dir, "losses_and_energies_logy.pdf", None)
        return

    plt.figure(figsize=(10, 6))
    pc_gammas = sorted([g for g in data["gamma_0s"] if g in data["pc_energies"]])
    bp_gammas = sorted([g for g in data["gamma_0s"] if g in data["bp_losses"]])
    all_gammas = sorted(set(pc_gammas + bp_gammas))
    n_gammas = len(all_gammas)
    if all_gammas:
        blue_cmap = plt.get_cmap("Blues")
        red_cmap = plt.get_cmap("Reds")
        for gamma_0 in pc_gammas:
            energies = np.array(data["pc_energies"][gamma_0]).flatten()
            iterations = np.arange(1, len(energies) + 1)
            idx = all_gammas.index(gamma_0)
            color = blue_cmap(gamma_plots.get_color_val(idx, n_gammas, "Blues"))
            plt.plot(
                iterations, energies, "-",
                alpha=gamma_plots.ALPHA, linewidth=gamma_plots.LINE_WIDTH, color=color,
            )
        for gamma_0 in bp_gammas:
            bp_loss = np.array(data["bp_losses"][gamma_0]).flatten()
            iterations = np.arange(1, len(bp_loss) + 1)
            idx = all_gammas.index(gamma_0)
            color = red_cmap(gamma_plots.get_color_val(idx, n_gammas, "Reds"))
            plt.plot(
                iterations, bp_loss, "--",
                alpha=gamma_plots.ALPHA, linewidth=gamma_plots.LINE_WIDTH, color=color,
            )
    plt.xlabel("$t$", fontsize=gamma_plots.FONT_SIZES["label"], labelpad=gamma_plots.LABEL_PAD)
    plt.ylabel(
        r"$l(\boldsymbol{\theta}_t)$",
        fontsize=gamma_plots.FONT_SIZES["label"],
        labelpad=gamma_plots.LABEL_PAD,
    )
    if log_x_scale:
        plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    legend_elements = [
        plt.Line2D(
            [0], [0], color=plt.get_cmap("Blues")(0.5), linestyle="-",
            linewidth=gamma_plots.LINE_WIDTH, label="PC",
        ),
        plt.Line2D(
            [0], [0], color=plt.get_cmap("Reds")(0.5), linestyle="--",
            linewidth=gamma_plots.LINE_WIDTH,
            label=(
                rf"BP, $N = {data['width']}$"
                if data.get("width") is not None
                else "BP"
            ),
        ),
    ]
    for idx, gamma_0 in enumerate(all_gammas):
        grey_val = 0.8 - (idx / (n_gammas - 1)) * 0.6 if n_gammas > 1 else 0.5
        legend_elements.append(
            plt.Line2D(
                [0], [0], color=(grey_val, grey_val, grey_val), linestyle="-",
                linewidth=gamma_plots.LINE_WIDTH,
                label=rf"$\gamma_0 = {gamma_0}$",
            )
        )
    plt.legend(
        handles=legend_elements,
        fontsize=gamma_plots.FONT_SIZES["legend"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis="both", labelsize=gamma_plots.FONT_SIZES["tick"])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    gamma_plots.save_plot(
        plot_dir, "losses_and_energies_logy.pdf", n_hidden, add_suffix=False
    )


def plot_width_sweep(data, plot_dir, param_type, use_skips, log_x_scale, plot_widths=None):
    os.makedirs(plot_dir, exist_ok=True)
    print(f"  Width sweep plots → {plot_dir}")
    if not data["pc_train_losses"] and not data["pc_energies"]:
        print("  Warning: no width-sweep data found")
        return

    loss_data = data_for_widths(data, plot_widths)
    overlay = pc_energies_on_mse_scale(loss_data, param_type, sweep="width")
    toy_plots.plot_losses(
        loss_data, plot_dir, "Blues", None, log_x_scale, param_type, use_skips
    )
    toy_plots.plot_losses_and_energies(
        overlay,
        plot_dir,
        "Blues",
        None,
        log_x_scale,
        param_type,
        use_skips,
        pc_legend_label=r"$\mathcal{F}^*(\boldsymbol{\theta})/N$ (PC)",
    )
    plot_losses_and_energies_logy(
        overlay, plot_dir, sweep="width", log_x_scale=log_x_scale
    )
    similarities = toy_plots.calculate_cosine_similarity(loss_data)
    if similarities:
        toy_plots.plot_cosine_similarity(loss_data, similarities, plot_dir, "Blues", None)
    else:
        print("  Warning: no cosine similarity data for width sweep")
    if data["pc_rescalings"]:
        toy_plots.plot_rescalings(data, plot_dir, "Blues", None, output_dim=1)
        plot_inv_rescaling_vs_width(
            data, plot_dir, data["gamma_0"], param_type
        )


def plot_gamma_sweep(data, plot_dir, n_hidden, log_x_scale, use_skips, param_type):
    os.makedirs(plot_dir, exist_ok=True)
    print(f"  Gamma sweep plots → {plot_dir}")
    if not data["pc_train_losses"] and not data["pc_energies"]:
        print("  Warning: no gamma-sweep data found")
        return

    overlay = pc_energies_on_mse_scale(data, param_type, sweep="gamma")
    plot_theory = (not use_skips) and (
        bool(data.get("dmft_losses")) or data.get("dmft_loss") is not None
    )
    gamma_plots.plot_losses(
        data, plot_dir, "Blues", n_hidden, log_x_scale, plot_theory=plot_theory
    )
    gamma_plots.plot_losses_and_energies(
        overlay,
        plot_dir,
        "Blues",
        n_hidden,
        log_x_scale,
        plot_theory=plot_theory,
        ylabel=r"$l(\boldsymbol{\theta}_t)$",
    )
    plot_losses_and_energies_logy(
        overlay, plot_dir, sweep="gamma", log_x_scale=log_x_scale, n_hidden=n_hidden
    )
    similarities = gamma_plots.calculate_cosine_similarity(data)
    if similarities:
        gamma_plots.plot_cosine_similarity(
            data, similarities, plot_dir, "Blues", n_hidden
        )
    else:
        print("  Warning: no cosine similarity data for gamma sweep")
    if data["pc_rescalings"]:
        gamma_plots.plot_rescalings(data, plot_dir, "Blues", n_hidden, output_dim=1)


def make_plot_dir(plot_root, seed, n_hidden, use_skips, param_type, activity_lr):
    return os.path.join(
        plot_root,
        f"seed_{seed}",
        f"{n_hidden}_n_hidden",
        f"{use_skips}_use_skips",
        param_type,
        f"{activity_lr}_activity_lr",
    )


def generate_plots(args):
    data_results_dir = os.path.join(args.results_dir, f"{args.input_dim}_input_dim")
    if not os.path.isdir(data_results_dir):
        print(f"No results at {data_results_dir}; skipping plots.")
        return

    plot_root = os.path.join(args.results_dir, "plots")
    os.makedirs(plot_root, exist_ok=True)

    for seed in range(args.seed, args.seed + args.n_seeds):
        for n_hidden in args.n_hiddens:
            for use_skips in args.use_skips:
                for param_type in args.param_types:
                    for activity_lr in args.activity_lrs:
                        base = make_plot_dir(
                            plot_root,
                            seed,
                            n_hidden,
                            use_skips,
                            param_type,
                            activity_lr,
                        )
                        plot_width = (
                            len(args.widths) > 1 or len(args.gamma_0s) == 1
                        )
                        plot_gamma = len(args.gamma_0s) > 1
                        if plot_width:
                            for gamma_0 in args.gamma_0s:
                                data = load_width_data(
                                    args,
                                    seed=seed,
                                    n_hidden=n_hidden,
                                    use_skips=use_skips,
                                    param_type=param_type,
                                    activity_lr=activity_lr,
                                    gamma_0=gamma_0,
                                )
                                plot_width_sweep(
                                    data,
                                    os.path.join(
                                        base, "width_sweep", f"{gamma_0}_gamma_0"
                                    ),
                                    param_type,
                                    use_skips,
                                    args.log_x_scale,
                                    plot_widths=getattr(args, "plot_widths", None),
                                )
                        if plot_gamma:
                            for width in args.widths:
                                data = load_gamma_data(
                                    args,
                                    seed=seed,
                                    n_hidden=n_hidden,
                                    use_skips=use_skips,
                                    param_type=param_type,
                                    activity_lr=activity_lr,
                                    width=width,
                                )
                                plot_gamma_sweep(
                                    data,
                                    os.path.join(
                                        base, "gamma_sweep", f"{width}_width"
                                    ),
                                    n_hidden,
                                    args.log_x_scale,
                                    use_skips,
                                    param_type,
                                )


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
