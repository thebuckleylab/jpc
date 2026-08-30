"""Plot the equilibrated-energy rescaling as a function of width."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiments.dmft.test_equilib_energy import main as run_equilib_energy

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    }
)

FIG_SIZE = (8, 6)
FONT_SIZES = {"label": 45, "legend": 25, "tick": 35}
LABEL_PAD = 15
LINE_WIDTH = 4
ALPHA = 0.7
MARKER_SIZE = 12


def plot_rescaling_per_width(
    widths,
    rescalings_by_config,
    save_path,
    depth,
    gamma,
):
    """Plot the equilibrated-energy rescaling ``s`` as a function of width N.

    ``rescalings_by_config`` maps a legend label to a list of rescaling values,
    one per width in ``widths``. With hidden precision κ = L and output
    precision λ = γ² N L the infinite-width init value is
    ``s = 1 / (gamma^2 N)`` (using ``E||w||^2 = N``), independent of depth.
    """
    widths = np.asarray(widths)
    cmap = plt.get_cmap("viridis")
    n_curves = len(rescalings_by_config)
    colors = [
        cmap(0.35 + 0.55 * i / max(n_curves - 1, 1)) for i in range(n_curves)
    ]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for i, (label, values) in enumerate(rescalings_by_config.items()):
        y = np.asarray(values, dtype=float)
        ax.plot(
            widths,
            y,
            "-o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=colors[i],
            alpha=ALPHA,
            label=label,
        )

    # s = 1/λ + ((L-1)/L)/(γ² N) = 1/(γ² N) when λ = γ² N L and κ = L.
    theory = (
        1.0 / depth + (depth - 1) / depth
    ) / (gamma ** 2 * widths.astype(float))
    ax.plot(
        widths,
        theory,
        "--",
        linewidth=LINE_WIDTH,
        color="black",
        label=r"Theory $1/(\gamma^2 N)$",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$N$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    ax.set_ylabel(r"$s(\boldsymbol{\theta})$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=2))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax.set_xticks(widths)
    ax.grid(True, which="both", ls="-", alpha=0.4)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    ax.legend(fontsize=FONT_SIZES["legend"])
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--input_dim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--param_types", type=str, nargs="+", default=["mupc"], choices=["mupc", "sp"])

    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    # Inference parameters
    parser.add_argument("--gammas", type=float, nargs="+", default=[1])
    parser.add_argument("--activity_lrs", type=float, nargs="+", default=[5e-1])
    parser.add_argument("--n_infer_iters", type=int, default=50)

    # Loop parameters
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])

    # Tolerance parameters
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)

    # Plotting parameters
    parser.add_argument("--save_dir", type=Path, default=Path("figures/equilib_energy"))

    args = parser.parse_args()

    # Track the equilibrated-energy rescaling s per width, one curve per config.
    rescalings_by_config: dict[tuple, list[float]] = {}
    multi_gamma = len(args.gammas) > 1
    multi_param = len(args.param_types) > 1
    multi_lr = len(args.activity_lrs) > 1

    for width in args.widths:
        for gamma in args.gammas:
            for param_type in args.param_types:
                for activity_lr in args.activity_lrs:
                    run_args = argparse.Namespace(**vars(args))
                    run_args.width = width
                    run_args.gamma = gamma
                    run_args.param_type = param_type
                    run_args.activity_lr = activity_lr
                    rescaling = run_equilib_energy(run_args)
                    config = (param_type, gamma, activity_lr)
                    rescalings_by_config.setdefault(config, []).append(rescaling)

    def config_label(config: tuple) -> str:
        param_type, gamma, activity_lr = config
        parts = ["Data" if not multi_param else param_type]
        if multi_gamma:
            parts.append(rf"$\gamma = {gamma}$")
        if multi_lr:
            parts.append(rf"$\eta = {activity_lr}$")
        return ", ".join(parts)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    plot_rescaling_per_width(
        widths=args.widths,
        rescalings_by_config={
            config_label(config): values
            for config, values in rescalings_by_config.items()
        },
        save_path=args.save_dir / "equilib_rescaling_per_N.pdf",
        depth=args.depth,
        gamma=args.gammas[0],
    )
