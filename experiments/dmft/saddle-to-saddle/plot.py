"""Plot origin-saddle MSE curves for energy-scaled µPC vs BP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

FONT_SIZES = {"label": 45, "legend": 25, "tick": 35}
LABEL_PAD = 15
LINE_WIDTH = 4
ALPHA = 0.7

_SERIES = (
    ("bp", "BP"),
    ("pc", "PC"),
)
_COLORS = {
    "bp": "#2ca02c",
    "pc": "#1f77b4",
}


_XLABEL = "$t$"
_YLABEL = r"$\mathcal{L}(\boldsymbol{\theta}_t)$"
_AXIS_COMBOS = (
    (False, False, "linx_liny"),
    (False, True, "linx_logy"),
    (True, False, "logx_liny"),
    (True, True, "logx_logy"),
)
_DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def run_dir(base: Path, width: int, n_layers: int, seed: int) -> Path:
    return base / f"N{width}" / f"L{n_layers}" / f"seed_{seed}"


def discover_grid(base: Path, seed: int) -> tuple[list[int], list[int]]:
    widths, layers = set(), set()
    for path in base.glob(f"N*/L*/seed_{seed}/history.npz"):
        widths.add(int(path.parents[2].name[1:]))
        layers.add(int(path.parents[1].name[1:]))
    return sorted(widths), sorted(layers)


def load_histories(base: Path, widths, n_layers_list, seed: int) -> dict:
    data = {width: {name: {} for name, _ in _SERIES} for width in widths}
    for width in widths:
        for n_layers in n_layers_list:
            path = run_dir(base, width, n_layers, seed) / "history.npz"
            if not path.exists():
                print(f"  Missing {path}")
                continue
            loaded = np.load(path)
            for name, _ in _SERIES:
                data[width][name][n_layers] = np.asarray(loaded[name]).flatten()
    return data


def _iterations(losses: np.ndarray) -> np.ndarray:
    return np.arange(1, len(losses) + 1)


def plot_width(
    data_for_width: dict,
    width: int,
    n_layers_list,
    save_dir: str | os.PathLike,
    filename: str,
    *,
    log_xaxis: bool = False,
    log_yaxis: bool = False,
) -> None:
    n_L = len(n_layers_list)
    fig, axes = plt.subplots(
        1,
        n_L,
        figsize=(5.5 * n_L, 4.2),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for ax, n_layers in zip(axes, n_layers_list):
        for name, label in _SERIES:
            losses = data_for_width.get(name, {}).get(n_layers)
            if losses is None:
                continue
            ax.plot(
                _iterations(losses),
                losses,
                "-",
                alpha=ALPHA,
                linewidth=LINE_WIDTH,
                color=_COLORS[name],
                label=label,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(rf"$L = {n_layers}$", fontsize=32, pad=16)
        ax.set_xlabel(_XLABEL, fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
        if not log_xaxis:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_xaxis:
            ax.set_xscale("log")
        if log_yaxis:
            ax.set_yscale("log", base=10)
    axes[0].set_ylabel(_YLABEL, fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(
            handles,
            labels,
            fontsize=FONT_SIZES["legend"],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
    plt.close(fig)


def plot_width_all_axis_combos(
    data_for_width: dict,
    width: int,
    n_layers_list,
    save_dir: str | os.PathLike,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for log_x, log_y, suffix in _AXIS_COMBOS:
        plot_width(
            data_for_width,
            width,
            n_layers_list,
            save_dir,
            f"losses_N{width}_{suffix}.pdf",
            log_xaxis=log_x,
            log_yaxis=log_y,
        )


def plot_toy(
    results_dir: Path,
    save_dir: Path,
    widths,
    n_layers_list,
    seed: int,
) -> None:
    if widths is None or n_layers_list is None:
        found_widths, found_layers = discover_grid(results_dir, seed)
        widths = widths or found_widths
        n_layers_list = n_layers_list or found_layers
    if not widths or not n_layers_list:
        raise FileNotFoundError(f"No N*/L*/seed_{seed}/history.npz under {results_dir}")
    print(f"Loading histories from {results_dir}")
    data = load_histories(results_dir, widths, n_layers_list, seed)
    for width in widths:
        plot_width_all_axis_combos(data[width], width, n_layers_list, save_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot origin-saddle MSE for BP and energy-scaled µPC."
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory with the N*/L*/seed_* history tree. Defaults to results/.",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Where to write PDFs. Defaults to the results directory.",
    )
    p.add_argument("--widths", type=int, nargs="+", default=None)
    p.add_argument("--n-layers", type=int, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else _DEFAULT_RESULTS
    save_dir = Path(args.save_dir) if args.save_dir else results_dir
    plot_toy(results_dir, save_dir, args.widths, args.n_layers, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
