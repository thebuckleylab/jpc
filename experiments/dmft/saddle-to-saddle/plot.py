"""Plot origin-saddle MSE curves for energy-scaled µPC vs BP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from experiments.dmft.src import plot_style as ps


def use_paper_style():
    """DMFT paper style from ``param_checks`` (mathtext, not usetex)."""
    ps.apply_paper_style()
    mpl.rcParams["text.usetex"] = False


use_paper_style()

# Same PC / BP colours as ``param_checks/plot_toy.py``.
_SERIES = (
    ("bp", ps.LABEL_BP),
    ("pc", ps.LABEL_PC),
)
_COLORS = {
    "bp": "#DC143C",
    "pc": "#4A90E2",
}
_AXIS_COMBOS = (
    (False, False, "linx_liny"),
    (False, True, "linx_logy"),
    (True, False, "logx_liny"),
    (True, True, "logx_logy"),
)
_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS = _HERE / "results"
_DEFAULT_MNIST = _DEFAULT_RESULTS / "mnist_origin_saddle"


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
    save_dir: str | Path,
    filename: str,
    *,
    log_xaxis: bool = False,
    log_yaxis: bool = False,
) -> None:
    n_L = len(n_layers_list)
    fig, axes = plt.subplots(
        1,
        n_L,
        figsize=(ps.TEXT_WIDTH_IN, ps.PANEL_THIRD[1]),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for ax, n_layers in zip(axes, n_layers_list):
        _plot_series(ax, {
            name: data_for_width.get(name, {}).get(n_layers) for name, _ in _SERIES
        })
        ax.set_title(rf"$L = {n_layers}$")
        ax.set_xlabel(ps.TEX["time"])
        ps.style_axes(ax)
        if log_xaxis:
            ax.set_xscale("log")
        if log_yaxis:
            ax.set_yscale("log")
    axes[0].set_ylabel(ps.LOSS_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="upper right")
    ps.save_figure(fig, str(Path(save_dir) / filename), formats=("pdf",))


def plot_width_all_axis_combos(
    data_for_width: dict,
    width: int,
    n_layers_list,
    save_dir: str | Path,
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


def _plot_series(ax, series: dict) -> None:
    for name, label in _SERIES:
        losses = series.get(name)
        if losses is None:
            continue
        losses = np.asarray(losses).flatten()
        ax.plot(
            _iterations(losses),
            losses,
            color=_COLORS[name],
            label=label,
        )


def load_flat_history(path: Path) -> dict:
    loaded = np.load(path)
    return {name: np.asarray(loaded[name]).flatten() for name, _ in _SERIES}


def plot_curves(
    history: dict,
    save_dir: str | Path,
    filename: str,
    *,
    log_xaxis: bool = False,
    log_yaxis: bool = False,
) -> None:
    fig = plt.figure(figsize=ps.PANEL_HALF)
    ax = fig.gca()
    _plot_series(ax, history)
    ax.set_xlabel(ps.TEX["time"])
    ax.set_ylabel(ps.LOSS_LABEL)
    if log_xaxis:
        ax.set_xscale("log")
    if log_yaxis:
        ax.set_yscale("log")
    ax.legend(loc="lower left")
    ps.style_axes(ax)
    ps.save_figure(fig, str(Path(save_dir) / filename), formats=("pdf",))


def plot_curves_all_axis_combos(history: dict, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for log_x, log_y, suffix in _AXIS_COMBOS:
        plot_curves(
            history,
            save_dir,
            f"losses_{suffix}.pdf",
            log_xaxis=log_x,
            log_yaxis=log_y,
        )


def plot_mnist(results_dir: Path, save_dir: Path) -> None:
    history_path = results_dir / "history.npz"
    if not history_path.exists():
        raise FileNotFoundError(f"No history.npz in {results_dir}")
    print(f"Loading {history_path}")
    plot_curves_all_axis_combos(load_flat_history(history_path), save_dir)


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


def _is_toy_tree(results_dir: Path, seed: int) -> bool:
    return any(results_dir.glob(f"N*/L*/seed_{seed}/history.npz"))


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot origin-saddle MSE for BP and energy-scaled µPC."
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "MNIST directory with history.npz, or a toy N*/L*/seed_* tree. "
            "Defaults to results/ and results/mnist_origin_saddle/."
        ),
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
    if args.results_dir:
        results_dir = Path(args.results_dir)
        save_dir = Path(args.save_dir) if args.save_dir else results_dir
        if (results_dir / "history.npz").exists():
            plot_mnist(results_dir, save_dir)
        elif _is_toy_tree(results_dir, args.seed):
            plot_toy(results_dir, save_dir, args.widths, args.n_layers, args.seed)
        else:
            raise FileNotFoundError(
                f"No history.npz or N*/L*/seed_{args.seed}/history.npz under {results_dir}"
            )
        print("Done.")
        return

    plotted = False
    if (_DEFAULT_MNIST / "history.npz").exists():
        save_dir = Path(args.save_dir) if args.save_dir else _DEFAULT_MNIST
        plot_mnist(_DEFAULT_MNIST, save_dir)
        plotted = True
    if _is_toy_tree(_DEFAULT_RESULTS, args.seed):
        save_dir = Path(args.save_dir) if args.save_dir else _DEFAULT_RESULTS
        plot_toy(_DEFAULT_RESULTS, save_dir, args.widths, args.n_layers, args.seed)
        plotted = True
    if not plotted:
        raise FileNotFoundError(
            f"No histories at {_DEFAULT_MNIST} or {_DEFAULT_RESULTS}. "
            "Pass --results-dir."
        )
    print("Done.")


if __name__ == "__main__":
    main()
