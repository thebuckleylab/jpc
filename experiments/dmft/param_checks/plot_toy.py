from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.dmft.src import plot_style as ps
from plot_toy import (
    LOSSES_AND_ENERGIES_YLABEL,
    data_for_widths,
    load_gamma_data,
    load_width_data,
    make_plot_dir,
    pc_energies_on_mse_scale,
)
from train_toy import add_common_args, pc_energy_legend, resolve_results_dir

# Figure 1 places these curves three across the text width.
PANEL = ps.PANEL_THIRD

# Colours from the param_checks plots: PC widths on Blues, BP in crimson,
# width legend entries on Greys. γ sweeps use Blues for PC and Reds for BP.
PC_LEGEND_COLOR = "#4A90E2"
BP_COLOR = "#DC143C"
DATA_COLOR = "#1E88E5"

BP_LOSS_LABEL = r"$\mathcal{L}(\boldsymbol{\theta})$ (BP)"
ALIGNMENT_YLABEL = "PC-BP alignment"
RESCALING_YLABEL = r"$s(\boldsymbol{\theta}_t)$"


def use_paper_style():
    """Install the DMFT paper style.

    Importing ``plot_toy`` loads the limits-paper plot modules, which turn
    on ``text.usetex``. The paper style uses mathtext, so usetex is turned
    back off after the rcParams update.
    """
    ps.apply_paper_style()
    mpl.rcParams["text.usetex"] = False


use_paper_style()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot toy µPC vs BP results from train_toy.py "
            "in the DMFT paper style."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--plot_widths",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Widths to draw on the width-sweep plots, in this order "
            "(default: every --widths). Widths listed here are loaded "
            "even if they are absent from --widths."
        ),
    )
    return parser.parse_args()


def _curve(values):
    values = np.asarray(values, dtype=float).ravel()
    return np.arange(1, values.size + 1), values


THEORY_LABEL = r"Theory ($N \rightarrow \infty$)"


def _cmap_val(idx, n):
    """Sample a sequential map the way the param_checks plots do."""
    if n <= 1:
        return 0.15
    return 0.15 + (idx / (n - 1)) * 0.85


def _sequential_colors(n, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(_cmap_val(idx, n)) for idx in range(n)]


def _expand_for_outside_legend(fig, ax, *, loc, bbox_to_anchor):
    """Grow the figure so an outside legend leaves the axes at panel size.

    Constrained layout fits the legend by shrinking the axes. The axes
    size is recorded without the legend, then the figure is enlarged
    until that size comes back.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.get_texts()]
    legend.remove()
    fig.canvas.draw()
    bare = ax.get_window_extent(fig.canvas.get_renderer())
    target_w = bare.width / fig.dpi
    target_h = bare.height / fig.dpi
    ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        borderaxespad=0.0,
    )
    for _ in range(4):
        fig.canvas.draw()
        box = ax.get_window_extent(fig.canvas.get_renderer())
        extra_w = target_w - box.width / fig.dpi
        extra_h = target_h - box.height / fig.dpi
        if extra_w <= 0.02 and extra_h <= 0.02:
            break
        width, height = fig.get_size_inches()
        fig.set_size_inches(
            width + max(extra_w, 0.0),
            height + max(extra_h, 0.0),
        )


def _save(
    fig,
    plot_dir,
    filename,
    *,
    outside_legend=False,
    legend_loc="upper left",
    legend_anchor=(1.02, 1.0),
):
    ax = fig.gca()
    ps.style_axes(ax)
    if outside_legend:
        _expand_for_outside_legend(
            fig, ax, loc=legend_loc, bbox_to_anchor=legend_anchor
        )
    path = ps.save_figure(
        fig, os.path.join(plot_dir, filename), formats=("pdf",)
    )
    print(f"    {path}")
    return path


def _line(label, color, linestyle="-"):
    return Line2D([], [], color=color, linestyle=linestyle, label=label)


def plot_keyed_series(
    keys,
    series,
    plot_dir,
    filename,
    ylabel,
    *,
    key_label,
    log_y=False,
    log_x=False,
):
    """One coloured curve per swept key (width or γ)."""
    present = [key for key in keys if key in series]
    if not present:
        return
    fig = plt.figure(figsize=PANEL)
    ax = fig.gca()
    colors = _sequential_colors(len(present), "Blues")
    for key, color in zip(present, colors):
        iterations, values = _curve(series[key])
        ax.plot(iterations, values, color=color, label=key_label(key))
    ax.set_xlabel(ps.TEX["time"])
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    _save(fig, plot_dir, filename)


def plot_width_pc_bp(
    data,
    plot_dir,
    filename,
    ylabel,
    *,
    pc_key,
    log_y,
    log_x,
    pc_label,
    bp_label,
    include_dmft,
    legend_loc="upper left",
    legend_anchor=(1.02, 1.0),
):
    """PC curves coloured by width, BP at the largest width, optional DMFT."""
    pc_series = data.get(pc_key) or {}
    widths = sorted(width for width in data["widths"] if width in pc_series)
    max_width = max(data["widths"]) if data["widths"] else None
    bp_series = data.get("bp_losses") or {}
    has_bp = max_width in bp_series
    has_dmft = include_dmft and data.get("dmft_loss") is not None
    if not widths and not has_bp and not has_dmft:
        return

    fig = plt.figure(figsize=PANEL)
    ax = fig.gca()

    blues = _sequential_colors(len(widths), "Blues")
    for width, color in zip(widths, blues):
        iterations, values = _curve(pc_series[width])
        ax.plot(iterations, values, color=color)
    if has_bp:
        iterations, values = _curve(bp_series[max_width])
        ax.plot(iterations, values, color=BP_COLOR)
    if has_dmft:
        iterations, values = _curve(data["dmft_loss"])
        ax.plot(
            iterations,
            values,
            color="black",
            linestyle="--",
        )

    handles = []
    if widths:
        handles.append(_line(pc_label, PC_LEGEND_COLOR))
    if has_bp:
        handles.append(_line(bp_label, BP_COLOR))
    legend_widths = sorted(set(widths + ([max_width] if has_bp else [])))
    greys = plt.get_cmap("Greys")
    n_legend = len(legend_widths)
    for idx, width in enumerate(legend_widths):
        gray_val = 0.3 + (idx / max(n_legend - 1, 1)) * 0.5 if n_legend > 1 else 0.5
        handles.append(_line(ps.width_label(width), greys(gray_val)))
    if has_dmft:
        handles.append(_line(THEORY_LABEL, "black", linestyle="--"))
    ax.legend(
        handles=handles,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        borderaxespad=0.0,
    )
    ax.set_xlabel(ps.TEX["time"])
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    _save(
        fig,
        plot_dir,
        filename,
        outside_legend=True,
        legend_loc=legend_loc,
        legend_anchor=legend_anchor,
    )


def plot_gamma_pc_bp(
    data,
    plot_dir,
    filename,
    ylabel,
    *,
    pc_key,
    log_y,
    log_x,
    pc_label,
    bp_label,
):
    """γ sweep: colour is γ, solid is PC, dashed is BP."""
    pc_series = data.get(pc_key) or {}
    bp_series = data.get("bp_losses") or {}
    pc_gammas = sorted(gamma for gamma in data["gamma_0s"] if gamma in pc_series)
    bp_gammas = sorted(gamma for gamma in data["gamma_0s"] if gamma in bp_series)
    all_gammas = sorted(set(pc_gammas + bp_gammas))
    if not all_gammas:
        return

    fig = plt.figure(figsize=PANEL)
    ax = fig.gca()
    n_gammas = len(all_gammas)
    blues = plt.get_cmap("Blues")
    reds = plt.get_cmap("Reds")

    def gamma_label(gamma_0):
        return rf"$\gamma = {ps.fmt_number(gamma_0)}$"

    for gamma_0 in pc_gammas:
        iterations, values = _curve(pc_series[gamma_0])
        idx = all_gammas.index(gamma_0)
        ax.plot(
            iterations,
            values,
            color=blues(_cmap_val(idx, n_gammas)),
            linestyle="-",
        )
    for gamma_0 in bp_gammas:
        iterations, values = _curve(bp_series[gamma_0])
        idx = all_gammas.index(gamma_0)
        ax.plot(
            iterations,
            values,
            color=reds(_cmap_val(idx, n_gammas)),
            linestyle="--",
        )

    handles = []
    if pc_gammas:
        handles.append(_line(pc_label, blues(0.5), linestyle="-"))
    if bp_gammas:
        handles.append(_line(bp_label, reds(0.5), linestyle="--"))
    for idx, gamma_0 in enumerate(all_gammas):
        grey_val = 0.8 - (idx / (n_gammas - 1)) * 0.6 if n_gammas > 1 else 0.5
        handles.append(_line(gamma_label(gamma_0), (grey_val, grey_val, grey_val)))
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlabel(ps.TEX["time"])
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    _save(fig, plot_dir, filename)


def plot_inv_rescaling_vs_width(data, plot_dir, gamma_0, param_type):
    """Plot ``1/s(θ_0)`` vs width.

    Energy-scaled µPC uses the infinite-width line ``γ² N``. ``mupc_old``
    uses a constant line at 1.
    """
    rescalings = data.get("pc_rescalings") or {}
    widths = sorted(width for width in data["widths"] if width in rescalings)
    first = []
    valid = []
    for width in widths:
        values = np.asarray(rescalings[width]).ravel()
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

    fig = plt.figure(figsize=PANEL)
    ax = fig.gca()
    ax.plot(
        valid,
        first,
        linestyle="none",
        marker="o",
        color=DATA_COLOR,
        label="Data",
        zorder=3,
    )
    if param_type != "sp":
        theory_widths = np.logspace(
            np.log10(min(valid)), np.log10(max(valid)), 100
        )
        if param_type == "mupc_old":
            theory = np.ones_like(theory_widths)
            theory_label = r"Theory: $\Theta_N(1)$"
        else:
            theory = (gamma_0 ** 2) * theory_widths
            theory_label = r"Theory: $\Theta(N)$"
        ax.plot(
            theory_widths,
            theory,
            linestyle="--",
            color=ps.COLOR_REFERENCE,
            label=theory_label,
            zorder=2,
        )
    ax.set_xlabel(ps.TEX["width"])
    ax.set_ylabel(r"$1/s(\boldsymbol{\theta})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    _save(fig, plot_dir, "inv_rescaling_vs_width.pdf")


def _show_dmft(data, param_type, use_skips):
    return (
        data.get("dmft_loss") is not None
        and param_type != "sp"
        and use_skips is not True
    )


def plot_width_sweep(data, plot_dir, param_type, use_skips, log_x_scale, plot_widths=None):
    os.makedirs(plot_dir, exist_ok=True)
    print(f"  Width sweep plots → {plot_dir}")
    if not data["pc_train_losses"] and not data["pc_energies"]:
        print("  Warning: no width-sweep data found")
        return

    loss_data = data_for_widths(data, plot_widths)
    overlay = pc_energies_on_mse_scale(loss_data, param_type, sweep="width")
    energy_legend = pc_energy_legend(param_type)
    show_dmft = _show_dmft(loss_data, param_type, use_skips)
    plot_width_pc_bp(
        loss_data,
        plot_dir,
        "losses.pdf",
        ps.LOSS_LABEL,
        pc_key="pc_train_losses",
        log_y=True,
        log_x=log_x_scale,
        pc_label=ps.LABEL_PC,
        bp_label=ps.LABEL_BP,
        include_dmft=show_dmft,
    )
    plot_width_pc_bp(
        overlay,
        plot_dir,
        "losses_and_energies.pdf",
        LOSSES_AND_ENERGIES_YLABEL,
        pc_key="pc_energies",
        log_y=False,
        log_x=log_x_scale,
        pc_label=energy_legend,
        bp_label=BP_LOSS_LABEL,
        include_dmft=show_dmft,
        legend_loc="center left",
        legend_anchor=(1.02, 0.5),
    )
    plot_width_pc_bp(
        overlay,
        plot_dir,
        "losses_and_energies_logy.pdf",
        LOSSES_AND_ENERGIES_YLABEL,
        pc_key="pc_energies",
        log_y=True,
        log_x=log_x_scale,
        pc_label=energy_legend,
        bp_label=BP_LOSS_LABEL,
        include_dmft=loss_data.get("dmft_loss") is not None,
    )
    similarities = {
        width: loss_data["grad_cosine_similarities"][width]
        for width in loss_data["widths"]
        if width in loss_data["grad_cosine_similarities"]
    }
    if similarities:
        plot_keyed_series(
            sorted(similarities),
            similarities,
            plot_dir,
            "grads_cosine_similarities.pdf",
            ALIGNMENT_YLABEL,
            key_label=ps.width_label,
        )
    else:
        print("  Warning: no cosine similarity data for width sweep")
    if loss_data["pc_rescalings"]:
        plot_keyed_series(
            sorted(width for width in loss_data["widths"] if width in loss_data["pc_rescalings"]),
            loss_data["pc_rescalings"],
            plot_dir,
            "pc_rescaling.pdf",
            RESCALING_YLABEL,
            key_label=ps.width_label,
        )
        plot_inv_rescaling_vs_width(
            loss_data, plot_dir, loss_data["gamma_0"], param_type
        )


def plot_gamma_sweep(data, plot_dir, log_x_scale, param_type):
    os.makedirs(plot_dir, exist_ok=True)
    print(f"  Gamma sweep plots → {plot_dir}")
    if not data["pc_train_losses"] and not data["pc_energies"]:
        print("  Warning: no gamma-sweep data found")
        return

    overlay = pc_energies_on_mse_scale(data, param_type, sweep="gamma")
    bp_label = ps.LABEL_BP
    if data.get("width") is not None:
        bp_with_width = rf"{ps.LABEL_BP}, {ps.width_label(data['width'])}"
    else:
        bp_with_width = ps.LABEL_BP

    def gamma_label(gamma_0):
        return rf"$\gamma = {ps.fmt_number(gamma_0)}$"

    plot_gamma_pc_bp(
        data,
        plot_dir,
        "losses.pdf",
        ps.LOSS_LABEL,
        pc_key="pc_train_losses",
        log_y=True,
        log_x=log_x_scale,
        pc_label=ps.LABEL_PC,
        bp_label=bp_label,
    )
    plot_gamma_pc_bp(
        overlay,
        plot_dir,
        "losses_and_energies.pdf",
        LOSSES_AND_ENERGIES_YLABEL,
        pc_key="pc_energies",
        log_y=False,
        log_x=log_x_scale,
        pc_label=ps.LABEL_PC,
        bp_label=bp_label,
    )
    plot_gamma_pc_bp(
        overlay,
        plot_dir,
        "losses_and_energies_logy.pdf",
        LOSSES_AND_ENERGIES_YLABEL,
        pc_key="pc_energies",
        log_y=True,
        log_x=log_x_scale,
        pc_label=ps.LABEL_PC,
        bp_label=bp_with_width,
    )
    similarities = {
        gamma_0: data["grad_cosine_similarities"][gamma_0]
        for gamma_0 in data["gamma_0s"]
        if gamma_0 in data["grad_cosine_similarities"]
    }
    if similarities:
        plot_keyed_series(
            sorted(similarities),
            similarities,
            plot_dir,
            "grads_cosine_similarities.pdf",
            ALIGNMENT_YLABEL,
            key_label=gamma_label,
        )
    else:
        print("  Warning: no cosine similarity data for gamma sweep")
    if data["pc_rescalings"]:
        plot_keyed_series(
            sorted(gamma for gamma in data["gamma_0s"] if gamma in data["pc_rescalings"]),
            data["pc_rescalings"],
            plot_dir,
            "pc_rescaling.pdf",
            RESCALING_YLABEL,
            key_label=gamma_label,
        )


def generate_plots(args):
    use_paper_style()
    args = resolve_results_dir(args)
    data_results_dir = os.path.join(args.results_dir, f"{args.input_dim}_input_dim")
    if not os.path.isdir(data_results_dir):
        print(f"No results at {data_results_dir}; skipping plots.")
        return

    plot_root = os.path.join(args.results_dir, "plots_paper")
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
                                    args.log_x_scale,
                                    param_type,
                                )


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
