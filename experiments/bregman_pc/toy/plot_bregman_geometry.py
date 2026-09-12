"""Intuition figures for the activation-matched Bregman divergence.

Assumes the matched potential from ``plot_matched_potential.py``. The
tangent-gap geometry of ``D_psi`` is panel (c) of ``matched_potential.pdf``.
PDFs are written to ``<save-dir>/<act-fn>/``.

Saved figures
-------------
``energy_vs_preact.pdf``
    Why matching removes ``phi'``: Euclidean energy plateaus under saturation
    while the Bregman energy (and its ``a``-derivative) does not.
``divergence_vs_prediction.pdf``
    ``D_psi(z, zhat)`` versus squared Euclidean error as functions of ``zhat``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_matched_potential import (
    ALPHA,
    ANNOT_FONT,
    BREGMAN_COLOR,
    FIG_SIZE,
    FONT_SIZES,
    GUIDE_COLOR,
    LINE_WIDTH,
    STD_PC_COLOR,
    TWO_FIG_SIZE,
    _act_grids,
    _draw_range_lines,
    _labels,
    _style_axes,
    act_save_dir,
    add_act_save_args,
    bregman,
    inv_phi,
    phi,
    phi_prime,
    save_fig,
)


def _mark_condition(ax, x, y, text, text_xy, va="bottom"):
    ax.scatter([x], [y], color=GUIDE_COLOR, s=70, zorder=6, clip_on=False)
    ax.text(
        text_xy[0],
        text_xy[1],
        text,
        fontsize=ANNOT_FONT,
        color=GUIDE_COLOR,
        ha="left",
        va=va,
    )


def plot_energy_vs_preact(name: str, save_dir: str) -> None:
    """Euclidean energy saturates in ``a``; the matched Bregman energy does not."""
    a, _, a_max = _act_grids(name)
    z = {"tanh": 0.6, "sigmoid": 0.75, "linear": 1.2}[name]
    zhat = phi(name, a)
    eucl = 0.5 * (z - zhat) ** 2
    breg = bregman(name, z, zhat)
    d_breg = -(z - zhat)
    d_eucl = d_breg * phi_prime(name, a)
    a_star = float(inv_phi(name, np.array(z)))
    labels = _labels(name)

    fig, axes = plt.subplots(1, 2, figsize=TWO_FIG_SIZE)
    fonts = FONT_SIZES

    axes[0].plot(
        a, eucl, color=STD_PC_COLOR, lw=LINE_WIDTH, alpha=ALPHA,
        label=labels["eucl_energy"],
    )
    axes[0].plot(
        a, breg, color=BREGMAN_COLOR, lw=LINE_WIDTH, alpha=ALPHA,
        label=labels["breg_energy"],
    )
    axes[0].set_xlim(-a_max, a_max)
    _style_axes(axes[0], r"$a$", "Energy", fonts)
    ymax = axes[0].get_ylim()[1]
    _mark_condition(
        axes[0],
        a_star,
        0.0,
        r"$a^\star=\phi^{-1}(z)$",
        text_xy=(a_star - 0.6, 0.06 * ymax),
    )

    axes[1].plot(
        a, d_eucl, color=STD_PC_COLOR, lw=LINE_WIDTH, alpha=ALPHA,
        label=labels["eucl_dda"],
    )
    axes[1].plot(
        a, d_breg, color=BREGMAN_COLOR, lw=LINE_WIDTH, alpha=ALPHA,
        label=labels["breg_dda"],
    )
    axes[1].axhline(0.0, color=GUIDE_COLOR, ls=":", lw=1.2)
    axes[1].set_xlim(-a_max, a_max)
    _style_axes(axes[1], r"$a$", "Preactivation gradient", fonts)
    ymin = axes[1].get_ylim()[0]
    _mark_condition(
        axes[1],
        a_star,
        0.0,
        r"$a^\star=\phi^{-1}(z)$",
        text_xy=(a_star + 0.05, 0.03 * ymin),
        va="top",
    )

    fig.tight_layout(w_pad=3.0)
    save_fig(fig, os.path.join(save_dir, "energy_vs_preact.pdf"))


def plot_divergence_vs_prediction(name: str, save_dir: str) -> None:
    """Matched vs Euclidean discrepancy as a function of the predicted activity."""
    _, zhat, _ = _act_grids(name)
    labels = _labels(name)
    z = {"tanh": 0.35, "sigmoid": 0.65, "linear": 0.8}[name]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(
        zhat,
        0.5 * (z - zhat) ** 2,
        color=STD_PC_COLOR,
        lw=LINE_WIDTH,
        alpha=ALPHA,
        label=labels["eucl_div"],
    )
    ax.plot(
        zhat,
        bregman(name, z, zhat),
        color=BREGMAN_COLOR,
        lw=LINE_WIDTH,
        alpha=ALPHA,
        label=labels["breg_div"],
    )
    _draw_range_lines(ax, name, vertical=True)
    _style_axes(ax, r"$\hat{z}$", "Energy")
    ax.legend(
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.05, 1.0),
    )
    _mark_condition(ax, z, 0.0, r"$\hat{z}=z$", text_xy=(z - 0.02, 0.2))
    save_fig(fig, os.path.join(save_dir, "divergence_vs_prediction.pdf"))


def plot_all(name: str, save_dir: str | os.PathLike) -> None:
    save_dir = act_save_dir(save_dir, name)
    plot_energy_vs_preact(name, save_dir)
    plot_divergence_vs_prediction(name, save_dir)


if __name__ == "__main__":
    args = add_act_save_args(
        argparse.ArgumentParser(
            description="Plot matched vs Euclidean energy (tanh by default)."
        )
    ).parse_args()
    plot_all(args.act_fn, args.save_dir)
