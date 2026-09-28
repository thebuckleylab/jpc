"""Intuition figures for the activation-matched Bregman potential.

The matching construction is ``psi'(z) = phi^{-1}(z)``, so ``psi`` is any
antiderivative of the inverse activation. Labels use the chosen ``phi``
(``tanh``, ``sigmoid``, or ``linear``). For tanh:

    tanh(u)
    psi'(z) = arctanh(z)
    [psi''(z)]^{-1} = 1 - z^2
    psi(z) = z arctanh(z) + (1/2) log(1 - z^2)

Linear recovers Euclidean PC: ``phi(a)=a``, ``psi(z)=z^2/2``.
PDFs are written to ``<save-dir>/<act-fn>/``.

Saved figures
-------------
``matched_potential.pdf``
    Activation, inverse, and potential in one row.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# fix-cm keeps Computer Modern but lets \fontsize exceed the ~25pt design
# sizes. Without it, 32pt and 44pt labels look identical.
_LATEX_PREAMBLE = (
    r"\usepackage{fix-cm}"
    r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{mathtools}"
)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": _LATEX_PREAMBLE,
    # Matplotlib's Agg PDF writer emits cmsy minus as character 0, which
    # many viewers drop. pdflatex via pgf keeps Computer Modern and writes
    # a normal minus.
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": _LATEX_PREAMBLE,
})

ACT_FNS = ("linear", "tanh", "sigmoid")
FIG_SIZE = (8, 6)
ROW_FIG_SIZE = (22, 6.5)
TWO_FIG_SIZE = (16, 6.2)
FONT_SIZES = {"label": 36, "legend": 24, "tick": 28, "title": 32, "annot": 26}
ANNOT_FONT = FONT_SIZES["annot"]
TITLE_PAD = 26
LABEL_PAD = 12
LINE_WIDTH = 4
ALPHA = 0.9
EPS = 1e-6
N = 600
CURVE_COLOR = "#1f77b4"
# Same cycle as plot.py: blue Bregman PC, orange Standard PC, green BP.
STD_PC_COLOR = "#ff7f0e"
BREGMAN_COLOR = CURVE_COLOR
POINT_COLOR = "#d62728"
GUIDE_COLOR = "0.55"
_DEFAULT_SAVE = Path(__file__).resolve().parent / "figures"
_SAVE_DIR_HELP = (
    "Parent directory for PDFs. Writes to <save-dir>/<act-fn>/. "
    "Defaults to experiments/bregman_pc/toy/figures."
)


def _clip_activity(name: str, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if name == "linear":
        return z
    if name == "tanh":
        return np.clip(z, -1.0 + EPS, 1.0 - EPS)
    return np.clip(z, EPS, 1.0 - EPS)


def phi(name: str, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if name == "linear":
        return a
    if name == "tanh":
        return np.tanh(a)
    return 1.0 / (1.0 + np.exp(-np.clip(a, -40.0, 40.0)))


def phi_prime(name: str, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if name == "linear":
        return np.ones_like(a)
    if name == "tanh":
        t = np.tanh(a)
        return 1.0 - t**2
    s = phi(name, a)
    return s * (1.0 - s)


def inv_phi(name: str, z: np.ndarray) -> np.ndarray:
    if name == "linear":
        return np.asarray(z, dtype=float)
    z = _clip_activity(name, z)
    if name == "tanh":
        return np.arctanh(z)
    return np.log(z) - np.log1p(-z)


def psi(name: str, z: np.ndarray) -> np.ndarray:
    """Matched potential with ``psi' = phi^{-1}`` (up to an additive constant)."""
    if name == "linear":
        z = np.asarray(z, dtype=float)
        return 0.5 * z**2
    z = _clip_activity(name, z)
    if name == "tanh":
        return z * np.arctanh(z) + 0.5 * np.log1p(-z**2)
    return z * np.log(z) + (1.0 - z) * np.log1p(-z)


def _labels(name: str) -> dict[str, str]:
    if name == "linear":
        return {
            "phi": r"$\mathrm{id}(a)$",
            "inv_phi": r"$z$",
            "psi": r"$\psi(z)$",
            "eucl_energy": r"$\frac{1}{2}(z-a)^{2}$",
            "breg_energy": r"$D_{\psi}(z,a)$",
            "eucl_dda": r"$-\varepsilon\,\phi'(a)$",
            "breg_dda": r"$-\varepsilon$",
            "eucl_div": r"$\frac{1}{2}(z-\hat{z})^{2}$",
            "breg_div": r"$D_{\psi}(z,\hat{z})$",
        }
    if name == "tanh":
        return {
            "phi": r"$\tanh(a)$",
            "inv_phi": r"$\mathrm{arctanh}(z)$",
            "psi": r"$\psi(z)$",
            "eucl_energy": r"$\frac{1}{2}(z-\tanh(a))^{2}$",
            "breg_energy": r"$D_{\psi}(z,\tanh(a))$",
            "eucl_dda": r"$-\varepsilon\,(1-\tanh^{2}(a))$",
            "breg_dda": r"$-\varepsilon$",
            "eucl_div": r"$\frac{1}{2}(z-\hat{z})^{2}$",
            "breg_div": r"$D_{\psi}(z,\hat{z})$",
        }
    return {
        "phi": r"$\sigma(a)$",
        "inv_phi": r"$\mathrm{logit}(z)$",
        "psi": r"$\psi(z)$",
        "eucl_energy": r"$\frac{1}{2}(z-\sigma(a))^{2}$",
        "breg_energy": r"$D_{\psi}(z,\sigma(a))$",
        "eucl_dda": r"$-\varepsilon\,\sigma'(a)$",
        "breg_dda": r"$-\varepsilon$",
        "eucl_div": r"$\frac{1}{2}(z-\hat{z})^{2}$",
        "breg_div": r"$D_{\psi}(z,\hat{z})$",
    }


def bregman(name: str, p, q) -> np.ndarray:
    p = np.asarray(p)
    q = np.asarray(q)
    return psi(name, p) - psi(name, q) - inv_phi(name, q) * (p - q)


def _act_grids(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    a_max = 5.0 if name == "sigmoid" else 3.5
    a = np.linspace(-a_max, a_max, N)
    if name == "linear":
        return a, a.copy(), a_max
    z_hi = float(phi(name, np.array(a_max)))
    if name == "tanh":
        z = np.linspace(-z_hi, z_hi, N)
    else:
        z_lo = float(phi(name, np.array(-a_max)))
        z = np.linspace(z_lo, z_hi, N)
    return a, z, a_max


def _range_guides(name: str) -> tuple[float, ...]:
    if name == "linear":
        return ()
    if name == "tanh":
        return (-1.0, 1.0)
    return (0.0, 1.0)


def _style_axes(ax, xlabel, ylabel, fonts=None):
    fonts = fonts or FONT_SIZES
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fonts["label"], labelpad=LABEL_PAD)
    ax.set_ylabel(ylabel, fontsize=fonts["label"], labelpad=LABEL_PAD, wrap=False)
    ax.yaxis.label.set_wrap(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=fonts["legend"], frameon=False, loc="best")
    ax.grid(True, which="both", ls="-", alpha=0.4)
    ax.tick_params(axis="both", labelsize=fonts["tick"])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _draw_range_lines(ax, name: str, *, vertical: bool):
    for v in _range_guides(name):
        if vertical:
            ax.axvline(v, color=GUIDE_COLOR, ls="--", lw=1.5, zorder=0.5)
        else:
            ax.axhline(v, color=GUIDE_COLOR, ls="--", lw=1.5, zorder=0.5)


def act_save_dir(save_dir: str | os.PathLike, name: str) -> str:
    path = Path(save_dir) / name
    os.makedirs(path, exist_ok=True)
    return str(path)


def save_fig(fig, path: str, *, tight: bool = True) -> None:
    kwargs = {"backend": "pgf"}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"wrote {path}")


def add_act_save_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--act-fn",
        type=str,
        default="tanh",
        choices=list(ACT_FNS),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(_DEFAULT_SAVE),
        help=_SAVE_DIR_HELP,
    )
    return parser


def _geometry_points(name: str) -> tuple[float, float]:
    """Activity ``z`` and prediction ``zhat`` used in the tangent-gap geometry."""
    if name == "tanh":
        return 0.72, -0.28
    if name == "linear":
        return 2.0, -0.8
    return 0.82, 0.22


def _draw_bregman_geometry(ax, name: str, z_grid: np.ndarray, fonts) -> None:
    """Tangent-gap geometry of ``D_psi(z, zhat)`` on the matched potential."""
    z, zhat = _geometry_points(name)
    psi_q = float(psi(name, np.array(zhat)))
    dpsi_q = float(inv_phi(name, np.array(zhat)))
    tangent = psi_q + dpsi_q * (z_grid - zhat)
    psi_p = float(psi(name, np.array(z)))
    tan_p = psi_q + dpsi_q * (z - zhat)

    ax.plot(
        z_grid,
        tangent,
        color=POINT_COLOR,
        lw=LINE_WIDTH,
        ls="--",
        alpha=ALPHA,
        label=r"tangent at $\hat{z}$",
        zorder=3,
    )
    ax.plot(
        [z, z], [tan_p, psi_p],
        color="black",
        lw=LINE_WIDTH,
        alpha=ALPHA,
        solid_capstyle="round",
        zorder=3,
    )
    ax.scatter([zhat], [psi_q], color=POINT_COLOR, s=90, zorder=5)
    ax.scatter([z], [psi_p], color="black", s=90, zorder=5)
    ax.scatter([z], [tan_p], color="black", s=90, zorder=5)

    y_mid = 0.5 * (tan_p + psi_p)
    x_pad = {"tanh": 0.05, "sigmoid": 0.03, "linear": 0.12}[name]
    ax.text(
        z + x_pad,
        y_mid,
        r"$D_\psi(z,\hat{z})$",
        fontsize=fonts.get("annot", ANNOT_FONT),
        color="black",
        ha="left",
        va="center",
    )
    ax.annotate(
        r"$\hat{z}$",
        xy=(zhat, psi_q),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=fonts.get("annot", ANNOT_FONT),
        color=POINT_COLOR,
    )
    ax.annotate(
        r"$z$",
        xy=(z, psi_p),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=fonts.get("annot", ANNOT_FONT),
        color="black",
    )


def plot_construction_row(name: str, save_dir: str) -> None:
    a, z, a_max = _act_grids(name)
    labels = _labels(name)
    fig, axes = plt.subplots(1, 3, figsize=ROW_FIG_SIZE)
    fonts = FONT_SIZES
    title_pad = TITLE_PAD

    _draw_range_lines(axes[0], name, vertical=False)
    axes[0].plot(a, phi(name, a), color=CURVE_COLOR, lw=LINE_WIDTH, alpha=ALPHA, zorder=3)
    axes[0].set_xlim(-a_max, a_max)
    if name == "linear":
        axes[0].set_ylim(-a_max, a_max)
    axes[0].set_title(r"(a) Activation function $\phi$", fontsize=fonts["title"], pad=title_pad)
    _style_axes(axes[0], r"$a$", labels["phi"], fonts)

    _draw_range_lines(axes[1], name, vertical=True)
    axes[1].plot(z, inv_phi(name, z), color=CURVE_COLOR, lw=LINE_WIDTH, alpha=ALPHA, zorder=3)
    axes[1].set_ylim(-a_max, a_max)
    axes[1].set_title(r"(b) $\psi'(z)\coloneqq\phi^{-1}(z)$", fontsize=fonts["title"], pad=title_pad)
    _style_axes(axes[1], r"$z$", labels["inv_phi"], fonts)

    _draw_range_lines(axes[2], name, vertical=True)
    axes[2].plot(
        z, psi(name, z), color=CURVE_COLOR, lw=LINE_WIDTH, alpha=ALPHA, label=labels["psi"], zorder=3
    )
    _draw_bregman_geometry(axes[2], name, z, fonts)
    axes[2].set_title(r"(c) Matched Bregman potential $\psi$", fontsize=fonts["title"], pad=title_pad)
    _style_axes(axes[2], r"$z$", labels["psi"], fonts)

    fig.tight_layout(w_pad=3.0)
    save_fig(fig, os.path.join(save_dir, "matched_potential.pdf"))


def plot_all(name: str, save_dir: str | os.PathLike) -> None:
    plot_construction_row(name, act_save_dir(save_dir, name))


if __name__ == "__main__":
    args = add_act_save_args(
        argparse.ArgumentParser(
            description="Plot the activation-matched Bregman potential (tanh by default)."
        )
    ).parse_args()
    plot_all(args.act_fn, args.save_dir)
