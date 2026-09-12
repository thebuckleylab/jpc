"""Shared publication style for the DMFT paper figures.

Panels are drawn at their final printed size so that ``\\includegraphics``
never rescales them: font sizes in this module are the font sizes on the
page. ``TEXT_WIDTH_IN`` is the ICLR text width, and ``PANEL_*`` give the
figure sizes for one, two, or three panels per row.

Figures are laid out with constrained layout and saved without a tight
bounding box, so the saved file is exactly ``figsize`` inches.

Notation follows the manuscript: ``$N$`` width, ``$H$`` hidden layers
with ``$\\ell = 1, \\dots, H$``, ``$\\gamma_0$`` the output scaling,
``$\\beta$`` the activity (inference) learning rate, ``$K$`` inference
steps, ``$t$`` training steps, ``$\\mathcal{L}$`` the loss, and
``$C^{h,\\ell}$`` / ``$C^{\\phi,\\ell}$`` the feature kernels.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# --- Geometry -------------------------------------------------------------

TEXT_WIDTH_IN = 5.5  # ICLR \textwidth

PANEL_FULL = (TEXT_WIDTH_IN, 2.20)
PANEL_HALF = (2.70, 1.95)
PANEL_THIRD = (1.83, 1.62)

#: Height of one row in a per-layer panel grid spanning the text width.
PER_LAYER_ROW_HEIGHT = 1.75

#: Font size for in-axes annotations (row labels, effective ranks).
ANNOTATION_SIZE = 6.5

#: Default marker size, also the reference for per-curve adjustments.
MARKER_SIZE = 2.8


# --- Colours --------------------------------------------------------------

COLOR_PC = "tab:blue"
COLOR_BP = "tab:orange"
COLOR_REFERENCE = "black"

#: Diverging map for kernel heatmaps, always centred on zero.
KERNEL_CMAP = "coolwarm"

#: Sequential map for swept scalars (widths, depths, gammas, K).
SEQUENCE_CMAP = "viridis"

# The top of viridis is a pale yellow that disappears against white.
_SEQUENCE_RANGE = (0.0, 0.88)


# --- Notation -------------------------------------------------------------

TEX = {
    "width": r"$N$",
    "n_hidden": r"$H$",
    "layer": r"$\ell$",
    "gamma_0": r"$\gamma_0$",
    "activity_lr": r"$\beta$",
    "n_infer_iters": r"$K$",
    "time": r"$t$",
    "loss": r"$\mathcal{L}$",
}

#: Axis label for a training-loss curve.
LOSS_LABEL = r"training loss $\mathcal{L}$"

#: Legend labels for the three ways a kernel / loss can be obtained.
LABEL_DMFT = "DMFT"
LABEL_NN = "NN"
LABEL_NN_CLOSED_FORM = "NN (closed-form)"
LABEL_PC = "PC"
LABEL_BP = "BP"


def feature_tex(symbol):
    """TeX fragment ``h`` or ``\\phi`` for feature-kernel labels."""
    return r"\phi" if symbol == "phi" else "h"


def fmt_sci(value, precision=2):
    """Format a float as LaTeX scientific notation, e.g. ``$2.5\\times10^{-1}$``.

    Values that are comfortably readable in fixed-point notation are left
    as plain decimals.
    """
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    if -2 <= exponent < 3:
        return f"{value:.{max(0, precision - exponent)}f}".rstrip("0").rstrip(".")
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{precision}f}\times 10^{{{exponent}}}"


def fmt_number(value):
    """Format a hyperparameter value without a trailing ``.0``."""
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def width_label(width):
    return rf"$N = {int(width)}$"


def layer_label(layer):
    """Legend label for a 0-based layer index."""
    return rf"$\ell = {int(layer) + 1}$"


def k_label(n_infer_iters, prefix=None):
    label = rf"$K = {int(n_infer_iters)}$"
    return f"{prefix}, {label}" if prefix else label


# --- Style ----------------------------------------------------------------

_RC = {
    # Typography: STIX is metric-compatible with Times, so figure text
    # matches the ICLR body font.
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "figure.titlesize": 8,
    "figure.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    # Lines and markers.
    "lines.linewidth": 1.1,
    "lines.markersize": MARKER_SIZE,
    "lines.markeredgewidth": 0.0,
    "errorbar.capsize": 1.5,
    # Axes and ticks.
    "axes.linewidth": 0.6,
    "axes.labelpad": 2.0,
    "axes.titlepad": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    # Grid.
    "axes.grid": False,
    "grid.linewidth": 0.4,
    "grid.color": "0.85",
    "grid.linestyle": "-",
    "grid.alpha": 1.0,
    # Legend.
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.25,
    "legend.columnspacing": 0.9,
    "legend.borderaxespad": 0.3,
    "legend.borderpad": 0.2,
    # Layout: constrained layout with a standard bbox keeps the saved
    # file exactly `figsize` inches.
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.015,
    "figure.constrained_layout.w_pad": 0.015,
    "figure.constrained_layout.hspace": 0.03,
    "figure.constrained_layout.wspace": 0.03,
    # Output.
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "standard",
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style():
    """Install the paper style globally. Safe to call more than once."""
    mpl.rcParams.update(_RC)


def sequence_colors(n, cmap_name=SEQUENCE_CMAP):
    """``n`` colours spanning a sequential map, skipping the pale extreme."""
    cmap = plt.get_cmap(cmap_name)
    lo, hi = _SEQUENCE_RANGE
    if n <= 1:
        return [cmap(lo)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def style_axes(ax, *, grid=True):
    """Apply the shared axes decoration (spines already handled by rcParams).

    Only major gridlines are drawn: on a log axis the minor decade lines
    crowd a panel of this size.
    """
    if grid:
        ax.grid(True, which="major")
        ax.set_axisbelow(True)


def integer_ticks(ax, values, axis="x"):
    """Pin an axis to integer positions, e.g. layer indices."""
    ticks = sorted({int(v) for v in np.asarray(values).ravel()})
    if axis == "x":
        ax.set_xticks(ticks)
    else:
        ax.set_yticks(ticks)


def symmetric_clim(arrays, *, vmin=None, vmax=None):
    """Colour limits centred on zero for a diverging kernel heatmap.

    Explicit ``vmin`` / ``vmax`` win. Otherwise the limits are
    ``(-m, m)`` with ``m`` the largest finite absolute value, so the
    midpoint of the diverging map sits at zero.
    """
    if vmin is not None or vmax is not None:
        limits = {}
        if vmin is not None:
            limits["vmin"] = vmin
        if vmax is not None:
            limits["vmax"] = vmax
        return limits
    stacked = np.concatenate(
        [np.asarray(a, dtype=float).ravel() for a in arrays]
    )
    finite = stacked[np.isfinite(stacked)]
    if not finite.size:
        return {}
    magnitude = float(np.max(np.abs(finite)))
    if magnitude == 0.0:
        return {}
    return {"vmin": -magnitude, "vmax": magnitude}


def per_layer_figsize(ncols, nrows, width=TEXT_WIDTH_IN):
    """Figure size for a per-layer panel grid spanning ``width`` inches."""
    return (width, PER_LAYER_ROW_HEIGHT * nrows)


_TEXT_WIDTH_CACHE = {}


def text_width_in(text, fontsize=None):
    """Printed width of ``text`` in inches, at the paper font.

    Used to reserve margins in hand-laid-out figures. Measured with the
    active backend where possible, falling back to an average-glyph
    estimate for backends without a renderer.
    """
    if not text:
        return 0.0
    if fontsize is None:
        fontsize = mpl.rcParams["font.size"]
    key = (text, float(fontsize))
    if key in _TEXT_WIDTH_CACHE:
        return _TEXT_WIDTH_CACHE[key]
    width = None
    try:
        fig = plt.figure(figsize=(1.0, 1.0), layout="none")
        artist = fig.text(0.0, 0.0, text, fontsize=fontsize)
        extent = artist.get_window_extent(fig.canvas.get_renderer())
        width = extent.width / fig.dpi
        plt.close(fig)
    except Exception:
        width = None
    if width is None:
        # Times-like glyphs average about half the point size in width.
        width = 0.5 * fontsize * len(text) / 72.0
    _TEXT_WIDTH_CACHE[key] = width
    return width


def save_figure(fig, save_path, *, formats=("pdf", "png")):
    """Save ``fig`` at its exact ``figsize`` in every requested format.

    ``save_path`` may carry any extension; siblings are written for the
    remaining formats. The path is returned unchanged so callers keep
    their existing return values.
    """
    stem, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # ``savefig`` restores a ``None`` layout engine as "use rcParams", which
    # would turn constrained layout on between PDF and PNG for figures
    # created with ``layout="none"``. Keep those unmanaged.
    rc = {}
    if fig.get_layout_engine() is None:
        rc = {
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
        }
    with mpl.rc_context(rc):
        for fmt in formats:
            fig.savefig(f"{stem}.{fmt}", format=fmt)
    plt.close(fig)
    return save_path
