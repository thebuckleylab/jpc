import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_style as ps

ps.apply_paper_style()


def _to_numpy(arr):
    if isinstance(arr, (list, tuple)):
        return [_to_numpy(x) for x in arr]
    return np.asarray(arr)


def _warn_if_nonfinite(name, arr):
    arr = np.asarray(arr)
    n_bad = np.size(arr) - np.count_nonzero(np.isfinite(arr))
    if n_bad:
        print(
            f"Warning: {name} has {n_bad}/{np.size(arr)} non-finite values; "
            "plots may appear empty."
        )


def feature_kernel_symbol(act_fn=None, use_phi=None):
    """Symbol for the compared feature kernel: ``"h"`` or ``"phi"``.

    Linear nets use ``C^h`` (``phi`` is the identity). Nonlinear nets use
    ``C^phi``. Pass ``use_phi`` to override; otherwise ``act_fn != "linear"``
    selects ``phi``.
    """
    if use_phi is not None:
        return "phi" if use_phi else "h"
    if act_fn is None or act_fn == "linear":
        return "h"
    return "phi"


def _feature_kernel_tex(symbol):
    """TeX fragment ``h`` or ``\\phi`` for feature-kernel labels."""
    return r"\phi" if symbol == "phi" else "h"


def _data_ylim(
    values, *, invert=False, pad_frac=0.08, min_pad=0.02, ymin_floor=None
):
    """Tight y-limits around ``values``, with optional inverted axis.

    ``invert=True`` puts smaller values at the top (so relative
    displacement 0 matches cosine/CKA ``1`` at the top).
    ``ymin_floor`` clamps the padded lower limit to a value the quantity
    cannot go below (e.g. 0 for CKA / cosine); it does not stretch the
    axis down to that floor, which would waste most of a narrow panel
    whenever the data sit near 1.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo = float(np.min(v))
    hi = float(np.max(v))
    pad = max((hi - lo) * pad_frac, min_pad)
    lo, hi = lo - pad, hi + pad
    if (
        ymin_floor is not None
        and float(np.min(v)) >= ymin_floor
    ):
        lo = max(lo, float(ymin_floor))
    if invert:
        return (hi, lo)
    return (lo, hi)


def _displacement_metric_spec(
    metric, feature_symbol, t0="0", t="T", layer=None
):
    """Return ``(value_col, ylabel, invert_ylim)`` for a displacement plot.

    ``metric`` is ``"cosine"`` (alignment with the initial kernel) or
    ``"rel_frob"`` (``||C_t - C_0||_F / ||C_0||_F``). Relative Frobenius
    inverts the y-axis so 0 (no displacement) sits at the top.

    ``layer`` is appended to the kernel superscript; pass ``None`` where
    the layer is already given by the x-axis or the panel label, which
    keeps the label short enough for a narrow panel.
    """
    sym = _feature_kernel_tex(feature_symbol)
    if layer:
        sym = f"{sym},{layer}"
    if metric == "cosine":
        ylabel = rf"$\cos(C^{{{sym}}}_{{{t0}}}, C^{{{sym}}}_{{{t}}})$"
        return "displacement", ylabel, False
    if metric == "rel_frob":
        ylabel = (
            rf"$\|\Delta C^{{{sym}}}\|_F / \|C^{{{sym}}}_{{{t0}}}\|_F$"
        )
        return "rel_displacement", ylabel, True
    raise ValueError(f"unknown displacement metric {metric!r}")


def _displacement_plot_filename(cosine_name, metric):
    if metric == "cosine":
        return cosine_name
    if metric == "rel_frob":
        stem, ext = os.path.splitext(cosine_name)
        if stem.endswith("_vs_time"):
            return f"{stem[:-8]}_rel_vs_time{ext}"
        if stem.endswith("_vs_loss"):
            return f"{stem[:-8]}_rel_vs_loss{ext}"
        return cosine_name.replace(
            "kernel_displacement", "kernel_rel_displacement", 1
        )
    raise ValueError(f"unknown displacement metric {metric!r}")


_CONCENTRATION_FILENAME_TAGS = {
    "cosine": None,
    "rel_frob": "rel",
    "cka": "cka",
}


def _insert_rel_before_vs(name, metric):
    """Insert a metric tag before the first ``_vs_`` in a plot filename.

    Cosine keeps the untagged name. Relative Frobenius inserts ``_rel``;
    CKA inserts ``_cka``.
    """
    if metric not in _CONCENTRATION_FILENAME_TAGS:
        raise ValueError(f"unknown metric {metric!r}")
    tag = _CONCENTRATION_FILENAME_TAGS[metric]
    if tag is None:
        return name
    stem, ext = os.path.splitext(name)
    idx = stem.find("_vs_")
    if idx >= 0:
        return f"{stem[:idx]}_{tag}{stem[idx:]}{ext}"
    return f"{stem}_{tag}{ext}"


def _concentration_metric_spec(metric, feature_symbol):
    """Return ``(value_col, std_col, ylabel, invert_ylim)`` for concentration.

    ``metric`` is ``"cosine"`` (mean pairwise cosine similarity),
    ``"cka"`` (mean pairwise linear CKA), or ``"rel_frob"`` (mean
    pairwise relative Frobenius). Relative Frobenius inverts the y-axis
    so 0 (identical kernels) sits at the top.
    """
    sym = _feature_kernel_tex(feature_symbol)
    if metric == "cosine":
        ylabel = rf"mean pairwise $\cos(C^{{{sym},\ell}})$"
        return "cosine_mean", "cosine_std", ylabel, False
    if metric == "cka":
        ylabel = rf"mean pairwise $\mathrm{{CKA}}(C^{{{sym},\ell}})$"
        return "cka_mean", "cka_std", ylabel, False
    if metric == "rel_frob":
        ylabel = rf"mean pairwise $\|\Delta C^{{{sym},\ell}}\|_F$"
        return "rel_mean", "rel_std", ylabel, True
    raise ValueError(f"unknown concentration metric {metric!r}")


def _maybe_set_ylim(ax, ylim):
    if ylim is not None:
        ax.set_ylim(*ylim)


def _markersize_for_n(n):
    """Marker size for a curve of ``n`` points at the printed panel size.

    Dense curves get smaller markers so the line stays visible; sparse
    curves keep the default from the shared style.
    """
    return 1.6 if n > 20 else ps.MARKER_SIZE


#: Line style per data source, so colour is free to encode a swept value.
_SOURCE_STYLES = {"dmft": "--", "nn": "-", "closed_form": "-."}

_SOURCE_LABELS = {
    "dmft": ps.LABEL_DMFT,
    "nn": ps.LABEL_NN,
    "closed_form": ps.LABEL_NN_CLOSED_FORM,
}


def _proxy_line(label, *, color="0.35", linestyle="-", marker=None):
    """Legend-only handle."""
    return Line2D(
        [], [], color=color, linestyle=linestyle, marker=marker, label=label
    )


def _source_legend(ax, value_handles, *, dmft=False, nn=False, closed_form=False):
    """Legend keyed by swept value (colour) plus data source (line style).

    Source entries are drawn in grey and only appear when more than one
    source is present, since a single source is stated in the caption.
    """
    drawn = [
        key
        for key, present in (
            ("dmft", dmft), ("nn", nn), ("closed_form", closed_form)
        )
        if present
    ]
    handles = list(value_handles)
    if len(drawn) > 1:
        handles += [
            _proxy_line(_SOURCE_LABELS[key], linestyle=_SOURCE_STYLES[key])
            for key in drawn
        ]
    if not handles:
        return
    # One column: the source labels are too wide for a panel this narrow
    # to carry two of them side by side.
    ax.legend(handles=handles)


def _apply_x_axis(ax, *, xlabel=None, xscale="linear", invert_x=False):
    """Set xlabel / xscale; invert at most once (safe with shared axes)."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if xscale == "log" and ax.get_xscale() != "log":
        ax.set_xscale("log")
    if invert_x and not ax.xaxis_inverted():
        ax.invert_xaxis()


def plot_dmft_loss(dmft_loss, plots_dir, gamma_0=None):
    """Plot DMFT loss over training iterations."""
    dmft_loss = np.asarray(dmft_loss).flatten()
    _warn_if_nonfinite("dmft_loss", dmft_loss)
    iterations = np.arange(1, len(dmft_loss) + 1)

    plt.figure(figsize=ps.PANEL_THIRD)
    plt.plot(iterations, dmft_loss, color=ps.COLOR_REFERENCE)
    plt.xlabel(ps.TEX["time"])
    plt.ylabel(ps.LOSS_LABEL)
    ps.style_axes(plt.gca())
    ps.save_figure(plt.gcf(), os.path.join(plots_dir, "dmft_loss.png"))


def _plot_layer_kernel_row(kernels, save_path, row_label, xlabel, ylabel):
    """One row of per-layer kernel heatmaps, laid out like the grids."""
    kernels = [np.asarray(k) for k in kernels]
    n_layers = len(kernels)
    clim_kw = ps.symmetric_clim(kernels)
    geom = _kernel_grid_geometry(
        1,
        n_layers,
        fig_width=ps.PANEL_HALF[0] if n_layers < 4 else ps.TEXT_WIDTH_IN,
        row_labels=(row_label,),
        has_axis_symbols=True,
        has_col_labels=True,
        cbar=True,
        cbar_label=None,
        cbar_ticklabels=_kernel_cbar_ticklabels(clim_kw),
    )
    fig = plt.figure(figsize=geom["figsize"], layout="none")
    for l, kernel in enumerate(kernels):
        ax = fig.add_axes(_kernel_cell_rect(geom, 0, l))
        im = ax.imshow(kernel, cmap=ps.KERNEL_CMAP, aspect="auto", **clim_kw)
        ax.set_title(ps.layer_label(l))
        _style_kernel_heatmap_ax(
            ax,
            xlabel=xlabel if l == 0 else None,
            ylabel=ylabel if l == 0 else None,
        )
    _place_kernel_row_labels(fig, geom, (row_label,))
    _add_kernel_colorbar(fig, geom, im, None)
    return ps.save_figure(fig, save_path)


def plot_H_kernels(all_H, plots_dir, gamma_0=None):
    """Plot final-time slice of H kernels for each layer."""
    all_H = _to_numpy(all_H)
    _warn_if_nonfinite("all_H", np.stack([np.asarray(H) for H in all_H]))
    _plot_layer_kernel_row(
        [np.asarray(H_l[-1, :, -1, :]) for H_l in all_H],
        os.path.join(plots_dir, "all_H_kernels.png"),
        row_label=ps.LABEL_DMFT,
        xlabel=r"$\mu$",
        ylabel=r"$\nu$",
    )


def plot_G_kernels(all_G, plots_dir, gamma_0=None):
    """Plot G kernels for each layer.

    Linear DMFT stores ``G`` as ``(T, T)``; nonlinear DMFT stores
    ``(T, P, T, P)``, in which case the final-time sample-sample block is shown.
    """
    all_G = _to_numpy(all_G)
    _warn_if_nonfinite("all_G", np.stack([np.asarray(G) for G in all_G]))
    kernels = []
    for G_l in all_G:
        G_arr = np.asarray(G_l)
        kernels.append(G_arr[-1, :, -1, :] if G_arr.ndim == 4 else G_arr)
    _plot_layer_kernel_row(
        kernels,
        os.path.join(plots_dir, "all_G_kernels.png"),
        row_label=ps.LABEL_DMFT,
        xlabel=r"$\mu$",
        ylabel=r"$\nu$",
    )


def plot_dmft_kernels_and_loss(
    all_H,
    all_G,
    dmft_loss,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
):
    """Plot DMFT loss and kernel matrices under the BP plots subdirectory."""
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    plots_dir = os.path.join(plots_dir, "bp")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dmft_loss(dmft_loss, plots_dir, gamma_0=gamma_0)
    plot_H_kernels(all_H, plots_dir, gamma_0=gamma_0)
    plot_G_kernels(all_G, plots_dir, gamma_0=gamma_0)
    return plots_dir


def _initial_sample_kernel(cov, num_inference_steps, num_training_steps, num_samples):
    """
    Extract the sample-sample block at the first inference step (k=0) and
    last training time from a flattened ((K+1)*T*P, (K+1)*T*P) covariance.

    The PC kernels store the full inference trajectory k=0,...,K, so the
    state dimension per (t, mu) block is K+1, not K.
    """
    K1 = num_inference_steps + 1
    T = num_training_steps
    P = num_samples
    tensor = np.asarray(cov).reshape(K1, T, P, K1, T, P)
    return tensor[0, -1, :, 0, -1, :]


def plot_pc_layer_kernels(
    kernels,
    plots_dir,
    filename,
    num_inference_steps,
    num_training_steps,
    num_samples,
    gamma_0=None,
    ylabel=ps.LABEL_DMFT,
):
    """Plot initial sample-sample kernels for each PC layer."""
    kernels = _to_numpy(kernels)
    _warn_if_nonfinite(
        filename,
        np.stack([np.asarray(k) for k in kernels]),
    )
    _plot_layer_kernel_row(
        [
            _initial_sample_kernel(
                cov_l,
                num_inference_steps=num_inference_steps,
                num_training_steps=num_training_steps,
                num_samples=num_samples,
            )
            for cov_l in kernels
        ],
        os.path.join(plots_dir, filename),
        row_label=ylabel,
        xlabel=r"$\mu$",
        ylabel=r"$\nu$",
    )


def plot_pc_dmft_kernels_and_loss(
    all_Ch,
    all_Cdelta,
    pc_dmft_loss,
    plots_dir,
    num_inference_steps,
    num_training_steps,
    num_samples,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
    feature_symbol="h",
):
    """Plot PC DMFT loss and initial sample-sample Ch/Cphi / Cdelta kernels.

    ``all_Ch`` is the linear ``C^h`` or nonlinear ``C^phi`` (callers often
    keep the name ``all_Ch`` even when the nonlinear solver returns
    ``all_Cphi``). ``feature_symbol`` selects the ylabel (``"h"`` or
    ``"phi"``).
    """
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    plots_dir = os.path.join(plots_dir, "pc")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dmft_loss(pc_dmft_loss, plots_dir, gamma_0=gamma_0)
    # Rename the generic loss file for clarity.
    generic_loss = os.path.join(plots_dir, "dmft_loss.png")
    pc_loss = os.path.join(plots_dir, "pc_dmft_loss.png")
    if os.path.exists(generic_loss):
        os.replace(generic_loss, pc_loss)

    sym = _feature_kernel_tex(feature_symbol)
    plot_pc_layer_kernels(
        kernels=all_Ch,
        plots_dir=plots_dir,
        filename=(
            "all_Cphi_kernels.png"
            if feature_symbol == "phi"
            else "all_Ch_kernels.png"
        ),
        num_inference_steps=num_inference_steps,
        num_training_steps=num_training_steps,
        num_samples=num_samples,
        gamma_0=gamma_0,
        ylabel=rf"$C^{{{sym}}}$",
    )
    plot_pc_layer_kernels(
        kernels=all_Cdelta,
        plots_dir=plots_dir,
        filename="all_Cdelta_kernels.png",
        num_inference_steps=num_inference_steps,
        num_training_steps=num_training_steps,
        num_samples=num_samples,
        gamma_0=gamma_0,
        ylabel=r"$C^\Delta$",
    )
    return plots_dir


def plot_pc_theory_vs_finite_loss(
    pc_dmft_loss,
    finite_df,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
    n_infer_iters=None,
    update_mode=None,
    skip_theory=False,
    skip_finite=False,
    figsize=None,
):
    """Overlay the PC DMFT theory loss curve with finite-size empirical losses.

    ``finite_df`` is expected to have columns ``width``, ``t`` and ``loss``,
    as produced by ``test_coord_check.get_coord_data(..., stats=["loss"])``
    run with the same hyperparameters used for ``pc_dmft_loss``. This lets us
    visually check that the finite-size PC networks converge to the DMFT
    (infinite-width) prediction as width grows.

    ``update_mode`` (e.g. ``"infer"`` / ``"theory"``) is optional metadata used
    in the output filename so multiple finite simulations can be compared
    without overwriting each other.

    If ``skip_theory`` is True (or ``pc_dmft_loss`` is None / all zeros), only
    finite overlays are drawn and the filename uses ``pc_finite_loss``.
    If ``skip_finite`` is True (or ``finite_df`` is empty), only the DMFT
    theory curve is drawn and the filename uses ``pc_theory_loss``.
    """
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    if n_infer_iters is not None:
        plots_dir = os.path.join(plots_dir, f"{n_infer_iters}_n_infer_iters")
    plots_dir = os.path.join(plots_dir, "pc")
    os.makedirs(plots_dir, exist_ok=True)

    # None / all-zeros placeholder: plot finite overlays only (e.g. --skip_theory).
    plot_theory = (not skip_theory) and pc_dmft_loss is not None
    if plot_theory:
        pc_dmft_loss = np.asarray(pc_dmft_loss).flatten()
        if np.allclose(pc_dmft_loss, 0.0):
            plot_theory = False
        else:
            _warn_if_nonfinite("pc_dmft_loss", pc_dmft_loss)

    plot_finite = (
        (not skip_finite)
        and finite_df is not None
        and len(finite_df)
        and "width" in finite_df.columns
    )
    widths = (
        sorted(finite_df["width"].unique()) if plot_finite else []
    )
    colors = ps.sequence_colors(len(widths))

    plt.figure(figsize=figsize or ps.PANEL_THIRD)
    if plot_finite:
        markersize = _markersize_for_n(finite_df["t"].nunique())
        for width, color in zip(widths, colors):
            sub = finite_df[finite_df["width"] == width].sort_values("t")
            plt.plot(
                sub["t"],
                sub["loss"],
                marker="o",
                markersize=markersize,
                color=color,
                label=ps.width_label(width),
            )
    if plot_theory:
        theory_t = np.arange(1, len(pc_dmft_loss) + 1)
        plt.plot(
            theory_t,
            pc_dmft_loss,
            color=ps.COLOR_REFERENCE,
            linestyle="--",
            label=ps.LABEL_DMFT,
        )
    plt.xlabel(ps.TEX["time"])
    plt.ylabel(ps.LOSS_LABEL)
    if skip_finite or (plot_theory and not plot_finite):
        filename = "pc_theory_loss"
    elif skip_theory or not plot_theory:
        filename = "pc_finite_loss"
    else:
        filename = "pc_theory_vs_finite_loss"
    if update_mode is not None:
        filename += f"_{update_mode}"
    plt.legend()
    ps.style_axes(plt.gca())
    save_path = os.path.join(plots_dir, f"{filename}.png")
    ps.save_figure(plt.gcf(), save_path)
    print(f"PC loss plot saved to {save_path}")
    return save_path


_SWEEP_META_COLS = (
    "n_hidden",
    "gamma_0",
    "activity_lr",
    "n_infer_iters",
    "param_type",
    "use_skips",
)

_SWEEP_VALUE_LABEL = {
    "n_hidden": lambda v: rf"$H = {int(v)}$",
    "gamma_0": lambda v: rf"$\gamma_0 = {ps.fmt_number(v)}$",
    "n_infer_iters": lambda v: rf"$K = {int(v)}$",
}

_SWEEP_FILENAME = {
    "n_hidden": "pc_loss_vs_n_hidden",
    "gamma_0": "pc_loss_vs_gamma_0",
    "n_infer_iters": "pc_loss_vs_n_infer_iters",
}


def _pc_loss_plots_dir(
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
):
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    if n_infer_iters is not None:
        plots_dir = os.path.join(plots_dir, f"{n_infer_iters}_n_infer_iters")
    plots_dir = os.path.join(plots_dir, "pc")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _mask_equal(df, col, value):
    """Boolean mask comparing ``df[col]`` to ``value``, with NA-safe equality."""
    series = df[col]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return series.isna()
    return series == value


def plot_pc_param_sweep_loss(
    theory_df,
    finite_df,
    plots_dir,
    swept_col,
    skip_theory=False,
    skip_finite=False,
    plot_closed_form=False,
    figsize=None,
):
    """Overlay theory and finite-size losses for every value of ``swept_col``.

    Finite curves use only the largest recorded width. Theory is dashed; finite
    infer is solid. If ``plot_closed_form`` is True, closed-form finite updates
    are added: one curve per swept value, except for ``n_infer_iters`` where
    closed-form is independent of ``K`` so a single extra curve is drawn.

    For ``n_infer_iters``, the DMFT theory curve is drawn only for the
    smallest ``K`` (larger ``K`` are finite overlays only).

    If ``skip_theory`` is True, only finite overlays are drawn. If
    ``skip_finite`` is True, only DMFT theory curves are drawn.
    """
    if swept_col not in _SWEEP_VALUE_LABEL:
        raise ValueError(
            f"swept_col must be one of {list(_SWEEP_VALUE_LABEL)}, got {swept_col!r}"
        )

    group_cols = [c for c in _SWEEP_META_COLS if c != swept_col]
    frames = []
    if (not skip_theory) and theory_df is not None and len(theory_df):
        frames.append(theory_df[group_cols])
    if (not skip_finite) and finite_df is not None and len(finite_df):
        infer_df = finite_df[finite_df["infer_mode"] == "infer"]
        if len(infer_df):
            frames.append(infer_df[group_cols])
        elif len(finite_df):
            frames.append(finite_df[group_cols])
    if not frames:
        return []

    groups = pd.concat(frames, ignore_index=True).drop_duplicates()
    save_paths = []
    value_label = _SWEEP_VALUE_LABEL[swept_col]
    filename = _SWEEP_FILENAME[swept_col]

    for _, group in groups.iterrows():
        def _in_group(df):
            mask = pd.Series(True, index=df.index)
            for col in group_cols:
                mask &= _mask_equal(df, col, group[col])
            return df.loc[mask]

        g_theory = (
            _in_group(theory_df)
            if theory_df is not None and len(theory_df)
            else theory_df
        )
        g_finite = (
            _in_group(finite_df)
            if (not skip_finite)
            and finite_df is not None
            and len(finite_df)
            else None
        )

        infer_finite = (
            g_finite[g_finite["infer_mode"] == "infer"]
            if g_finite is not None and len(g_finite)
            else g_finite
        )
        closed_finite = (
            g_finite[g_finite["infer_mode"] == "closed_form"]
            if g_finite is not None and len(g_finite)
            else g_finite
        )

        swept_values = []
        for source in (g_theory, infer_finite):
            if source is not None and len(source) and swept_col in source.columns:
                swept_values.extend(source[swept_col].dropna().unique().tolist())
        # Preserve numeric order while keeping first-seen type.
        swept_values = sorted(set(swept_values), key=lambda v: (float(v), str(v)))
        if not swept_values:
            continue

        max_width = None
        if infer_finite is not None and len(infer_finite):
            max_width = int(infer_finite["width"].max())
        elif closed_finite is not None and len(closed_finite):
            max_width = int(closed_finite["width"].max())

        colors = ps.sequence_colors(len(swept_values))
        markersize = _markersize_for_n(
            infer_finite["t"].nunique()
            if infer_finite is not None and len(infer_finite)
            else 0
        )

        plt.figure(figsize=figsize or ps.PANEL_THIRD)
        ax = plt.gca()
        plot_theory = (
            (not skip_theory)
            and g_theory is not None
            and len(g_theory)
        )
        # Colour encodes the swept value; line style encodes the source.
        value_handles = []
        drew_dmft = drew_nn = drew_closed_form = False
        # K-sweep: DMFT is shown only for the smallest inference-step count.
        min_swept = min(swept_values, key=lambda v: float(v))
        for value, color in zip(swept_values, colors):
            plot_theory_this = plot_theory and (
                swept_col != "n_infer_iters"
                or float(value) == float(min_swept)
            )
            if plot_theory_this:
                sub_th = g_theory[g_theory[swept_col] == value].sort_values("t")
                if len(sub_th):
                    y = np.asarray(sub_th["loss"])
                    if not np.allclose(y, 0.0):
                        _warn_if_nonfinite(f"pc_dmft_loss[{swept_col}={value}]", y)
                        ax.plot(
                            sub_th["t"],
                            y,
                            color=color,
                            linestyle=_SOURCE_STYLES["dmft"],
                        )
                        drew_dmft = True
            if infer_finite is not None and len(infer_finite):
                sub_fi = infer_finite[
                    (infer_finite[swept_col] == value)
                    & (infer_finite["width"] == max_width)
                ].sort_values("t")
                if len(sub_fi):
                    ax.plot(
                        sub_fi["t"],
                        sub_fi["loss"],
                        color=color,
                        linestyle=_SOURCE_STYLES["nn"],
                        marker="o",
                        markersize=markersize,
                    )
                    drew_nn = True
            if (
                plot_closed_form
                and swept_col != "n_infer_iters"
                and closed_finite is not None
                and len(closed_finite)
            ):
                sub_cf = closed_finite[
                    (closed_finite[swept_col] == value)
                    & (closed_finite["width"] == max_width)
                ].sort_values("t")
                if len(sub_cf):
                    ax.plot(
                        sub_cf["t"],
                        sub_cf["loss"],
                        color=color,
                        linestyle=_SOURCE_STYLES["closed_form"],
                        marker="s",
                        markersize=markersize,
                    )
                    drew_closed_form = True
            value_handles.append(
                _proxy_line(value_label(value), color=color)
            )

        if (
            plot_closed_form
            and swept_col == "n_infer_iters"
            and closed_finite is not None
            and len(closed_finite)
        ):
            cf_width = (
                int(closed_finite["width"].max())
                if max_width is None
                else max_width
            )
            sub_cf = closed_finite[closed_finite["width"] == cf_width].sort_values(
                "t"
            )
            # Closed-form does not depend on K; keep a single curve.
            sub_cf = sub_cf.drop_duplicates(subset=["t"], keep="first")
            if len(sub_cf):
                ax.plot(
                    sub_cf["t"],
                    sub_cf["loss"],
                    color=ps.COLOR_REFERENCE,
                    linestyle=_SOURCE_STYLES["closed_form"],
                )
                drew_closed_form = True

        ax.set_xlabel(ps.TEX["time"])
        ax.set_ylabel(ps.LOSS_LABEL)
        _source_legend(
            ax,
            value_handles,
            dmft=drew_dmft,
            nn=drew_nn,
            closed_form=drew_closed_form,
        )
        ps.style_axes(ax)

        out_dir = _pc_loss_plots_dir(
            plots_dir,
            n_hidden=group["n_hidden"] if swept_col != "n_hidden" else None,
            gamma_0=group["gamma_0"] if swept_col != "gamma_0" else None,
            activity_lr=group["activity_lr"],
            n_infer_iters=(
                group["n_infer_iters"] if swept_col != "n_infer_iters" else None
            ),
        )
        save_path = os.path.join(out_dir, f"{filename}.png")
        ps.save_figure(plt.gcf(), save_path)
        print(f"PC sweep loss plot saved to {save_path}")
        save_paths.append(save_path)

    return save_paths


def plot_bp_theory_vs_finite_loss(
    dmft_loss,
    finite_df,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    skip_theory=False,
):
    """Overlay the BP DMFT theory loss curve with finite-size empirical losses.

    ``finite_df`` is expected to have columns ``width``, ``t`` and ``loss``,
    as produced by ``train_bpn`` over the same hyperparameters used for
    ``dmft_loss``. This lets us visually check that the finite-size BP
    networks converge to the DMFT (infinite-width) prediction as width grows.

    If ``skip_theory`` is True (or ``dmft_loss`` is None / all zeros), only
    finite overlays are drawn and the title/filename use ``bp_finite_loss``.
    """
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    plots_dir = os.path.join(plots_dir, "bp")
    os.makedirs(plots_dir, exist_ok=True)

    # None / all-zeros placeholder: plot finite overlays only (e.g. --skip_theory).
    plot_theory = (not skip_theory) and dmft_loss is not None
    if plot_theory:
        dmft_loss = np.asarray(dmft_loss).flatten()
        if np.allclose(dmft_loss, 0.0):
            plot_theory = False
        else:
            _warn_if_nonfinite("dmft_loss", dmft_loss)

    widths = sorted(finite_df["width"].unique())
    colors = ps.sequence_colors(len(widths))

    plt.figure(figsize=ps.PANEL_THIRD)
    for width, color in zip(widths, colors):
        sub = finite_df[finite_df["width"] == width].sort_values("t")
        plt.plot(
            sub["t"],
            sub["loss"],
            marker="o",
            color=color,
            label=ps.width_label(width),
        )
    if plot_theory:
        theory_t = np.arange(1, len(dmft_loss) + 1)
        plt.plot(
            theory_t,
            dmft_loss,
            color=ps.COLOR_REFERENCE,
            linestyle=_SOURCE_STYLES["dmft"],
            label=ps.LABEL_DMFT,
        )
    plt.xlabel(ps.TEX["time"])
    plt.ylabel(ps.LOSS_LABEL)
    if skip_theory or not plot_theory:
        filename = "bp_finite_loss"
    else:
        filename = "bp_theory_vs_finite_loss"
    plt.legend()
    ps.style_axes(plt.gca())
    save_path = os.path.join(plots_dir, f"{filename}.png")
    ps.save_figure(plt.gcf(), save_path)
    print(f"BP loss plot saved to {save_path}")
    return save_path

def plot_grad_cosine_similarities(
    similarities_by_width,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
):
    """Plot PC–BP gradient cosine similarity over training time.

    ``similarities_by_width`` maps width -> 1D array of cosine similarities
    indexed by training step. Saved under the plots tree (not under
    ``*_input_dim``), so ``--cleanup_npy`` will not delete it.
    """
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    os.makedirs(plots_dir, exist_ok=True)

    widths = sorted(similarities_by_width.keys())
    if not widths:
        print("No cosine similarity curves to plot.")
        return None

    colors = ps.sequence_colors(len(widths))

    plt.figure(figsize=ps.PANEL_THIRD)
    for width, color in zip(widths, colors):
        values = np.asarray(similarities_by_width[width]).flatten()
        _warn_if_nonfinite(f"cos_sim width={width}", values)
        t = np.arange(1, len(values) + 1)
        plt.plot(
            t,
            values,
            marker="o",
            color=color,
            label=ps.width_label(width),
        )
    plt.xlabel(ps.TEX["time"])
    plt.ylabel(r"$\cos(\nabla_{\theta}\mathcal{L}_{\mathrm{BP}}, "
               r"\nabla_{\theta}\mathcal{F}^*_{\mathrm{PC}})$")
    plt.ylim(-1.05, 1.05)
    plt.legend()
    ps.style_axes(plt.gca())
    save_path = os.path.join(plots_dir, "grad_cosine_similarities.png")
    ps.save_figure(plt.gcf(), save_path)
    print(f"PC–BP gradient cosine similarity plot saved to {save_path}")
    return save_path


def plot_pc_kernel_width_alignment(
    align_df,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
    n_infer_iters=None,
    feature_symbol="h",
    figsize=None,
):
    """Plot finite-vs-theory kernel alignment vs width for every layer.

    ``align_df`` must have columns ``width``, ``layer``, ``kernel``
    (``"h"`` for the feature kernel ``C^h``/``C^phi``, or ``"delta"``),
    ``alignment``, and optionally ``seed``. Mean ± std over seeds is shown
    when multiple trials are present. Compares last-training-time
    hidden-layer kernels (feature kernel at ``k=0``, ``C^Δ`` at the last
    inference step ``k=K``) for all ``H`` hidden layers (readout omitted).

    ``feature_symbol`` is ``"h"`` (linear) or ``"phi"`` (nonlinear) and
    only affects axis labels / filenames.
    """
    plots_dir = _pc_loss_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
    )
    if align_df is None or len(align_df) == 0:
        print("No kernel-alignment records to plot.")
        return []

    sym = _feature_kernel_tex(feature_symbol)
    feature_filename = (
        "pc_kernel_convergence_Cphi_vs_width.png"
        if feature_symbol == "phi"
        else "pc_kernel_convergence_Ch_vs_width.png"
    )
    # The layer index lives in the legend, so it is dropped from the
    # kernel symbol to keep the y-label short enough for a narrow panel.
    specs = (
        (
            "h",
            rf"$\cos(C^{{{sym}}}_{{\mathrm{{DMFT}}}}, "
            rf"C^{{{sym}}}_{{\mathrm{{NN}}}})$",
            feature_filename,
        ),
        (
            "delta",
            r"$\cos(C^{\Delta}_{\mathrm{DMFT}}, C^{\Delta}_{\mathrm{NN}})$",
            "pc_kernel_convergence_Cdelta_vs_width.png",
        ),
    )
    saved = []
    for kernel_id, ylabel, filename in specs:
        sub = align_df[align_df["kernel"] == kernel_id]
        if sub.empty:
            continue

        plt.figure(figsize=figsize or ps.PANEL_THIRD)
        ax = plt.gca()
        layers = sorted(sub["layer"].unique())
        colors = ps.sequence_colors(len(layers))
        for layer, color in zip(layers, colors):
            sl = sub[sub["layer"] == layer]
            widths = sorted(sl["width"].unique())
            means = []
            stds = []
            for width in widths:
                vals = sl.loc[sl["width"] == width, "alignment"].to_numpy(
                    dtype=float
                )
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=0))
            means = np.asarray(means)
            stds = np.asarray(stds)
            _warn_if_nonfinite(
                f"kernel alignment {kernel_id} layer={int(layer)}", means
            )
            ax.errorbar(
                widths,
                means,
                stds,
                marker="o",
                color=color,
                label=ps.layer_label(layer),
            )
        ax.set_xscale("log")
        ax.set_xlabel(ps.TEX["width"])
        ax.set_ylabel(ylabel)
        ax.legend(ncol=2 if len(layers) > 3 else 1)
        ps.style_axes(ax)
        save_path = os.path.join(plots_dir, filename)
        ps.save_figure(plt.gcf(), save_path)
        print(f"PC kernel alignment plot saved to {save_path}")
        saved.append(save_path)
    return saved


def _alignment_plots_dir(
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    dir_name="alignment",
):
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    if n_infer_iters is not None:
        plots_dir = os.path.join(plots_dir, f"{n_infer_iters}_n_infer_iters")
    plots_dir = os.path.join(plots_dir, dir_name)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_kernel_displacement(
    displacement_df,
    plots_dir,
    n_hidden=None,
    activity_lr=None,
    feature_symbol="h",
):
    """Cosine similarity between the initial (``t=0``) and final (last
    training step) feature kernel at every layer, at ``k=0`` for PC --
    i.e. how much each layer's feature kernel moved during training.

    ``displacement_df`` must have columns ``layer``, ``method`` (``"pc"``
    or ``"bp"``), ``displacement``, ``gamma_0`` and ``n_infer_iters``.

    If a single ``(gamma_0, n_infer_iters)`` combination is present, PC
    and BP are overlaid as two curves. If several combinations are
    present (``gamma_0s`` and/or ``n_infer_iters`` swept), only PC is
    drawn, with one curve per combination (BP does not depend on
    ``n_infer_iters`` and is dropped in this case).

    ``feature_symbol`` is ``"h"`` (linear) or ``"phi"`` (nonlinear).
    """
    if displacement_df is None or len(displacement_df) == 0:
        print("No kernel displacement records to plot.")
        return None

    combos = (
        displacement_df[["gamma_0", "n_infer_iters"]]
        .drop_duplicates()
        .sort_values(["gamma_0", "n_infer_iters"])
    )
    overlay = len(combos) > 1
    sym = _feature_kernel_tex(feature_symbol)

    plt.figure(figsize=ps.PANEL_THIRD)
    ax = plt.gca()
    if not overlay:
        # Per-column access (not row-wise .iloc) to avoid pandas upcasting
        # gamma_0 (float) and n_infer_iters (int) to a common dtype.
        gamma_0 = combos["gamma_0"].iloc[0]
        n_infer_iters = int(combos["n_infer_iters"].iloc[0])

        pc_sub = displacement_df[
            displacement_df["method"] == "pc"
        ].sort_values("layer")
        layers = np.asarray(pc_sub["layer"], dtype=float) + 1
        values = np.asarray(pc_sub["displacement"], dtype=float)
        _warn_if_nonfinite("pc displacement", values)
        ax.plot(
            layers, values, marker="o", color=ps.COLOR_PC, label=ps.LABEL_PC
        )

        bp_sub = displacement_df[
            displacement_df["method"] == "bp"
        ].sort_values("layer")
        if len(bp_sub):
            bp_layers = np.asarray(bp_sub["layer"], dtype=float) + 1
            bp_values = np.asarray(bp_sub["displacement"], dtype=float)
            _warn_if_nonfinite("bp displacement", bp_values)
            ax.plot(
                bp_layers,
                bp_values,
                marker="s",
                color=ps.COLOR_BP,
                label=ps.LABEL_BP,
            )

        out_dir = _alignment_plots_dir(
            plots_dir,
            n_hidden=n_hidden,
            gamma_0=gamma_0,
            activity_lr=activity_lr,
            n_infer_iters=n_infer_iters,
        )
    else:
        colors = ps.sequence_colors(len(combos))
        pc_df = displacement_df[displacement_df["method"] == "pc"]
        # itertuples (not iterrows) keeps each field's own dtype, since
        # iterrows would coerce a mixed float/int row to a common dtype.
        for row, color in zip(combos.itertuples(index=False), colors):
            sub = pc_df[
                (pc_df["gamma_0"] == row.gamma_0)
                & (pc_df["n_infer_iters"] == row.n_infer_iters)
            ].sort_values("layer")
            if not len(sub):
                continue
            layers = np.asarray(sub["layer"], dtype=float) + 1
            values = np.asarray(sub["displacement"], dtype=float)
            _warn_if_nonfinite(
                f"pc displacement (gamma_0={row.gamma_0}, "
                f"K={row.n_infer_iters})",
                values,
            )
            label = (
                rf"$\gamma_0 = {ps.fmt_number(row.gamma_0)}$, "
                rf"$K = {int(row.n_infer_iters)}$"
            )
            ax.plot(layers, values, marker="o", color=color, label=label)

        out_dir = _alignment_plots_dir(
            plots_dir, n_hidden=n_hidden, activity_lr=activity_lr
        )

    ax.set_xlabel(ps.TEX["layer"])
    ax.set_ylabel(rf"$\cos(C^{{{sym}}}_{{0}}, C^{{{sym}}}_{{T}})$")
    ps.integer_ticks(ax, np.asarray(displacement_df["layer"], dtype=float) + 1)
    _maybe_set_ylim(
        ax,
        _data_ylim(
            displacement_df["displacement"], invert=False, ymin_floor=0.0
        ),
    )
    ax.legend()
    ps.style_axes(ax)
    save_path = os.path.join(out_dir, "kernel_displacement_vs_layer.png")
    ps.save_figure(plt.gcf(), save_path)
    print(f"Kernel displacement plot saved to {save_path}")
    return save_path


def plot_pc_k_sweep_displacement(
    displacement_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    metric="cosine",
    figsize=None,
):
    """Overlay per-layer feature-kernel displacement for a ``K`` sweep.

    ``displacement_df`` must have columns ``layer``, ``kind``
    (``"dmft"``, ``"infer"``, or ``"closed_form"``), ``n_infer_iters``,
    and ``displacement`` (cosine) or ``rel_displacement`` (relative
    Frobenius), selected by ``metric``. One curve per series: DMFT
    (smallest ``K``, dashed), finite-size infer (solid, increasing
    ``K``), and closed-form (linear case, dash-dot).

    ``feature_symbol`` is ``"h"`` (linear) or ``"phi"`` (nonlinear).
    ``metric`` is ``"cosine"`` or ``"rel_frob"``.
    """
    if displacement_df is None or len(displacement_df) == 0:
        print("No K-sweep kernel displacement records to plot.")
        return None

    value_col, ylabel, invert = _displacement_metric_spec(
        metric, feature_symbol
    )
    ylim = _data_ylim(displacement_df[value_col], invert=invert)
    plt.figure(figsize=figsize or ps.PANEL_THIRD)
    ax = plt.gca()
    infer_ks = sorted(
        {
            int(k)
            for k in displacement_df.loc[
                displacement_df["kind"] == "infer", "n_infer_iters"
            ].dropna().unique()
        }
    )
    k_palette = ps.sequence_colors(len(infer_ks))
    k_colors = {k: k_palette[i] for i, k in enumerate(infer_ks)}
    all_layers = np.asarray(displacement_df["layer"], dtype=float) + 1
    value_handles = []
    drew_dmft = drew_nn = drew_closed_form = False

    dmft_sub = displacement_df[displacement_df["kind"] == "dmft"]
    if len(dmft_sub):
        dmft_k = int(dmft_sub["n_infer_iters"].iloc[0])
        sub = dmft_sub.sort_values("layer")
        layers = np.asarray(sub["layer"], dtype=float) + 1
        values = np.asarray(sub[value_col], dtype=float)
        _warn_if_nonfinite(f"dmft displacement (K={dmft_k})", values)
        ax.plot(
            layers,
            values,
            color=k_colors.get(dmft_k, ps.COLOR_REFERENCE),
            linestyle=_SOURCE_STYLES["dmft"],
            marker="o",
        )
        drew_dmft = True

    for k in infer_ks:
        sub = displacement_df[
            (displacement_df["kind"] == "infer")
            & (displacement_df["n_infer_iters"].astype(int) == int(k))
        ].sort_values("layer")
        if not len(sub):
            continue
        layers = np.asarray(sub["layer"], dtype=float) + 1
        values = np.asarray(sub[value_col], dtype=float)
        _warn_if_nonfinite(f"infer displacement (K={k})", values)
        ax.plot(
            layers,
            values,
            color=k_colors[k],
            linestyle=_SOURCE_STYLES["nn"],
            marker="o",
        )
        drew_nn = True
        value_handles.append(_proxy_line(ps.k_label(k), color=k_colors[k]))

    cf_sub = displacement_df[displacement_df["kind"] == "closed_form"]
    if len(cf_sub):
        sub = cf_sub.sort_values("layer")
        layers = np.asarray(sub["layer"], dtype=float) + 1
        values = np.asarray(sub[value_col], dtype=float)
        _warn_if_nonfinite("closed-form displacement", values)
        ax.plot(
            layers,
            values,
            color=ps.COLOR_REFERENCE,
            linestyle=_SOURCE_STYLES["closed_form"],
            marker="s",
        )
        drew_closed_form = True

    ax.set_xlabel(ps.TEX["layer"])
    ax.set_ylabel(ylabel)
    ps.integer_ticks(ax, all_layers)
    _maybe_set_ylim(ax, ylim)
    _source_legend(
        ax,
        value_handles,
        dmft=drew_dmft,
        nn=drew_nn,
        closed_form=drew_closed_form,
    )
    ps.style_axes(ax)
    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        dir_name=dir_name,
    )
    save_path = os.path.join(
        out_dir,
        _displacement_plot_filename(
            "kernel_displacement_vs_layer.png", metric
        ),
    )
    ps.save_figure(plt.gcf(), save_path)
    print(f"K-sweep kernel displacement plot saved to {save_path}")
    return save_path


def plot_kernel_displacement_per_timepoint(
    displacement_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    metric="cosine",
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
    filename="kernel_displacement_vs_time.png",
    t_sub="t",
):
    """Per-layer feature-kernel displacement from ``t0`` vs ``x_col``.

    ``displacement_df`` must have columns ``x_col``, ``layer``, ``method``
    (``"pc"`` or ``"bp"``), and ``displacement`` (cosine similarity of
    the kernel at the current point vs ``t0``) or ``rel_displacement``
    (relative Frobenius), selected by ``metric``. One subplot per hidden
    layer. For three or fewer layers the panels are arranged in a single
    row; otherwise an auto grid is used. Saved as ``filename`` (cosine)
    or the matching ``*_rel_vs_*`` name.

    ``feature_symbol`` is ``"h"`` (linear) or ``"phi"`` (nonlinear).
    ``t_sub`` is the TeX subscript for the current kernel (``t`` or
    ``L``). High-to-low loss axes should pass ``invert_x=True`` so
    training still reads left to right.
    """
    if displacement_df is None or len(displacement_df) == 0:
        print("No kernel displacement records to plot.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )

    layers = sorted(displacement_df["layer"].unique())
    n_l = len(layers)
    value_col, ylabel, invert = _displacement_metric_spec(
        metric, feature_symbol, t0="0", t=t_sub
    )
    ylim = _data_ylim(
        displacement_df[value_col],
        invert=invert,
        ymin_floor=0.0 if metric == "cosine" else None,
    )
    markersize = _markersize_for_n(displacement_df[x_col].nunique())

    fig, axes, nrows, ncols = _per_layer_axes(n_l)
    for i, layer in enumerate(layers):
        ax = axes[i // ncols, i % ncols]
        sub = displacement_df[displacement_df["layer"] == layer]
        _plot_pc_bp_series(
            ax,
            sub,
            x_col=x_col,
            value_col=value_col,
            markersize=markersize,
            label_prefix=f"displacement (layer={layer}) ",
        )
        _panel_label(ax, layer)
        _apply_x_axis(ax, xscale=xscale, invert_x=invert_x)
        _maybe_set_ylim(ax, ylim)
        ps.style_axes(ax)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        if i == 0:
            ax.legend()

    for j in range(n_l, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    _shared_xlabel(fig, xlabel)

    save_path = os.path.join(
        out_dir,
        _displacement_plot_filename(filename, metric),
    )
    ps.save_figure(fig, save_path)
    print(f"Kernel displacement plot saved to {save_path}")
    return save_path


_PC_BP_STYLES = (
    ("pc", ps.COLOR_PC, "o", ps.LABEL_PC),
    ("bp", ps.COLOR_BP, "s", ps.LABEL_BP),
)

_SPLIT_STYLES = (
    ("train", ps.COLOR_PC, "o", "-", "train"),
    ("test", ps.COLOR_BP, "s", "--", "test"),
)


def _per_layer_grid(n_l):
    """Rows and columns for one panel per layer, a single row if ``n_l <= 3``."""
    if n_l <= 3:
        return 1, n_l
    ncols = int(np.ceil(np.sqrt(n_l)))
    return int(np.ceil(n_l / ncols)), ncols


def _per_layer_axes(n_l, sharey=True):
    """One panel per layer, spanning the text width."""
    nrows, ncols = _per_layer_grid(n_l)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=ps.per_layer_figsize(ncols, nrows),
        squeeze=False,
        sharex=True,
        sharey=sharey,
    )
    return fig, axes, nrows, ncols


def _panel_label(ax, layer):
    """Identify a per-layer panel without a descriptive title."""
    ax.set_title(ps.layer_label(layer))


def _shared_xlabel(fig, xlabel):
    """One x label for a whole panel grid, since every panel shares the axis."""
    if xlabel:
        fig.supxlabel(xlabel)


def _snapshot_label(x_col, x):
    """Identify a panel holding one training snapshot, by ``t`` or loss."""
    if x_col == "loss":
        return rf"$\mathcal{{L}} = {ps.fmt_sci(x)}$"
    return rf"$t = {int(x)}$"


def _plot_pc_bp_series(
    ax,
    sub,
    *,
    x_col,
    value_col,
    markersize=None,
    label_prefix="",
    std_col=None,
):
    """Overlay PC and Backprop curves on ``ax`` from a per-layer slice.

    If ``std_col`` is set and present on ``sub``, draw sample-SD error
    bars instead of a plain line.
    """
    for method, color, marker, label in _PC_BP_STYLES:
        msub = sub[sub["method"] == method].sort_values(x_col)
        if not len(msub):
            continue
        x = np.asarray(msub[x_col], dtype=float)
        values = np.asarray(msub[value_col], dtype=float)
        _warn_if_nonfinite(f"{label_prefix}{method}", values)
        line_kw = dict(marker=marker, color=color, label=label)
        if markersize is not None:
            line_kw["markersize"] = markersize
        if std_col is not None and std_col in msub.columns:
            ax.errorbar(
                x,
                values,
                yerr=np.asarray(msub[std_col], dtype=float),
                **line_kw,
            )
        else:
            ax.plot(x, values, **line_kw)


def _plot_train_test_series(
    ax, sub, *, x_col, value_col, markersize=None, label_prefix=""
):
    """Overlay train and test curves on ``ax`` from a per-layer slice."""
    for split, color, marker, linestyle, label in _SPLIT_STYLES:
        ssub = sub[sub["split"] == split].sort_values(x_col)
        if not len(ssub):
            continue
        x = np.asarray(ssub[x_col], dtype=float)
        values = np.asarray(ssub[value_col], dtype=float)
        _warn_if_nonfinite(f"{label_prefix}{split}", values)
        line_kw = dict(
            marker=marker, color=color, linestyle=linestyle, label=label
        )
        if markersize is not None:
            line_kw["markersize"] = markersize
        ax.plot(x, values, **line_kw)


def plot_pc_bp_metric_vs_time(
    metric_df,
    plots_dir,
    *,
    value_col,
    ylabel,
    title,
    filename,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    invert_ylim=False,
    ymin_floor=None,
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
    std_col=None,
):
    """Per-layer PC vs backprop line plot vs ``x_col``.

    Same layout as ``plot_kernel_displacement_per_timepoint``: one
    subplot per hidden layer, blue circles for PC and orange squares
    for backprop. ``metric_df`` must have columns ``x_col``, ``layer``,
    ``method`` (``"pc"`` or ``"bp"``), and ``value_col``. Optional
    ``std_col`` draws sample-SD error bars.
    """
    if metric_df is None or len(metric_df) == 0:
        print(f"No records to plot for {filename}.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    layers = sorted(metric_df["layer"].unique())
    n_l = len(layers)
    ylim_vals = metric_df[value_col]
    if std_col is not None and std_col in metric_df.columns:
        std = metric_df[std_col]
        ylim_vals = np.concatenate(
            [
                np.asarray(ylim_vals - std, dtype=float),
                np.asarray(ylim_vals + std, dtype=float),
            ]
        )
    ylim = _data_ylim(
        ylim_vals, invert=invert_ylim, ymin_floor=ymin_floor
    )
    markersize = _markersize_for_n(metric_df[x_col].nunique())

    fig, axes, nrows, ncols = _per_layer_axes(n_l)
    for i, layer in enumerate(layers):
        ax = axes[i // ncols, i % ncols]
        sub = metric_df[metric_df["layer"] == layer]
        _plot_pc_bp_series(
            ax,
            sub,
            x_col=x_col,
            value_col=value_col,
            markersize=markersize,
            label_prefix=f"{value_col} (layer={layer}) ",
            std_col=std_col,
        )
        _panel_label(ax, layer)
        _apply_x_axis(ax, xscale=xscale, invert_x=invert_x)
        _maybe_set_ylim(ax, ylim)
        ps.style_axes(ax)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        if i == 0:
            ax.legend()

    for j in range(n_l, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    _shared_xlabel(fig, xlabel)

    save_path = os.path.join(out_dir, filename)
    ps.save_figure(fig, save_path)
    print(f"{title} saved to {save_path}")
    return save_path


def _axis_kwargs(x_col, xlabel, xscale, invert_x, ymin_floor=None):
    kw = dict(
        x_col=x_col, xlabel=xlabel, xscale=xscale, invert_x=invert_x
    )
    if ymin_floor is not None:
        kw["ymin_floor"] = ymin_floor
    return kw


def plot_kernel_target_alignment_vs_time(
    alignment_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    ylabel=None,
    filename="kernel_target_alignment_vs_time.png",
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
):
    """Per-layer CKA of PC / BP feature kernels with the target kernel.

    ``alignment_df`` must have columns ``x_col``, ``layer``, ``method``,
    and ``alignment``. The target kernel is ``C^y = Y Y^T``.
    """
    sym = _feature_kernel_tex(feature_symbol)
    if ylabel is None:
        ylabel = rf"$\mathrm{{CKA}}(C^{{{sym},\ell}}, C^{{y}})$"
    return plot_pc_bp_metric_vs_time(
        alignment_df,
        plots_dir,
        value_col="alignment",
        ylabel=ylabel,
        title="Feature-kernel alignment with the target",
        filename=filename,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        width=width,
        dir_name=dir_name,
        **_axis_kwargs(x_col, xlabel, xscale, invert_x, ymin_floor=0.0),
    )


def plot_kernel_target_change_alignment_vs_time(
    alignment_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    ylabel=None,
    filename="kernel_target_change_alignment_vs_time.png",
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
):
    """Per-layer CKA of PC / BP kernel *changes* with the target kernel.

    ``alignment_df`` must have columns ``x_col``, ``layer``, ``method``,
    and ``alignment``. The init point should be omitted (``ΔC=0``).
    """
    sym = _feature_kernel_tex(feature_symbol)
    if ylabel is None:
        ylabel = rf"$\mathrm{{CKA}}(\Delta C^{{{sym},\ell}}, C^{{y}})$"
    return plot_pc_bp_metric_vs_time(
        alignment_df,
        plots_dir,
        value_col="alignment",
        ylabel=ylabel,
        title="Feature-kernel change alignment with the target",
        filename=filename,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        width=width,
        dir_name=dir_name,
        **_axis_kwargs(x_col, xlabel, xscale, invert_x, ymin_floor=0.0),
    )


def plot_kernel_input_alignment_vs_time(
    alignment_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    ylabel=None,
    filename="kernel_input_alignment_vs_time.png",
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
):
    """Per-layer CKA of PC / BP feature kernels with the input kernel.

    ``alignment_df`` must have columns ``x_col``, ``layer``, ``method``,
    and ``alignment``. The input kernel is ``C^x = X X^T``.
    """
    sym = _feature_kernel_tex(feature_symbol)
    if ylabel is None:
        ylabel = rf"$\mathrm{{CKA}}(C^{{{sym},\ell}}, C^{{x}})$"
    return plot_pc_bp_metric_vs_time(
        alignment_df,
        plots_dir,
        value_col="alignment",
        ylabel=ylabel,
        title="Feature-kernel alignment with the input",
        filename=filename,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        width=width,
        dir_name=dir_name,
        **_axis_kwargs(x_col, xlabel, xscale, invert_x, ymin_floor=0.0),
    )


def plot_leading_evec_label_overlap_vs_time(
    overlap_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    ylabel=None,
    filename="evec_leading_overlap_label_vs_time.png",
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
):
    """Per-layer overlap of the centered leading eigenvector with labels.

    ``overlap_df`` must have columns ``x_col``, ``layer``, ``method``,
    and ``overlap``. Default ylabel is ``|cos(v_1, y)|``.
    """
    if ylabel is None:
        ylabel = r"$\left|\cos(v_1^{\ell}, y)\right|$"
    return plot_pc_bp_metric_vs_time(
        overlap_df,
        plots_dir,
        value_col="overlap",
        ylabel=ylabel,
        title="Leading-eigenvector overlap with the labels",
        filename=filename,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        width=width,
        dir_name=dir_name,
        **_axis_kwargs(x_col, xlabel, xscale, invert_x, ymin_floor=0.0),
    )


def plot_kernel_effective_rank_vs_time(
    rank_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    ylabel=None,
    filename="kernel_effective_rank_vs_time.png",
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
):
    """Per-layer participation-ratio effective rank of PC / BP kernels.

    ``rank_df`` must have columns ``x_col``, ``layer``, ``method``, and
    ``effective_rank``.
    """
    sym = _feature_kernel_tex(feature_symbol)
    if ylabel is None:
        ylabel = rf"$R_{{\mathrm{{eff}}}}(C^{{{sym},\ell}})$"
    return plot_pc_bp_metric_vs_time(
        rank_df,
        plots_dir,
        value_col="effective_rank",
        ylabel=ylabel,
        title="Feature-kernel effective rank (participation ratio)",
        filename=filename,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        width=width,
        dir_name=dir_name,
        **_axis_kwargs(x_col, xlabel, xscale, invert_x),
    )


def plot_kernel_spectrum(
    spectrum_df,
    plots_dir,
    *,
    title,
    filename,
    ylabel,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    annotate_rank=False,
    xlabel=r"eigenvalue index $i$",
):
    """Per-layer kernel eigenspectrum, PC vs backprop, log y-scale.

    ``spectrum_df`` must have columns ``layer``, ``method``, ``index``
    (1-based, descending eigenvalues), and ``eigenvalue``. If
    ``annotate_rank`` is True, ``effective_rank`` is written on each
    panel.
    """
    if spectrum_df is None or len(spectrum_df) == 0:
        print(f"No records to plot for {filename}.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    layers = sorted(spectrum_df["layer"].unique())
    n_l = len(layers)
    n_idx = int(spectrum_df["index"].nunique())
    markersize = None if n_idx > 80 else _markersize_for_n(n_idx)

    fig, axes, nrows, ncols = _per_layer_axes(n_l, sharey=False)
    for i, layer in enumerate(layers):
        ax = axes[i // ncols, i % ncols]
        sub = spectrum_df[spectrum_df["layer"] == layer]
        rank_notes = []
        for method, color, marker, label in _PC_BP_STYLES:
            msub = sub[sub["method"] == method].sort_values("index")
            if not len(msub):
                continue
            idx = np.asarray(msub["index"], dtype=float)
            values = np.asarray(msub["eigenvalue"], dtype=float)
            _warn_if_nonfinite(
                f"{label} spectrum (layer={layer})", values
            )
            values = np.where(values > 0.0, values, np.nan)
            plot_kw = dict(color=color, label=label)
            if markersize is None:
                plot_kw["marker"] = None
            else:
                plot_kw["marker"] = marker
                plot_kw["markersize"] = markersize
            ax.plot(idx, values, **plot_kw)
            if annotate_rank and "effective_rank" in msub.columns:
                r_eff = float(msub["effective_rank"].iloc[0])
                rank_notes.append(
                    rf"$R_{{\mathrm{{eff}}}}^{{\mathrm{{{label}}}}}="
                    rf"{r_eff:.2f}$"
                )
        ax.set_yscale("log")
        _panel_label(ax, layer)
        ps.style_axes(ax)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        if i == 0:
            ax.legend()
        if rank_notes:
            ax.text(
                0.97,
                0.97,
                "\n".join(rank_notes),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=ps.ANNOTATION_SIZE,
            )

    for j in range(n_l, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    _shared_xlabel(fig, xlabel)

    save_path = os.path.join(out_dir, filename)
    ps.save_figure(fig, save_path)
    print(f"{title} saved to {save_path}")
    return save_path


def plot_temporal_kernel_effective_rank_vs_layer(
    rank_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
):
    """Participation-ratio effective rank of sample-traced kernels vs layer.

    ``rank_df`` must have columns ``layer``, ``method``, and
    ``effective_rank``.
    """
    if rank_df is None or len(rank_df) == 0:
        print("No temporal-kernel effective-rank records to plot.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    sym = _feature_kernel_tex(feature_symbol)
    ylim = _data_ylim(rank_df["effective_rank"], invert=False)

    plt.figure(figsize=ps.PANEL_THIRD)
    ax = plt.gca()
    plot_df = rank_df.copy()
    plot_df["layer_display"] = plot_df["layer"].astype(int) + 1
    _plot_pc_bp_series(
        ax,
        plot_df,
        x_col="layer_display",
        value_col="effective_rank",
        label_prefix="temporal rank ",
    )
    layers = sorted(plot_df["layer_display"].unique())
    ps.integer_ticks(ax, layers)
    ax.set_xlabel(ps.TEX["layer"])
    ax.set_ylabel(
        rf"$R_{{\mathrm{{eff}}}}(C^{{{sym},\ell}}_{{\mathrm{{temp}}}})$"
    )
    _maybe_set_ylim(ax, ylim)
    ps.style_axes(ax)
    ax.legend()
    save_path = os.path.join(
        out_dir, "temporal_kernel_effective_rank_vs_layer.png"
    )
    ps.save_figure(plt.gcf(), save_path)
    print(f"Temporal-kernel effective rank saved to {save_path}")
    return save_path


_K_MARKERS = ("o", "D", "^", "s", "v", "P")


def plot_pc_bp_subspace_overlap_vs_time(
    overlap_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
    filename="pc_bp_subspace_overlap_vs_time.png",
):
    """Per-layer PC–BP top-``k`` eigenspace overlap vs ``x_col``.

    ``overlap_df`` must have columns ``x_col``, ``layer``, ``k``, and
    ``overlap`` (mean of ``cos^2 θ_i``). One subplot per layer; one
    curve per ``k``.
    """
    if overlap_df is None or len(overlap_df) == 0:
        print("No PC-BP subspace-overlap records to plot.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    layers = sorted(overlap_df["layer"].unique())
    ks = sorted(overlap_df["k"].unique())
    n_l = len(layers)
    ylim = _data_ylim(overlap_df["overlap"], invert=False, ymin_floor=0.0)
    markersize = _markersize_for_n(overlap_df[x_col].nunique())
    k_palette = ps.sequence_colors(len(ks))
    k_colors = {k: k_palette[i] for i, k in enumerate(ks)}

    fig, axes, nrows, ncols = _per_layer_axes(n_l)
    for i, layer in enumerate(layers):
        ax = axes[i // ncols, i % ncols]
        sub = overlap_df[overlap_df["layer"] == layer]
        for j, k in enumerate(ks):
            ksub = sub[sub["k"] == k].sort_values(x_col)
            if not len(ksub):
                continue
            x = np.asarray(ksub[x_col], dtype=float)
            values = np.asarray(ksub["overlap"], dtype=float)
            _warn_if_nonfinite(
                f"subspace overlap (layer={layer}, k={int(k)})", values
            )
            ax.plot(
                x,
                values,
                marker=_K_MARKERS[j % len(_K_MARKERS)],
                markersize=markersize,
                color=k_colors[k],
                label=rf"$k={int(k)}$",
            )
        _panel_label(ax, layer)
        _apply_x_axis(ax, xscale=xscale, invert_x=invert_x)
        _maybe_set_ylim(ax, ylim)
        ps.style_axes(ax)
        if i % ncols == 0:
            ax.set_ylabel(
                r"$\frac{1}{k}\sum_{i=1}^{k}\cos^{2}\theta_i$"
            )
        if i == 0:
            ax.legend()

    for j in range(n_l, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    _shared_xlabel(fig, xlabel)

    save_path = os.path.join(out_dir, filename)
    ps.save_figure(fig, save_path)
    print(f"PC-BP subspace overlap plot saved to {save_path}")
    return save_path


def plot_temporal_pc_bp_alignment_vs_layer(
    alignment_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
):
    """PC–BP CKA of sample-traced temporal kernels vs layer.

    ``alignment_df`` must have columns ``layer`` and ``alignment``.
    """
    if alignment_df is None or len(alignment_df) == 0:
        print("No temporal PC-BP alignment records to plot.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    sym = _feature_kernel_tex(feature_symbol)
    plot_df = alignment_df.copy().sort_values("layer")
    layers = plot_df["layer"].astype(int) + 1
    values = np.asarray(plot_df["alignment"], dtype=float)
    _warn_if_nonfinite("temporal pc-bp alignment", values)
    ylim = _data_ylim(values, invert=False, ymin_floor=0.0)

    plt.figure(figsize=ps.PANEL_THIRD)
    ax = plt.gca()
    ax.plot(
        layers,
        values,
        marker="o",
        color=ps.COLOR_PC,
    )
    ps.integer_ticks(ax, list(layers))
    ax.set_xlabel(ps.TEX["layer"])
    ax.set_ylabel(
        rf"$\mathrm{{CKA}}(C^{{{sym},\ell}}_{{\mathrm{{temp,PC}}}}, "
        rf"C^{{{sym},\ell}}_{{\mathrm{{temp,BP}}}})$"
    )
    _maybe_set_ylim(ax, ylim)
    ps.style_axes(ax)
    save_path = os.path.join(
        out_dir, "temporal_pc_bp_kernel_alignment_vs_layer.png"
    )
    ps.save_figure(plt.gcf(), save_path)
    print(f"Temporal PC-BP kernel alignment saved to {save_path}")
    return save_path


def plot_pc_bp_kernel_alignment_test_vs_layer(
    alignment_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    ylabel=None,
    title=None,
    filename="pc_bp_kernel_alignment_test.png",
):
    """PC–BP CKA of snapshot feature kernels vs layer, train vs test.

    ``alignment_df`` must have columns ``layer``, ``split`` (``train`` /
    ``test``), and ``alignment``.
    """
    if alignment_df is None or len(alignment_df) == 0:
        print("No train/test PC-BP alignment records to plot.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    sym = _feature_kernel_tex(feature_symbol)
    plot_df = alignment_df.copy()
    plot_df["layer_display"] = plot_df["layer"].astype(int) + 1
    _warn_if_nonfinite(
        "train/test pc-bp alignment",
        np.asarray(plot_df["alignment"], dtype=float),
    )
    ylim = _data_ylim(plot_df["alignment"], invert=False, ymin_floor=0.0)

    plt.figure(figsize=ps.PANEL_THIRD)
    ax = plt.gca()
    _plot_train_test_series(
        ax,
        plot_df,
        x_col="layer_display",
        value_col="alignment",
        label_prefix="pc-bp alignment ",
    )
    layers = sorted(plot_df["layer_display"].unique())
    ps.integer_ticks(ax, layers)
    ax.set_xlabel(ps.TEX["layer"])
    if ylabel is None:
        ylabel = (
            rf"$\mathrm{{CKA}}(C^{{{sym},\ell}}_{{\mathrm{{PC}}}}, "
            rf"C^{{{sym},\ell}}_{{\mathrm{{BP}}}})$"
        )
    ax.set_ylabel(ylabel)
    _maybe_set_ylim(ax, ylim)
    ps.style_axes(ax)
    ax.legend()
    save_path = os.path.join(out_dir, filename)
    ps.save_figure(plt.gcf(), save_path)
    print(f"Train/test PC-BP kernel alignment saved to {save_path}")
    return save_path


def plot_kernel_target_alignment_test_vs_layer(
    alignment_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    ylabel=None,
    title=None,
    filename="kernel_target_alignment_test.png",
):
    """CKA of snapshot feature kernels with ``C^y``, train vs test.

    Two panels (PC, backprop). ``alignment_df`` must have columns
    ``layer``, ``method``, ``split`` (``train`` / ``test``), and
    ``alignment``. The target kernel is ``C^y = Y Y^T`` of that split.
    """
    if alignment_df is None or len(alignment_df) == 0:
        print("No train/test kernel-target alignment records to plot.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    sym = _feature_kernel_tex(feature_symbol)
    plot_df = alignment_df.copy()
    plot_df["layer_display"] = plot_df["layer"].astype(int) + 1
    _warn_if_nonfinite(
        "train/test kernel-target alignment",
        np.asarray(plot_df["alignment"], dtype=float),
    )
    ylim = _data_ylim(plot_df["alignment"], invert=False, ymin_floor=0.0)
    layers = sorted(plot_df["layer_display"].unique())
    if ylabel is None:
        ylabel = rf"$\mathrm{{CKA}}(C^{{{sym},\ell}}, C^{{y}})$"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=ps.per_layer_figsize(2, 1),
        sharex=True,
        sharey=True,
    )
    for ax, method, panel in zip(
        axes, ("pc", "bp"), (ps.LABEL_PC, ps.LABEL_BP)
    ):
        sub = plot_df[plot_df["method"] == method]
        _plot_train_test_series(
            ax,
            sub,
            x_col="layer_display",
            value_col="alignment",
            label_prefix=f"target alignment ({method}) ",
        )
        ps.integer_ticks(ax, layers)
        ax.set_title(panel)
        _maybe_set_ylim(ax, ylim)
        ps.style_axes(ax)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    _shared_xlabel(fig, ps.TEX["layer"])
    save_path = os.path.join(out_dir, filename)
    ps.save_figure(fig, save_path)
    print(f"Train/test kernel-target alignment saved to {save_path}")
    return save_path


_TARGET_INPUT_STYLES = (
    ("pc", "target", ps.COLOR_PC, "o", "-", r"PC, $C^{y}$"),
    ("bp", "target", ps.COLOR_BP, "s", "-", r"BP, $C^{y}$"),
    ("pc", "input", ps.COLOR_PC, "^", "--", r"PC, $C^{x}$"),
    ("bp", "input", ps.COLOR_BP, "D", "--", r"BP, $C^{x}$"),
)


def plot_kernel_target_input_alignment_final(
    alignment_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    ylabel=None,
    title=None,
    filename="kernel_target_input_alignment_final.png",
    figsize=None,
):
    """CKA of train feature kernels with ``C^y`` and ``C^x``, vs layer.

    Four curves on one axis: PC/BP vs the target kernel and vs the input
    kernel. ``alignment_df`` must have columns ``layer``, ``method``
    (``"pc"`` / ``"bp"``), ``ref`` (``"target"`` / ``"input"``), and
    ``alignment``.
    """
    if alignment_df is None or len(alignment_df) == 0:
        print("No kernel-target/input alignment records to plot.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    sym = _feature_kernel_tex(feature_symbol)
    plot_df = alignment_df.copy()
    plot_df["layer_display"] = plot_df["layer"].astype(int) + 1
    _warn_if_nonfinite(
        "kernel-target/input alignment",
        np.asarray(plot_df["alignment"], dtype=float),
    )
    ylim = _data_ylim(plot_df["alignment"], invert=False, ymin_floor=0.0)
    layers = sorted(plot_df["layer_display"].unique())
    if ylabel is None:
        ylabel = rf"$\mathrm{{CKA}}(C^{{{sym},\ell}}, C^{{y/x}})$"

    plt.figure(figsize=figsize or ps.PANEL_THIRD)
    ax = plt.gca()
    for method, ref, color, marker, linestyle, label in _TARGET_INPUT_STYLES:
        sub = plot_df[
            (plot_df["method"] == method) & (plot_df["ref"] == ref)
        ].sort_values("layer_display")
        if not len(sub):
            continue
        x = np.asarray(sub["layer_display"], dtype=float)
        values = np.asarray(sub["alignment"], dtype=float)
        _warn_if_nonfinite(f"{method} {ref} alignment", values)
        ax.plot(
            x,
            values,
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    ps.integer_ticks(ax, layers)
    ax.set_xlabel(ps.TEX["layer"])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        lo, hi = ylim
        ylim = (lo, hi + 0.28 * (hi - lo))
    _maybe_set_ylim(ax, ylim)
    ax.legend(ncol=2, loc="upper right")
    ps.style_axes(ax)
    save_path = os.path.join(out_dir, filename)
    ps.save_figure(plt.gcf(), save_path)
    print(f"Kernel-target/input alignment saved to {save_path}")
    return save_path


def plot_pc_bp_alignment_vs_time(
    alignment_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    ylabel=None,
    filename="pc_bp_kernel_alignment_vs_time.png",
    title=None,
    ylim=None,
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
    figsize=None,
):
    """PC-BP feature-kernel CKA over ``x_col``.

    ``alignment_df`` must have columns ``x_col``, ``layer``, and
    ``alignment``. One curve is drawn per hidden layer.

    ``feature_symbol`` is ``"h"`` (linear) or ``"phi"`` (nonlinear).
    Pass ``ylabel`` / ``filename`` / ``title`` / ``ylim`` to reuse this
    figure for kernel-*change* CKA (``C(t) - C(0)``). ``ylim`` defaults
    to a tight pad around the plotted values.
    """
    if alignment_df is None or len(alignment_df) == 0:
        print("No PC-BP kernel alignment records to plot.")
        return None

    layers = sorted(alignment_df["layer"].unique())
    colors = ps.sequence_colors(len(layers))
    sym = _feature_kernel_tex(feature_symbol)
    if ylabel is None:
        ylabel = (
            rf"$\mathrm{{CKA}}(C^{{{sym},\ell}}_{{\mathrm{{PC}}}}, "
            rf"C^{{{sym},\ell}}_{{\mathrm{{BP}}}})$"
        )

    plt.figure(figsize=figsize or ps.PANEL_THIRD)
    ax = plt.gca()
    markersize = _markersize_for_n(alignment_df[x_col].nunique())
    for layer, color in zip(layers, colors):
        sub = alignment_df[alignment_df["layer"] == layer].sort_values(x_col)
        x = np.asarray(sub[x_col], dtype=float)
        values = np.asarray(sub["alignment"], dtype=float)
        _warn_if_nonfinite(f"pc-bp alignment layer={int(layer)}", values)
        ax.plot(
            x,
            values,
            marker="o",
            markersize=markersize,
            color=color,
            label=ps.layer_label(layer),
        )

    _apply_x_axis(ax, xlabel=xlabel, xscale=xscale, invert_x=invert_x)
    ax.set_ylabel(ylabel)
    if ylim is None:
        ylim = _data_ylim(
            alignment_df["alignment"], invert=False, ymin_floor=0.0
        )
    _maybe_set_ylim(ax, ylim)
    ax.legend(ncol=2 if len(layers) > 3 else 1)
    ps.style_axes(ax)
    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    save_path = os.path.join(out_dir, filename)
    ps.save_figure(plt.gcf(), save_path)
    print(f"PC-BP kernel alignment plot saved to {save_path}")
    return save_path


def plot_kernel_concentration_per_timepoint(
    conc_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    n_seeds=None,
    dir_name="alignment",
    feature_symbol="h",
    x_col="t",
    metric="cosine",
    filename="kernel_concentration_vs_layer_grid.png",
):
    """Mean pairwise kernel similarity across seeds, vs layer.

    One subplot per ``x_col``. ``conc_df`` must have columns ``x_col``,
    ``layer``, ``method`` (``"pc"`` or ``"bp"``), and the mean/std
    columns for ``metric`` (``"cosine"``, ``"cka"``, or ``"rel_frob"``).
    High cosine / CKA (low relative Frobenius) means kernels concentrate
    across inits. Error bars are the sample SD of pairwise values.
    """
    if conc_df is None or len(conc_df) == 0:
        print("No kernel-concentration records to plot.")
        return None

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    value_col, std_col, ylabel, invert = _concentration_metric_spec(
        metric, feature_symbol
    )
    x_values = sorted(conc_df[x_col].unique())
    if x_col == "loss":
        x_values = list(reversed(x_values))
    n_x = len(x_values)
    ncols = int(np.ceil(np.sqrt(n_x)))
    nrows = int(np.ceil(n_x / ncols))
    ylim = _data_ylim(
        conc_df[value_col], invert=invert, ymin_floor=0.0
    )
    has_std = std_col in conc_df.columns

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=ps.per_layer_figsize(ncols, nrows),
        squeeze=False,
        sharey=True,
    )
    for i, x in enumerate(x_values):
        ax = axes[i // ncols, i % ncols]
        sub = conc_df[conc_df[x_col] == x]
        for method, color, marker, label in _PC_BP_STYLES:
            msub = sub[sub["method"] == method].sort_values("layer")
            if not len(msub):
                continue
            layers = np.asarray(msub["layer"], dtype=float) + 1
            values = np.asarray(msub[value_col], dtype=float)
            _warn_if_nonfinite(f"{method} concentration ({x_col}={x})", values)
            plot_kw_line = dict(marker=marker, color=color, label=label)
            if has_std:
                ax.errorbar(
                    layers,
                    values,
                    yerr=np.asarray(msub[std_col], dtype=float),
                    **plot_kw_line,
                )
            else:
                ax.plot(layers, values, **plot_kw_line)
        ax.set_title(_snapshot_label(x_col, x))
        ps.integer_ticks(
            ax, np.asarray(conc_df["layer"], dtype=float) + 1
        )
        _maybe_set_ylim(ax, ylim)
        ps.style_axes(ax)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        if i == 0:
            ax.legend()

    for j in range(n_x, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    _shared_xlabel(fig, ps.TEX["layer"])

    save_path = os.path.join(
        out_dir, _insert_rel_before_vs(filename, metric)
    )
    ps.save_figure(fig, save_path)
    print(f"Kernel concentration grid saved to {save_path}")
    return save_path


def plot_kernel_concentration_vs_time(
    conc_df,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    n_seeds=None,
    dir_name="alignment",
    feature_symbol="h",
    x_col="t",
    xlabel="$t$",
    xscale="linear",
    invert_x=False,
    metric="cosine",
    filename="kernel_concentration_vs_time.png",
):
    """Mean pairwise kernel similarity across seeds vs ``x_col``.

    One subplot per hidden layer with PC vs BP, matching
    ``plot_kernel_target_alignment_vs_time``. ``metric`` is
    ``"cosine"``, ``"cka"``, or ``"rel_frob"``. Error bars are the
    sample SD of pairwise values.
    """
    if conc_df is None or len(conc_df) == 0:
        print("No kernel-concentration records to plot.")
        return None

    value_col, std_col, ylabel, invert = _concentration_metric_spec(
        metric, feature_symbol
    )
    title = "Feature-kernel concentration across seeds"
    if n_seeds is not None:
        title += rf", $n_{{\mathrm{{seeds}}}}={int(n_seeds)}$"
    has_std = std_col in conc_df.columns
    return plot_pc_bp_metric_vs_time(
        conc_df,
        plots_dir,
        value_col=value_col,
        ylabel=ylabel,
        title=title,
        filename=_insert_rel_before_vs(filename, metric),
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        width=width,
        dir_name=dir_name,
        invert_ylim=invert,
        ymin_floor=0.0,
        x_col=x_col,
        xlabel=xlabel,
        xscale=xscale,
        invert_x=invert_x,
        std_col=std_col if has_std else None,
    )


def plot_pc_bp_loss(
    pc_losses,
    bp_losses,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    figsize=None,
):
    """Overlay finite-size PC and BP training-loss curves vs time."""
    pc_losses = np.asarray(pc_losses).flatten()
    bp_losses = np.asarray(bp_losses).flatten()
    _warn_if_nonfinite("pc_losses", pc_losses)
    _warn_if_nonfinite("bp_losses", bp_losses)

    plt.figure(figsize=figsize or ps.PANEL_THIRD)
    ax = plt.gca()
    t_pc = np.arange(len(pc_losses))
    t_bp = np.arange(len(bp_losses))
    markersize = _markersize_for_n(max(len(pc_losses), len(bp_losses)))
    ax.plot(
        t_pc,
        pc_losses,
        marker="o",
        markersize=markersize,
        color=ps.COLOR_PC,
        label=ps.LABEL_PC,
    )
    ax.plot(
        t_bp,
        bp_losses,
        marker="s",
        markersize=markersize,
        color=ps.COLOR_BP,
        label=ps.LABEL_BP,
    )
    ax.set_xlabel(ps.TEX["time"])
    ax.set_ylabel(ps.LOSS_LABEL)
    ax.legend()
    ps.style_axes(ax)
    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    save_path = os.path.join(out_dir, "pc_bp_loss.png")
    ps.save_figure(plt.gcf(), save_path)
    print(f"PC-BP loss plot saved to {save_path}")
    return save_path


def plot_pc_bp_loss_matched_times(
    pc_losses,
    bp_losses,
    pc_times,
    bp_times,
    loss_grid,
    plots_dir,
    n_hidden=None,
    gamma_0=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    dir_name="alignment",
    heatmap_idx=None,
    yscale="log",
    filename="pc_bp_loss.png",
    figsize=None,
):
    """PC vs BP loss vs time, with first-crossing markers at each ``L*``.

    Small markers show every target loss on the dense grid; larger
    markers (and faint horizontal lines) highlight the heatmap subset.
    ``yscale`` is ``"log"`` or ``"linear"``.
    """
    pc_losses = np.asarray(pc_losses, dtype=float).flatten()
    bp_losses = np.asarray(bp_losses, dtype=float).flatten()
    pc_times = np.asarray(pc_times, dtype=int).flatten()
    bp_times = np.asarray(bp_times, dtype=int).flatten()
    loss_grid = np.asarray(loss_grid, dtype=float).flatten()
    _warn_if_nonfinite("pc_losses", pc_losses)
    _warn_if_nonfinite("bp_losses", bp_losses)
    if heatmap_idx is None:
        heatmap_idx = []
    heatmap_idx = [int(i) for i in heatmap_idx]

    plt.figure(figsize=figsize or ps.PANEL_THIRD)
    ax = plt.gca()
    t_pc = np.arange(len(pc_losses))
    t_bp = np.arange(len(bp_losses))
    ax.plot(t_pc, pc_losses, color=ps.COLOR_PC, label=ps.LABEL_PC, zorder=2)
    ax.plot(t_bp, bp_losses, color=ps.COLOR_BP, label=ps.LABEL_BP, zorder=2)
    ax.scatter(
        pc_times, loss_grid, marker="o", s=1.5, color=ps.COLOR_PC, zorder=3
    )
    ax.scatter(
        bp_times, loss_grid, marker="s", s=1.5, color=ps.COLOR_BP, zorder=3
    )
    handles = list(ax.get_legend_handles_labels()[0])
    if heatmap_idx:
        hm_pc = pc_times[heatmap_idx]
        hm_bp = bp_times[heatmap_idx]
        hm_L = loss_grid[heatmap_idx]
        ax.scatter(
            hm_pc,
            hm_L,
            marker="o",
            s=22,
            facecolors="none",
            edgecolors=ps.COLOR_PC,
            linewidths=0.8,
            zorder=4,
        )
        ax.scatter(
            hm_bp,
            hm_L,
            marker="s",
            s=22,
            facecolors="none",
            edgecolors=ps.COLOR_BP,
            linewidths=0.8,
            zorder=4,
        )
        for L_star in hm_L:
            ax.axhline(L_star, color="0.6", lw=0.5, ls=":", zorder=1)
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor="0.35",
                markeredgewidth=0.8,
                label=r"matched $\mathcal{L}^*$",
            )
        )

    ax.set_xlabel(ps.TEX["time"])
    ax.set_ylabel(ps.LOSS_LABEL)
    if yscale == "log":
        ax.set_yscale("log")
    ax.legend(handles=handles)
    ps.style_axes(ax)
    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    save_path = os.path.join(out_dir, filename)
    ps.save_figure(plt.gcf(), save_path)
    print(f"PC-BP loss-matched times plot saved to {save_path}")
    return save_path


def plot_pc_last_layer_displacement_vs_gamma(
    displacement_df,
    plots_dir,
    n_hidden=None,
    activity_lr=None,
    width=None,
    dir_name="alignment",
    feature_symbol="h",
    metric="cosine",
    figsize=None,
):
    """Last-hidden-layer displacement vs ``gamma_0``, one curve per ``K``.

    ``displacement_df`` must have columns ``layer``, ``kind``
    (``"dmft"``, ``"infer"``, or ``"closed_form"``), ``n_infer_iters``,
    ``gamma_0``, and ``displacement`` / ``rel_displacement`` (selected
    by ``metric``). Only the deepest hidden layer (``layer == max(layer)``,
    i.e. ``ℓ = H``) is drawn. DMFT is the smallest ``K`` (dashed);
    finite-size infer is solid for increasing ``K``; closed-form is
    dash-dot in the linear case.

    ``feature_symbol`` is ``"h"`` (linear) or ``"phi"`` (nonlinear).
    ``metric`` is ``"cosine"`` or ``"rel_frob"``.
    """
    if displacement_df is None or len(displacement_df) == 0:
        print("No last-layer kernel displacement records to plot.")
        return None

    last_layer = int(displacement_df["layer"].max())
    df = displacement_df[displacement_df["layer"] == last_layer]
    if not len(df):
        print("No last-layer kernel displacement records to plot.")
        return None

    value_col, ylabel, invert = _displacement_metric_spec(
        metric, feature_symbol, layer="H"
    )
    ylim = _data_ylim(df[value_col], invert=invert)
    plt.figure(figsize=figsize or ps.PANEL_THIRD)
    ax = plt.gca()
    infer_ks = sorted(
        {
            int(k)
            for k in df.loc[
                df["kind"] == "infer", "n_infer_iters"
            ].dropna().unique()
        }
    )
    k_palette = ps.sequence_colors(len(infer_ks))
    k_colors = {k: k_palette[i] for i, k in enumerate(infer_ks)}
    value_handles = []
    drew_dmft = drew_nn = drew_closed_form = False

    def _xy(sub):
        sub = sub.sort_values("gamma_0")
        x = np.asarray(sub["gamma_0"], dtype=float)
        y = np.asarray(sub[value_col], dtype=float)
        return x, y

    dmft_sub = df[df["kind"] == "dmft"]
    if len(dmft_sub):
        dmft_k = int(dmft_sub["n_infer_iters"].iloc[0])
        x, y = _xy(dmft_sub)
        _warn_if_nonfinite(
            f"dmft last-layer displacement (K={dmft_k})", y
        )
        ax.plot(
            x,
            y,
            color=k_colors.get(dmft_k, ps.COLOR_REFERENCE),
            linestyle=_SOURCE_STYLES["dmft"],
            marker="o",
        )
        drew_dmft = True

    for k in infer_ks:
        sub = df[
            (df["kind"] == "infer")
            & (df["n_infer_iters"].astype(int) == int(k))
        ]
        if not len(sub):
            continue
        x, y = _xy(sub)
        _warn_if_nonfinite(f"infer last-layer displacement (K={k})", y)
        ax.plot(
            x,
            y,
            color=k_colors[k],
            linestyle=_SOURCE_STYLES["nn"],
            marker="o",
        )
        drew_nn = True
        value_handles.append(_proxy_line(ps.k_label(k), color=k_colors[k]))

    cf_sub = df[df["kind"] == "closed_form"]
    if len(cf_sub):
        x, y = _xy(cf_sub)
        _warn_if_nonfinite("closed-form last-layer displacement", y)
        ax.plot(
            x,
            y,
            color=ps.COLOR_REFERENCE,
            linestyle=_SOURCE_STYLES["closed_form"],
            marker="s",
        )
        drew_closed_form = True

    ax.set_xlabel(ps.TEX["gamma_0"])
    ax.set_ylabel(ylabel)
    _maybe_set_ylim(ax, ylim)
    _source_legend(
        ax,
        value_handles,
        dmft=drew_dmft,
        nn=drew_nn,
        closed_form=drew_closed_form,
    )
    ps.style_axes(ax)
    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        activity_lr=activity_lr,
        dir_name=dir_name,
    )
    save_path = os.path.join(
        out_dir,
        _displacement_plot_filename(
            "kernel_displacement_last_layer_vs_gamma.png", metric
        ),
    )
    ps.save_figure(plt.gcf(), save_path)
    print(f"Last-layer displacement vs gamma plot saved to {save_path}")
    return save_path


def plot_pc_bp_kernel_alignment(
    alignment_df,
    plots_dir,
    n_hidden=None,
    activity_lr=None,
    feature_symbol="h",
):
    """Cosine similarity between the final PC and final BP feature kernels
    at every layer (both at the last training step; PC at ``k=0``).

    ``alignment_df`` must have columns ``layer``, ``alignment``,
    ``gamma_0`` and ``n_infer_iters``. One curve is drawn per
    ``(gamma_0, n_infer_iters)`` combination present in the data.

    ``feature_symbol`` is ``"h"`` (linear) or ``"phi"`` (nonlinear).
    """
    if alignment_df is None or len(alignment_df) == 0:
        print("No PC-BP kernel alignment records to plot.")
        return None

    combos = (
        alignment_df[["gamma_0", "n_infer_iters"]]
        .drop_duplicates()
        .sort_values(["gamma_0", "n_infer_iters"])
    )
    colors = ps.sequence_colors(len(combos))
    sym = _feature_kernel_tex(feature_symbol)

    plt.figure(figsize=ps.PANEL_THIRD)
    ax = plt.gca()
    # itertuples (not iterrows) keeps each field's own dtype, since
    # iterrows would coerce a mixed float/int row to a common dtype.
    for row, color in zip(combos.itertuples(index=False), colors):
        sub = alignment_df[
            (alignment_df["gamma_0"] == row.gamma_0)
            & (alignment_df["n_infer_iters"] == row.n_infer_iters)
        ].sort_values("layer")
        if not len(sub):
            continue
        layers = np.asarray(sub["layer"], dtype=float) + 1
        values = np.asarray(sub["alignment"], dtype=float)
        _warn_if_nonfinite(
            f"pc-bp alignment (gamma_0={row.gamma_0}, "
            f"K={row.n_infer_iters})",
            values,
        )
        label = (
            rf"$\gamma_0 = {ps.fmt_number(row.gamma_0)}$, "
            rf"$K = {int(row.n_infer_iters)}$"
        )
        ax.plot(layers, values, marker="o", color=color, label=label)

    if len(combos) == 1:
        # Per-column access (not row-wise .iloc) to avoid pandas upcasting
        # gamma_0 (float) and n_infer_iters (int) to a common dtype.
        gamma_0 = combos["gamma_0"].iloc[0]
        n_infer_iters = int(combos["n_infer_iters"].iloc[0])
        out_dir = _alignment_plots_dir(
            plots_dir,
            n_hidden=n_hidden,
            gamma_0=gamma_0,
            activity_lr=activity_lr,
            n_infer_iters=n_infer_iters,
        )
    else:
        out_dir = _alignment_plots_dir(
            plots_dir, n_hidden=n_hidden, activity_lr=activity_lr
        )

    ax.set_xlabel(ps.TEX["layer"])
    ax.set_ylabel(
        rf"$\cos(C^{{{sym},\ell}}_{{\mathrm{{PC}}}}, "
        rf"C^{{{sym},\ell}}_{{\mathrm{{BP}}}})$"
    )
    ps.integer_ticks(ax, np.asarray(alignment_df["layer"], dtype=float) + 1)
    _maybe_set_ylim(
        ax,
        _data_ylim(alignment_df["alignment"], invert=False, ymin_floor=0.0),
    )
    ax.legend()
    ps.style_axes(ax)
    save_path = os.path.join(out_dir, "pc_bp_kernel_alignment_vs_layer.png")
    ps.save_figure(plt.gcf(), save_path)
    print(f"PC-BP kernel alignment plot saved to {save_path}")
    return save_path


#: Gap between neighbouring heatmaps, as a fraction of one heatmap.
_KERNEL_GRID_GAP = 0.10

#: Inches reserved around the heatmaps: axis symbols on the outer panels,
#: column labels above the grid, and the parts of the colour bar other
#: than its tick labels.
_KERNEL_AXIS_SYMBOL_W = 0.22
_KERNEL_AXIS_SYMBOL_H = 0.16
_KERNEL_COL_LABEL_H = 0.17
_KERNEL_ROW_LABEL_PAD = 0.08
_KERNEL_CBAR_W = 0.035
_KERNEL_CBAR_PAD = 0.07
_KERNEL_CBAR_TICK_PAD = 0.07
_KERNEL_CBAR_LABEL_W = 0.14
#: Half-height of a colour-bar end tick (e.g. ``-1``), so it is not
#: clipped when axis symbols are omitted.
_KERNEL_CBAR_TICK_HALF_H = 0.08


def _kernel_grid_width(n_rows, n_layers):
    """Default printed width: tall grids span the text width, small ones less.

    Matches how these panels are laid out in the paper: a ``K`` sweep
    fills a row on its own, a two- or three-row grid shares a row with
    one other grid, and a small grid sits three to a row.
    """
    if n_rows >= 4:
        return ps.TEXT_WIDTH_IN
    if n_layers >= 4:
        return ps.PANEL_HALF[0]
    return ps.PANEL_THIRD[0]


def _kernel_grid_geometry(
    n_rows,
    n_layers,
    *,
    fig_width,
    row_labels=(),
    has_axis_symbols,
    has_col_labels,
    cbar,
    cbar_label,
    cbar_ticklabels=("-1.0",),
):
    """Exact inch geometry for a kernel grid, so the heatmaps stay square.

    Constrained layout cannot keep the cells square and the figure at a
    fixed printed size at the same time, and a long row label would eat
    into one row only. Everything outside the heatmaps is therefore
    measured here and the axes are placed by hand.

    Returns the figure size and the arguments for ``subplots_adjust``,
    plus the grid rectangle and cell size in inches.
    """
    label_w = max(
        (ps.text_width_in(label, ps.ANNOTATION_SIZE) for label in row_labels),
        default=0.0,
    )
    if label_w:
        # Pad on both sides: clear of the figure edge, clear of the panels.
        label_w += 2 * _KERNEL_ROW_LABEL_PAD
    left = label_w + (_KERNEL_AXIS_SYMBOL_W if has_axis_symbols else 0.02)
    bottom = _KERNEL_AXIS_SYMBOL_H if has_axis_symbols else 0.02
    top = _KERNEL_COL_LABEL_H if has_col_labels else 0.02
    right = 0.02
    if cbar:
        # End ticks sit on the colour-bar edges; keep them inside the
        # figure even when the heatmaps have no axis-symbol margin.
        bottom = max(bottom, _KERNEL_CBAR_TICK_HALF_H)
        top = max(top, _KERNEL_CBAR_TICK_HALF_H)
        tick_w = max(
            (
                ps.text_width_in(t, mpl.rcParams["ytick.labelsize"])
                for t in cbar_ticklabels
            ),
            default=0.0,
        )
        right = (
            _KERNEL_CBAR_PAD
            + _KERNEL_CBAR_W
            + _KERNEL_CBAR_TICK_PAD
            + tick_w
            + (_KERNEL_CBAR_LABEL_W if cbar_label else 0.0)
        )

    gap = _KERNEL_GRID_GAP
    grid_w = fig_width - left - right
    cell = grid_w / (n_layers + gap * (n_layers - 1))
    grid_h = cell * (n_rows + gap * (n_rows - 1))
    fig_height = top + grid_h + bottom
    return dict(
        figsize=(fig_width, fig_height),
        left=left,
        bottom=bottom,
        grid_w=grid_w,
        grid_h=grid_h,
        cell=cell,
        gap=gap,
        label_w=label_w,
        cbar_x=left + grid_w + _KERNEL_CBAR_PAD,
    )


def _kernel_cell_rect(geom, row, col):
    """Figure-fraction rectangle of one heatmap in the grid.

    The heatmaps are placed one by one rather than through a gridspec:
    saving a PDF re-runs the figure layout, which would move gridspec
    axes but leaves explicitly positioned axes alone.
    """
    width, height = geom["figsize"]
    cell, gap = geom["cell"], geom["gap"]
    x = geom["left"] + col * (1.0 + gap) * cell
    y = geom["bottom"] + geom["grid_h"] - (row * (1.0 + gap) + 1.0) * cell
    return [x / width, y / height, cell / width, cell / height]


def _place_kernel_row_labels(fig, geom, row_labels):
    """Row labels in the left margin, at the vertical centre of each row."""
    width, height = geom["figsize"]
    cell, gap = geom["cell"], geom["gap"]
    top_of_grid = geom["bottom"] + geom["grid_h"]
    for r, label in enumerate(row_labels):
        if not label:
            continue
        y = top_of_grid - (r * (1.0 + gap) + 0.5) * cell
        fig.text(
            (geom["label_w"] - _KERNEL_ROW_LABEL_PAD) / width,
            y / height,
            label,
            ha="right",
            va="center",
            fontsize=ps.ANNOTATION_SIZE,
        )


def _add_kernel_colorbar(fig, geom, mappable, cbar_label):
    """One colour bar spanning the height of the grid."""
    width_in, height_in = geom["figsize"]
    cax = fig.add_axes(
        [
            geom["cbar_x"] / width_in,
            geom["bottom"] / height_in,
            _KERNEL_CBAR_W / width_in,
            geom["grid_h"] / height_in,
        ]
    )
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.outline.set_linewidth(mpl.rcParams["axes.linewidth"])
    if cbar_label:
        cbar.set_label(cbar_label)
    return cbar


def _kernel_cbar_ticklabels(clim_kw):
    """Widest tick labels a shared colour bar will need, for margin sizing.

    The tick locator runs after the figure size is fixed, so the margin
    is sized for the widest label it could reasonably choose.
    """
    labels = ["-0.75"]
    for key in ("vmin", "vmax"):
        value = clim_kw.get(key)
        if value is not None:
            labels.append(f"{value:.2g}")
    return tuple(labels)


def _style_kernel_heatmap_ax(
    ax,
    *,
    xlabel=None,
    ylabel=None,
    mark_origin=False,
):
    """Hide ticks and optionally place axis symbols flush against the heatmap."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        pad=0,
        bottom=False,
        left=False,
        top=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=2)
    if mark_origin:
        ax.annotate(
            "0",
            xy=(0.0, 0.0),
            xycoords="axes fraction",
            xytext=(-2, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=ps.ANNOTATION_SIZE,
            annotation_clip=False,
        )


def plot_final_kernel_grid(
    kernel_rows,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    filename="final_kernels_grid.png",
    vmin=None,
    vmax=None,
    title="Final feature kernels",
    dir_name="alignment",
    origin="upper",
    xlabel=None,
    ylabel=None,
    cbar=True,
    cbar_label=None,
    mark_origin=False,
    fig_width=None,
    center_zero=True,
):
    """Grid of feature kernels (one heatmap per layer).

    ``kernel_rows`` is a list of ``(row_label, kernels)`` pairs. Each
    ``kernels`` is a sequence of 2-D arrays, one per layer (column),
    typically ``P x P`` (sample-sample) or ``T x T`` (sample-traced).
    Every panel shares one set of colour limits. By default those limits
    are symmetric about zero so the midpoint of the diverging map is
    zero; set ``center_zero=False`` to stretch over the finite data
    range instead. Pass ``vmin`` / ``vmax`` to set them explicitly.
    ``origin`` is forwarded to ``imshow`` (``"upper"`` puts index ``0``
    at the top-left; ``"lower"`` puts it at the bottom-left).

    Tick marks are omitted and row labels sit in the left margin. Axis
    symbols, if given, are drawn once on the bottom-left panel.
    ``mark_origin`` draws a ``0`` at the origin of that panel. When
    ``cbar`` is True, one colour bar spans the height of the grid.

    ``fig_width`` is the printed width in inches; it defaults to one,
    two, or three panels per text-width row depending on how many rows
    and columns the grid has. ``title`` is used only for logging, since
    figure captions carry the description.
    """
    if not kernel_rows:
        raise ValueError("kernel_rows must contain at least one row.")

    rows = []
    n_layers = None
    for label, kernels in kernel_rows:
        kernels = _to_numpy(kernels)
        if n_layers is None:
            n_layers = len(kernels)
        elif len(kernels) != n_layers:
            raise ValueError(
                f"row '{label}' has {len(kernels)} layers but the first "
                f"row has {n_layers}."
            )
        _warn_if_nonfinite(
            f"{label} kernels",
            np.stack([np.asarray(k) for k in kernels]),
        )
        rows.append((label, kernels))

    # Every panel shares the colour limits so rows stay comparable. A
    # diverging map is centred on zero unless ``center_zero`` is False
    # (raw kernels with an arbitrary scale).
    clim_fn = ps.symmetric_clim if center_zero else ps.data_clim
    clim_kw = clim_fn(
        [k for _, kernels in rows for k in kernels], vmin=vmin, vmax=vmax
    )
    if not cbar:
        cbar_label = None
    n_rows = len(rows)
    row_labels = [label for label, _ in rows]
    geom = _kernel_grid_geometry(
        n_rows,
        n_layers,
        fig_width=fig_width or _kernel_grid_width(n_rows, n_layers),
        row_labels=row_labels,
        has_axis_symbols=xlabel is not None or ylabel is not None,
        has_col_labels=True,
        cbar=cbar,
        cbar_label=cbar_label,
        cbar_ticklabels=_kernel_cbar_ticklabels(clim_kw),
    )
    fig = plt.figure(figsize=geom["figsize"], layout="none")
    images = []
    for r, (label, kernels) in enumerate(rows):
        for l, kernel in enumerate(kernels):
            ax = fig.add_axes(_kernel_cell_rect(geom, r, l))
            arr = np.asarray(kernel)
            im = ax.imshow(
                arr,
                cmap=ps.KERNEL_CMAP,
                origin=origin,
                aspect="auto",
                **clim_kw,
            )
            images.append(im)
            if r == 0:
                ax.set_title(ps.layer_label(l))
            # Axis symbols on the bottom-left panel only: every panel
            # shares them, and the grid is tight enough that repeating
            # them costs more than it explains.
            corner = r == n_rows - 1 and l == 0
            _style_kernel_heatmap_ax(
                ax,
                xlabel=xlabel if corner else None,
                ylabel=ylabel if corner else None,
                mark_origin=mark_origin and corner,
            )
    _place_kernel_row_labels(fig, geom, row_labels)
    if cbar and images:
        _add_kernel_colorbar(fig, geom, images[0], cbar_label)

    out_dir = _alignment_plots_dir(
        plots_dir,
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        dir_name=dir_name,
    )
    save_path = os.path.join(out_dir, filename)
    ps.save_figure(fig, save_path)
    print(f"{title} saved to {save_path}")
    return save_path


def plot_temporal_kernel_grid(
    kernel_rows,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
    activity_lr=None,
    n_infer_iters=None,
    width=None,
    filename="temporal_pc_kernels_grid.png",
    vmin=None,
    vmax=None,
    dir_name="alignment",
    origin="lower",
    title="Sample-traced feature kernels",
    xlabel=None,
    ylabel=None,
    cbar=True,
    cbar_label=None,
    mark_origin=False,
    fig_width=None,
    center_zero=True,
):
    """Grid of sample-traced (``T x T``) feature kernels at ``k=0``.

    ``origin="lower"`` (default) places index ``0`` at the bottom-left
    of each heatmap, i.e. the time axes increase upward/rightward.
    """
    return plot_final_kernel_grid(
        kernel_rows,
        plots_dir=plots_dir,
        gamma_0=gamma_0,
        n_hidden=n_hidden,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        width=width,
        filename=filename,
        vmin=vmin,
        vmax=vmax,
        title=title,
        dir_name=dir_name,
        origin=origin,
        xlabel=xlabel,
        ylabel=ylabel,
        cbar=cbar,
        cbar_label=cbar_label,
        mark_origin=mark_origin,
        fig_width=fig_width,
        center_zero=center_zero,
    )


def load_and_plot(results_dir, gamma_0, plots_dir=None, n_hidden=None):
    """Load saved DMFT results from results_dir and generate plots."""
    suffix = f"{gamma_0}_gamma_0"
    all_H = np.load(
        os.path.join(results_dir, f"all_H_{suffix}.npy"), allow_pickle=True
    )
    all_G = np.load(
        os.path.join(results_dir, f"all_G_{suffix}.npy"), allow_pickle=True
    )
    dmft_loss = np.load(
        os.path.join(results_dir, f"dmft_loss_{suffix}.npy"), allow_pickle=True
    )

    # object arrays from saving lists
    if all_H.dtype == object:
        all_H = list(all_H)
    if all_G.dtype == object:
        all_G = list(all_G)

    if plots_dir is None:
        plots_dir = os.path.join(results_dir, "plots")

    plot_dmft_kernels_and_loss(
        all_H=all_H,
        all_G=all_G,
        dmft_loss=dmft_loss,
        plots_dir=plots_dir,
        gamma_0=gamma_0,
        n_hidden=n_hidden,
    )
    return plots_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--plots_dir", type=str, default=None)
    parser.add_argument("--gamma_0", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=None)
    args = parser.parse_args()

    out_dir = load_and_plot(
        results_dir=args.results_dir,
        gamma_0=args.gamma_0,
        plots_dir=args.plots_dir,
        n_hidden=args.n_hidden,
    )
    print(f"Plots saved to {out_dir}")

# python plot_dmft_results.py --results_dir "results" --plots_dir "plots" --gamma_0 1