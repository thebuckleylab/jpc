import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_dmft_loss(dmft_loss, plots_dir, gamma_0=None):
    """Plot DMFT loss over training iterations."""
    dmft_loss = np.asarray(dmft_loss).flatten()
    _warn_if_nonfinite("dmft_loss", dmft_loss)
    iterations = np.arange(1, len(dmft_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, dmft_loss, color="black", linewidth=2)
    plt.xlabel("$t$")
    plt.ylabel("DMFT loss")
    if gamma_0 is not None:
        plt.title(f"$\\gamma_0 = {gamma_0}$")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "dmft_loss.png"), bbox_inches="tight")
    plt.close()


def plot_H_kernels(all_H, plots_dir, gamma_0=None):
    """Plot final-time slice of H kernels for each layer."""
    all_H = _to_numpy(all_H)
    n_layers = len(all_H)
    _warn_if_nonfinite("all_H", np.stack([np.asarray(H) for H in all_H]))

    fig, axes = plt.subplots(
        n_layers, 1, figsize=(6, 2 * n_layers), squeeze=False
    )
    for l, H_l in enumerate(all_H):
        ax = axes[l, 0]
        kernel = np.asarray(H_l[-1, :, -1, :])
        ax.imshow(kernel, cmap="coolwarm")
        if l == 0:
            ax.set_ylabel("Theory")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Layer {l}")

    if gamma_0 is not None:
        fig.suptitle(f"$\\gamma_0 = {gamma_0}$", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "all_H_kernels.png"), bbox_inches="tight")
    plt.close(fig)


def plot_G_kernels(all_G, plots_dir, gamma_0=None):
    """Plot G kernels for each layer.

    Linear DMFT stores ``G`` as ``(T, T)``; nonlinear DMFT stores
    ``(T, P, T, P)``, in which case the final-time sample-sample block is shown.
    """
    all_G = _to_numpy(all_G)
    n_layers = len(all_G)
    _warn_if_nonfinite("all_G", np.stack([np.asarray(G) for G in all_G]))

    fig, axes = plt.subplots(
        n_layers, 1, figsize=(6, 2 * n_layers), squeeze=False
    )
    for l, G_l in enumerate(all_G):
        ax = axes[l, 0]
        G_arr = np.asarray(G_l)
        if G_arr.ndim == 4:
            G_arr = G_arr[-1, :, -1, :]
        ax.imshow(G_arr, cmap="coolwarm")
        if l == 0:
            ax.set_ylabel("Theory")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Layer {l}")

    if gamma_0 is not None:
        fig.suptitle(f"$\\gamma_0 = {gamma_0}$", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "all_G_kernels.png"), bbox_inches="tight")
    plt.close(fig)


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
    ylabel="Theory",
):
    """Plot initial sample-sample kernels for each PC layer."""
    kernels = _to_numpy(kernels)
    n_layers = len(kernels)
    _warn_if_nonfinite(
        filename,
        np.stack([np.asarray(k) for k in kernels]),
    )

    fig, axes = plt.subplots(
        n_layers, 1, figsize=(6, 2 * n_layers), squeeze=False
    )
    for l, cov_l in enumerate(kernels):
        ax = axes[l, 0]
        kernel = _initial_sample_kernel(
            cov_l,
            num_inference_steps=num_inference_steps,
            num_training_steps=num_training_steps,
            num_samples=num_samples,
        )
        ax.imshow(kernel, cmap="coolwarm")
        if l == 0:
            ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Layer {l + 1}")

    if gamma_0 is not None:
        fig.suptitle(f"$\\gamma_0 = {gamma_0}$", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), bbox_inches="tight")
    plt.close(fig)


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
):
    """Plot PC DMFT loss and initial sample-sample Ch / Cdelta kernels."""
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

    plot_pc_layer_kernels(
        kernels=all_Ch,
        plots_dir=plots_dir,
        filename="all_Ch_kernels.png",
        num_inference_steps=num_inference_steps,
        num_training_steps=num_training_steps,
        num_samples=num_samples,
        gamma_0=gamma_0,
        ylabel=r"$C^h$",
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
):
    """Overlay the PC DMFT theory loss curve with finite-size empirical losses.

    ``finite_df`` is expected to have columns ``width``, ``t`` and ``loss``,
    as produced by ``test_coord_check.get_coord_data(..., stats=["loss"])``
    run with the same hyperparameters used for ``pc_dmft_loss``. This lets us
    visually check that the finite-size PC networks converge to the DMFT
    (infinite-width) prediction as width grows.

    ``update_mode`` (e.g. ``"infer"`` / ``"theory"``) is optional metadata used
    in the plot title and output filename so multiple finite simulations can be
    compared without overwriting each other.

    If ``skip_theory`` is True (or ``pc_dmft_loss`` is None / all zeros), only
    finite overlays are drawn and the title/filename use ``pc_finite_loss``.
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

    widths = sorted(finite_df["width"].unique())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(widths) - 1)) for i in range(len(widths))]

    plt.figure(figsize=(8, 6))
    for width, color in zip(widths, colors):
        sub = finite_df[finite_df["width"] == width].sort_values("t")
        plt.plot(
            sub["t"],
            sub["loss"],
            marker="o",
            color=color,
            alpha=0.8,
            label=f"width={width}",
        )
    if plot_theory:
        theory_t = np.arange(1, len(pc_dmft_loss) + 1)
        plt.plot(
            theory_t,
            pc_dmft_loss,
            color="black",
            linewidth=2.5,
            linestyle="--",
            label="DMFT theory",
        )
    plt.xlabel("$t$")
    plt.ylabel("PC training loss (MSE)")
    if skip_theory or not plot_theory:
        title = "PC finite-size simulation"
        filename = "pc_finite_loss"
    else:
        title = "PC theory vs finite-size simulation"
        filename = "pc_theory_vs_finite_loss"
    if update_mode is not None:
        title += f" ({update_mode})"
        filename += f"_{update_mode}"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    if activity_lr is not None:
        title += f", activity lr$={activity_lr}$"
    if n_infer_iters is not None:
        title += f", $K={n_infer_iters}$"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"{filename}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
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
    "n_hidden": lambda v: f"$H={int(v)}$",
    "gamma_0": lambda v: f"$\\gamma_0={v}$",
    "n_infer_iters": lambda v: f"$K={int(v)}$",
}

_SWEEP_AXIS_TITLE = {
    "n_hidden": r"$H$",
    "gamma_0": r"$\gamma_0$",
    "n_infer_iters": r"$K$",
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
    plot_closed_form=False,
):
    """Overlay theory and finite-size losses for every value of ``swept_col``.

    Finite curves use only the largest recorded width. Theory is dashed; finite
    infer is solid. If ``plot_closed_form`` is True, closed-form finite updates
    are added: one curve per swept value, except for ``n_infer_iters`` where
    closed-form is independent of ``K`` so a single extra curve is drawn.
    """
    if swept_col not in _SWEEP_VALUE_LABEL:
        raise ValueError(
            f"swept_col must be one of {list(_SWEEP_VALUE_LABEL)}, got {swept_col!r}"
        )

    group_cols = [c for c in _SWEEP_META_COLS if c != swept_col]
    frames = []
    if theory_df is not None and len(theory_df):
        frames.append(theory_df[group_cols])
    if finite_df is not None and len(finite_df):
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
    axis_title = _SWEEP_AXIS_TITLE[swept_col]
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
            if finite_df is not None and len(finite_df)
            else finite_df
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

        cmap = plt.get_cmap("viridis")
        colors = [
            cmap(i / max(1, len(swept_values) - 1))
            for i in range(len(swept_values))
        ]

        plt.figure(figsize=(8, 6))
        plot_theory = (
            (not skip_theory)
            and g_theory is not None
            and len(g_theory)
        )
        for value, color in zip(swept_values, colors):
            label = value_label(value)
            if plot_theory:
                sub_th = g_theory[g_theory[swept_col] == value].sort_values("t")
                if len(sub_th):
                    y = np.asarray(sub_th["loss"])
                    if not np.allclose(y, 0.0):
                        _warn_if_nonfinite(f"pc_dmft_loss[{swept_col}={value}]", y)
                        plt.plot(
                            sub_th["t"],
                            y,
                            color=color,
                            linestyle="--",
                            linewidth=2.0,
                            label=f"theory, {label}",
                        )
            if infer_finite is not None and len(infer_finite):
                sub_fi = infer_finite[
                    (infer_finite[swept_col] == value)
                    & (infer_finite["width"] == max_width)
                ].sort_values("t")
                if len(sub_fi):
                    plt.plot(
                        sub_fi["t"],
                        sub_fi["loss"],
                        color=color,
                        linestyle="-",
                        marker="o",
                        markersize=4,
                        alpha=0.85,
                        label=f"finite infer, {label}",
                    )
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
                    plt.plot(
                        sub_cf["t"],
                        sub_cf["loss"],
                        color=color,
                        linestyle=":",
                        marker="s",
                        markersize=4,
                        alpha=0.85,
                        label=f"finite closed-form, {label}",
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
                plt.plot(
                    sub_cf["t"],
                    sub_cf["loss"],
                    color="black",
                    linestyle="-.",
                    linewidth=2.2,
                    label=f"finite closed-form ($N={cf_width}$)",
                )

        plt.xlabel("$t$")
        plt.ylabel("PC training loss (MSE)")
        title = f"PC theory vs finite-size vs {axis_title}"
        if skip_theory or not plot_theory:
            title = f"PC finite-size vs {axis_title}"
        if max_width is not None:
            title += f" ($N={max_width}$)"
        plt.title(title)
        plt.legend(fontsize=8, ncol=1)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

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
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
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
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(widths) - 1)) for i in range(len(widths))]

    plt.figure(figsize=(8, 6))
    for width, color in zip(widths, colors):
        sub = finite_df[finite_df["width"] == width].sort_values("t")
        plt.plot(
            sub["t"],
            sub["loss"],
            marker="o",
            color=color,
            alpha=0.8,
            label=f"width={width}",
        )
    if plot_theory:
        theory_t = np.arange(1, len(dmft_loss) + 1)
        plt.plot(
            theory_t,
            dmft_loss,
            color="black",
            linewidth=2.5,
            linestyle="--",
            label="DMFT theory",
        )
    plt.xlabel("$t$")
    plt.ylabel("BP training loss (MSE)")
    if skip_theory or not plot_theory:
        title = "BP finite-size simulation"
        filename = "bp_finite_loss"
    else:
        title = "BP theory vs finite-size simulation"
        filename = "bp_theory_vs_finite_loss"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, f"{filename}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
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

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(widths) - 1)) for i in range(len(widths))]

    plt.figure(figsize=(8, 6))
    for width, color in zip(widths, colors):
        values = np.asarray(similarities_by_width[width]).flatten()
        _warn_if_nonfinite(f"cos_sim width={width}", values)
        t = np.arange(1, len(values) + 1)
        plt.plot(
            t,
            values,
            marker="o",
            color=color,
            alpha=0.8,
            label=f"width={width}",
        )
    plt.xlabel("$t$")
    plt.ylabel(r"$\cos(\nabla_{\theta}\mathcal{L}_{\mathrm{BP}}, "
               r"\nabla_{\theta}\mathcal{F}^*_{\mathrm{PC}})$")
    title = "PC–BP gradient cosine similarity"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    if activity_lr is not None:
        title += f", activity lr$={activity_lr}$"
    plt.title(title)
    plt.ylim(-1.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "grad_cosine_similarities.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"PC–BP gradient cosine similarity plot saved to {save_path}")
    return save_path


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