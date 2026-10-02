"""Render example paper figures from synthetic data, to check the plot style.

The simulations behind the paper figures take cluster time, and the
analysis scripts delete their raw fields once the plots are written, so
this script fabricates records with the same schema as the real ones and
calls the real plotting functions. The numbers are meaningless; only the
typography, panel sizes, labels and legends are. Sweep values, depths,
widths and learning rates match the commands in ``execute.sh`` that
``extract_figures.py`` collects.

    python preview_figures.py [--out preview_figures] [--extras]

Paper panels are written directly under ``--out`` as ``fig2a.png`` and
so on, one file per entry of ``extract_figures.FIGURES``. Pass
``--extras`` to also write the panels the analysis scripts draw but
that list does not collect, under ``--out/extra`` as ``figX1.png``,
``figX2.png``, ...:

    figX1   theory vs finite PC loss
    figX2   linear relative-Frobenius displacement vs layer
    figX3   linear relative-Frobenius last-layer displacement vs gamma
    figX4   nonlinear relative-Frobenius displacement vs layer
    figX5   nonlinear relative-Frobenius last-layer displacement vs gamma
    figX6   linear C^Δ vs width, L = 2
    figX7   linear C^Δ vs width, L = 3
    figX8   linear C^Δ vs width, L = 4
    figX9   linear C^Δ vs width, L = 5
    figX10  nonlinear C^Δ vs width, L = 3
    figX11  C^φ vs width at the linear activity learning rate, L = 3
    figX12  loss-matched PC vs BP loss with L* markers
    figX13  single-seed MNIST epoch metrics
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from extract_figures import FIGURES
from src import plot_style as ps
from src.plot_dmft_results import (
    plot_final_kernel_grid,
    plot_kernel_concentration_vs_time,
    plot_kernel_displacement_per_timepoint,
    plot_kernel_input_alignment_vs_time,
    plot_kernel_spectrum,
    plot_kernel_target_alignment_vs_time,
    plot_kernel_target_input_alignment_final,
    plot_pc_bp_alignment_vs_time,
    plot_pc_bp_loss,
    plot_pc_bp_loss_matched_times,
    plot_pc_k_sweep_displacement,
    plot_pc_kernel_width_alignment,
    plot_pc_last_layer_displacement_vs_gamma,
    plot_pc_param_sweep_loss,
    plot_pc_theory_vs_finite_loss,
    plot_temporal_kernel_grid,
)

RNG = np.random.default_rng(0)

# execute.sh: linear convergence (results_D / G / W / KG / S).
_LIN_WIDTHS = (10, 25, 100, 250, 1000, 2500, 10000)
_LIN_GAMMAS = (0.1, 0.5, 1.0)
_LIN_DEPTHS = (2, 3, 4, 5)
_LIN_KS = (5, 20, 50, 200, 500)
_LIN_LR = 0.01
_LIN_K = 5
_LIN_T = 20
_LIN_P = 20
_LIN_L = 5
_LIN_N = 10000
_LIN_SEEDS = 5

# execute.sh: nonlinear convergence (results_nonlin_G / KG / W).
_NL_GAMMAS = (0.1, 0.5, 1.0)
_NL_KS = (5, 10, 20, 50, 200, 500)
_NL_LR = 0.05
_NL_K = 10
_NL_T = 30
_NL_L = 3
_NL_N = 10000
_NL_SEEDS = 5

# execute.sh: alignment on tiny-CIFAR10, tanh, loss-matched linear suite.
_AL_L = 3
_AL_N = 10000
_AL_GAMMA = 1.0
_AL_LR = 0.1
_AL_K = 500
_AL_T = 1001
_AL_P = 40
_AL_SEEDS = 3
_AL_N_LOSS = 200  # max(11, round(1001 / 5))
_AL_HEATMAP = (0, 100, 199)

# execute.sh: classification benchmarks.
_BENCH_EPOCHS = 10
_BENCH_SEEDS = 3

_KERNEL_GRID_KW = dict(cbar=False, center_zero=False)

_OLD_LAYOUT = (
    "loss_sweeps",
    "theory_vs_finite",
    "width_alignment",
    "displacement",
    "kernel_grids",
    "kernel_grids_k_sweep",
    "kernel_grids_lstar",
    "alignment",
    "benchmark",
)


def _loss_curve(n_t, scale=1.0, floor=0.02, noise=0.0):
    """A loss curve that decays and flattens, like the real ones."""
    t = np.arange(n_t)
    y = floor + scale * np.exp(-3.0 * t / max(1, n_t - 1))
    if noise:
        y = y * (1.0 + noise * RNG.standard_normal(n_t))
    return np.clip(y, 1e-4, None)


def _kernel(size, rank=3):
    """A PSD kernel with decaying spectrum, normalised to unit diagonal."""
    factors = RNG.standard_normal((size, rank))
    K = factors @ factors.T + 0.3 * np.eye(size)
    d = np.sqrt(np.diag(K))
    return K / np.outer(d, d)


def _kernels_per_layer(n_layers, size, rank=3):
    return [_kernel(size, rank=rank + layer) for layer in range(n_layers)]


def _publish(save_path, dest_dir, stem):
    """Copy the PNG written for ``save_path`` to ``dest_dir/stem.png``."""
    if save_path is None:
        raise RuntimeError(f"no plot was written for {stem}")
    src = Path(save_path).with_suffix(".png")
    if not src.is_file():
        raise FileNotFoundError(src)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / f"{stem}.png")


def _saved_named(paths, name_part):
    matches = [p for p in paths if p and name_part in Path(p).name]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one path containing {name_part!r}, got {paths}"
        )
    return matches[0]


def _clear_previous_layout(out_dir):
    for name in _OLD_LAYOUT:
        path = out_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    extra = out_dir / "extra"
    if extra.is_dir():
        shutil.rmtree(extra)


# --- Loss sweeps (fig2a, fig2b, fig2d, fig3a, fig3b) ----------------------


def _sweep_frames(
    swept_col,
    values,
    *,
    fixed,
    n_t,
    width,
    include_theory,
    include_closed_form,
):
    """Theory / finite frames for ``plot_pc_param_sweep_loss``."""
    theory, finite = [], []
    for i, value in enumerate(values):
        meta = dict(fixed)
        meta[swept_col] = value
        scale = 1.0 / (1.0 + 0.35 * i)
        if include_theory:
            for t, loss in enumerate(_loss_curve(n_t, scale=scale)):
                theory.append(dict(meta, t=t, loss=float(loss)))
        noise = 0.35 / np.sqrt(width)
        curve = _loss_curve(n_t, scale=scale * 1.08, noise=noise)
        for t, loss in enumerate(curve):
            finite.append(
                dict(
                    meta,
                    t=t,
                    loss=float(loss),
                    width=width,
                    infer_mode="infer",
                )
            )
    if include_closed_form:
        # Closed form does not depend on K; one curve, tied to the
        # smallest K so it shares the infer group's other metadata.
        cf_meta = dict(fixed)
        cf_meta[swept_col] = values[0]
        for t, loss in enumerate(_loss_curve(n_t, scale=0.85, floor=0.03)):
            finite.append(
                dict(
                    cf_meta,
                    t=t,
                    loss=float(loss),
                    width=width,
                    infer_mode="closed_form",
                )
            )
    theory_df = pd.DataFrame(theory) if theory else None
    return theory_df, pd.DataFrame(finite)


def _plot_loss_sweep(
    work,
    stem,
    swept_col,
    values,
    *,
    fixed,
    n_t,
    width,
    include_theory,
    plot_closed_form,
):
    theory_df, finite_df = _sweep_frames(
        swept_col,
        values,
        fixed=fixed,
        n_t=n_t,
        width=width,
        include_theory=include_theory,
        include_closed_form=plot_closed_form,
    )
    saved = plot_pc_param_sweep_loss(
        theory_df,
        finite_df,
        str(work),
        swept_col,
        skip_theory=not include_theory,
        plot_closed_form=plot_closed_form,
    )
    if len(saved) != 1:
        raise RuntimeError(f"{stem}: expected one loss sweep, got {saved}")
    return saved[0]


def _meta(n_hidden, gamma_0, activity_lr, n_infer_iters):
    return dict(
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        param_type="mupc",
        use_skips=False,
    )


# --- Width alignment (fig2c, fig3c, supfig1, C^Δ extras) ------------------


def _width_frame(n_hidden, kernels, *, gap0, n_seeds, widths):
    rows = []
    for layer in range(n_hidden):
        for width in widths:
            gap = (gap0 + 0.04 * layer) / np.sqrt(width / widths[0])
            for seed in range(n_seeds):
                for kernel in kernels:
                    rows.append(
                        dict(
                            width=width,
                            layer=layer,
                            kernel=kernel,
                            seed=seed,
                            alignment=1.0
                            - gap * (1.0 + 0.08 * RNG.standard_normal()),
                        )
                    )
    return pd.DataFrame(rows)


def _plot_width(
    work,
    *,
    n_hidden,
    activity_lr,
    n_infer_iters,
    feature_symbol,
    include_delta,
    gap0,
    n_seeds,
    widths,
):
    kernels = ("h", "delta") if include_delta else ("h",)
    saved = plot_pc_kernel_width_alignment(
        _width_frame(
            n_hidden, kernels, gap0=gap0, n_seeds=n_seeds, widths=widths
        ),
        str(work),
        gamma_0=1.0,
        n_hidden=n_hidden,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        feature_symbol=feature_symbol,
    )
    return list(saved)


# --- Displacement (fig2e, fig2f, supfig3, relative-Frobenius extras) ------


def _displacement_frame(n_hidden, ks, gammas, *, closed_form):
    """Per-layer displacement. KG runs pass ``--skip_theory``, so no DMFT."""
    rows = []
    for gamma_0 in gammas:
        for layer in range(n_hidden):
            depth = (layer + 1) / n_hidden
            for K in ks:
                drift = 0.06 * depth * gamma_0**2 * (1.0 + 8.0 / K)
                rows.append(
                    dict(
                        layer=layer,
                        kind="infer",
                        n_infer_iters=int(K),
                        gamma_0=float(gamma_0),
                        displacement=1.0 - drift,
                        rel_displacement=drift,
                    )
                )
            if closed_form:
                drift = 0.04 * depth * gamma_0**2
                rows.append(
                    dict(
                        layer=layer,
                        kind="closed_form",
                        n_infer_iters=0,
                        gamma_0=float(gamma_0),
                        displacement=1.0 - drift,
                        rel_displacement=drift,
                    )
                )
    return pd.DataFrame(rows)


def _plot_displacement_pair(
    work,
    frame,
    *,
    n_hidden,
    activity_lr,
    feature_symbol,
    layer_gamma,
    include_rel=False,
):
    """Layer and last-layer-vs-gamma displacement plots.

    Relative Frobenius is an extra panel; cosine is the paper figure.
    """
    layer_df = frame[frame["gamma_0"] == layer_gamma]
    kw = dict(
        n_hidden=n_hidden,
        activity_lr=activity_lr,
        dir_name="convergence",
        feature_symbol=feature_symbol,
    )
    cosine_layer = plot_pc_k_sweep_displacement(
        layer_df,
        str(work),
        gamma_0=layer_gamma,
        metric="cosine",
        **kw,
    )
    cosine_gamma = plot_pc_last_layer_displacement_vs_gamma(
        frame, str(work), metric="cosine", **kw
    )
    rel_layer = rel_gamma = None
    if include_rel:
        rel_layer = plot_pc_k_sweep_displacement(
            layer_df,
            str(work),
            gamma_0=layer_gamma,
            metric="rel_frob",
            **kw,
        )
        rel_gamma = plot_pc_last_layer_displacement_vs_gamma(
            frame, str(work), metric="rel_frob", **kw
        )
    return cosine_layer, rel_layer, cosine_gamma, rel_gamma


# --- Kernel grids (fig2g, fig2h, supfig2, fig3g–i) ------------------------


def _plot_convergence_grids(work):
    """results_S single-K grid, its temporal twin, and the results_KG K sweep."""
    layers, size = _LIN_L, _LIN_P
    single_rows = [
        (ps.LABEL_DMFT, _kernels_per_layer(layers, size)),
        (ps.LABEL_NN, _kernels_per_layer(layers, size)),
        (ps.LABEL_NN_CLOSED_FORM, _kernels_per_layer(layers, size)),
    ]
    common = dict(
        plots_dir=str(work),
        gamma_0=1.0,
        n_hidden=layers,
        activity_lr=_LIN_LR,
        width=_LIN_N,
        dir_name="convergence",
        **_KERNEL_GRID_KW,
    )
    final = plot_final_kernel_grid(
        single_rows,
        n_infer_iters=_LIN_K,
        filename="final_pc_kernels_grid.png",
        title=r"Final $C^{h}$ feature kernels",
        **common,
    )
    temporal = plot_temporal_kernel_grid(
        [
            (label, _kernels_per_layer(layers, _LIN_T, rank=2))
            for label, _ in single_rows
        ],
        n_infer_iters=_LIN_K,
        title=r"Sample-traced $C^{h}$ feature kernels",
        **common,
    )
    # results_KG uses --skip_theory, so the grid has no DMFT row.
    k_rows = [
        (ps.k_label(K, prefix=ps.LABEL_NN), _kernels_per_layer(layers, size))
        for K in _LIN_KS
    ]
    k_rows.append(
        (ps.LABEL_NN_CLOSED_FORM, _kernels_per_layer(layers, size))
    )
    k_sweep = plot_final_kernel_grid(
        k_rows,
        filename="final_pc_kernels_grid.png",
        title=r"Final $C^{h}$ feature kernels",
        **common,
    )
    return final, temporal, k_sweep


# --- Alignment (fig3d–i, supfig4–7, figX12) -------------------------------


_ALIGN_KW = dict(
    n_hidden=_AL_L,
    gamma_0=_AL_GAMMA,
    activity_lr=_AL_LR,
    n_infer_iters=_AL_K,
    width=_AL_N,
    feature_symbol="phi",
    dir_name="alignment",
)

_BY_LOSS_LINEAR = dict(
    x_col="loss",
    xlabel=ps.TEX["loss"],
    xscale="linear",
    invert_x=True,
)


def _pc_bp_losses():
    pc = _loss_curve(_AL_T, scale=1.0, floor=0.05)
    bp = _loss_curve(_AL_T, scale=1.08, floor=0.055)
    return pc, bp


def _linear_loss_grid(pc_losses, bp_losses):
    """Decreasing overlap grid, as ``_overlap_loss_grid(..., scale='linear')``."""
    high = float(min(pc_losses[0], bp_losses[0]))
    low = float(max(pc_losses[-1], bp_losses[-1]))
    return np.linspace(high, low, _AL_N_LOSS)


def _first_crossing(losses, loss_grid):
    losses = np.asarray(losses, dtype=float)
    times = []
    last = len(losses) - 1
    for loss_star in loss_grid:
        hit = np.flatnonzero(losses <= loss_star)
        times.append(int(hit[0]) if hit.size else last)
    return np.asarray(times, dtype=int)


def _layer_rows(loss_grid, value_col, *, n_hidden=_AL_L):
    rows = []
    for method, offset in (("pc", 0.0), ("bp", 0.06)):
        for layer in range(n_hidden):
            for i, loss in enumerate(loss_grid):
                base = 0.25 + 0.55 * i / (len(loss_grid) - 1)
                row = dict(loss=float(loss), layer=layer, method=method)
                row[value_col] = base * (1.0 - 0.12 * layer) - offset
                rows.append(row)
    return pd.DataFrame(rows)


def _plot_alignment(work, *, include_loss_matched=False):
    pc_losses, bp_losses = _pc_bp_losses()
    loss_grid = _linear_loss_grid(pc_losses, bp_losses)
    pc_times = _first_crossing(pc_losses, loss_grid)
    bp_times = _first_crossing(bp_losses, loss_grid)
    plots = str(work)

    by_time = plot_pc_bp_loss(
        pc_losses,
        bp_losses,
        plots,
        n_hidden=_AL_L,
        gamma_0=_AL_GAMMA,
        activity_lr=_AL_LR,
        n_infer_iters=_AL_K,
        width=_AL_N,
        dir_name="alignment",
    )
    # The by-time curve is fig3d. The loss-matched curve (L* markers) is
    # an extra panel.
    loss_matched = None
    if include_loss_matched:
        loss_matched = plot_pc_bp_loss_matched_times(
            pc_losses,
            bp_losses,
            pc_times,
            bp_times,
            loss_grid,
            plots,
            n_hidden=_AL_L,
            gamma_0=_AL_GAMMA,
            activity_lr=_AL_LR,
            n_infer_iters=_AL_K,
            width=_AL_N,
            dir_name="alignment_loss_matched",
            heatmap_idx=list(_AL_HEATMAP),
            yscale="log",
            filename="pc_bp_loss_matched.png",
        )

    align_rows = []
    for layer in range(_AL_L):
        for i, loss in enumerate(loss_grid):
            align_rows.append(
                dict(
                    loss=float(loss),
                    layer=layer,
                    alignment=0.95
                    - 0.35 * i / (len(loss_grid) - 1)
                    - 0.06 * layer,
                )
            )
    cka = plot_pc_bp_alignment_vs_time(
        pd.DataFrame(align_rows),
        plots,
        filename="pc_bp_kernel_alignment_vs_loss.png",
        **_ALIGN_KW,
        **_BY_LOSS_LINEAR,
    )

    ti_rows = []
    for method, offset in (("pc", 0.0), ("bp", 0.05)):
        for ref, scale in (("target", 1.0), ("input", 0.55)):
            for layer in range(_AL_L):
                ti_rows.append(
                    dict(
                        layer=layer,
                        method=method,
                        ref=ref,
                        alignment=scale * (0.62 - 0.1 * layer) - offset,
                    )
                )
    target_input = plot_kernel_target_input_alignment_final(
        pd.DataFrame(ti_rows),
        plots,
        title="Feature-kernel alignment with the target and input",
        **_ALIGN_KW,
    )

    grids = []
    for index in _AL_HEATMAP:
        loss_star = float(loss_grid[index])
        t_pc = int(pc_times[index])
        t_bp = int(bp_times[index])
        grids.append(
            plot_final_kernel_grid(
                [
                    (ps.LABEL_PC, _kernels_per_layer(_AL_L, _AL_P)),
                    (ps.LABEL_BP, _kernels_per_layer(_AL_L, _AL_P)),
                ],
                plots_dir=plots,
                gamma_0=_AL_GAMMA,
                n_hidden=_AL_L,
                activity_lr=_AL_LR,
                n_infer_iters=_AL_K,
                width=_AL_N,
                filename=f"feature_kernels_grid_lstar{index}.png",
                vmin=-1.0,
                vmax=1.0,
                dir_name="alignment",
                title=(
                    rf"$C^{{\phi}}$ feature kernels "
                    rf"($\mathcal{{L}}={loss_star:.2e}$, "
                    rf"$t_{{\mathrm{{PC}}}}={t_pc}$, "
                    rf"$t_{{\mathrm{{BP}}}}={t_bp}$, correlation)"
                ),
            )
        )

    target = plot_kernel_target_alignment_vs_time(
        _layer_rows(loss_grid, "alignment"),
        plots,
        filename="kernel_target_alignment_vs_loss.png",
        **_ALIGN_KW,
        **_BY_LOSS_LINEAR,
    )
    inp = plot_kernel_input_alignment_vs_time(
        _layer_rows(loss_grid, "alignment"),
        plots,
        filename="kernel_input_alignment_vs_loss.png",
        **_ALIGN_KW,
        **_BY_LOSS_LINEAR,
    )
    disp = _layer_rows(loss_grid, "displacement")
    disp["rel_displacement"] = 1.0 - disp["displacement"].clip(upper=0.99)
    displacement = plot_kernel_displacement_per_timepoint(
        disp,
        plots,
        filename="kernel_displacement_vs_loss.png",
        t_sub=r"\mathcal{L}",
        **_ALIGN_KW,
        **_BY_LOSS_LINEAR,
    )

    spec_rows = []
    for method in ("pc", "bp"):
        for layer in range(_AL_L):
            eigenvalues = np.geomspace(1.0, 1e-3, _AL_P) * (
                1.0 + 0.15 * layer
            )
            for i, value in enumerate(eigenvalues, start=1):
                spec_rows.append(
                    dict(
                        layer=layer,
                        method=method,
                        index=i,
                        eigenvalue=float(value),
                        effective_rank=4.5 + layer,
                    )
                )
    spectrum = plot_kernel_spectrum(
        pd.DataFrame(spec_rows),
        plots,
        title=(
            rf"$C^{{\phi}}$ feature-kernel spectrum at last overlap"
        ),
        filename="kernel_spectrum_final.png",
        ylabel=r"$\lambda_i(C^{\phi,\ell})$",
        annotate_rank=True,
        **_ALIGN_KW,
    )

    conc = _layer_rows(loss_grid, "cka_mean")
    conc["cka_std"] = 0.02
    concentration = plot_kernel_concentration_vs_time(
        conc,
        plots,
        metric="cka",
        n_seeds=_AL_SEEDS,
        filename="kernel_concentration_vs_loss.png",
        **_ALIGN_KW,
        **_BY_LOSS_LINEAR,
    )
    return {
        "by_time": by_time,
        "loss_matched": loss_matched,
        "cka": cka,
        "target_input": target_input,
        "grids": grids,
        "target": target,
        "input": inp,
        "displacement": displacement,
        "spectrum": spectrum,
        "concentration": concentration,
    }


# --- Benchmark (supfig8, figX13) ------------------------------------------


def _synthetic_benchmark_history(
    n_epochs=_BENCH_EPOCHS,
    n_mini=41,
    n_steps=80,
    seed=0,
    pc_offset=0.0,
    acc_end=97.0,
    loss_floor=0.08,
):
    """History dict matching ``run_benchmark`` / ``plot_metrics`` keys."""
    rng = np.random.default_rng(seed)
    n_eval = n_epochs + 1
    epochs = np.arange(n_eval, dtype=float)
    mini = np.linspace(0.0, float(n_epochs), n_mini)

    def loss_curve(n, start, floor):
        t = np.arange(n)
        y = floor + (start - floor) * np.exp(-3.0 * t / max(1, n - 1))
        y = y * (1.0 + 0.03 * rng.standard_normal(n)) + 0.08 * pc_offset
        return np.clip(y, 1e-4, None)

    def acc_curve(n, start, end):
        t = np.arange(n) / max(1, n - 1)
        y = start + (end - start) * (1.0 - np.exp(-4.0 * t))
        y = y + 0.4 * rng.standard_normal(n) - 1.5 * pc_offset
        return np.clip(y, 0.0, 100.0)

    return {
        "epoch_eval": epochs,
        "epoch_train": epochs,
        "mini_epoch": mini,
        "pc_train_loss_epoch": loss_curve(n_eval, 2.4, loss_floor * 0.5),
        "bp_train_loss_epoch": loss_curve(n_eval, 2.35, loss_floor * 0.55),
        "pc_train_acc_epoch": acc_curve(n_eval, 12.0, acc_end + 1.0),
        "bp_train_acc_epoch": acc_curve(n_eval, 11.0, acc_end + 0.5),
        "pc_test_loss": loss_curve(n_eval, 2.3, loss_floor),
        "bp_test_loss": loss_curve(n_eval, 2.25, loss_floor + 0.01),
        "pc_test_acc": acc_curve(n_eval, 14.0, acc_end),
        "bp_test_acc": acc_curve(n_eval, 13.0, acc_end - 0.5),
        "pc_train_loss_mini": loss_curve(n_mini, 2.4, loss_floor * 0.5),
        "bp_train_loss_mini": loss_curve(n_mini, 2.35, loss_floor * 0.55),
        "pc_train_acc_mini": acc_curve(n_mini, 12.0, acc_end + 1.0),
        "bp_train_acc_mini": acc_curve(n_mini, 11.0, acc_end + 0.5),
        "pc_test_loss_mini": loss_curve(n_mini, 2.3, loss_floor),
        "bp_test_loss_mini": loss_curve(n_mini, 2.25, loss_floor + 0.01),
        "pc_test_acc_mini": acc_curve(n_mini, 14.0, acc_end),
        "bp_test_acc_mini": acc_curve(n_mini, 13.0, acc_end - 0.5),
        "pc_train_loss_step": loss_curve(n_steps, 2.4, loss_floor * 0.5),
        "bp_train_loss_step": loss_curve(n_steps, 2.35, loss_floor * 0.55),
        "pc_train_acc_step": acc_curve(n_steps, 12.0, acc_end + 1.0),
        "bp_train_acc_step": acc_curve(n_steps, 11.0, acc_end + 0.5),
    }


def _plot_benchmarks(work, *, include_single=False):
    from train_benchmark import plot_metrics, plot_metrics_mean_sem

    def histories(seed0, acc_end, loss_floor):
        return [
            _synthetic_benchmark_history(
                seed=seed0 + s,
                pc_offset=0.15 * s,
                acc_end=acc_end,
                loss_floor=loss_floor,
            )
            for s in range(_BENCH_SEEDS)
        ]

    mnist_dir = work / "mnist"
    fashion_dir = work / "fashion"
    mnist_mean = plot_metrics_mean_sem(
        histories(0, acc_end=97.0, loss_floor=0.08), str(mnist_dir)
    )[0]
    fashion_mean = plot_metrics_mean_sem(
        histories(10, acc_end=90.0, loss_floor=0.15), str(fashion_dir)
    )[0]
    mnist_single = None
    if include_single:
        mnist_single = plot_metrics(
            _synthetic_benchmark_history(
                seed=0, acc_end=97.0, loss_floor=0.08
            ),
            str(mnist_dir / "single"),
        )[0]
    return mnist_mean, fashion_mean, mnist_single


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="preview_figures")
    parser.add_argument(
        "--extras",
        action="store_true",
        help=(
            "Also write the panels extract_figures.py does not collect, "
            "under OUT/extra."
        ),
    )
    args = parser.parse_args()

    # Callers look up save_figure on the plot_style module, so this
    # applies to the DMFT plotters and to train_benchmark.
    _orig_save_figure = ps.save_figure

    def _save_png(fig, save_path, *, formats=("png",)):
        del formats
        return _orig_save_figure(fig, save_path, formats=("png",))

    ps.save_figure = _save_png

    out_dir = Path(args.out).resolve()
    extra_dir = out_dir / "extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_layout(out_dir)
    for pdf in out_dir.glob("*.pdf"):
        pdf.unlink()

    with tempfile.TemporaryDirectory(prefix="preview_paper_") as tmp:
        work = Path(tmp)
        paper = {}

        paper["fig2a"] = _plot_loss_sweep(
            work,
            "fig2a",
            "n_hidden",
            _LIN_DEPTHS,
            fixed=_meta(_LIN_L, 1.0, _LIN_LR, _LIN_K),
            n_t=_LIN_T,
            width=_LIN_N,
            include_theory=True,
            plot_closed_form=False,
        )
        paper["fig2b"] = _plot_loss_sweep(
            work,
            "fig2b",
            "gamma_0",
            _LIN_GAMMAS,
            fixed=_meta(_LIN_L, 1.0, _LIN_LR, _LIN_K),
            n_t=_LIN_T,
            width=_LIN_N,
            include_theory=True,
            plot_closed_form=False,
        )
        paper["fig2d"] = _plot_loss_sweep(
            work,
            "fig2d",
            "n_infer_iters",
            _LIN_KS,
            fixed=_meta(_LIN_L, 1.0, _LIN_LR, _LIN_KS[0]),
            n_t=_LIN_T,
            width=_LIN_N,
            include_theory=False,
            plot_closed_form=True,
        )
        paper["fig3a"] = _plot_loss_sweep(
            work,
            "fig3a",
            "gamma_0",
            _NL_GAMMAS,
            fixed=_meta(_NL_L, 1.0, _NL_LR, _NL_K),
            n_t=_NL_T,
            width=_NL_N,
            include_theory=True,
            plot_closed_form=False,
        )
        paper["fig3b"] = _plot_loss_sweep(
            work,
            "fig3b",
            "n_infer_iters",
            _NL_KS,
            fixed=_meta(_NL_L, 1.0, _NL_LR, _NL_KS[0]),
            n_t=_NL_T,
            width=_NL_N,
            include_theory=False,
            plot_closed_form=False,
        )

        width_stems = {
            2: "supfig1a",
            3: "supfig1b",
            4: "supfig1c",
            5: "fig2c",
        }
        delta_stems = {2: "figX6", 3: "figX7", 4: "figX8", 5: "figX9"}
        extras = {}
        for n_hidden, stem in width_stems.items():
            saved = _plot_width(
                work,
                n_hidden=n_hidden,
                activity_lr=_LIN_LR,
                n_infer_iters=_LIN_K,
                feature_symbol="h",
                include_delta=args.extras,
                gap0=0.12,
                n_seeds=_LIN_SEEDS,
                widths=_LIN_WIDTHS,
            )
            paper[stem] = _saved_named(saved, "Ch_vs_width")
            if args.extras:
                extras[delta_stems[n_hidden]] = _saved_named(saved, "Cdelta")

        saved = _plot_width(
            work,
            n_hidden=_NL_L,
            activity_lr=_NL_LR,
            n_infer_iters=_NL_K,
            feature_symbol="phi",
            include_delta=args.extras,
            gap0=0.18,
            n_seeds=_NL_SEEDS,
            widths=_LIN_WIDTHS,
        )
        paper["fig3c"] = _saved_named(saved, "Cphi_vs_width")
        if args.extras:
            extras["figX10"] = _saved_named(saved, "Cdelta")
            saved = _plot_width(
                work,
                n_hidden=_NL_L,
                activity_lr=_LIN_LR,
                n_infer_iters=_LIN_K,
                feature_symbol="phi",
                include_delta=False,
                gap0=0.28,
                n_seeds=_LIN_SEEDS,
                widths=_LIN_WIDTHS,
            )
            extras["figX11"] = _saved_named(saved, "Cphi_vs_width")

        lin_disp = _displacement_frame(
            _LIN_L, _LIN_KS, _LIN_GAMMAS, closed_form=True
        )
        fig2e, figX2, fig2f, figX3 = _plot_displacement_pair(
            work,
            lin_disp,
            n_hidden=_LIN_L,
            activity_lr=_LIN_LR,
            feature_symbol="h",
            layer_gamma=1.0,
            include_rel=args.extras,
        )
        paper["fig2e"] = fig2e
        paper["fig2f"] = fig2f
        if args.extras:
            extras["figX2"] = figX2
            extras["figX3"] = figX3

        nl_disp = _displacement_frame(
            _NL_L, _NL_KS, _NL_GAMMAS, closed_form=False
        )
        sup3a, figX4, sup3b, figX5 = _plot_displacement_pair(
            work,
            nl_disp,
            n_hidden=_NL_L,
            activity_lr=_NL_LR,
            feature_symbol="phi",
            layer_gamma=1.0,
            include_rel=args.extras,
        )
        paper["supfig3a"] = sup3a
        paper["supfig3b"] = sup3b
        if args.extras:
            extras["figX4"] = figX4
            extras["figX5"] = figX5

        fig2g, fig2h, supfig2 = _plot_convergence_grids(work)
        paper["fig2g"] = fig2g
        paper["fig2h"] = fig2h
        paper["supfig2"] = supfig2

        aligned = _plot_alignment(work, include_loss_matched=args.extras)
        paper["fig3d"] = aligned["by_time"]
        paper["fig3e"] = aligned["cka"]
        paper["fig3f"] = aligned["target_input"]
        for letter, path in zip("ghi", aligned["grids"]):
            paper[f"fig3{letter}"] = path
        paper["supfig5a"] = aligned["target"]
        paper["supfig5b"] = aligned["input"]
        paper["supfig6"] = aligned["displacement"]
        paper["supfig7"] = aligned["spectrum"]
        paper["supfig4"] = aligned["concentration"]
        if args.extras:
            extras["figX12"] = aligned["loss_matched"]
            theory_t = 40
            theory_rows = []
            for width in (256, 1024, 4096, 16384):
                curve = _loss_curve(theory_t, noise=0.6 / np.sqrt(width))
                for t, loss in enumerate(curve):
                    theory_rows.append(
                        dict(width=width, t=t, loss=float(loss))
                    )
            extras["figX1"] = plot_pc_theory_vs_finite_loss(
                _loss_curve(theory_t),
                pd.DataFrame(theory_rows),
                str(work),
                gamma_0=1.0,
                n_hidden=3,
                activity_lr=0.05,
                n_infer_iters=10,
                update_mode="infer",
            )

        mnist, fashion, single = _plot_benchmarks(
            work, include_single=args.extras
        )
        paper["supfig8a"] = mnist
        paper["supfig8b"] = fashion
        if args.extras:
            extras["figX13"] = single

        expected = [stem for _, stem in FIGURES]
        missing = [stem for stem in expected if stem not in paper]
        if missing:
            raise RuntimeError(f"missing paper panels: {missing}")
        unexpected = sorted(set(paper) - set(expected))
        if unexpected:
            raise RuntimeError(f"unexpected paper panels: {unexpected}")

        for stem, path in paper.items():
            _publish(path, out_dir, stem)
        if args.extras:
            for stem, path in extras.items():
                _publish(path, extra_dir, stem)

    print(f"Paper panels ({len(paper)}) written under {out_dir}")
    if args.extras:
        print(f"Extra panels ({len(extras)}) written under {extra_dir}")


if __name__ == "__main__":
    main()
