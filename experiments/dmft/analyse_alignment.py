"""Finite-size PC vs backprop feature-kernel alignment analysis.

Trains one finite-width PC network and one finite-width BP network on the
same data (for a single, given set of hyperparameters — no DMFT theory,
no sweeps), then compares their hidden-layer feature kernels across
training. Linear nets use ``C^{h,\\ell}`` (``phi`` is the identity);
nonlinear nets use ``C^{\\phi,\\ell} = \\phi(h^\\ell)\\phi(h^\\ell)^T / N``.
PC inference is selected via ``--pc_infer_mode``: ``infer`` runs iterative
activity optimisation (``--n_infer_iters`` gradient steps per training
step); ``closed_form`` solves the linear equilibrium activities directly
(requires ``--act_fn linear``).

For ``--n_timepoints`` equally spaced training steps ``t`` (including
``t=0``, the untrained feedforward network, and ``t=T-1``, the last
training step), this produces feature-kernel heatmap grids. Displacement
and alignment line plots use a denser time grid: every training step if
``T <= 31``, otherwise every 2 steps (always including the endpoints).
Time-indexed figures are written under ``alignment/by_time/``.

Pass ``--skip_loss_matched`` to write only the time-indexed suite.
Otherwise a parallel suite is written under ``alignment/by_loss/``.
Those figures use a shared target-loss grid ``L*`` on the overlap of
the PC and BP training-loss curves (from the common init loss down to
the last loss both methods have reached). Each ``L*`` is paired with
the first time each method crosses it. Line plots, PC–BP pairing,
spectra, and sample-traced temporal kernels use a dense ``L*`` grid
with ``max(11, round(T / --n_loss_divisor))`` points (default divisor 5,
capped at ``T``); heatmaps use an ``--n_timepoints`` subset of that
same grid. ``--loss_scale`` (``log`` or ``linear``) sets both how
``L*`` is spaced and the x-axis scale of vs-``L`` plots (high loss on
the left). Verbose run output is written to
``alignment/analyse_alignment.log``.

- PC vs BP training-loss curves
  (``alignment/by_time/pc_bp_loss.png``). Loss-matched runs also write
  ``alignment/by_loss/pc_bp_loss.png`` with markers at the
  first-crossing times of each ``L*``.
- Feature-kernel grid: one figure per heatmap timepoint, a ``P x P``
  heatmap per hidden layer, top row PC / bottom row backprop, kernels
  converted to correlations ``R_{ij} = C_{ij} / sqrt(C_{ii} C_{jj})``
  (``alignment/by_time/feature_kernels_grid_t{t}.png``; loss-matched
  analogue ``alignment/by_loss/feature_kernels_grid_lstar{i}.png``).
- Kernel displacement vs time / loss: one subplot per layer, cosine
  similarity of each layer's feature kernel vs. at ``t=0``, PC and BP
  curves (``kernel_displacement_vs_time.png`` /
  ``kernel_displacement_vs_loss.png``), plus relative Frobenius
  ``||C - C_0||_F / ||C_0||_F`` (``kernel_displacement_rel_vs_*``).
- Kernel-target alignment: linear CKA of each PC / BP feature kernel
  with ``C^y = Y Y^T``, plus CKA of the kernel *change* ``C - C(0)``
  with ``C^y``, omitting the init point.
- Kernel-input alignment: CKA with ``C^x = X X^T``.
- Leading-eigenvector overlap with the labels.
- Feature-kernel effective rank (participation ratio).
- PC-BP alignment: CKA between the PC and backprop feature kernels at
  each layer (time-matched or loss-matched pairing), plus CKA of the
  kernel *changes*, plus top-``k`` eigenspace overlap (``k=1,3,5``).
- Final-kernel spectrum: last training step (by time) or last overlap
  ``L*`` (by loss).
- Sample-traced temporal kernels over the full time trajectory (by
  time) or over the loss-matched frames (by loss).
- Train vs test snapshot: PC–BP CKA and kernel–target CKA vs layer,
  using the training kernels at the last time / last overlap ``L*``
  and feedforward feature kernels on a held-out set of the same size
  (``pc_bp_kernel_alignment_test.png``,
  ``kernel_target_alignment_test.png``), plus a test-set kernel grid
  (``feature_kernels_grid_test.png``).
- If ``--n_seeds > 1``: kernel concentration across weight-init seeds
  vs layer and vs time / loss. Skipped when ``--n_seeds`` is 1.

``--seed`` draws two independent RNG streams (dataset, weight init). PC
is initialized from the weight-init stream; BP copies those Linear
weights so both networks start from the same parameters. At ``t=0`` the
script asserts that PC–BP kernel CKA is close to 1 (the mismatch
``1 - CKA`` is small). PC and BP always train on the same ``(X, y)``.
A held-out test set of the same size is drawn from a child of the
dataset stream (official test split for ``tiny-CIFAR10`` / CIFAR-10 /
Fashion-MNIST). Test feature kernels are feedforward ``h`` at ``k=0``
at the same training snapshots as the train kernels: last step
``t=T-1`` (by time) or last overlap ``L*`` (by loss; PC and BP may
use different times). Additional seeds (``--n_seeds``) vary only the
weight-init stream.

Use the ``PC_dmft_env`` conda environment:
    /data/ndcn-computational-neuroscience/mert5001/envs/PC_dmft_env/bin/python analyse_alignment.py
"""

import os
import sys
import atexit
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import set_seed
from experiments.limits_paper.utils import setup_bp_experiment
from experiments.dmft.utils import (
    CIFAR_GRAY_DIM,
    MLP,
    centered_kernel_alignment,
    cleanup_experiment_dirs,
    copy_mlp_linear_params,
    cosine_similarity,
    create_tiny_cifar10_dataset,
    create_toy_dataset,
    kernel_eigs,
    kernels_to_correlations,
    leading_evec_label_overlap,
    participation_ratio,
    participation_ratio_from_eigs,
    relative_frobenius_displacement,
    subspace_overlap,
    train_bpn,
)
from theory_pc_nonlin_utils import get_nonlinearity
from analyse_convergence import (
    _train_finite_pc,
    _feature_kernels_from_h,
    _sample_traced_feature_kernels_from_h_traj,
)
from plot_dmft_results import (
    _alignment_plots_dir,
    feature_kernel_symbol,
    plot_final_kernel_grid,
    plot_temporal_kernel_grid,
    plot_kernel_displacement_per_timepoint,
    plot_kernel_concentration_per_timepoint,
    plot_kernel_concentration_vs_time,
    plot_kernel_effective_rank_vs_time,
    plot_kernel_input_alignment_vs_time,
    plot_kernel_spectrum,
    plot_kernel_target_alignment_vs_time,
    plot_kernel_target_change_alignment_vs_time,
    plot_leading_evec_label_overlap_vs_time,
    plot_pc_bp_alignment_vs_time,
    plot_pc_bp_loss,
    plot_pc_bp_loss_matched_times,
    plot_pc_bp_subspace_overlap_vs_time,
    plot_temporal_kernel_effective_rank_vs_layer,
    plot_temporal_pc_bp_alignment_vs_layer,
    plot_pc_bp_kernel_alignment_test_vs_layer,
    plot_kernel_target_alignment_test_vs_layer,
)

_TERMINAL = sys.stdout
_log_f = None


def _term(msg=""):
    """Print to the terminal; also echo to the log if stdout is redirected."""
    print(msg, file=_TERMINAL, flush=True)
    if sys.stdout is not _TERMINAL:
        print(msg, flush=True)


def _restore_stdout():
    global _log_f
    if sys.stdout is not _TERMINAL:
        sys.stdout = _TERMINAL
    if _log_f is not None and not _log_f.closed:
        _log_f.close()
    _log_f = None


def _start_log(path):
    global _log_f
    _log_f = open(path, "w")
    atexit.register(_restore_stdout)
    sys.stdout = _log_f
    print(f"analyse_alignment log: {path}")


def _select_timepoints(n_train_iters, n_timepoints):
    """``n_timepoints`` equally spaced training-step indices.

    Always includes both endpoints (``t=0`` and ``t=n_train_iters - 1``).
    If ``n_timepoints >= n_train_iters``, every training step is used.
    Used for feature-kernel heatmap grids.
    """
    n_timepoints = max(2, min(n_timepoints, n_train_iters))
    idx = np.round(
        np.linspace(0, n_train_iters - 1, n_timepoints)
    ).astype(int)
    return np.unique(idx).tolist()


def _select_curve_timepoints(n_train_iters, stride=2, every_t_max=31):
    """Dense time grid for displacement and alignment line plots.

    Every training step if ``T <= every_t_max``, otherwise every
    ``stride`` steps (always including ``t=0`` and ``t=T-1``).
    """
    if n_train_iters <= every_t_max:
        return list(range(n_train_iters))
    idx = list(range(0, n_train_iters, stride))
    last = n_train_iters - 1
    if idx[-1] != last:
        idx.append(last)
    return idx


_DIR_BY_TIME = os.path.join("alignment", "by_time")
_DIR_BY_LOSS = os.path.join("alignment", "by_loss")
_DEFAULT_N_LOSS_DIVISOR = 5
_MIN_N_LOSS_POINTS = 11


def _n_loss_grid_points(
    n_train_iters, divisor, min_points=_MIN_N_LOSS_POINTS
):
    """Number of L* points: ``max(min_points, round(T / divisor))``, at most ``T``.

    Always includes both endpoints when that count is used in a linspace.
    """
    n_train_iters = int(n_train_iters)
    divisor = max(1, int(divisor))
    n = int(round(n_train_iters / float(divisor)))
    n = max(int(min_points), n)
    return max(2, min(n, n_train_iters))


def _min_positive(*arrays):
    """Smallest strictly positive finite value across ``arrays``."""
    vals = np.concatenate(
        [np.asarray(a, dtype=float).reshape(-1) for a in arrays]
    )
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return float(np.finfo(np.float64).tiny)
    return float(vals.min())


def _overlap_loss_grid(pc_losses, bp_losses, n_points, scale="log"):
    """Decreasing target-loss grid on the PC/BP overlap.

    ``L_hi`` is the shared init loss ``min(L_PC[0], L_BP[0])``.
    ``L_lo`` is the last overlap point ``max(L_PC[-1], L_BP[-1])``.
    ``scale`` is ``"log"`` or ``"linear"``. Always includes both
    endpoints when there is a decreasing overlap.
    """
    pc = np.asarray(pc_losses, dtype=float).reshape(-1)
    bp = np.asarray(bp_losses, dtype=float).reshape(-1)
    L_hi = float(np.nanmin([pc[0], bp[0]]))
    L_lo = float(np.nanmax([pc[-1], bp[-1]]))
    n_points = max(2, int(n_points))
    if not np.isfinite(L_hi) or not np.isfinite(L_lo):
        raise ValueError("non-finite losses; cannot build L* grid")
    if L_lo >= L_hi:
        print(
            f"Warning: no decreasing loss overlap "
            f"(L_hi={L_hi:.4e}, L_lo={L_lo:.4e}); using init only."
        )
        return np.asarray([L_hi], dtype=float)
    if scale == "log":
        L_hi_p = L_hi if L_hi > 0.0 else _min_positive(pc, bp)
        L_lo_p = L_lo if L_lo > 0.0 else _min_positive(pc, bp)
        if L_lo_p >= L_hi_p:
            return np.asarray([L_hi_p], dtype=float)
        grid = np.logspace(np.log10(L_lo_p), np.log10(L_hi_p), n_points)[::-1]
        grid[0] = L_hi_p
        grid[-1] = L_lo_p
    elif scale == "linear":
        grid = np.linspace(L_hi, L_lo, n_points)
        grid[0] = L_hi
        grid[-1] = L_lo
    else:
        raise ValueError(f"unknown loss scale {scale!r}")
    _, idx = np.unique(np.round(grid, decimals=15), return_index=True)
    return np.asarray(grid[np.sort(idx)], dtype=float)


def _first_crossing_times(losses, loss_grid):
    """First ``t`` with ``L(t) <= L*`` for each target on ``loss_grid``.

    If a target is never reached, the last finite step is used and a
    warning is printed.
    """
    losses = np.asarray(losses, dtype=float).reshape(-1)
    finite = np.flatnonzero(np.isfinite(losses))
    last = int(finite[-1]) if finite.size else max(0, len(losses) - 1)
    times = []
    for L_star in np.asarray(loss_grid, dtype=float).reshape(-1):
        hit = np.flatnonzero(np.isfinite(losses) & (losses <= L_star))
        if hit.size:
            times.append(int(hit[0]))
        else:
            print(
                f"Warning: loss never crossed L*={L_star:.4e}; "
                f"using last step t={last}."
            )
            times.append(last)
    return times


def _ensure_kernels_at_times(kernels_by_t, h_traj, times, phi_fn):
    """Compute missing feature kernels in ``kernels_by_t`` for ``times``."""
    for t in times:
        t = int(t)
        if t not in kernels_by_t:
            kernels_by_t[t] = _feature_kernels_from_h(h_traj[:, t], phi_fn)
    return kernels_by_t


def _collect_alignment_records(
    pc_kernels_by_t,
    bp_kernels_by_t,
    pairs,
    *,
    t0,
    C_y,
    C_x,
    Y_target,
    n_hidden,
    x_col="t",
):
    """Metrics along ``pairs`` of ``(x, t_pc, t_bp)``.

    Displacement is always from the shared init kernels at ``t0``.
    Change-alignment rows omit the init pair (``t_pc == t0 == t_bp``).
    """
    displacement_records = []
    alignment_records = []
    change_alignment_records = []
    target_alignment_records = []
    rank_records = []
    change_target_alignment_records = []
    input_alignment_records = []
    evec_overlap_records = []
    subspace_overlap_records = []
    pc_C0_layers = pc_kernels_by_t[t0]
    bp_C0_layers = bp_kernels_by_t[t0]
    max_k = max(1, int(C_y.shape[0]) - 1)
    for x, t_pc, t_bp in pairs:
        t_pc = int(t_pc)
        t_bp = int(t_bp)
        is_init = t_pc == t0 and t_bp == t0
        for l in range(n_hidden):
            pc_C0 = pc_C0_layers[l]
            bp_C0 = bp_C0_layers[l]
            pc_Ct = pc_kernels_by_t[t_pc][l]
            bp_Ct = bp_kernels_by_t[t_bp][l]
            displacement_records.append({
                x_col: x,
                "layer": l,
                "method": "pc",
                "displacement": float(
                    cosine_similarity(pc_C0, pc_Ct, eps=1e-30)
                ),
                "rel_displacement": float(
                    relative_frobenius_displacement(
                        pc_C0, pc_Ct, eps=1e-30
                    )
                ),
            })
            displacement_records.append({
                x_col: x,
                "layer": l,
                "method": "bp",
                "displacement": float(
                    cosine_similarity(bp_C0, bp_Ct, eps=1e-30)
                ),
                "rel_displacement": float(
                    relative_frobenius_displacement(
                        bp_C0, bp_Ct, eps=1e-30
                    )
                ),
            })
            alignment_records.append({
                x_col: x,
                "layer": l,
                "alignment": float(
                    centered_kernel_alignment(pc_Ct, bp_Ct, eps=1e-30)
                ),
            })
            if not is_init:
                change_alignment_records.append({
                    x_col: x,
                    "layer": l,
                    "alignment": float(
                        centered_kernel_alignment(
                            np.asarray(pc_Ct) - np.asarray(pc_C0),
                            np.asarray(bp_Ct) - np.asarray(bp_C0),
                            eps=1e-30,
                        )
                    ),
                })
            for method, Ct in (("pc", pc_Ct), ("bp", bp_Ct)):
                target_alignment_records.append({
                    x_col: x,
                    "layer": l,
                    "method": method,
                    "alignment": float(
                        centered_kernel_alignment(Ct, C_y, eps=1e-30)
                    ),
                })
                rank_records.append({
                    x_col: x,
                    "layer": l,
                    "method": method,
                    "effective_rank": float(participation_ratio(Ct)),
                })
                input_alignment_records.append({
                    x_col: x,
                    "layer": l,
                    "method": method,
                    "alignment": float(
                        centered_kernel_alignment(Ct, C_x, eps=1e-30)
                    ),
                })
                evec_overlap_records.append({
                    x_col: x,
                    "layer": l,
                    "method": method,
                    "overlap": float(
                        leading_evec_label_overlap(Ct, Y_target)
                    ),
                })
            if not is_init:
                for method, Ct, C0 in (
                    ("pc", pc_Ct, pc_C0),
                    ("bp", bp_Ct, bp_C0),
                ):
                    change_target_alignment_records.append({
                        x_col: x,
                        "layer": l,
                        "method": method,
                        "alignment": float(
                            centered_kernel_alignment(
                                np.asarray(Ct) - np.asarray(C0),
                                C_y,
                                eps=1e-30,
                            )
                        ),
                    })
            for k in _SUBSPACE_KS:
                if k > max_k:
                    continue
                subspace_overlap_records.append({
                    x_col: x,
                    "layer": l,
                    "k": k,
                    "overlap": float(
                        subspace_overlap(pc_Ct, bp_Ct, k, centered=True)
                    ),
                })
    return {
        "displacement": pd.DataFrame(displacement_records),
        "alignment": pd.DataFrame(alignment_records),
        "change_alignment": pd.DataFrame(change_alignment_records),
        "target_alignment": pd.DataFrame(target_alignment_records),
        "change_target_alignment": pd.DataFrame(
            change_target_alignment_records
        ),
        "input_alignment": pd.DataFrame(input_alignment_records),
        "evec_overlap": pd.DataFrame(evec_overlap_records),
        "rank": pd.DataFrame(rank_records),
        "subspace_overlap": pd.DataFrame(subspace_overlap_records),
    }


def _plot_alignment_suite(
    records,
    plot_kw,
    feat_tex,
    evec_ylabel,
    *,
    vs,
    x_col,
    xlabel,
    xscale="linear",
    invert_x=False,
):
    """Line plots for displacement, alignment, rank, and PC–BP pairing."""
    suffix = "time" if vs == "time" else "loss"
    t_sub = "t" if vs == "time" else "L"
    arg = "t" if vs == "time" else "L"
    axis = dict(x_col=x_col, xlabel=xlabel, xscale=xscale, invert_x=invert_x)
    plot_kernel_displacement_per_timepoint(
        records["displacement"],
        filename=f"kernel_displacement_vs_{suffix}.png",
        t_sub=t_sub,
        **axis,
        **plot_kw,
    )
    plot_kernel_displacement_per_timepoint(
        records["displacement"],
        metric="rel_frob",
        filename=f"kernel_displacement_vs_{suffix}.png",
        t_sub=t_sub,
        **axis,
        **plot_kw,
    )
    plot_pc_bp_alignment_vs_time(
        records["alignment"],
        filename=f"pc_bp_kernel_alignment_vs_{suffix}.png",
        **axis,
        **plot_kw,
    )
    plot_pc_bp_alignment_vs_time(
        records["change_alignment"],
        ylabel=(
            rf"$\mathrm{{CKA}}(\Delta C^{{{feat_tex},\ell}}"
            rf"_{{\mathrm{{PC}}}}({arg}), "
            rf"\Delta C^{{{feat_tex},\ell}}_{{\mathrm{{BP}}}}({arg}))$"
        ),
        filename=f"pc_bp_kernel_change_alignment_vs_{suffix}.png",
        title=(
            "PC vs backprop feature-kernel change alignment over training"
        ),
        **axis,
        **plot_kw,
    )
    plot_kernel_target_alignment_vs_time(
        records["target_alignment"],
        filename=f"kernel_target_alignment_vs_{suffix}.png",
        **axis,
        **plot_kw,
    )
    plot_kernel_target_change_alignment_vs_time(
        records["change_target_alignment"],
        filename=f"kernel_target_change_alignment_vs_{suffix}.png",
        **axis,
        **plot_kw,
    )
    plot_kernel_input_alignment_vs_time(
        records["input_alignment"],
        filename=f"kernel_input_alignment_vs_{suffix}.png",
        **axis,
        **plot_kw,
    )
    plot_leading_evec_label_overlap_vs_time(
        records["evec_overlap"],
        ylabel=evec_ylabel,
        filename=f"evec_leading_overlap_label_vs_{suffix}.png",
        **axis,
        **plot_kw,
    )
    plot_kernel_effective_rank_vs_time(
        records["rank"],
        filename=f"kernel_effective_rank_vs_{suffix}.png",
        **axis,
        **plot_kw,
    )
    plot_pc_bp_subspace_overlap_vs_time(
        records["subspace_overlap"],
        filename=f"pc_bp_subspace_overlap_vs_{suffix}.png",
        **axis,
        **plot_kw,
    )


def _plot_temporal_kernel_figures(
    pc_h_traj,
    bp_h_traj,
    phi_fn,
    plot_kw,
    feat_tex,
    *,
    xlabel,
    ylabel,
    title_note="",
):
    """Sample-traced temporal kernels, spectrum, rank, and PC–BP CKA."""
    pc_temporal_kernels = _sample_traced_feature_kernels_from_h_traj(
        pc_h_traj, phi_fn
    )
    bp_temporal_kernels = _sample_traced_feature_kernels_from_h_traj(
        bp_h_traj, phi_fn
    )
    title = (
        rf"Sample-traced $C^{{{feat_tex}}}$ feature kernels "
        rf"(correlation)"
    )
    if title_note:
        title += title_note
    plot_temporal_kernel_grid(
        [
            ("PC", kernels_to_correlations(pc_temporal_kernels)),
            ("Backprop", kernels_to_correlations(bp_temporal_kernels)),
        ],
        filename="temporal_kernels_grid.png",
        vmin=-1.0,
        vmax=1.0,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        **{k: plot_kw[k] for k in (
            "plots_dir", "gamma_0", "n_hidden", "activity_lr",
            "n_infer_iters", "width", "dir_name",
        ) if k in plot_kw},
    )
    temporal_spectrum_df = pd.DataFrame(
        _spectrum_records(pc_temporal_kernels, "pc")
        + _spectrum_records(bp_temporal_kernels, "bp")
    )
    spec_title = rf"Sample-traced $C^{{{feat_tex}}}$ feature-kernel spectrum"
    if title_note:
        spec_title += title_note
    plot_kernel_spectrum(
        temporal_spectrum_df,
        ylabel=rf"$\lambda_i(C^{{{feat_tex},\ell}}_{{\mathrm{{temp}}}})$",
        title=spec_title,
        filename="temporal_kernel_spectrum.png",
        annotate_rank=True,
        **plot_kw,
    )
    temporal_rank_records = (
        temporal_spectrum_df.groupby(["layer", "method"], as_index=False)
        .agg(effective_rank=("effective_rank", "first"))
    )
    plot_temporal_kernel_effective_rank_vs_layer(
        temporal_rank_records,
        **plot_kw,
    )
    temporal_cka_records = []
    for l, (C_pc, C_bp) in enumerate(
        zip(pc_temporal_kernels, bp_temporal_kernels)
    ):
        cka = float(centered_kernel_alignment(C_pc, C_bp, eps=1e-30))
        temporal_cka_records.append({
            "layer": l,
            "alignment": cka,
        })
    plot_temporal_pc_bp_alignment_vs_layer(
        pd.DataFrame(temporal_cka_records),
        **plot_kw,
    )


def _collect_train_test_alignment_records(
    pc_train_kernels,
    bp_train_kernels,
    pc_test_kernels,
    bp_test_kernels,
    C_y_train,
    C_y_test,
    n_hidden,
):
    """PC–BP CKA and kernel–target CKA on train vs test snapshots."""
    pc_bp_records = []
    target_records = []
    for split, pc_ks, bp_ks, C_y in (
        ("train", pc_train_kernels, bp_train_kernels, C_y_train),
        ("test", pc_test_kernels, bp_test_kernels, C_y_test),
    ):
        for l in range(n_hidden):
            pc_bp_records.append({
                "layer": l,
                "split": split,
                "alignment": float(
                    centered_kernel_alignment(
                        pc_ks[l], bp_ks[l], eps=1e-30
                    )
                ),
            })
            for method, Ct in (("pc", pc_ks[l]), ("bp", bp_ks[l])):
                target_records.append({
                    "layer": l,
                    "method": method,
                    "split": split,
                    "alignment": float(
                        centered_kernel_alignment(Ct, C_y, eps=1e-30)
                    ),
                })
    return pd.DataFrame(pc_bp_records), pd.DataFrame(target_records)


def _plot_train_test_kernel_suite(
    pc_train_kernels,
    bp_train_kernels,
    pc_test_kernels,
    bp_test_kernels,
    C_y_train,
    C_y_test,
    plot_kw,
    feat_tex,
    n_hidden,
    *,
    title_note="",
):
    """Train vs test PC–BP / target CKA vs layer, plus test kernel grid."""
    pc_bp_df, target_df = _collect_train_test_alignment_records(
        pc_train_kernels,
        bp_train_kernels,
        pc_test_kernels,
        bp_test_kernels,
        C_y_train,
        C_y_test,
        n_hidden,
    )
    pc_bp_title = "PC vs backprop feature-kernel alignment (train vs test)"
    target_title = "Feature-kernel alignment with the target (train vs test)"
    if title_note:
        pc_bp_title += title_note
        target_title += title_note
    plot_pc_bp_kernel_alignment_test_vs_layer(
        pc_bp_df, title=pc_bp_title, **plot_kw
    )
    plot_kernel_target_alignment_test_vs_layer(
        target_df, title=target_title, **plot_kw
    )
    grid_title = (
        rf"$C^{{{feat_tex}}}$ feature kernels (test, correlation)"
    )
    if title_note:
        grid_title = (
            rf"$C^{{{feat_tex}}}$ feature kernels "
            rf"(test{title_note}, correlation)"
        )
    plot_final_kernel_grid(
        [
            ("PC", kernels_to_correlations(pc_test_kernels)),
            ("Backprop", kernels_to_correlations(bp_test_kernels)),
        ],
        filename="feature_kernels_grid_test.png",
        share_clim=True,
        vmin=-1.0,
        vmax=1.0,
        title=grid_title,
        **{k: plot_kw[k] for k in (
            "plots_dir", "gamma_0", "n_hidden", "activity_lr",
            "n_infer_iters", "width", "dir_name",
        ) if k in plot_kw},
    )


_INIT_CKA_ATOL = 1e-3
_SUBSPACE_KS = (1, 3, 5)


def _sample_gram(M):
    """Gram over samples: rows of ``M`` are samples, ``C = M M^T``."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        M = M[:, None]
    return M @ M.T


def _target_kernel(Y_target):
    """Label Gram ``C^y = Y Y^T`` with sample axis first."""
    return _sample_gram(Y_target)


def _input_kernel(X_input):
    """Input Gram ``C^x = X X^T`` with sample axis first."""
    return _sample_gram(X_input)


def _spectrum_records(kernels, method):
    """One row per descending eigenvalue of each layer's kernel."""
    records = []
    for l, C in enumerate(kernels):
        eigs = kernel_eigs(C)
        r_eff = participation_ratio_from_eigs(eigs)
        for i, lam in enumerate(eigs):
            records.append({
                "layer": l,
                "method": method,
                "index": i + 1,
                "eigenvalue": float(lam),
                "effective_rank": r_eff,
            })
    return records


def _assert_init_pc_bp_cka(pc_kernels, bp_kernels, *, seed, atol=_INIT_CKA_ATOL):
    """Assert that PC and BP feature kernels match at initialisation.

    After copying PC weights onto BP, ``1 - CKA`` should be small at
    ``t=0`` for every hidden layer.
    """
    if len(pc_kernels) != len(bp_kernels):
        raise AssertionError(
            "PC/BP init kernel counts differ: "
            f"{len(pc_kernels)} vs {len(bp_kernels)}"
        )
    for l, (C_pc, C_bp) in enumerate(zip(pc_kernels, bp_kernels)):
        cka = float(centered_kernel_alignment(C_pc, C_bp, eps=1e-30))
        gap = abs(1.0 - cka)
        print(
            f"  init CKA (seed={seed}, layer={l + 1}): {cka:.6f} "
            f"(1-CKA={gap:.2e})"
        )
        if gap > atol:
            raise AssertionError(
                f"PC vs BP init kernels differ at layer {l + 1} "
                f"(seed={seed}): CKA={cka:.6f}, expected "
                f"1-CKA <= {atol}"
            )


def _mean_pairwise_cka(kernels):
    """Mean and std of pairwise CKA over a list of kernels (one per seed)."""
    vals = []
    for i in range(len(kernels)):
        for j in range(i + 1, len(kernels)):
            vals.append(
                centered_kernel_alignment(kernels[i], kernels[j], eps=1e-30)
            )
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return float(vals.mean()), std


def _train_finite_bp(
    key,
    *,
    results_dir,
    input_dim,
    output_dim,
    n_samples,
    n_hidden,
    use_skips,
    act_fn,
    param_type,
    param_lr,
    gamma_0,
    param_optim_id,
    n_train_iters,
    width,
    loss_id,
    seed,
    X_input,
    Y_target,
    collect_h_k0=True,
    init_from=None,
    X_eval=None,
    phi_fn=None,
    collect_eval_kernels=False,
    momentum=0.9,
):
    """Run one finite-width BP training job.

    Returns ``(losses, h_k0_traj, eval_kernels)``. ``h_k0_traj`` has
    shape ``(n_hidden, T, P, N)`` — the hidden pre-activations ``h^l``
    (see ``bp_hidden_preactivations`` in ``experiments.dmft.utils``) at
    every training step, before that step's parameter update — or
    ``None`` if ``collect_h_k0`` is False. ``eval_kernels`` is a list of
    per-layer test-set feature kernels, one list per training step, or
    ``None`` if ``collect_eval_kernels`` is False.

    If ``init_from`` is given (a ``jpc.make_mlp`` layer list or BP
    ``MLP``), Linear weights are copied onto the BP network after
    construction so PC and BP share the same initial parameters.
    """
    save_dir = setup_bp_experiment(
        results_dir=results_dir,
        input_dim=input_dim,
        n_samples=n_samples,
        n_hidden=n_hidden,
        use_skips=use_skips,
        act_fn=act_fn,
        param_type=param_type,
        optim_id=param_optim_id,
        param_lr=param_lr,
        gamma_0=gamma_0,
        n_train_iters=n_train_iters,
        width=width,
        loss_id=loss_id,
        seed=seed,
    )
    model = MLP(
        key=key,
        d_in=input_dim,
        N=width,
        L=n_hidden + 1,
        d_out=output_dim,
        act_fn=act_fn,
        param_type=param_type,
        gamma=gamma_0,
        use_bias=False,
        use_skips=use_skips,
    )
    if init_from is not None:
        model = copy_mlp_linear_params(init_from, model)
        _term("Copied PC initial Linear weights onto the BP network.")
    h_k0_steps = [] if collect_h_k0 else None
    eval_kernels = [] if collect_eval_kernels else None
    if collect_eval_kernels:
        if X_eval is None or phi_fn is None:
            raise ValueError(
                "collect_eval_kernels requires X_eval and phi_fn"
            )

        def _record_eval_kernels(hs):
            h = np.stack(
                [np.asarray(x, dtype=np.float32) for x in hs], axis=0
            )
            eval_kernels.append(_feature_kernels_from_h(h, phi_fn))

        h_k0_eval_callback = _record_eval_kernels
    else:
        h_k0_eval_callback = None
    train_bpn(
        model=model,
        use_skips=use_skips,
        X_input=X_input,
        Y_target=Y_target,
        width=width,
        gamma_0=gamma_0,
        param_type=param_type,
        optim_id=param_optim_id,
        param_lr=param_lr,
        n_train_iters=n_train_iters,
        save_dir=save_dir,
        store_grads=False,
        loss_id=loss_id,
        h_k0_steps=h_k0_steps,
        X_eval=X_eval,
        h_k0_eval_callback=h_k0_eval_callback,
        momentum=momentum,
    )
    losses = np.load(f"{save_dir}/losses.npy")
    h_k0_traj = np.stack(h_k0_steps, axis=1) if collect_h_k0 else None
    return losses, h_k0_traj, eval_kernels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "tiny-CIFAR10", "Fashion-MNIST", "CIFAR10"])
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=20)

    # Model parameters
    parser.add_argument("--act_fn", type=str, default="linear", choices=["linear", "tanh", "relu"])
    parser.add_argument("--param_type", type=str, default="mupc", choices=["mupc", "sp", "my-mup"])
    parser.add_argument("--use_skips", action="store_true", default=False)
    parser.add_argument("--n_hidden", type=int, default=5)
    parser.add_argument("--width", type=int, default=2048)

    # Training parameters (shared)
    parser.add_argument("--gamma_0", type=float, default=1.0)
    parser.add_argument("--n_train_iters", type=int, default=20)
    parser.add_argument("--loss_id", type=str, default="mse", choices=["mse", "ce"])

    # BP training parameters
    parser.add_argument(
        "--param_lr",
        type=float,
        default=0.05,
        help=(
            "Backprop parameter learning rate "
            "(Adam: divided by √N, or √(N L) with skips; "
            "GD / SGD+momentum: µP bakes γ² N into the optimiser)."
        ),
    )
    parser.add_argument(
        "--param_optim",
        type=str,
        default="gd",
        choices=["gd", "adam", "sgd_momentum"],
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for --param_optim sgd_momentum (ignored otherwise).",
    )

    # PC training / inference parameters
    parser.add_argument(
        "--param_lr_pc",
        type=float,
        default=0.5,
        help=(
            "PC parameter learning rate (Adam: divided like BP; "
            "GD / SGD+momentum: used as-is)."
        ),
    )
    parser.add_argument(
        "--pc_infer_mode",
        type=str,
        default="infer",
        choices=["infer", "closed_form"],
        help=(
            "PC inference mode. 'infer' runs --n_infer_iters steps of "
            "gradient descent on the activities each training step. "
            "'closed_form' solves the linear equilibrium activities "
            "directly (requires --act_fn linear)."
        ),
    )
    parser.add_argument("--n_infer_iters", type=int, default=5)
    parser.add_argument("--activity_lr", type=float, default=0.05)
    parser.add_argument(
        "--nonlin_beta",
        type=float,
        default=1.0,
        help="Steepness for the tanh feature nonlinearity phi.",
    )

    # Timepoints
    parser.add_argument(
        "--n_timepoints",
        type=int,
        default=6,
        help=(
            "Number of equally spaced training-step timepoints "
            "(including t=0 and t=T-1) used for the feature-kernel "
            "heatmap grids. Displacement and alignment line plots use a "
            "denser grid (every step if T<=31, else every 2 steps). "
            "The same count is used as a subset of the L* grid for "
            "loss-matched heatmaps."
        ),
    )
    parser.add_argument(
        "--skip_loss_matched",
        action="store_true",
        default=False,
        help=(
            "Skip loss-matched figures under alignment/by_loss/. "
            "Time-indexed figures under alignment/by_time/ are always "
            "written."
        ),
    )
    parser.add_argument(
        "--loss_scale",
        type=str,
        default="log",
        choices=["log", "linear"],
        help=(
            "Spacing of the target-loss grid L* and x-axis scale of "
            "vs-L plots. Default log (loss typically spans decades)."
        ),
    )
    parser.add_argument(
        "--n_loss_divisor",
        type=int,
        default=_DEFAULT_N_LOSS_DIVISOR,
        help=(
            "L* grid size is max(11, round(T / n_loss_divisor)), "
            "including the start and end, capped at T. Default "
            f"{_DEFAULT_N_LOSS_DIVISOR}."
        ),
    )

    # Loop parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help=(
            "Number of independent weight-init seeds (dataset is shared). "
            "If greater than 1, plot kernel concentration across seeds "
            "for both PC and BP. Default 1 skips that analysis."
        ),
    )

    parser.add_argument(
        "--cleanup_npy",
        action="store_true",
        default=True,
        help=(
            "After the run, delete finite-sim result directories "
            "(*_input_dim under results_dir), keeping plot pngs."
        ),
    )
    args = parser.parse_args()

    if args.pc_infer_mode == "closed_form" and args.act_fn != "linear":
        parser.error(
            "--pc_infer_mode closed_form requires --act_fn linear."
        )
    if args.n_seeds < 1:
        parser.error("--n_seeds must be >= 1.")
    if args.n_loss_divisor < 1:
        parser.error("--n_loss_divisor must be >= 1.")

    os.makedirs(args.results_dir, exist_ok=True)
    plot_loss_matched = not args.skip_loss_matched

    plots_dir = os.path.join(
        args.results_dir,
        "plots",
        f"{args.width}_width",
        f"{args.pc_infer_mode}_pc_infer_mode",
    )
    align_root = _alignment_plots_dir(
        plots_dir,
        n_hidden=args.n_hidden,
        gamma_0=args.gamma_0,
        activity_lr=args.activity_lr,
        n_infer_iters=args.n_infer_iters,
        dir_name="alignment",
    )
    log_path = os.path.join(align_root, "analyse_alignment.log")
    _start_log(log_path)

    # Two independent children of --seed: dataset and weight init. BP
    # copies the PC Linear weights after construction, so both networks
    # start from the same parameters (the BP constructor key is unused
    # for the copied weights). A further split of the dataset stream
    # draws an independent test set of the same size.
    data_key, model_parent = jax.random.split(jax.random.PRNGKey(args.seed))
    train_key, test_key = jax.random.split(data_key)

    if args.dataset == "toy":
        X, y = create_toy_dataset(
            key=train_key, D=args.input_dim, P=args.n_samples
        )
        X_test, y_test = create_toy_dataset(
            key=test_key, D=args.input_dim, P=args.n_samples
        )
        input_dim = args.input_dim
        output_dim = 1
    elif args.dataset == "tiny-CIFAR10":
        input_dim = CIFAR_GRAY_DIM
        X, y = create_tiny_cifar10_dataset(
            key=train_key, D=input_dim, P=args.n_samples, train=True
        )
        X_test, y_test = create_tiny_cifar10_dataset(
            key=test_key, D=input_dim, P=args.n_samples, train=False
        )
        output_dim = 1
        print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    else:
        from torch import Generator as TorchGenerator

        loader_gen = TorchGenerator()
        loader_gen.manual_seed(int(np.asarray(train_key)[0]) & 0x7FFFFFFF)
        train_loader, test_loader = get_dataloaders(
            args.dataset, args.n_samples, generator=loader_gen
        )
        img_batch, label_batch = next(iter(train_loader))
        img_batch_test, label_batch_test = next(iter(test_loader))

        input_dim = img_batch.shape[1]
        output_dim = label_batch.shape[1]
        print(f"Input dim: {input_dim}, Output dim: {output_dim}")

        X = img_batch.numpy().T
        y = label_batch.numpy()
        X_test = img_batch_test.numpy().T
        y_test = label_batch_test.numpy()

    X_input = jnp.asarray(X.T, dtype=jnp.float32)
    Y_target = y[:, None] if y.ndim == 1 else y
    Y_target = jnp.asarray(Y_target, dtype=jnp.float32)
    X_test_input = jnp.asarray(X_test.T, dtype=jnp.float32)
    Y_test = y_test[:, None] if y_test.ndim == 1 else y_test
    Y_test = jnp.asarray(Y_test, dtype=jnp.float32)
    C_y = _target_kernel(Y_target)
    C_y_test = _target_kernel(Y_test)
    C_x = _input_kernel(X_input)
    n_label_cols = 1 if np.asarray(Y_target).ndim == 1 else int(
        np.asarray(Y_target).shape[1]
    )
    print(
        f"Target kernel C^y shape: {C_y.shape}, "
        f"input kernel C^x shape: {C_x.shape}"
    )
    print(f"Test target kernel C^y_test shape: {C_y_test.shape}")
    if int(X_test_input.shape[0]) != int(X_input.shape[0]):
        raise ValueError(
            "test set size must match the training set: "
            f"got P_train={int(X_input.shape[0])}, "
            f"P_test={int(X_test_input.shape[0])}"
        )
    loss_id = (
        "mse" if args.dataset in ("toy", "tiny-CIFAR10") else args.loss_id
    )

    n_hidden = args.n_hidden
    T_train = args.n_train_iters
    width = args.width
    infer_mode = "optim" if args.pc_infer_mode == "infer" else "closed_form"
    n_seeds = args.n_seeds
    phi_fn, _ = get_nonlinearity(args.act_fn, beta=args.nonlin_beta)
    feat_sym = feature_kernel_symbol(args.act_fn)
    feat_tex = r"\phi" if feat_sym == "phi" else "h"
    heatmap_timepoints = _select_timepoints(T_train, args.n_timepoints)
    curve_timepoints = _select_curve_timepoints(T_train)
    kernel_times = sorted(set(heatmap_timepoints) | set(curve_timepoints))
    print(
        f"\nHeatmap timepoints t = {heatmap_timepoints} (of T = {T_train})"
    )
    print(f"Curve timepoints t = {curve_timepoints}")
    print(f"Feature kernel: C^{feat_sym}")
    print(f"n_seeds = {n_seeds}")
    if plot_loss_matched:
        n_L_preview = _n_loss_grid_points(T_train, args.n_loss_divisor)
        print(
            f"Loss-matched plots enabled "
            f"(scale={args.loss_scale}, n_loss_divisor={args.n_loss_divisor}, "
            f"n_L*={n_L_preview})"
        )
    else:
        print(
            "Loss-matched plots skipped "
            "(omit --skip_loss_matched to enable)."
        )
    print()

    plot_kw = dict(
        plots_dir=plots_dir,
        gamma_0=args.gamma_0,
        n_hidden=n_hidden,
        activity_lr=args.activity_lr,
        n_infer_iters=args.n_infer_iters,
        width=width,
        feature_symbol=feat_sym,
    )
    plot_kw_time = dict(plot_kw, dir_name=_DIR_BY_TIME)
    plot_kw_loss = dict(plot_kw, dir_name=_DIR_BY_LOSS)

    pc_kernels_by_seed = []
    bp_kernels_by_seed = []
    pc_kernels_loss_by_seed = []
    bp_kernels_loss_by_seed = []
    loss_grid = None
    heatmap_loss_idx = None
    plot_seed = args.seed

    for seed in range(args.seed, args.seed + n_seeds):
        _term(f"\n=== seed {seed} ===")
        set_seed(seed)
        model_key = jax.random.fold_in(model_parent, int(seed))

        _term(
            f"\nRunning finite-size PC ({args.pc_infer_mode}) simulation "
            f"(N={width}, H={n_hidden})...\n"
        )
        pc_losses, pc_fields = _train_finite_pc(
            model_key,
            results_dir=args.results_dir,
            input_dim=input_dim,
            output_dim=output_dim,
            n_samples=args.n_samples,
            n_hidden=n_hidden,
            use_skips=args.use_skips,
            act_fn=args.act_fn,
            param_type=args.param_type,
            param_lr=args.param_lr_pc,
            gamma_0=args.gamma_0,
            param_optim_id=args.param_optim,
            momentum=args.momentum,
            n_train_iters=T_train,
            infer_mode=infer_mode,
            n_infer_iters=args.n_infer_iters,
            activity_lr=args.activity_lr,
            width=width,
            loss_id=loss_id,
            seed=seed,
            X_input=X_input,
            Y_target=Y_target,
            collect_fields=False,
            collect_h_k0=True,
            collect_init_model=True,
            X_eval=X_test_input if seed == plot_seed else None,
            phi_fn=phi_fn if seed == plot_seed else None,
            collect_eval_kernels=(seed == plot_seed),
        )
        pc_h_traj = pc_fields["h_k0_traj"]  # (n_hidden, T, P, N)
        pc_eval_kernels = pc_fields.get("eval_kernels")
        _term(
            f"PC final training loss: "
            f"{float(np.asarray(pc_losses).flatten()[-1]):.4e}"
        )

        _term(
            f"\nRunning finite-size BP simulation "
            f"(N={width}, H={n_hidden})...\n"
        )
        bp_losses, bp_h_traj, bp_eval_kernels = _train_finite_bp(
            model_key,
            results_dir=args.results_dir,
            input_dim=input_dim,
            output_dim=output_dim,
            n_samples=args.n_samples,
            n_hidden=n_hidden,
            use_skips=args.use_skips,
            act_fn=args.act_fn,
            param_type=args.param_type,
            param_lr=args.param_lr,
            gamma_0=args.gamma_0,
            param_optim_id=args.param_optim,
            momentum=args.momentum,
            n_train_iters=T_train,
            width=width,
            loss_id=loss_id,
            seed=seed,
            X_input=X_input,
            Y_target=Y_target,
            collect_h_k0=True,
            init_from=pc_fields["init_model"],
            X_eval=X_test_input if seed == plot_seed else None,
            phi_fn=phi_fn if seed == plot_seed else None,
            collect_eval_kernels=(seed == plot_seed),
        )
        _term(
            f"BP final training loss: "
            f"{float(np.asarray(bp_losses).flatten()[-1]):.4e}"
        )
        pc_losses = np.asarray(pc_losses, dtype=float).flatten()
        bp_losses = np.asarray(bp_losses, dtype=float).flatten()
        if pc_eval_kernels is not None and bp_eval_kernels is not None:
            if (
                len(pc_eval_kernels) != T_train
                or len(bp_eval_kernels) != T_train
            ):
                raise RuntimeError(
                    "eval kernel length mismatch: "
                    f"PC={len(pc_eval_kernels)}, BP={len(bp_eval_kernels)}, "
                    f"T={T_train}"
                )

        pc_kernels_by_t = {
            t: _feature_kernels_from_h(pc_h_traj[:, t], phi_fn)
            for t in kernel_times
        }
        bp_kernels_by_t = {
            t: _feature_kernels_from_h(bp_h_traj[:, t], phi_fn)
            for t in kernel_times
        }
        t0 = 0 if 0 in pc_kernels_by_t else kernel_times[0]
        print("Sanity-checking PC vs BP feature kernels at t=0...")
        _assert_init_pc_bp_cka(
            pc_kernels_by_t[t0], bp_kernels_by_t[t0], seed=seed
        )
        pc_kernels_by_seed.append(pc_kernels_by_t)
        bp_kernels_by_seed.append(bp_kernels_by_t)

        t_pc_L = None
        t_bp_L = None
        if plot_loss_matched:
            if loss_grid is None:
                n_L = _n_loss_grid_points(T_train, args.n_loss_divisor)
                loss_grid = _overlap_loss_grid(
                    pc_losses, bp_losses, n_L, args.loss_scale
                )
                heatmap_loss_idx = _select_timepoints(
                    len(loss_grid), args.n_timepoints
                )
                print(
                    f"\nLoss-matched L* grid ({args.loss_scale}, "
                    f"n={len(loss_grid)}): "
                    + ", ".join(f"{L:.3e}" for L in loss_grid)
                )
                print(
                    f"Loss-matched heatmap L* indices = {heatmap_loss_idx}"
                )
            t_pc_L = _first_crossing_times(pc_losses, loss_grid)
            t_bp_L = _first_crossing_times(bp_losses, loss_grid)
            _ensure_kernels_at_times(
                pc_kernels_by_t, pc_h_traj, t_pc_L, phi_fn
            )
            _ensure_kernels_at_times(
                bp_kernels_by_t, bp_h_traj, t_bp_L, phi_fn
            )
            pc_kernels_loss_by_seed.append(
                [pc_kernels_by_t[int(t)] for t in t_pc_L]
            )
            bp_kernels_loss_by_seed.append(
                [bp_kernels_by_t[int(t)] for t in t_bp_L]
            )
            if seed == plot_seed:
                print("Loss-matched first-crossing times (plot seed):")
                for i, (L_star, tpc, tbp) in enumerate(
                    zip(loss_grid, t_pc_L, t_bp_L)
                ):
                    mark = " [heatmap]" if i in heatmap_loss_idx else ""
                    print(
                        f"  L*={L_star:.4e}  t_PC={tpc:4d}  "
                        f"t_BP={tbp:4d}{mark}"
                    )

        if seed != plot_seed:
            continue

        plot_pc_bp_loss(
            pc_losses,
            bp_losses,
            plots_dir=plots_dir,
            n_hidden=n_hidden,
            gamma_0=args.gamma_0,
            activity_lr=args.activity_lr,
            n_infer_iters=args.n_infer_iters,
            width=width,
            dir_name=_DIR_BY_TIME,
        )

        for t in heatmap_timepoints:
            plot_final_kernel_grid(
                [
                    ("PC", kernels_to_correlations(pc_kernels_by_t[t])),
                    ("Backprop", kernels_to_correlations(bp_kernels_by_t[t])),
                ],
                plots_dir=plots_dir,
                gamma_0=args.gamma_0,
                n_hidden=n_hidden,
                activity_lr=args.activity_lr,
                n_infer_iters=args.n_infer_iters,
                width=width,
                filename=f"feature_kernels_grid_t{t}.png",
                share_clim=True,
                vmin=-1.0,
                vmax=1.0,
                title=(
                    rf"$C^{{{feat_tex}}}$ feature kernels "
                    rf"($t={t}$, correlation)"
                ),
                dir_name=_DIR_BY_TIME,
            )

        time_pairs = [(t, t, t) for t in curve_timepoints]
        time_records = _collect_alignment_records(
            pc_kernels_by_t,
            bp_kernels_by_t,
            time_pairs,
            t0=t0,
            C_y=C_y,
            C_x=C_x,
            Y_target=Y_target,
            n_hidden=n_hidden,
            x_col="t",
        )
        evec_ylabel_time = (
            r"$\left|\cos(v_1^{\ell}(t), y)\right|$"
            if n_label_cols == 1
            else r"$\|U_y^{\top} v_1^{\ell}(t)\|$"
        )
        _plot_alignment_suite(
            time_records,
            plot_kw_time,
            feat_tex,
            evec_ylabel_time,
            vs="time",
            x_col="t",
            xlabel="$t$",
        )

        t_final = kernel_times[-1]
        final_spectrum_df = pd.DataFrame(
            _spectrum_records(pc_kernels_by_t[t_final], "pc")
            + _spectrum_records(bp_kernels_by_t[t_final], "bp")
        )
        plot_kernel_spectrum(
            final_spectrum_df,
            ylabel=rf"$\lambda_i(C^{{{feat_tex},\ell}})$",
            title=(
                rf"Final $C^{{{feat_tex}}}$ feature-kernel spectrum "
                rf"($t={t_final}$)"
            ),
            filename="kernel_spectrum_final.png",
            annotate_rank=True,
            **plot_kw_time,
        )

        print("Temporal kernels (time-indexed)...")
        _plot_temporal_kernel_figures(
            pc_h_traj,
            bp_h_traj,
            phi_fn,
            plot_kw_time,
            feat_tex,
            xlabel=r"$t$",
            ylabel=r"$t'$",
        )

        print("Train vs test kernels (time-indexed)...")
        _plot_train_test_kernel_suite(
            pc_kernels_by_t[t_final],
            bp_kernels_by_t[t_final],
            pc_eval_kernels[t_final],
            bp_eval_kernels[t_final],
            C_y,
            C_y_test,
            plot_kw_time,
            feat_tex,
            n_hidden,
            title_note=rf" ($t={t_final}$)",
        )

        if plot_loss_matched:
            plot_pc_bp_loss_matched_times(
                pc_losses,
                bp_losses,
                t_pc_L,
                t_bp_L,
                loss_grid,
                plots_dir=plots_dir,
                n_hidden=n_hidden,
                gamma_0=args.gamma_0,
                activity_lr=args.activity_lr,
                n_infer_iters=args.n_infer_iters,
                width=width,
                dir_name=_DIR_BY_LOSS,
                heatmap_idx=heatmap_loss_idx,
                yscale=args.loss_scale,
            )
            for i in heatmap_loss_idx:
                L_star = float(loss_grid[i])
                t_pc_i = int(t_pc_L[i])
                t_bp_i = int(t_bp_L[i])
                plot_final_kernel_grid(
                    [
                        (
                            "PC",
                            kernels_to_correlations(
                                pc_kernels_by_t[t_pc_i]
                            ),
                        ),
                        (
                            "Backprop",
                            kernels_to_correlations(
                                bp_kernels_by_t[t_bp_i]
                            ),
                        ),
                    ],
                    plots_dir=plots_dir,
                    gamma_0=args.gamma_0,
                    n_hidden=n_hidden,
                    activity_lr=args.activity_lr,
                    n_infer_iters=args.n_infer_iters,
                    width=width,
                    filename=f"feature_kernels_grid_lstar{i}.png",
                    share_clim=True,
                    vmin=-1.0,
                    vmax=1.0,
                    title=(
                        rf"$C^{{{feat_tex}}}$ feature kernels "
                        rf"($L={L_star:.2e}$, $t_{{\mathrm{{PC}}}}={t_pc_i}$, "
                        rf"$t_{{\mathrm{{BP}}}}={t_bp_i}$, correlation)"
                    ),
                    dir_name=_DIR_BY_LOSS,
                )

            loss_pairs = list(zip(loss_grid, t_pc_L, t_bp_L))
            loss_records = _collect_alignment_records(
                pc_kernels_by_t,
                bp_kernels_by_t,
                loss_pairs,
                t0=t0,
                C_y=C_y,
                C_x=C_x,
                Y_target=Y_target,
                n_hidden=n_hidden,
                x_col="loss",
            )
            evec_ylabel_loss = (
                r"$\left|\cos(v_1^{\ell}(L), y)\right|$"
                if n_label_cols == 1
                else r"$\|U_y^{\top} v_1^{\ell}(L)\|$"
            )
            _plot_alignment_suite(
                loss_records,
                plot_kw_loss,
                feat_tex,
                evec_ylabel_loss,
                vs="loss",
                x_col="loss",
                xlabel="$L$",
                xscale=args.loss_scale,
                invert_x=True,
            )

            i_lo = len(loss_grid) - 1
            L_lo = float(loss_grid[i_lo])
            t_pc_lo = int(t_pc_L[i_lo])
            t_bp_lo = int(t_bp_L[i_lo])
            overlap_spectrum_df = pd.DataFrame(
                _spectrum_records(pc_kernels_by_t[t_pc_lo], "pc")
                + _spectrum_records(bp_kernels_by_t[t_bp_lo], "bp")
            )
            plot_kernel_spectrum(
                overlap_spectrum_df,
                ylabel=rf"$\lambda_i(C^{{{feat_tex},\ell}})$",
                title=(
                    rf"$C^{{{feat_tex}}}$ feature-kernel spectrum "
                    rf"at last overlap ($L={L_lo:.2e}$)"
                ),
                filename="kernel_spectrum_final.png",
                annotate_rank=True,
                **plot_kw_loss,
            )

            print("Temporal kernels (loss-matched)...")
            _plot_temporal_kernel_figures(
                pc_h_traj[:, np.asarray(t_pc_L, dtype=int)],
                bp_h_traj[:, np.asarray(t_bp_L, dtype=int)],
                phi_fn,
                plot_kw_loss,
                feat_tex,
                xlabel=r"$L$",
                ylabel=r"$L'$",
                title_note=" (loss-matched)",
            )

            print("Train vs test kernels (loss-matched)...")
            _plot_train_test_kernel_suite(
                pc_kernels_by_t[t_pc_lo],
                bp_kernels_by_t[t_bp_lo],
                pc_eval_kernels[t_pc_lo],
                bp_eval_kernels[t_bp_lo],
                C_y,
                C_y_test,
                plot_kw_loss,
                feat_tex,
                n_hidden,
                title_note=(
                    rf" ($L={L_lo:.2e}$, $t_{{\mathrm{{PC}}}}={t_pc_lo}$, "
                    rf"$t_{{\mathrm{{BP}}}}={t_bp_lo}$)"
                ),
            )

    if n_seeds > 1:
        _term(
            f"\nKernel concentration across {n_seeds} seeds "
            f"(mean pairwise CKA)..."
        )
        conc_records = []
        for t in curve_timepoints:
            for l in range(n_hidden):
                for method, by_seed in (
                    ("pc", pc_kernels_by_seed),
                    ("bp", bp_kernels_by_seed),
                ):
                    kernels = [seed_kernels[t][l] for seed_kernels in by_seed]
                    cka_mean, cka_std = _mean_pairwise_cka(kernels)
                    conc_records.append({
                        "t": t,
                        "layer": l,
                        "method": method,
                        "cka_mean": cka_mean,
                        "cka_std": cka_std,
                    })
        conc_df = pd.DataFrame(conc_records)
        plot_kernel_concentration_per_timepoint(
            conc_df, n_seeds=n_seeds, **plot_kw_time
        )
        plot_kernel_concentration_vs_time(
            conc_df, n_seeds=n_seeds, **plot_kw_time
        )
        if plot_loss_matched and loss_grid is not None:
            conc_loss_records = []
            for i, L_star in enumerate(loss_grid):
                for l in range(n_hidden):
                    for method, by_seed in (
                        ("pc", pc_kernels_loss_by_seed),
                        ("bp", bp_kernels_loss_by_seed),
                    ):
                        kernels = [
                            seed_kernels[i][l] for seed_kernels in by_seed
                        ]
                        cka_mean, cka_std = _mean_pairwise_cka(kernels)
                        conc_loss_records.append({
                            "loss": float(L_star),
                            "star_idx": i,
                            "layer": l,
                            "method": method,
                            "cka_mean": cka_mean,
                            "cka_std": cka_std,
                        })
            conc_loss_df = pd.DataFrame(conc_loss_records)
            conc_loss_grid_df = conc_loss_df[
                conc_loss_df["star_idx"].isin(heatmap_loss_idx)
            ]
            plot_kernel_concentration_per_timepoint(
                conc_loss_grid_df,
                n_seeds=n_seeds,
                x_col="loss",
                **plot_kw_loss,
            )
            plot_kernel_concentration_vs_time(
                conc_loss_df,
                n_seeds=n_seeds,
                x_col="loss",
                xlabel="$L$",
                xscale=args.loss_scale,
                invert_x=True,
                filename="kernel_concentration_vs_loss.png",
                **plot_kw_loss,
            )
    else:
        _term(
            "\nSkipping kernel-concentration analysis "
            "(pass --n_seeds > 1 to enable)."
        )

    if args.cleanup_npy:
        removed_dirs = cleanup_experiment_dirs(args.results_dir)
        if removed_dirs:
            _term(
                f"\nRemoved {len(removed_dirs)} experiment dir(s) "
                f"under {args.results_dir} (png plots kept):"
            )
            for d in removed_dirs:
                _term(f"  - {d}")
        else:
            _term(f"\nNo *_input_dim dirs to remove under {args.results_dir}.")


######### LINEAR ##########
###########################

# Infer mode (default)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 20 --n_hidden 5 --width 10000 --gamma_0 1.0 --param_lr 0.1 --param_lr_pc 0.2 --activity_lr 0.01 --pc_infer_mode infer --n_infer_iters 5 --n_train_iters 21

# Closed-form inference for PC
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 20 --n_hidden 5 --width 10000 --gamma_0 1.0 --param_lr 0.1 --param_lr_pc 0.2 --pc_infer_mode closed_form --n_train_iters 21

# Closed-form inference for PC (tiny-CIFAR10)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --pc_infer_mode closed_form --n_train_iters 501 --dataset tiny-CIFAR10


############ NONLINEAR ##################
#########################################

# Infer mode (default)
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.5 --param_lr_pc 1.0 --activity_lr 0.05 --pc_infer_mode infer --n_infer_iters 100 --n_train_iters 31 --act_fn tanh --dataset tiny-CIFAR10

# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.2 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 200 --n_train_iters 101 --act_fn tanh --dataset tiny-CIFAR10

# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 40 --n_hidden 3 --width 10000 --gamma_0 1.0 --param_lr 0.05 --param_lr_pc 0.5 --activity_lr 0.1 --pc_infer_mode infer --n_infer_iters 200 --n_train_iters 801 --act_fn tanh --dataset tiny-CIFAR10

# --skip_loss_matched --loss_scale linear --n_loss_divisor 10