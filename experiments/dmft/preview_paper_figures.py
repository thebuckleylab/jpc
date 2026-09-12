"""Render every paper panel from synthetic data, to check the plot style.

The simulations behind the paper figures take cluster time, and the
analysis scripts delete their raw fields once the plots are written, so
this script fabricates records with the same schema as the real ones and
calls the real plotting functions. The numbers are meaningless; only the
typography, panel sizes, labels and legends are.

    python preview_paper_figures.py [--out preview_figures]

Panels land under ``--out`` in the same directory layout the analysis
scripts use, as PDF (for the paper) plus PNG (for a quick look).
"""

import argparse
import os

import numpy as np
import pandas as pd

import plot_style as ps
from plot_dmft_results import (
    plot_final_kernel_grid,
    plot_kernel_concentration_vs_time,
    plot_kernel_displacement_per_timepoint,
    plot_kernel_input_alignment_vs_time,
    plot_kernel_spectrum,
    plot_kernel_target_alignment_vs_time,
    plot_kernel_target_input_alignment_final,
    plot_pc_bp_alignment_vs_time,
    plot_pc_bp_loss_matched_times,
    plot_pc_k_sweep_displacement,
    plot_pc_kernel_width_alignment,
    plot_pc_last_layer_displacement_vs_gamma,
    plot_pc_param_sweep_loss,
    plot_pc_theory_vs_finite_loss,
    plot_temporal_kernel_grid,
)

RNG = np.random.default_rng(0)

#: A loss curve that decays and flattens, like the real ones.
def _loss_curve(n_t, scale=1.0, floor=0.02, noise=0.0):
    t = np.arange(n_t)
    y = floor + scale * np.exp(-3.0 * t / max(1, n_t - 1))
    if noise:
        y = y * (1.0 + noise * RNG.standard_normal(n_t))
    return y


def _kernel(size, rank=3):
    """A PSD kernel with decaying spectrum, normalised to unit diagonal."""
    factors = RNG.standard_normal((size, rank))
    K = factors @ factors.T + 0.3 * np.eye(size)
    d = np.sqrt(np.diag(K))
    return K / np.outer(d, d)


def _kernels_per_layer(n_layers, size, rank=3):
    return [_kernel(size, rank=rank + l) for l in range(n_layers)]


# --- Figure 1 / 2: PC loss sweeps ----------------------------------------


def _sweep_frames(swept_col, values, *, n_hidden=5, gamma_0=1.0,
                  n_infer_iters=5, activity_lr=0.01, n_t=40,
                  widths=(512, 4096)):
    """Theory / finite frames with the columns ``plot_pc_param_sweep_loss`` needs."""
    meta = dict(
        n_hidden=n_hidden,
        gamma_0=gamma_0,
        activity_lr=activity_lr,
        n_infer_iters=n_infer_iters,
        param_type="mupc",
        use_skips=False,
    )
    theory, finite = [], []
    for i, value in enumerate(values):
        row_meta = dict(meta)
        row_meta[swept_col] = value
        scale = 1.0 / (1.0 + 0.35 * i)
        theory_loss = _loss_curve(n_t, scale=scale)
        for t, loss in zip(range(n_t), theory_loss):
            theory.append(dict(row_meta, t=t, loss=loss))
        for width in widths:
            noise = 0.35 / np.sqrt(width)
            for infer_mode in ("infer", "closed_form"):
                curve = _loss_curve(n_t, scale=scale, noise=noise)
                for t, loss in zip(range(n_t), curve):
                    finite.append(
                        dict(
                            row_meta,
                            t=t,
                            loss=loss,
                            width=width,
                            infer_mode=infer_mode,
                        )
                    )
    return pd.DataFrame(theory), pd.DataFrame(finite)


def preview_loss_sweeps(out_dir):
    for swept_col, values in (
        ("n_hidden", (2, 3, 4, 5)),
        ("gamma_0", (0.1, 0.5, 1.0, 2.0)),
        ("n_infer_iters", (5, 10, 20, 50)),
    ):
        theory_df, finite_df = _sweep_frames(swept_col, values)
        plot_pc_param_sweep_loss(
            theory_df,
            finite_df,
            os.path.join(out_dir, "loss_sweeps"),
            swept_col,
            plot_closed_form=True,
        )


def preview_theory_vs_finite(out_dir):
    n_t = 40
    rows = []
    for width in (256, 1024, 4096, 16384):
        curve = _loss_curve(n_t, noise=0.6 / np.sqrt(width))
        for t, loss in zip(range(n_t), curve):
            rows.append(dict(width=width, t=t, loss=loss))
    plot_pc_theory_vs_finite_loss(
        _loss_curve(n_t),
        pd.DataFrame(rows),
        os.path.join(out_dir, "theory_vs_finite"),
        gamma_0=1.0,
        n_hidden=3,
        activity_lr=0.05,
        n_infer_iters=10,
        update_mode="infer",
    )


# --- Figure 1 / 2 / Supp 1: kernel convergence vs width -------------------


def preview_width_alignment(out_dir, feature_symbol="h", n_hidden=5):
    rows = []
    widths = (128, 512, 2048, 8192, 32768)
    for layer in range(n_hidden):
        for width in widths:
            gap = (0.12 + 0.04 * layer) / np.sqrt(width / 128)
            for seed in range(3):
                for kernel in ("h", "delta"):
                    rows.append(
                        dict(
                            width=width,
                            layer=layer,
                            kernel=kernel,
                            seed=seed,
                            alignment=1.0
                            - gap * (1.0 + 0.1 * RNG.standard_normal()),
                        )
                    )
    plot_pc_kernel_width_alignment(
        pd.DataFrame(rows),
        os.path.join(out_dir, "width_alignment"),
        gamma_0=1.0,
        n_hidden=n_hidden,
        activity_lr=0.01,
        n_infer_iters=5,
        feature_symbol=feature_symbol,
    )


# --- Figure 1 / Supp 3: displacement --------------------------------------


def _displacement_rows(n_hidden, ks, gammas):
    rows = []
    for gamma_0 in gammas:
        for layer in range(n_hidden):
            depth = (layer + 1) / n_hidden
            for kind, ks_for_kind in (
                ("dmft", ks[:1]),
                ("infer", ks),
                ("closed_form", [max(ks)]),
            ):
                for K in ks_for_kind:
                    drift = 0.06 * depth * gamma_0**2 * (1.0 + 8.0 / K)
                    rows.append(
                        dict(
                            layer=layer,
                            kind=kind,
                            n_infer_iters=K,
                            gamma_0=gamma_0,
                            displacement=1.0 - drift,
                            rel_displacement=drift,
                        )
                    )
    return pd.DataFrame(rows)


def preview_displacement(out_dir, n_hidden=5):
    ks = [5, 10, 20, 50]
    df = _displacement_rows(n_hidden, ks, [1.0])
    for metric in ("cosine", "rel_frob"):
        plot_pc_k_sweep_displacement(
            df,
            os.path.join(out_dir, "displacement"),
            n_hidden=n_hidden,
            gamma_0=1.0,
            activity_lr=0.01,
            dir_name="convergence",
            metric=metric,
        )
    gamma_df = _displacement_rows(
        n_hidden, ks, [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    )
    for metric in ("cosine", "rel_frob"):
        plot_pc_last_layer_displacement_vs_gamma(
            gamma_df,
            os.path.join(out_dir, "displacement"),
            n_hidden=n_hidden,
            activity_lr=0.01,
            dir_name="convergence",
            metric=metric,
        )


# --- Figure 1 Row 3 / Supp 2: kernel grids --------------------------------


def preview_kernel_grids(out_dir):
    n_layers, size = 5, 24
    # Figure 1 Row 3: three sources, one row each.
    rows = [
        (ps.LABEL_DMFT, _kernels_per_layer(n_layers, size)),
        (ps.LABEL_NN, _kernels_per_layer(n_layers, size)),
        (ps.LABEL_NN_CLOSED_FORM, _kernels_per_layer(n_layers, size)),
    ]
    plot_final_kernel_grid(
        rows,
        plots_dir=os.path.join(out_dir, "kernel_grids"),
        gamma_0=1.0,
        n_hidden=n_layers,
        activity_lr=0.01,
        n_infer_iters=5,
        filename="final_pc_kernels_grid.png",
        dir_name="convergence",
        cbar=False,
        center_zero=False,
    )
    plot_temporal_kernel_grid(
        rows,
        plots_dir=os.path.join(out_dir, "kernel_grids"),
        gamma_0=1.0,
        n_hidden=n_layers,
        activity_lr=0.01,
        n_infer_iters=5,
        dir_name="convergence",
        cbar=False,
        center_zero=False,
    )
    # Supplementary Figure 2: a K sweep, so more rows than columns.
    k_rows = [(ps.k_label(5, prefix=ps.LABEL_DMFT),
               _kernels_per_layer(n_layers, size))]
    k_rows += [
        (ps.k_label(K, prefix=ps.LABEL_NN), _kernels_per_layer(n_layers, size))
        for K in (5, 10, 20, 50)
    ]
    k_rows.append(
        (ps.LABEL_NN_CLOSED_FORM, _kernels_per_layer(n_layers, size))
    )
    plot_final_kernel_grid(
        k_rows,
        plots_dir=os.path.join(out_dir, "kernel_grids_k_sweep"),
        gamma_0=1.0,
        n_hidden=n_layers,
        activity_lr=0.01,
        filename="final_pc_kernels_grid.png",
        dir_name="convergence",
        cbar=False,
        center_zero=False,
    )
    # Figure 2 Row 3: PC vs BP at matched loss.
    for i in (0, 100, 199):
        plot_final_kernel_grid(
            [
                (ps.LABEL_PC, _kernels_per_layer(3, size)),
                (ps.LABEL_BP, _kernels_per_layer(3, size)),
            ],
            plots_dir=os.path.join(out_dir, "kernel_grids_lstar"),
            gamma_0=1.0,
            n_hidden=3,
            activity_lr=0.1,
            n_infer_iters=500,
            filename=f"feature_kernels_grid_lstar{i}.png",
            vmin=-1.0,
            vmax=1.0,
            dir_name="alignment",
        )


# --- Figure 2 Row 2 / Supp 4-7: alignment ---------------------------------

_ALIGN_KW = dict(
    n_hidden=3,
    gamma_0=1.0,
    activity_lr=0.1,
    n_infer_iters=500,
    width=10000,
    dir_name="alignment",
)

#: A loss-matched axis, as used by the ``by_loss_linear`` figures.
_LOSS_GRID = np.geomspace(1.0, 0.02, 12)

_BY_LOSS_AXIS = dict(
    x_col="loss", xlabel=ps.LOSS_LABEL, xscale="log", invert_x=True
)


def _pc_bp_layer_rows(value_col, n_hidden=3, extra=None):
    rows = []
    for method, offset in (("pc", 0.0), ("bp", 0.06)):
        for layer in range(n_hidden):
            for i, loss in enumerate(_LOSS_GRID):
                base = 0.25 + 0.6 * i / (len(_LOSS_GRID) - 1)
                value = base * (1.0 - 0.1 * layer) - offset
                row = dict(loss=loss, layer=layer, method=method)
                row[value_col] = value
                if extra:
                    row.update(extra(value))
                rows.append(row)
    return pd.DataFrame(rows)


def preview_alignment(out_dir):
    plots_dir = os.path.join(out_dir, "alignment")

    n_t = 200
    pc_losses = _loss_curve(n_t, scale=1.0, floor=0.01)
    bp_losses = _loss_curve(n_t, scale=1.0, floor=0.01) * 1.15
    loss_grid = np.geomspace(pc_losses[1], pc_losses[-1] * 1.4, 200)
    pc_times = np.searchsorted(-pc_losses, -loss_grid).clip(0, n_t - 1)
    bp_times = np.searchsorted(-bp_losses, -loss_grid).clip(0, n_t - 1)
    plot_pc_bp_loss_matched_times(
        pc_losses,
        bp_losses,
        pc_times,
        bp_times,
        loss_grid,
        plots_dir,
        heatmap_idx=[0, 100, 199],
        **{k: v for k, v in _ALIGN_KW.items() if k != "dir_name"},
        dir_name="alignment",
    )

    # Figure 2 Row 2, middle: PC-BP kernel CKA vs loss.
    align_rows = []
    for layer in range(3):
        for i, loss in enumerate(_LOSS_GRID):
            align_rows.append(
                dict(
                    loss=loss,
                    layer=layer,
                    alignment=0.99 - 0.25 * (i / len(_LOSS_GRID))
                    - 0.05 * layer,
                )
            )
    plot_pc_bp_alignment_vs_time(
        pd.DataFrame(align_rows),
        plots_dir,
        filename="pc_bp_kernel_alignment_vs_loss.png",
        **_ALIGN_KW,
        **_BY_LOSS_AXIS,
    )

    # Figure 2 Row 2, right: alignment with target and input kernels.
    ti_rows = []
    for method, offset in (("pc", 0.0), ("bp", 0.05)):
        for ref, scale in (("target", 1.0), ("input", 0.55)):
            for layer in range(3):
                ti_rows.append(
                    dict(
                        layer=layer,
                        method=method,
                        ref=ref,
                        alignment=scale * (0.6 - 0.08 * layer) - offset,
                    )
                )
    plot_kernel_target_input_alignment_final(
        pd.DataFrame(ti_rows), plots_dir, **_ALIGN_KW
    )

    # Supplementary Figures 4 and 5: per-layer panels vs loss.
    plot_kernel_target_alignment_vs_time(
        _pc_bp_layer_rows("alignment"),
        plots_dir,
        filename="kernel_target_alignment_vs_loss.png",
        **_ALIGN_KW,
        **_BY_LOSS_AXIS,
    )
    plot_kernel_input_alignment_vs_time(
        _pc_bp_layer_rows("alignment"),
        plots_dir,
        filename="kernel_input_alignment_vs_loss.png",
        **_ALIGN_KW,
        **_BY_LOSS_AXIS,
    )
    disp_df = _pc_bp_layer_rows("displacement")
    disp_df["rel_displacement"] = 1.0 - disp_df["displacement"]
    plot_kernel_displacement_per_timepoint(
        disp_df,
        plots_dir,
        filename="kernel_displacement_vs_loss.png",
        t_sub=r"\mathcal{L}",
        **_ALIGN_KW,
        **_BY_LOSS_AXIS,
    )

    # Supplementary Figure 6: eigenspectra.
    spec_rows = []
    for method in ("pc", "bp"):
        for layer in range(3):
            eigenvalues = np.geomspace(1.0, 1e-4, 64) * (1.0 + 0.1 * layer)
            for i, value in enumerate(eigenvalues, start=1):
                spec_rows.append(
                    dict(
                        layer=layer,
                        method=method,
                        index=i,
                        eigenvalue=value,
                        effective_rank=6.5 + layer,
                    )
                )
    plot_kernel_spectrum(
        pd.DataFrame(spec_rows),
        plots_dir,
        title="Feature-kernel spectrum",
        filename="kernel_spectrum_final.png",
        ylabel=r"$\lambda_i(C^{h,\ell})$",
        annotate_rank=True,
        **_ALIGN_KW,
    )

    # Supplementary Figure 7: concentration across seeds.
    conc_df = _pc_bp_layer_rows("cka_mean")
    conc_df["cka_std"] = 0.03
    plot_kernel_concentration_vs_time(
        conc_df,
        plots_dir,
        metric="cka",
        n_seeds=5,
        filename="kernel_concentration_vs_loss.png",
        **_ALIGN_KW,
        **_BY_LOSS_AXIS,
    )


# --- Figure 3: PC vs BP classification benchmark --------------------------


def _synthetic_benchmark_history(
    n_epochs=10, n_mini=41, n_steps=80, seed=0, pc_offset=0.0
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
        y = y + 0.5 * rng.standard_normal(n) - 1.5 * pc_offset
        return np.clip(y, 0.0, 100.0)

    return {
        "epoch_eval": epochs,
        "epoch_train": epochs,
        "mini_epoch": mini,
        "pc_train_loss_epoch": loss_curve(n_eval, 2.4, 0.04),
        "bp_train_loss_epoch": loss_curve(n_eval, 2.35, 0.05),
        "pc_train_acc_epoch": acc_curve(n_eval, 12.0, 98.0),
        "bp_train_acc_epoch": acc_curve(n_eval, 11.0, 97.5),
        "pc_test_loss": loss_curve(n_eval, 2.3, 0.08),
        "bp_test_loss": loss_curve(n_eval, 2.25, 0.09),
        "pc_test_acc": acc_curve(n_eval, 14.0, 97.0),
        "bp_test_acc": acc_curve(n_eval, 13.0, 96.5),
        "pc_train_loss_mini": loss_curve(n_mini, 2.4, 0.04),
        "bp_train_loss_mini": loss_curve(n_mini, 2.35, 0.05),
        "pc_train_acc_mini": acc_curve(n_mini, 12.0, 98.0),
        "bp_train_acc_mini": acc_curve(n_mini, 11.0, 97.5),
        "pc_test_loss_mini": loss_curve(n_mini, 2.3, 0.08),
        "bp_test_loss_mini": loss_curve(n_mini, 2.25, 0.09),
        "pc_test_acc_mini": acc_curve(n_mini, 14.0, 97.0),
        "bp_test_acc_mini": acc_curve(n_mini, 13.0, 96.5),
        "pc_train_loss_step": loss_curve(n_steps, 2.4, 0.04),
        "bp_train_loss_step": loss_curve(n_steps, 2.35, 0.05),
        "pc_train_acc_step": acc_curve(n_steps, 12.0, 98.0),
        "bp_train_acc_step": acc_curve(n_steps, 11.0, 97.5),
    }


def preview_benchmark(out_dir):
    from train_benchmark import plot_metrics, plot_metrics_mean_sem

    plots_dir = os.path.join(out_dir, "benchmark")
    plot_metrics(_synthetic_benchmark_history(seed=0), plots_dir)
    plot_metrics_mean_sem(
        [_synthetic_benchmark_history(seed=s, pc_offset=0.15 * s) for s in range(3)],
        plots_dir,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="preview_figures")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    preview_loss_sweeps(out_dir)
    preview_theory_vs_finite(out_dir)
    for n_hidden in (2, 3, 4, 5):
        preview_width_alignment(out_dir, n_hidden=n_hidden)
    preview_width_alignment(out_dir, feature_symbol="phi", n_hidden=3)
    preview_displacement(out_dir, n_hidden=5)
    preview_displacement(out_dir, n_hidden=3)
    preview_kernel_grids(out_dir)
    preview_alignment(out_dir)
    preview_benchmark(out_dir)
    print(f"\nPreview panels written under {out_dir}")


if __name__ == "__main__":
    main()
