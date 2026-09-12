"""Copy paper figures into ``figures/pdf`` and ``figures/png``.

Each entry is a source path relative to this script's directory, already
named as ``.pdf``. The matching ``.png`` is taken from the same stem.
Missing files are skipped; a found / not-found list is printed and
written to ``figures/extract_status.txt``.

    python extract_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "figures"

# (source relative to HERE, destination stem without suffix)
FIGURES = [
    # Figure 2
    (
        "results_convergence/results_D/plots/gamma_1.0/activity_lr_0.01/"
        "5_n_infer_iters/pc/pc_loss_vs_n_hidden.pdf",
        "fig2a",
    ),
    (
        "results_convergence/results_G/plots/5_n_hidden/activity_lr_0.01/"
        "5_n_infer_iters/pc/pc_loss_vs_gamma_0.pdf",
        "fig2b",
    ),
    (
        "results_convergence/results_W/plots/5_n_hidden/gamma_1.0/"
        "activity_lr_0.01/5_n_infer_iters/pc/pc_kernel_convergence_Ch_vs_width.pdf",
        "fig2c",
    ),
    (
        "results_convergence/results_KG/plots/5_n_hidden/gamma_1.0/"
        "activity_lr_0.01/pc/pc_loss_vs_n_infer_iters.pdf",
        "fig2d",
    ),
    (
        "results_convergence/results_KG/plots/5_n_hidden/gamma_1.0/"
        "activity_lr_0.01/convergence/kernel_displacement_vs_layer.pdf",
        "fig2e",
    ),
    (
        "results_convergence/results_KG/plots/5_n_hidden/activity_lr_0.01/"
        "convergence/kernel_displacement_last_layer_vs_gamma.pdf",
        "fig2f",
    ),
    (
        "results_convergence/results_S/plots/5_n_hidden/gamma_1.0/"
        "activity_lr_0.01/5_n_infer_iters/convergence/final_pc_kernels_grid.pdf",
        "fig2g",
    ),
    (
        "results_convergence/results_S/plots/5_n_hidden/gamma_1.0/"
        "activity_lr_0.01/5_n_infer_iters/convergence/temporal_pc_kernels_grid.pdf",
        "fig2h",
    ),
    # Figure 3
    (
        "results_convergence/results_nonlin_G/plots/3_n_hidden/"
        "activity_lr_0.05/10_n_infer_iters/pc/pc_loss_vs_gamma_0.pdf",
        "fig3a",
    ),
    (
        "results_convergence/results_nonlin_KG/plots/3_n_hidden/gamma_1.0/"
        "activity_lr_0.05/pc/pc_loss_vs_n_infer_iters.pdf",
        "fig3b",
    ),
    (
        "results_convergence/results_nonlin_W/plots/3_n_hidden/gamma_1.0/"
        "activity_lr_0.05/10_n_infer_iters/pc/"
        "pc_kernel_convergence_Cphi_vs_width.pdf",
        "fig3c",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_time/"
        "pc_bp_loss.pdf",
        "fig3d",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "pc_bp_kernel_alignment_vs_loss.pdf",
        "fig3e",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "kernel_target_input_alignment_final.pdf",
        "fig3f",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "feature_kernels_grid_lstar0.pdf",
        "fig3g",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "feature_kernels_grid_lstar100.pdf",
        "fig3h",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "feature_kernels_grid_lstar199.pdf",
        "fig3i",
    ),
    # Supplementary
    (
        "results_convergence/results_W/plots/2_n_hidden/gamma_1.0/"
        "activity_lr_0.01/5_n_infer_iters/pc/pc_kernel_convergence_Ch_vs_width.pdf",
        "supfig1a",
    ),
    (
        "results_convergence/results_W/plots/3_n_hidden/gamma_1.0/"
        "activity_lr_0.01/5_n_infer_iters/pc/pc_kernel_convergence_Ch_vs_width.pdf",
        "supfig1b",
    ),
    (
        "results_convergence/results_W/plots/4_n_hidden/gamma_1.0/"
        "activity_lr_0.01/5_n_infer_iters/pc/pc_kernel_convergence_Ch_vs_width.pdf",
        "supfig1c",
    ),
    (
        "results_convergence/results_KG/plots/5_n_hidden/gamma_1.0/"
        "activity_lr_0.01/convergence/final_pc_kernels_grid.pdf",
        "supfig2",
    ),
    (
        "results_convergence/results_nonlin_KG/plots/3_n_hidden/gamma_1.0/"
        "activity_lr_0.05/convergence/kernel_displacement_vs_layer.pdf",
        "supfig3a",
    ),
    (
        "results_convergence/results_nonlin_KG/plots/3_n_hidden/"
        "activity_lr_0.05/convergence/kernel_displacement_last_layer_vs_gamma.pdf",
        "supfig3b",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "kernel_target_alignment_vs_loss.pdf",
        "supfig4a",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "kernel_input_alignment_vs_loss.pdf",
        "supfig4b",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "kernel_displacement_vs_loss.pdf",
        "supfig5",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "kernel_spectrum_final.pdf",
        "supfig6",
    ),
    (
        "results_alignment/plots/10000_width/infer_pc_infer_mode/3_n_hidden/"
        "gamma_1.0/activity_lr_0.1/500_n_infer_iters/alignment/by_loss_linear/"
        "kernel_concentration_cka_vs_loss.pdf",
        "supfig7",
    ),
    (
        "results_benchmark/results_mnist/MNIST/mlp/ce/256_width/2_n_hidden/"
        "relu_act_fn/mupc_param_type/1.0_gamma/adam_param_optim/0.1_param_lr/"
        "0.1_param_lr_pc/64_batch_size/10_n_epochs/20_n_infer_iters/"
        "0.01_activity_lr/False_use_skips/False_skip_pc/False_skip_bp/"
        "seeds_0_2/plots/pc_bp_epoch_metrics_mean_sem.pdf",
        "supfig8a",
    ),
    (
        "results_benchmark/results_fashion_mnist/Fashion-MNIST/mlp/ce/"
        "256_width/2_n_hidden/relu_act_fn/mupc_param_type/1.0_gamma/"
        "adam_param_optim/0.3_param_lr/0.3_param_lr_pc/128_batch_size/"
        "10_n_epochs/20_n_infer_iters/0.001_activity_lr/False_use_skips/"
        "False_skip_pc/False_skip_bp/seeds_0_2/plots/"
        "pc_bp_epoch_metrics_mean_sem.pdf",
        "supfig8b",
    ),
]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    pdf_dir = OUT_DIR / "pdf"
    png_dir = OUT_DIR / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    found: list[str] = []
    missing: list[str] = []

    for rel, stem in FIGURES:
        src_pdf = HERE / rel
        src_png = src_pdf.with_suffix(".png")
        for src, dst in (
            (src_pdf, pdf_dir / f"{stem}.pdf"),
            (src_png, png_dir / f"{stem}.png"),
        ):
            label = f"{dst.relative_to(OUT_DIR)}  <-  {src.relative_to(HERE)}"
            if copy_if_exists(src, dst):
                found.append(label)
            else:
                missing.append(label)

    lines = [
        f"Copied {len(found)} file(s) into {OUT_DIR}",
        f"Missing {len(missing)} file(s)",
        "",
        "=== found ===",
        *(found or ["(none)"]),
        "",
        "=== not found ===",
        *(missing or ["(none)"]),
        "",
    ]
    report = "\n".join(lines)
    (OUT_DIR / "extract_status.txt").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
