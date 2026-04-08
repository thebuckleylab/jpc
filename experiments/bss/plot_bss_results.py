"""
Plot results from train_bss: train loss, SINR over training, and final per-source SNR.
Saves each in a separate PDF (same style as plot_theory_results).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Plot styling (same as plot_theory_results.py)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})
FIG_SIZE = (8, 6)
FONT_SIZES = {"label": 45, "legend": 25, "tick": 35}
LABEL_PAD = 15
LINE_WIDTH = 4
ALPHA = 0.7


def _save_plot(save_dir, filename):
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
    plt.close()


def _find_results_dir(root_dir):
    """Find a directory under root_dir that contains train_losses.npy (most recent mtime)."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if "train_losses.npy" in filenames and "sinrs.npy" in filenames:
            mtime = os.path.getmtime(os.path.join(dirpath, "train_losses.npy"))
            found.append((mtime, dirpath))
    if not found:
        return None
    found.sort(key=lambda x: -x[0])
    return found[0][1]


def load_results(save_dir):
    """Load BSS experiment .npy arrays."""
    out = {}
    for name in [
        "train_losses",
        "train_losses_smooth",
        "sinrs",
        "snr_per_source",
        "ground_truth_sources",
        "reconstructed_sources",
    ]:
        path = os.path.join(save_dir, f"{name}.npy")
        if os.path.isfile(path):
            out[name] = np.load(path)
        else:
            out[name] = None
    return out


def _setup_axes(ax):
    """Apply shared axis styling (same as plot_theory_results)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    ax.grid(True, which="both", ls="-", alpha=0.4)


def plot_bss_results(save_dir, out_path=None):
    data = load_results(save_dir)

    if data["train_losses"] is None:
        if os.path.isdir("bss_results"):
            resolved = _find_results_dir("bss_results")
            if resolved is not None:
                save_dir = resolved
                data = load_results(save_dir)
        if data["train_losses"] is None:
            raise FileNotFoundError(
                "No train_losses.npy found in default path or under bss_results/. "
                "Run train_bss.py first, or pass save_dir explicitly."
            )

    out_dir = save_dir if out_path is None else out_path
    os.makedirs(out_dir, exist_ok=True)

    n_steps = len(data["train_losses"])
    steps = np.arange(1, n_steps + 1, dtype=float)

    # 1. Train loss (raw + smoothed for readability)
    plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()
    ax.plot(steps, data["train_losses"], label="Train (raw)", color="C0", linewidth=0.8, alpha=0.35)
    smooth = data["train_losses_smooth"]
    if smooth is not None and len(smooth) == n_steps:
        ax.plot(steps, smooth, label="Train (smoothed)", color="C0", linewidth=LINE_WIDTH, alpha=ALPHA)
    else:
        # Fallback: compute rolling mean in-place
        window = min(50, max(1, n_steps // 20))
        smooth = np.convolve(data["train_losses"], np.ones(window) / window, mode="same")
        ax.plot(steps, smooth, label="Train (smoothed)", color="C0", linewidth=LINE_WIDTH, alpha=ALPHA)
    ax.set_xlabel("Step $t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    ax.set_ylabel(r"$\mathcal{L}(\boldsymbol{\theta}_t)$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    ax.legend(fontsize=FONT_SIZES["legend"])
    _setup_axes(ax)
    _save_plot(out_dir, "losses.pdf")

    # 2. SINR (dB) at evaluation steps
    sinrs = data["sinrs"]
    if sinrs is not None and len(sinrs) > 0:
        plt.figure(figsize=FIG_SIZE)
        ax = plt.gca()
        if len(sinrs) == 1:
            sinr_steps = np.array([1.0])
        else:
            sinr_steps = 1 + np.arange(len(sinrs)) * ((n_steps - 1) / max(1, len(sinrs) - 1))
        ax.plot(
            sinr_steps,
            sinrs,
            color="darkgreen",
            linewidth=LINE_WIDTH,
            alpha=ALPHA,
            marker="o",
            markersize=6,
        )
        ax.set_xlabel("Step $t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
        ax.set_ylabel("SINR (dB)", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
        _setup_axes(ax)
        _save_plot(out_dir, "sinr.pdf")

    # 3. Final per-source SNR (bar chart)
    snr_per_source = data["snr_per_source"]
    if snr_per_source is not None and len(snr_per_source) > 0:
        plt.figure(figsize=FIG_SIZE)
        ax = plt.gca()
        n_sources = len(snr_per_source)
        x = np.arange(1, n_sources + 1)
        ax.bar(x, snr_per_source, color="C0", alpha=ALPHA, edgecolor="black", linewidth=1.5)
        ax.set_xlabel("Source index", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
        ax.set_ylabel("SNR (dB)", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_ylim(0, None)
        _setup_axes(ax)
        _save_plot(out_dir, "snr_per_source.pdf")

    # 4. Reconstructed vs ground truth sources (time series, one subplot per source)
    # Plots all samples in the saved arrays (n_val from train_bss).
    S_gt = data["ground_truth_sources"]
    Y_rec = data["reconstructed_sources"]
    if S_gt is not None and Y_rec is not None and S_gt.shape == Y_rec.shape:
        n_sources, n_samples = S_gt.shape
        t = np.arange(n_samples, dtype=float)
        fig, axes = plt.subplots(n_sources, 1, sharex=True, figsize=(8, 3 * n_sources))
        if n_sources == 1:
            axes = [axes]
        line_width_sources = 1.0
        for i, ax in enumerate(axes):
            ax.plot(t, S_gt[i, :], label="Ground truth", color="C0", linewidth=line_width_sources, alpha=ALPHA)
            ax.plot(t, Y_rec[i, :], label="Reconstructed", color="C1", linestyle="--", linewidth=line_width_sources, alpha=ALPHA)
            ax.set_ylabel(f"Source {i + 1}", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
            ax.legend(fontsize=FONT_SIZES["legend"])
            _setup_axes(ax)
        axes[-1].set_xlabel("Sample index", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
        _save_plot(out_dir, "sources_reconstructed.pdf")

    files = ["losses.pdf"]
    if sinrs is not None and len(sinrs) > 0:
        files.append("sinr.pdf")
    if snr_per_source is not None and len(snr_per_source) > 0:
        files.append("snr_per_source.pdf")
    if S_gt is not None and Y_rec is not None and S_gt.shape == Y_rec.shape:
        files.append("sources_reconstructed.pdf")
    print(f"Saved plots to {out_dir}: {', '.join(files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot BSS training results from train_bss.py."
    )
    parser.add_argument(
        "save_dir",
        type=str,
        nargs="?",
        default=None,
        help="Directory containing .npy outputs from train_bss. Default: search bss_results/ for latest run.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for the PDFs. Default: save_dir (same as data).",
    )
    args = parser.parse_args()
    save_dir = args.save_dir
    if save_dir is None and os.path.isdir("bss_results"):
        save_dir = _find_results_dir("bss_results")
    if save_dir is None:
        raise FileNotFoundError(
            "No BSS results directory found. Run train_bss.py or pass save_dir explicitly."
        )
    plot_bss_results(save_dir, out_path=args.out)


if __name__ == "__main__":
    main()
