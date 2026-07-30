import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


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
    """Plot G kernels for each layer."""
    all_G = _to_numpy(all_G)
    n_layers = len(all_G)
    _warn_if_nonfinite("all_G", np.stack([np.asarray(G) for G in all_G]))

    fig, axes = plt.subplots(
        n_layers, 1, figsize=(6, 2 * n_layers), squeeze=False
    )
    for l, G_l in enumerate(all_G):
        ax = axes[l, 0]
        ax.imshow(np.asarray(G_l), cmap="coolwarm")
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
    """Plot DMFT loss and kernel matrices, saving to plots_dir."""
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dmft_loss(dmft_loss, plots_dir, gamma_0=gamma_0)
    plot_H_kernels(all_H, plots_dir, gamma_0=gamma_0)
    plot_G_kernels(all_G, plots_dir, gamma_0=gamma_0)


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