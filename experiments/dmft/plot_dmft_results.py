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
    update_mode=None,
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
    """
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    if activity_lr is not None:
        plots_dir = os.path.join(plots_dir, f"activity_lr_{activity_lr}")
    plots_dir = os.path.join(plots_dir, "pc")
    os.makedirs(plots_dir, exist_ok=True)

    pc_dmft_loss = np.asarray(pc_dmft_loss).flatten()
    _warn_if_nonfinite("pc_dmft_loss", pc_dmft_loss)
    theory_t = np.arange(1, len(pc_dmft_loss) + 1)

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
    title = "PC theory vs finite-size simulation"
    if update_mode is not None:
        title += f" ({update_mode})"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    if activity_lr is not None:
        title += f", activity lr$={activity_lr}$"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    filename = "pc_theory_vs_finite_loss"
    if update_mode is not None:
        filename += f"_{update_mode}"
    save_path = os.path.join(plots_dir, f"{filename}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"PC theory vs finite-size loss plot saved to {save_path}")
    return save_path


def plot_bp_theory_vs_finite_loss(
    dmft_loss,
    finite_df,
    plots_dir,
    gamma_0=None,
    n_hidden=None,
):
    """Overlay the BP DMFT theory loss curve with finite-size empirical losses.

    ``finite_df`` is expected to have columns ``width``, ``t`` and ``loss``,
    as produced by ``train_bpn`` over the same hyperparameters used for
    ``dmft_loss``. This lets us visually check that the finite-size BP
    networks converge to the DMFT (infinite-width) prediction as width grows.
    """
    if n_hidden is not None:
        plots_dir = os.path.join(plots_dir, f"{n_hidden}_n_hidden")
    if gamma_0 is not None:
        plots_dir = os.path.join(plots_dir, f"gamma_{gamma_0}")
    plots_dir = os.path.join(plots_dir, "bp")
    os.makedirs(plots_dir, exist_ok=True)

    dmft_loss = np.asarray(dmft_loss).flatten()
    _warn_if_nonfinite("dmft_loss", dmft_loss)
    theory_t = np.arange(1, len(dmft_loss) + 1)

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
    title = "BP theory vs finite-size simulation"
    if gamma_0 is not None:
        title += f", $\\gamma_0={gamma_0}$"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "bp_theory_vs_finite_loss.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"BP theory vs finite-size loss plot saved to {save_path}")
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