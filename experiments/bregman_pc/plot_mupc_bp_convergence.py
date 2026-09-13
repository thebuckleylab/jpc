"""Plot μPC BP-convergence cosine heatmaps in the limits-paper width/depth style."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

LABEL_PAD = 20

_METHODS = (
    (
        "grad_cosine_similarities.npy",
        r"$\cos(\nabla_{\boldsymbol{\theta}_t}\mathcal{F}_B^*, \nabla_{\boldsymbol{\theta}_t}\mathcal{L})$",
        "bregman_pc_grad_cosine_similarity_heatmap",
    ),
    (
        "std_pc_grad_cosine_similarities.npy",
        r"$\cos(\nabla_{\boldsymbol{\theta}_t}\mathcal{F}^*, \nabla_{\boldsymbol{\theta}_t}\mathcal{L})$",
        "std_pc_grad_cosine_similarity_heatmap",
    ),
)


def _load_npy_safe(path):
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return np.array([
            x.item() if hasattr(x, "item") else float(x) for x in arr
        ])
    return np.asarray(arr).flatten()


def discover_available_configs(results_dir, seed=0, filename="grad_cosine_similarities.npy"):
    """Map (width, n_hidden) to the seed directory that contains ``filename``."""
    configs = {}
    seed_str = str(seed)
    for root, _, files in os.walk(results_dir):
        if filename not in files:
            continue
        if root.split(os.sep)[-1] != seed_str:
            continue
        width = None
        n_hidden = None
        for part in root.split(os.sep):
            if part.endswith("_width"):
                try:
                    width = int(part.replace("_width", ""))
                except ValueError:
                    pass
            elif part.endswith("_n_hidden"):
                try:
                    n_hidden = int(part.replace("_n_hidden", ""))
                except ValueError:
                    pass
        if width is not None:
            configs[(width, n_hidden)] = root
    return configs


def _available_seeds(results_dir, filename):
    seeds = set()
    for root, _, files in os.walk(results_dir):
        if filename not in files:
            continue
        last_dir = root.split(os.sep)[-1]
        try:
            seeds.add(int(last_dir))
        except ValueError:
            pass
    return sorted(seeds)


def _cosine_at_step(path, time_step):
    cosine_sims = _load_npy_safe(path)
    if cosine_sims is None or len(cosine_sims) == 0:
        return None
    idx = min(time_step, len(cosine_sims) - 1)
    value = cosine_sims[idx]
    if isinstance(value, np.ndarray):
        value = float(value.item() if value.ndim == 0 else value[0])
    else:
        value = float(value)
    if value < 0:
        return 0.0
    return value


def collect_cosine_grid(
    results_dir,
    filename,
    seeds,
    widths_filter=None,
    n_hiddens_filter=None,
    time_step=0,
):
    if not seeds:
        print("  Warning: No seeds found")
        return None

    print(f"  Found seeds: {seeds}, averaging over {len(seeds)} seeds")
    seed_configs = {
        seed_val: discover_available_configs(
            results_dir, seed=seed_val, filename=filename
        )
        for seed_val in seeds
    }
    all_keys = set().union(*seed_configs.values()) if seed_configs else set()
    if widths_filter:
        all_keys = {(w, h) for (w, h) in all_keys if w in set(widths_filter)}
    if n_hiddens_filter:
        n_hiddens_filter_set = set(n_hiddens_filter)
        all_keys = {
            (w, h) for (w, h) in all_keys
            if (h if h is not None else 0) in n_hiddens_filter_set
        }
    if not all_keys:
        print("  Warning: No configurations found")
        return None

    widths = []
    depths = []
    cosine_similarities = []
    for width, n_hidden in sorted(all_keys):
        values = []
        for seed_val in seeds:
            path = seed_configs[seed_val].get((width, n_hidden))
            if path is None:
                continue
            value = _cosine_at_step(os.path.join(path, filename), time_step)
            if value is not None:
                values.append(value)
        if not values:
            continue
        widths.append(width)
        depths.append((n_hidden + 1) if n_hidden is not None else 1)
        cosine_similarities.append(float(np.mean(values)))

    if not cosine_similarities:
        print("  Warning: No cosine similarity data available")
        return None

    print(f"  Loaded {len(cosine_similarities)} data points")
    print(
        "  Cosine similarity range: "
        f"{np.min(cosine_similarities):.6f} - {np.max(cosine_similarities):.6f}"
    )
    return np.array(widths), np.array(depths), np.array(cosine_similarities)


def plot_cosine_heatmap(
    widths_arr,
    depths_arr,
    cosine_sims_arr,
    plot_dir,
    filename_stem,
    cbar_label,
    time_step,
    axis_widths=None,
    axis_depths=None,
):
    unique_widths = np.array(axis_widths) if axis_widths is not None else np.sort(np.unique(widths_arr))
    unique_depths = np.array(axis_depths) if axis_depths is not None else np.sort(np.unique(depths_arr))
    cosine_grid = np.full((len(unique_widths), len(unique_depths)), np.nan, dtype=float)
    for width, depth, cosine_sim in zip(widths_arr, depths_arr, cosine_sims_arr):
        width_idx = np.where(unique_widths == width)[0][0]
        depth_idx = np.where(unique_depths == depth)[0][0]
        cosine_grid[width_idx, depth_idx] = cosine_sim

    fig, ax = plt.subplots(figsize=(10, 8))
    masked = np.ma.masked_invalid(cosine_grid)
    im = ax.imshow(
        masked.T,
        aspect="auto",
        origin="lower",
        cmap="viridis_r",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"$t={time_step}$", fontsize=55, pad=20)

    ax.set_xticks(np.arange(len(unique_widths)))
    ax.set_xticklabels([f"${w}$" for w in unique_widths], fontsize=35)
    ax.set_yticks(np.arange(len(unique_depths)))
    ax.set_yticklabels([f"${d}$" for d in unique_depths], fontsize=35)
    ax.set_xlabel("Width $N$", fontsize=55, labelpad=LABEL_PAD)
    ax.set_ylabel("Depth $L$", fontsize=55, labelpad=LABEL_PAD)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=55, rotation=270, labelpad=80)
    cbar.ax.tick_params(labelsize=40)
    ax.tick_params(axis="x", labelsize=35)
    ax.tick_params(axis="y", labelsize=35)
    for spine in ax.spines.values():
        spine.set_visible(False)

    os.makedirs(plot_dir, exist_ok=True)
    filename = f"{filename_stem}_t{time_step}.pdf"
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, filename),
        bbox_inches="tight",
        pad_inches=0.9,
        dpi=300,
    )
    plt.close()
    print(f"  Saved cosine similarity heatmap to {os.path.join(plot_dir, filename)}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot μPC BP-convergence cosine heatmaps vs width and depth"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results",
            "mupc_bp_convergence",
        ),
        help="Directory containing mupc_bp_convergence outputs",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: results_dir/plots)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to average over (default: all discovered)",
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[2, 8, 32, 128, 512, 2048],
        help="Widths N to include",
    )
    parser.add_argument(
        "--n_hiddens",
        type=int,
        nargs="+",
        default=[1, 3, 7, 15, 31],
        help="Hidden-layer counts H; plotted depth is L = H + 1",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        nargs="+",
        default=[0, 50, 100],
        help="Training-step indices (clamped to the last saved step)",
    )
    args = parser.parse_args()

    plot_dir = args.plot_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for filename, cbar_label, stem in _METHODS:
        print(f"\n{'=' * 80}")
        print(f"Method file: {filename}")
        print(f"{'=' * 80}")
        seeds = args.seeds or _available_seeds(args.results_dir, filename)
        for time_step in args.time_steps:
            print(f"\n  Plotting cosine similarity heatmap at time step t={time_step}")
            collected = collect_cosine_grid(
                args.results_dir,
                filename,
                seeds,
                widths_filter=args.widths,
                n_hiddens_filter=args.n_hiddens,
                time_step=time_step,
            )
            if collected is None:
                continue
            widths_arr, depths_arr, cosine_sims_arr = collected
            plot_cosine_heatmap(
                widths_arr,
                depths_arr,
                cosine_sims_arr,
                plot_dir,
                stem,
                cbar_label,
                time_step,
                axis_widths=args.widths,
                axis_depths=[h + 1 for h in args.n_hiddens],
            )


if __name__ == "__main__":
    main()
