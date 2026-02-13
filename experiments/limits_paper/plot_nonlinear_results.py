import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Reuse *exactly* the styling and helper utilities from plot_main_results.
# This ensures consistent rcParams, fontsizes, line widths, legends, etc.
import plot_main_results as pmr


@dataclass(frozen=True)
class RunConfig:
    input_dim: Optional[int] = None
    n_samples: Optional[int] = None
    n_hidden: Optional[int] = None
    use_skips: Optional[bool] = None
    act_fn: Optional[str] = None
    param_type: Optional[str] = None
    param_optim: Optional[str] = None  # PC path component: "{id}_param_optim"
    optim_id: Optional[str] = None     # BP path component: "{id}_optim_id"
    param_lr: Optional[float] = None
    gamma_0: Optional[float] = None
    n_train_iters: Optional[int] = None
    infer_mode: Optional[str] = None
    activity_lr: Optional[float] = None
    seed: Optional[int] = None


def _parse_bool(s: str) -> Optional[bool]:
    s = str(s).lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _float_close(a: float, b: float, rtol: float = 1e-7, atol: float = 1e-12) -> bool:
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


def _extract_from_parts(parts: Iterable[str]) -> Dict[str, Any]:
    """Extract known hyperparameters from a directory path split into parts."""
    out: Dict[str, Any] = {}
    for p in parts:
        if p.endswith("_input_dim"):
            try:
                out["input_dim"] = int(p.replace("_input_dim", ""))
            except ValueError:
                pass
        elif p.endswith("_n_samples"):
            try:
                out["n_samples"] = int(p.replace("_n_samples", ""))
            except ValueError:
                pass
        elif p.endswith("_n_hidden"):
            try:
                out["n_hidden"] = int(p.replace("_n_hidden", ""))
            except ValueError:
                pass
        elif p.endswith("_use_skips"):
            b = _parse_bool(p.replace("_use_skips", ""))
            if b is not None:
                out["use_skips"] = b
        elif p.endswith("_act_fn"):
            out["act_fn"] = p.replace("_act_fn", "")
        elif p.endswith("_param_type"):
            out["param_type"] = p.replace("_param_type", "")
        elif p.endswith("_param_optim"):
            out["param_optim"] = p.replace("_param_optim", "")
        elif p.endswith("_optim_id"):
            out["optim_id"] = p.replace("_optim_id", "")
        elif p.endswith("_param_lr"):
            try:
                out["param_lr"] = float(p.replace("_param_lr", ""))
            except ValueError:
                pass
        elif p.endswith("_gamma_0"):
            try:
                out["gamma_0"] = float(p.replace("_gamma_0", ""))
            except ValueError:
                pass
        elif p.endswith("_n_train_iters"):
            try:
                out["n_train_iters"] = int(p.replace("_n_train_iters", ""))
            except ValueError:
                pass
        elif p.endswith("_infer_mode"):
            out["infer_mode"] = p.replace("_infer_mode", "")
        elif p.endswith("_activity_lr"):
            try:
                out["activity_lr"] = float(p.replace("_activity_lr", ""))
            except ValueError:
                pass
        elif p.endswith("_width"):
            try:
                out["width"] = int(p.replace("_width", ""))
            except ValueError:
                pass
    return out


def _matches_config(extracted: Dict[str, Any], cfg: RunConfig, is_pc: bool) -> bool:
    """Check whether extracted path config matches the requested config."""
    # Integers/strings: exact match when specified.
    # Note: BP directories do not include infer_mode/activity_lr, so only match those for PC.
    keys = ["input_dim", "n_samples", "n_hidden", "act_fn", "param_type", "n_train_iters"]
    if is_pc:
        keys.append("infer_mode")
    for k in keys:
        desired = getattr(cfg, k)
        if desired is None:
            continue
        if extracted.get(k) != desired:
            return False

    # use_skips
    if cfg.use_skips is not None and extracted.get("use_skips") != cfg.use_skips:
        return False

    # Optim identifier differs between PC and BP dirs
    if is_pc and cfg.param_optim is not None:
        if extracted.get("param_optim") != cfg.param_optim:
            return False
    if (not is_pc) and cfg.optim_id is not None:
        if extracted.get("optim_id") != cfg.optim_id:
            return False

    # Floats: tolerant match when specified.
    for k in ("param_lr", "gamma_0"):
        desired = getattr(cfg, k)
        if desired is None:
            continue
        got = extracted.get(k)
        if got is None or not _float_close(float(got), float(desired)):
            return False

    # activity_lr only exists in PC path; match if requested.
    if is_pc and cfg.activity_lr is not None:
        got = extracted.get("activity_lr")
        if got is None or not _float_close(float(got), float(cfg.activity_lr)):
            return False

    return True


def _find_pc_dirs(dataset_results_dir: str, cfg: RunConfig) -> Dict[int, str]:
    """Return {width -> pc_seed_dir} for matching PC runs (dir contains energies.npy)."""
    pc_dirs: Dict[int, str] = {}
    seed_str = str(cfg.seed) if cfg.seed is not None else None

    for root, _, files in os.walk(dataset_results_dir):
        if "energies.npy" not in files:
            continue
        if seed_str is not None and root.split(os.sep)[-1] != seed_str:
            continue
        parts = root.split(os.sep)
        extracted = _extract_from_parts(parts)
        width = extracted.get("width")
        if width is None:
            continue
        if not _matches_config(extracted, cfg, is_pc=True):
            continue
        pc_dirs[int(width)] = root

    return pc_dirs


def _find_bp_dirs(dataset_results_dir: str, cfg: RunConfig) -> Dict[int, str]:
    """Return {width -> bp_seed_dir} for matching BP runs (dir contains losses.npy)."""
    bp_dirs: Dict[int, str] = {}
    seed_str = str(cfg.seed) if cfg.seed is not None else None

    for root, _, files in os.walk(dataset_results_dir):
        if "losses.npy" not in files:
            continue
        if seed_str is not None and root.split(os.sep)[-1] != seed_str:
            continue
        parts = root.split(os.sep)
        extracted = _extract_from_parts(parts)
        width = extracted.get("width")
        if width is None:
            continue
        if not _matches_config(extracted, cfg, is_pc=False):
            continue
        bp_dirs[int(width)] = root

    return bp_dirs


def load_nonlinear_data(dataset_results_dir: str, cfg: RunConfig, widths: Optional[List[int]] = None) -> Dict[str, Any]:
    """Load nonlinear training curves (PC loss/energy + BP loss + grad cosine sims) by width."""
    pc_dirs = _find_pc_dirs(dataset_results_dir, cfg)
    bp_dirs = _find_bp_dirs(dataset_results_dir, cfg)

    if widths is None:
        widths_to_use = sorted(pc_dirs.keys())
    else:
        widths_to_use = [w for w in widths if w in pc_dirs]

    data: Dict[str, Any] = {
        "widths": widths_to_use,
        "pc_energies": {},
        "pc_train_losses": {},
        "bp_losses": {},
        "grad_cosine_similarities": {},
    }

    for w in widths_to_use:
        pc_dir = pc_dirs.get(w)
        if pc_dir:
            energies = pmr._load_npy_safe(os.path.join(pc_dir, "energies.npy"))
            train_losses = pmr._load_npy_safe(os.path.join(pc_dir, "train_losses.npy"))
            cos_sims = pmr._load_npy_safe(os.path.join(pc_dir, "grad_cosine_similarities.npy"))
            if energies is not None:
                data["pc_energies"][w] = energies
            if train_losses is not None:
                data["pc_train_losses"][w] = train_losses
            if cos_sims is not None:
                data["grad_cosine_similarities"][w] = cos_sims

        bp_dir = bp_dirs.get(w)
        if bp_dir:
            losses = pmr._load_npy_safe(os.path.join(bp_dir, "losses.npy"))
            if losses is not None:
                data["bp_losses"][w] = losses

    return data


def plot_pc_losses(data: Dict[str, Any], plot_dir: str, colormap_name: str = "Blues", log_x_scale: bool = False) -> None:
    """PC train loss curves vs t, colored by width (same style as plot_main_results)."""
    plt.figure(figsize=(12.5, 6))
    widths_list = sorted([w for w in data["widths"] if w in data["pc_train_losses"]])
    if widths_list:
        cmap = plt.get_cmap(colormap_name)
        n_widths = len(widths_list)
        for idx, w in enumerate(widths_list):
            y = np.array(data["pc_train_losses"][w]).flatten()
            x = np.arange(1, len(y) + 1)
            color = cmap(pmr.get_color_val(idx, n_widths, colormap_name))
            plt.plot(x, y, "-", alpha=pmr.ALPHA, linewidth=pmr.LINE_WIDTH, color=color, label="")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("$t$", fontsize=pmr.FONT_SIZES["label"], labelpad=pmr.LABEL_PAD)
    plt.ylabel(r"$\mathcal{L}(\boldsymbol{\theta}_t)$", fontsize=pmr.FONT_SIZES["label"], labelpad=pmr.LABEL_PAD)
    if log_x_scale:
        plt.xscale("log", base=10)

    # Legend encoding widths in grayscale (same convention as plot_main_results)
    if widths_list:
        gray_cmap = plt.get_cmap("Greys")
        legend_handles, legend_labels = [], []
        for idx, w in enumerate(widths_list):
            gray_val = 0.3 + (idx / max(len(widths_list) - 1, 1)) * 0.5 if len(widths_list) > 1 else 0.5
            legend_handles.append(plt.Line2D([0], [0], color=gray_cmap(gray_val), linewidth=pmr.LINE_WIDTH))
            legend_labels.append(f"$N = {w}$")
        plt.legend(handles=legend_handles, labels=legend_labels, fontsize=pmr.FONT_SIZES["legend"],
                   bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis="both", labelsize=pmr.FONT_SIZES["tick"])
    plt.yscale("log", base=10)
    pmr.save_plot(plot_dir, "pc_losses.pdf", n_hidden=None, add_suffix=False)


def plot_losses_and_energies(data: Dict[str, Any], plot_dir: str, colormap_name: str = "Blues", log_x_scale: bool = False) -> None:
    """Overlay PC energies (by width) and BP loss at max width (same style as plot_main_results)."""
    plt.figure(figsize=(12.5, 6))

    widths_with_energy = sorted([w for w in data["widths"] if w in data["pc_energies"]])
    max_width = max(data["widths"]) if data["widths"] else None

    pc_legend_color = "#4A90E2"
    bp_color = "#DC143C"

    # PC energies
    if widths_with_energy:
        blues_cmap = plt.get_cmap(colormap_name)
        n_widths = len(widths_with_energy)
        for idx, w in enumerate(widths_with_energy):
            y = np.array(data["pc_energies"][w]).flatten()
            x = np.arange(1, len(y) + 1)
            color = blues_cmap(pmr.get_color_val(idx, n_widths, colormap_name))
            plt.plot(x, y, "-", alpha=pmr.ALPHA, linewidth=pmr.LINE_WIDTH, color=color, label="")

    # BP loss at widest N (matches plot_main_results convention)
    if max_width is not None and max_width in data["bp_losses"]:
        y = np.array(data["bp_losses"][max_width]).flatten()
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, "-", color=bp_color, linewidth=pmr.LINE_WIDTH, alpha=pmr.ALPHA, label="")

    # Custom legend: PC/BP + widths
    legend_handles, legend_labels = [], []
    if widths_with_energy:
        legend_handles.append(plt.Line2D([0], [0], color=pc_legend_color, linewidth=pmr.LINE_WIDTH, alpha=pmr.ALPHA))
        legend_labels.append(r"$\mathcal{F}(\mathbf{z}_{T_{\text{max}}})$ (PC)")
    if max_width is not None and max_width in data["bp_losses"]:
        legend_handles.append(plt.Line2D([0], [0], color=bp_color, linewidth=pmr.LINE_WIDTH, alpha=pmr.ALPHA))
        legend_labels.append(r"$\mathcal{L}(\boldsymbol{\theta})$ (BP)")

    all_widths = sorted(set(widths_with_energy + ([max_width] if max_width is not None and max_width in data["bp_losses"] else [])))
    if all_widths:
        gray_cmap = plt.get_cmap("Greys")
        for idx, w in enumerate(all_widths):
            gray_val = 0.3 + (idx / max(len(all_widths) - 1, 1)) * 0.5 if len(all_widths) > 1 else 0.5
            legend_handles.append(plt.Line2D([0], [0], color=gray_cmap(gray_val), linewidth=pmr.LINE_WIDTH))
            legend_labels.append(f"$N = {w}$")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("$t$", fontsize=pmr.FONT_SIZES["label"], labelpad=pmr.LABEL_PAD)
    plt.ylabel(r"$l(\boldsymbol{\theta}_t)$", fontsize=pmr.FONT_SIZES["label"], labelpad=pmr.LABEL_PAD)
    if log_x_scale:
        plt.xscale("log", base=10)
    if legend_handles:
        plt.legend(handles=legend_handles, labels=legend_labels, fontsize=pmr.FONT_SIZES["legend"],
                   bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis="both", labelsize=pmr.FONT_SIZES["tick"])
    pmr.save_plot(plot_dir, "losses_and_energies.pdf", n_hidden=None, add_suffix=False)


def plot_cosine_similarity(data: Dict[str, Any], plot_dir: str, colormap_name: str = "viridis") -> None:
    """Cosine similarity curves vs t (same style as plot_main_results)."""
    plt.figure(figsize=pmr.FIG_SIZE)
    widths_list = sorted([w for w in data["widths"] if w in data["grad_cosine_similarities"]])
    if widths_list:
        cmap = plt.get_cmap(colormap_name)
        n_widths = len(widths_list)
        for idx, w in enumerate(widths_list):
            y = np.array(data["grad_cosine_similarities"][w]).flatten()
            x = np.arange(1, len(y) + 1)
            color = cmap(pmr.get_color_val(idx, n_widths, colormap_name))
            plt.plot(x, y, label=f"$N = {w}$", alpha=pmr.ALPHA, linewidth=pmr.LINE_WIDTH, color=color)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("$t$", fontsize=pmr.FONT_SIZES["label"], labelpad=pmr.LABEL_PAD)
    plt.ylabel(
        r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, \nabla_{\boldsymbol{\theta}} \mathcal{F}\right)$",
        fontsize=pmr.FONT_SIZES["label"],
        labelpad=pmr.LABEL_PAD,
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles=handles, labels=labels, fontsize=pmr.FONT_SIZES["legend"],
                   bbox_to_anchor=(1.0, 0.0), loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis="both", labelsize=pmr.FONT_SIZES["tick"])
    pmr.save_plot(plot_dir, "grads_cosine_similarities.pdf", n_hidden=None, add_suffix=False)


def plot_cosine_similarity_by_activity_lr(
    depth_to_activity_lr_to_values: Dict[int, Dict[float, np.ndarray]],
    plot_dir: str,
    colormap_name: str = "viridis",
    filename: str = "grads_cosine_similarities_by_activity_lr.pdf",
    use_colorbar: bool = True,
) -> None:
    """Plot cosine similarity vs t for smallest and largest depth, sweeping activity_lr.
    
    Uses green colorscale for shallow depth and orange for deep depth, with beta values
    varying the intensity within each colorscale. If use_colorbar is True, a greyscale
    colorbar on the right shows beta values. If False, the legend lists one entry per beta
    (with grey colorscale) and is placed outside the plot on the right.
    """
    # Increase figure width when using legend instead of colorbar to accommodate outside legend
    fig_width = pmr.FIG_SIZE[0] * 1.3 if not use_colorbar else pmr.FIG_SIZE[0]
    plt.figure(figsize=(fig_width, pmr.FIG_SIZE[1]))

    if not depth_to_activity_lr_to_values:
        return

    depths = sorted(depth_to_activity_lr_to_values.keys())
    all_activity_lrs = []  # Initialize for colorbar
    legend_handles = []  # Initialize for legend
    legend_labels = []  # Initialize for legend
    if len(depths) < 2:
        # Fallback: if only one depth, use original behavior
        if depths:
            activity_lr_to_values = depth_to_activity_lr_to_values[depths[0]]
            activity_lrs = sorted(activity_lr_to_values.keys())
            all_activity_lrs = activity_lrs  # Set for potential colorbar
            if activity_lrs:
                cmap = plt.get_cmap(colormap_name)
                n_series = len(activity_lrs)
                for idx, lr in enumerate(activity_lrs):
                    y = np.array(activity_lr_to_values[lr]).flatten()
                    x = np.arange(1, len(y) + 1)
                    color = cmap(pmr.get_color_val(idx, n_series, colormap_name))
                    plt.plot(x, y, label=rf"$\beta = {lr}$", alpha=pmr.ALPHA, linewidth=pmr.LINE_WIDTH, color=color)
    else:
        # Use smallest and largest depths
        min_depth = depths[0]
        max_depth = depths[-1]
        
        # Green colormap for shallow depth, orange for deep depth
        green_cmap = plt.get_cmap("Greens")
        orange_cmap = plt.get_cmap("Oranges")
        grey_cmap = plt.get_cmap("Greys")
        
        # Collect all activity_lrs across both depths
        all_activity_lrs = set()
        for depth in [min_depth, max_depth]:
            all_activity_lrs.update(depth_to_activity_lr_to_values[depth].keys())
        all_activity_lrs = sorted(all_activity_lrs)
        
        # Plot lines for min_depth (green/shallow) and max_depth (orange/deep)
        for depth, cmap_to_use in [(min_depth, green_cmap), (max_depth, orange_cmap)]:
            activity_lr_to_values = depth_to_activity_lr_to_values[depth]
            activity_lrs = sorted(activity_lr_to_values.keys())
            
            for lr in activity_lrs:
                if lr not in activity_lr_to_values:
                    continue
                y = np.array(activity_lr_to_values[lr]).flatten()
                x = np.arange(1, len(y) + 1)
                # Color intensity based on beta value position in all_activity_lrs
                # Use the depth colormap (green or orange) with beta variation
                beta_idx = all_activity_lrs.index(lr)
                color_val = 0.3 + 0.5 * (beta_idx / max(len(all_activity_lrs) - 1, 1)) if len(all_activity_lrs) > 1 else 0.5
                depth_color = cmap_to_use(color_val)
                plt.plot(x, y, alpha=pmr.ALPHA, linewidth=pmr.LINE_WIDTH, color=depth_color)

        # Create custom legend: depth entries with green/orange
        # Add depth entries (L = n_hidden + 1)
        if min_depth in depth_to_activity_lr_to_values:
            legend_handles.append(plt.Line2D([0], [0], color=green_cmap(0.7), linewidth=pmr.LINE_WIDTH, alpha=pmr.ALPHA))
            legend_labels.append(f"$L = {min_depth + 1}$")
        if max_depth in depth_to_activity_lr_to_values:
            legend_handles.append(plt.Line2D([0], [0], color=orange_cmap(0.7), linewidth=pmr.LINE_WIDTH, alpha=pmr.ALPHA))
            legend_labels.append(f"$L = {max_depth + 1}$")
        # If not using colorbar, add one legend entry per beta (grey colorscale)
        if not use_colorbar:
            for idx, lr in enumerate(all_activity_lrs):
                grey_val = 0.3 + (idx / max(len(all_activity_lrs) - 1, 1)) * 0.5 if len(all_activity_lrs) > 1 else 0.5
                legend_handles.append(plt.Line2D([0], [0], color=grey_cmap(grey_val), linewidth=pmr.LINE_WIDTH))
                legend_labels.append(rf"$\beta = {lr}$")

    ax = plt.gca()
    
    # Create colorbar for beta values on the right side (only when use_colorbar)
    if use_colorbar and all_activity_lrs:
        # Use greyscale colormap to represent beta values
        beta_cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=min(all_activity_lrs), vmax=max(all_activity_lrs))
        sm = ScalarMappable(cmap=beta_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$\beta$", fontsize=pmr.FONT_SIZES["label"], labelpad=pmr.LABEL_PAD)
        cbar.ax.tick_params(labelsize=pmr.FONT_SIZES["tick"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("$t$", fontsize=pmr.FONT_SIZES["label"], labelpad=pmr.LABEL_PAD)
    plt.ylabel(
        r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, \nabla_{\boldsymbol{\theta}} \mathcal{F}\right)$",
        fontsize=pmr.FONT_SIZES["label"],
        labelpad=pmr.LABEL_PAD,
    )
    if legend_handles:
        if use_colorbar:
            plt.legend(handles=legend_handles, labels=legend_labels, fontsize=pmr.FONT_SIZES["legend"], loc="lower right")
        else:
            plt.legend(
                handles=legend_handles,
                labels=legend_labels,
                fontsize=pmr.FONT_SIZES["legend"],
                bbox_to_anchor=(1.04, 0.5),
                loc="center left",
            )
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis="both", labelsize=pmr.FONT_SIZES["tick"])
    pmr.save_plot(plot_dir, filename, n_hidden=None, add_suffix=False)


def _format_float_dir_name(v: float) -> str:
    # Match common folder naming from f"{float}_activity_lr" in train.py (argparse -> float).
    # Using Python's default str() is the closest behavior.
    return str(float(v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--plot_dir", type=str, default="nonlinear_plots")

    # Dataset selection (mirrors plot_main_results)
    parser.add_argument("--datasets", type=str, nargs="+", default=["CIFAR10"])
    parser.add_argument("--dataset_input_dims", type=int, nargs="+", default=[3072])
    parser.add_argument("--n_samples", type=int, default=64)

    # Sweep axes requested by the user
    parser.add_argument("--n_hiddens", type=int, nargs="+", default=[1, 3, 7, 15])
    parser.add_argument("--activity_lrs", type=float, nargs="+", default=[1e-1, 5e-1, 1, 2, 5, 10, 20, 50])
    parser.add_argument("--use_skips", nargs="+", default=[False, True])
    parser.add_argument("--activity_lrs_beta_plot", type=float, nargs="+", default=[1e-1, 5e-1, 1, 5, 10, 20])

    # Fixed (or lightly varying) experiment params used to locate runs
    parser.add_argument("--act_fns", type=str, nargs="+", default=["tanh", "relu"])
    parser.add_argument("--param_types", type=str, nargs="+", default=["mupc"])
    parser.add_argument("--param_optim", type=str, default="gd")
    parser.add_argument("--param_lr", type=float, default=0.025)
    parser.add_argument("--gamma_0s", type=float, nargs="+", default=[1.0])
    parser.add_argument("--n_train_iters", type=int, default=100)
    parser.add_argument("--infer_mode", type=str, default="optim")

    # Plot controls
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128, 2048])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--colormap", type=str, default="Blues")
    parser.add_argument("--log_x_scale", action="store_true", default=False)
    parser.add_argument(
        "--no_colorbar",
        action="store_true",
        default=True,
        help="Use legend with one entry per beta (outside right) instead of a colorbar.",
    )

    args = parser.parse_args()

    if len(args.dataset_input_dims) != len(args.datasets):
        raise ValueError("Length of --dataset_input_dims must match --datasets")

    os.makedirs(args.plot_dir, exist_ok=True)

    # Normalize use_skips list (argparse may parse as strings depending on CLI usage)
    parsed_use_skips: List[bool] = []
    for v in args.use_skips:
        if isinstance(v, bool):
            parsed_use_skips.append(v)
        else:
            b = _parse_bool(v)
            if b is None:
                raise ValueError(f"Invalid --use_skips value: {v}. Use True/False.")
            parsed_use_skips.append(b)

    for dataset, input_dim in zip(args.datasets, args.dataset_input_dims):
        dataset_results_dir = os.path.join(args.results_dir, f"{input_dim}_input_dim")
        if not os.path.exists(dataset_results_dir):
            print(f"Warning: results directory not found: {dataset_results_dir}. Skipping {dataset}.")
            continue

        dataset_plot_dir = os.path.join(args.plot_dir, dataset)
        os.makedirs(dataset_plot_dir, exist_ok=True)

        for act_fn in args.act_fns:
            act_fn_plot_dir = os.path.join(dataset_plot_dir, f"{act_fn}_act_fn")
            os.makedirs(act_fn_plot_dir, exist_ok=True)

            for param_type in args.param_types:
                for use_skips in parsed_use_skips:
                    for gamma_0 in args.gamma_0s:
                        # Collect data across all n_hiddens for the beta plot
                        target_width: Optional[int] = max(args.widths) if args.widths else None
                        depth_to_cos_sims_by_lr: Dict[int, Dict[float, np.ndarray]] = {}
                        
                        activity_lrs_for_beta_plot = (
                            args.activity_lrs_beta_plot if args.activity_lrs_beta_plot is not None else args.activity_lrs
                        )
                        activity_lrs_for_per_lr_plots = args.activity_lrs
                        activity_lrs_union = sorted(
                            set(float(x) for x in (list(activity_lrs_for_per_lr_plots) + list(activity_lrs_for_beta_plot)))
                        )

                        # First pass: collect data for beta plot and create per-n_hidden plots
                        for n_hidden in args.n_hiddens:
                            n_hidden_dir = os.path.join(act_fn_plot_dir, f"{n_hidden}_n_hidden")
                            os.makedirs(n_hidden_dir, exist_ok=True)
                            
                            skip_dir = os.path.join(n_hidden_dir, f"{use_skips}_use_skips")
                            os.makedirs(skip_dir, exist_ok=True)
                            
                            gamma_dir = os.path.join(skip_dir, f"{gamma_0}_gamma_0")
                            os.makedirs(gamma_dir, exist_ok=True)
                            
                            cos_sims_by_lr_for_depth: Dict[float, np.ndarray] = {}

                            for activity_lr in activity_lrs_union:
                                # Keep directory naming consistent with training (float string)
                                act_lr_dir = os.path.join(gamma_dir, f"{_format_float_dir_name(activity_lr)}_activity_lr", param_type)
                                os.makedirs(act_lr_dir, exist_ok=True)

                                cfg = RunConfig(
                                    input_dim=input_dim,
                                    n_samples=args.n_samples,
                                    n_hidden=n_hidden,
                                    use_skips=use_skips,
                                    act_fn=act_fn,
                                    param_type=param_type,
                                    param_optim=args.param_optim,
                                    optim_id=args.param_optim,   # BP uses *_optim_id
                                    param_lr=args.param_lr,
                                    gamma_0=gamma_0,
                                    n_train_iters=args.n_train_iters,
                                    infer_mode=args.infer_mode,
                                    activity_lr=activity_lr,
                                    seed=args.seed,
                                )

                                data = load_nonlinear_data(dataset_results_dir, cfg, widths=args.widths)
                                if not data["widths"]:
                                    print(
                                        f"Warning: no PC runs found for dataset={dataset}, H={n_hidden}, "
                                        f"use_skips={use_skips}, activity_lr={activity_lr}, gamma_0={gamma_0}, param_type={param_type}."
                                    )
                                    continue

                                if float(activity_lr) in set(float(x) for x in activity_lrs_for_per_lr_plots):
                                    # Plots requested: losses_and_energies + cosine similarities
                                    plot_losses_and_energies(
                                        data, act_lr_dir, colormap_name=args.colormap, log_x_scale=args.log_x_scale
                                    )
                                    plot_cosine_similarity(data, act_lr_dir, colormap_name=args.colormap)

                                # Store cosine sims for largest width for the activity-lr sweep plot
                                if target_width is None and data["widths"]:
                                    target_width = max(data["widths"])
                                if target_width is not None:
                                    vals = data["grad_cosine_similarities"].get(target_width)
                                    if vals is not None and float(activity_lr) in set(float(x) for x in activity_lrs_for_beta_plot):
                                        cos_sims_by_lr_for_depth[float(activity_lr)] = np.array(vals).flatten()
                            
                            if cos_sims_by_lr_for_depth:
                                depth_to_cos_sims_by_lr[n_hidden] = cos_sims_by_lr_for_depth
                        
                        # Create the beta plot with smallest and largest depths
                        if depth_to_cos_sims_by_lr:
                            # Save at act_fn level (common across all n_hiddens)
                            skip_dir = os.path.join(act_fn_plot_dir, f"{use_skips}_use_skips")
                            gamma_dir = os.path.join(skip_dir, f"{gamma_0}_gamma_0")
                            gamma_param_dir = os.path.join(gamma_dir, param_type)
                            os.makedirs(gamma_param_dir, exist_ok=True)
                            
                            plot_cosine_similarity_by_activity_lr(
                                depth_to_cos_sims_by_lr,
                                gamma_param_dir,
                                colormap_name=args.colormap,
                                filename="grads_cosine_similarities_by_activity_lr_and_depth.pdf",
                                use_colorbar=not args.no_colorbar,
                            )

    print("Done.")
