import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"
})

FIG_SIZE = (8, 6)
FONT_SIZES = {"label": 45, "legend": 25, "tick": 35}
LABEL_PAD = 15
LINE_WIDTH = 4
ALPHA = 0.7


def _is_sequential_colormap(colormap_name):
    """Check if a colormap is sequential (aligned with plot_toy_results.py)."""
    sequential = {
        "viridis", "plasma", "inferno", "magma", "cividis",
        "Reds", "Blues", "Greens", "Oranges", "Purples", "Greys",
        "YlOrRd", "YlOrBr", "YlGnBu", "YlGn", "RdPu",
        "BuGn", "BuPu", "GnBu", "PuBu", "PuBuGn", "PuRd", "OrRd",
        "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "cool", "hot",
        "copper", "bone", "pink", "spring", "summer", "autumn", "winter",
    }
    return colormap_name in sequential


def _get_color_val(idx, n_widths, colormap_name="viridis"):
    """Get color value for colormap (aligned with plot_toy_results.py)."""
    if _is_sequential_colormap(colormap_name):
        return 0.15 + (idx / max(n_widths - 1, 1)) * 0.85 if n_widths > 1 else 0.15
    return (idx / max(n_widths - 1, 1)) * 0.85 if n_widths > 1 else 0


def _layer_cos_sim_display_name(layer_name):
    """Canonical display name for a CNN layer: Conv1, Conv2, Conv3, Readout, etc."""
    raw = layer_name or "Layer"
    name = str(raw).strip().lower()

    # Map generic 'Layer 0/1/2/3' style names onto Conv1/2/3/Readout
    if name.startswith("layer"):
        digits = "".join(c for c in name if c.isdigit())
        try:
            idx = int(digits)
        except ValueError:
            idx = None
        layer_map = {0: "Conv1", 1: "Conv2", 2: "Conv3", 3: "Readout"}
        if idx in layer_map:
            return layer_map[idx]

    # Handle stage-style names from the CNN runs, e.g. 'stage1_conv', 'stage2_conv', 'stage3_conv'
    if name.startswith("stage") and "conv" in name:
        digits = "".join(c for c in name if c.isdigit())
        return f"Conv{digits or '1'}"

    # Handle simpler conv names, e.g. 'conv1', 'conv2'
    if name.startswith("conv"):
        digits = "".join(c for c in name if c.isdigit())
        return f"Conv{digits or '1'}"

    if name == "readout":
        return "Readout"

    return raw[:1].upper() + raw[1:]


def _layer_cos_sim_ylabel(layer_name):
    """Format layer name for y-axis using canonical display name."""
    return f"Grad cos. {_layer_cos_sim_display_name(layer_name)}"


def _legend_n_label(width):
    """Legend label for width: display total N as 9 * width (CNN channel count)."""
    n = 9 * width
    n_display = int(n) if n == int(n) else n
    return f"$N = {n_display}$"


def _setup_plot(xlabel, ylabel, log_scale=False):
    """Setup common plot styling (aligned with plot_toy_results.py)."""
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel(xlabel, fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    plt.ylabel(ylabel, fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=FONT_SIZES["legend"])
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    if log_scale:
        plt.yscale("log", base=10)


def _run_sibling_path(run_dir, name):
    """Path to sibling .npy file saved by test_theory (e.g. .../0_rescalings.npy)."""
    return f"{run_dir}_{name}.npy"


def _parse_activity_lr_from_path(path):
    """Extract activity_lr from path segment '*_activity_lr' (e.g. 0.5_activity_lr). Returns float or None."""
    for part in path.split(os.sep):
        if part.endswith("_activity_lr"):
            try:
                return float(part.replace("_activity_lr", ""))
            except ValueError:
                pass
    return None


def _parse_n_infer_iters_from_path(path):
    """Extract n_infer_iters from path segment '*_n_infer_iters' (e.g. 10_n_infer_iters). Returns int or None."""
    for part in path.split(os.sep):
        if part.endswith("_n_infer_iters"):
            try:
                return int(part.replace("_n_infer_iters", ""))
            except ValueError:
                pass
    return None


def _parse_use_amortiser_from_path(path):
    """Extract use_amortiser from path segment 'use_amortiser_True' or 'use_amortiser_False'. Returns bool or None."""
    for part in path.split(os.sep):
        if part == "use_amortiser_True":
            return True
        if part == "use_amortiser_False":
            return False
    return None


def _seed_from_run_dir(run_dir):
    """Extract seed index from run_dir (e.g. .../use_amortiser_False/2 -> 2)."""
    base = os.path.basename(os.path.normpath(run_dir))
    if base.isdigit():
        return int(base)
    if base.startswith("seed_"):
        try:
            return int(base.replace("seed_", ""))
        except ValueError:
            pass
    return None


def _filter_width_runs_by_seed(width_runs, seed):
    """Keep only run dirs matching seed. Raises if any width has no matching run."""
    filtered = []
    missing = []
    for width, run_dirs in width_runs:
        matched = [d for d in run_dirs if _seed_from_run_dir(d) == seed]
        if not matched:
            missing.append(width)
        else:
            filtered.append((width, matched))
    if missing:
        available = ", ".join(
            f"N={w}: {sorted({_seed_from_run_dir(d) for d in ds})}"
            for w, ds in width_runs
        )
        raise FileNotFoundError(
            f"No run found for --seed {seed} at widths {missing}. Available seeds: {available}."
        )
    return filtered


def discover_width_run_dirs(results_dir):
    """Discover all (width, run_dir) under results_dir where run_dir contains steps.npy.
    test_theory saves steps.npy inside the run dir and *_rescalings.npy etc. as siblings.
    Extracts width from '*_width', activity_lr from '*_activity_lr', n_infer_iters from '*_n_infer_iters',
    use_amortiser from 'use_amortiser_True' / 'use_amortiser_False'.
    Returns a dict: (activity_lr, n_infer_iters, use_amortiser) -> [(width, run_dir), ...], with None for missing path segments.
    """
    found = []  # (width, activity_lr, n_infer_iters, use_amortiser, run_dir)
    for root, _, files in os.walk(results_dir):
        if "steps.npy" not in files:
            continue
        width = None
        for part in root.split(os.sep):
            if part.endswith("_width"):
                try:
                    width = int(part.replace("_width", ""))
                    break
                except ValueError:
                    pass
        if width is None:
            continue
        activity_lr = _parse_activity_lr_from_path(root)
        n_infer_iters = _parse_n_infer_iters_from_path(root)
        use_amortiser = _parse_use_amortiser_from_path(root)
        found.append((width, activity_lr, n_infer_iters, use_amortiser, root))

    # Group by (activity_lr, n_infer_iters, use_amortiser); within each group keep *all* run_dirs per width.
    # We interpret multiple run_dirs for the same width as different seeds/replications to be averaged later.
    by_group = {}  # (activity_lr, n_infer_iters, use_amortiser) -> {width: [run_dir, ...]}
    for w, alr, n_infer, use_am, d in found:
        key = (alr, n_infer, use_am)
        by_group.setdefault(key, {}).setdefault(w, []).append(d)

    # Convert to (activity_lr, n_infer_iters, use_amortiser) -> [(width, [run_dir, ...]), ...] sorted by width
    return {
        key: [(w, by_group[key][w]) for w in sorted(by_group[key].keys())]
        for key in sorted(
            by_group.keys(),
            key=lambda k: (
                k[0] is None,
                k[0] or 0,
                k[1] is None,
                k[1] if k[1] is not None else 0,
                k[2] is None,
                k[2] if k[2] is not None else False,
            ),
        )
    }


def _file_suffix(activity_lr, n_infer_iters, use_amortiser, n_activity_lr_groups, n_infer_iters_groups, n_use_amortiser_groups):
    """Filename suffix for per-activity_lr, per-n_infer_iters and per-use_amortiser outputs when multiple groups exist."""
    parts = []
    if n_activity_lr_groups > 1:
        if activity_lr is None:
            parts.append("_no_activity_lr")
        else:
            parts.append(f"_{activity_lr}_activity_lr")
    if n_infer_iters_groups > 1 and n_infer_iters is not None:
        parts.append(f"_{n_infer_iters}_n_infer_iters")
    if n_use_amortiser_groups > 1 and use_amortiser is not None:
        parts.append("_amortiser" if use_amortiser else "_no_amortiser")
    return "".join(parts)


def main(args):
    run_dir = args.run_dir
    colormap_name = getattr(args, "colormap", "Blues") or "Blues"
    plot_widths = getattr(args, "plot_widths", None)
    plot_widths_set = set(plot_widths) if plot_widths else None
    selected_seed = getattr(args, "seed", None)

    # Discover all runs: group by (activity_lr, n_infer_iters) -> [(width, run_dir), ...]
    activity_lr_groups = discover_width_run_dirs(run_dir)
    if selected_seed is not None:
        filtered_groups = {}
        for key, width_runs in activity_lr_groups.items():
            filtered_groups[key] = (
                _filter_width_runs_by_seed(width_runs, selected_seed) if width_runs else width_runs
            )
        activity_lr_groups = filtered_groups
    if not activity_lr_groups:
        # No runs found under run_dir; treat run_dir as single run (backward compat)
        steps_path = os.path.join(run_dir, "steps.npy")
        if not os.path.isfile(steps_path):
            raise FileNotFoundError(
                f"No run data found in {run_dir!r}: missing {steps_path}. "
                "Pass either (1) a run directory that contains steps.npy (and sibling *_rescalings.npy etc.), "
                "or (2) the base results directory (e.g. theory_results) under which test_theory.py "
                "saved runs (subdirs with *_width and steps.npy)."
            )
        activity_lr_groups = {(None, None, None): []}
        if selected_seed is not None:
            dir_seed = _seed_from_run_dir(run_dir)
            if dir_seed is not None and dir_seed != selected_seed:
                raise FileNotFoundError(
                    f"--seed {selected_seed} does not match run_dir seed {dir_seed} ({run_dir!r})."
                )

    os.makedirs(args.output_dir, exist_ok=True)

    def aggregate_seed_curves(curves):
        """Aggregate a list of 1D arrays (different seeds) into mean and SEM.

        Curves can have different lengths; we truncate to the minimum available length.
        Returns (mean, sem, n_seeds, n_time).
        """
        if not curves:
            return None, None, 0, 0
        curves = [np.asarray(c).flatten() for c in curves if c is not None]
        if not curves:
            return None, None, 0, 0
        n_time = min(len(c) for c in curves)
        if n_time <= 0:
            return None, None, 0, 0
        stacked = np.stack([c[:n_time] for c in curves], axis=0)  # (n_seeds, n_time)
        mean = stacked.mean(axis=0)
        if stacked.shape[0] <= 1:
            sem = np.zeros_like(mean)
        else:
            sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        return mean, sem, stacked.shape[0], n_time

    def aggregate_seed_arrays_2d(arrays_2d):
        """Aggregate list of (T, D) arrays into mean/SEM over seeds, truncating to min T.

        Returns (mean, sem, n_seeds, n_time) with shape (T, D) for mean/sem.
        """
        if not arrays_2d:
            return None, None, 0, 0
        arrays_2d = [np.asarray(a) for a in arrays_2d if a is not None]
        if not arrays_2d:
            return None, None, 0, 0
        n_time = min(a.shape[0] for a in arrays_2d)
        if n_time <= 0:
            return None, None, 0, 0
        # Ensure consistent feature dimension by truncating to min D if needed
        n_dim = min(a.shape[1] if a.ndim > 1 else 1 for a in arrays_2d)
        stacked = np.stack(
            [(a[:n_time, :n_dim] if a.ndim > 1 else a[:n_time].reshape(-1, 1)) for a in arrays_2d],
            axis=0,
        )  # (n_seeds, n_time, n_dim)
        mean = stacked.mean(axis=0)
        if stacked.shape[0] <= 1:
            sem = np.zeros_like(mean)
        else:
            sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        return mean, sem, stacked.shape[0], n_time

    def plot_mean_with_sem(x, mean, sem, *, color, label=None, linestyle="-", alpha_line=ALPHA, alpha_band=0.18):
        """Line plot of mean with SEM band."""
        plt.plot(x, mean, linestyle, color=color, linewidth=LINE_WIDTH, alpha=alpha_line, label=label)
        if sem is not None and np.any(np.isfinite(sem)) and np.max(sem) > 0:
            plt.fill_between(x, mean - sem, mean + sem, color=color, alpha=alpha_band, linewidth=0)

    def load_numerical_energy(run_dir):
        """Load per-step numerical/experimental energy curve from a run dir.

        Supports both older sibling naming (*_experiment_energies.npy) and the CNN test_theory
        naming (numerical_energies.npy inside the run dir).
        """
        # Preferred: sibling file saved by some runs
        p = _run_sibling_path(run_dir, "experiment_energies")
        if os.path.isfile(p):
            return np.load(p).flatten()
        # CNN test_theory convention
        p = os.path.join(run_dir, "numerical_energies.npy")
        if os.path.isfile(p):
            return np.load(p).flatten()
        return None

    def load_theory_energy(run_dir):
        """Load per-step theory energy curve from a run dir (if available)."""
        p = _run_sibling_path(run_dir, "theory_energies")
        if os.path.isfile(p):
            return np.load(p).flatten()
        p = os.path.join(run_dir, "theory_energies.npy")
        if os.path.isfile(p):
            return np.load(p).flatten()
        return None

    def load_bp_losses(run_dir):
        """Load BP losses from a run dir (if available)."""
        p = os.path.join(run_dir, "bp_losses.npy")
        if os.path.isfile(p):
            return np.load(p).flatten()
        p = _run_sibling_path(run_dir, "bp_losses")
        if os.path.isfile(p):
            return np.load(p).flatten()
        return None

    def load_cos_sims(run_dir):
        path_overall = os.path.join(run_dir, "grad_cosine_similarities.npy")
        path_per_layer = os.path.join(run_dir, "grad_cosine_similarities_per_layer.npy")
        path_overall_theory_bp = os.path.join(run_dir, "grad_cosine_similarities_theory_bp.npy")
        path_per_layer_theory_bp = os.path.join(run_dir, "grad_cosine_similarities_theory_bp_per_layer.npy")
        path_names = os.path.join(run_dir, "grad_cosine_similarities_layer_names.npy")

        names = np.load(path_names, allow_pickle=True) if os.path.isfile(path_names) else None
        if names is not None and isinstance(names, np.ndarray):
            names = list(names)
        if not names:
            names = ["conv1", "conv2", "readout"]

        overall = None
        per_layer = None
        if os.path.isfile(path_overall) and os.path.isfile(path_per_layer):
            overall = np.load(path_overall).flatten()
            per_layer = np.load(path_per_layer)  # (n_steps, n_layers)

        overall_theory_bp = None
        per_layer_theory_bp = None
        if os.path.isfile(path_overall_theory_bp) and os.path.isfile(path_per_layer_theory_bp):
            overall_theory_bp = np.load(path_overall_theory_bp).flatten()
            per_layer_theory_bp = np.load(path_per_layer_theory_bp)

        if overall is None and overall_theory_bp is None:
            return None, None, names, None, None
        return overall, per_layer, names, overall_theory_bp, per_layer_theory_bp

    n_groups = len(activity_lr_groups)
    n_activity_lr_groups = len(set(k[0] for k in activity_lr_groups.keys()))
    n_infer_iters_groups = len(set(k[1] for k in activity_lr_groups.keys()))
    n_use_amortiser_groups = len(set(k[2] for k in activity_lr_groups.keys()))

    for (activity_lr, n_infer_iters, use_amortiser), width_runs in activity_lr_groups.items():
        if plot_widths_set is not None and width_runs:
            width_runs = [(w, ds) for (w, ds) in width_runs if w in plot_widths_set]
            if not width_runs:
                available = sorted({w for (w, _) in activity_lr_groups[(activity_lr, n_infer_iters, use_amortiser)]})
                raise ValueError(
                    f"--plot_widths={sorted(plot_widths_set)} selected no discovered widths. "
                    f"Available widths for this group: {available}. "
                    "Either omit --plot_widths to include all discovered widths, or pass one of the available widths."
                )
        suffix = _file_suffix(activity_lr, n_infer_iters, use_amortiser, n_activity_lr_groups, n_infer_iters_groups, n_use_amortiser_groups)
        if selected_seed is not None:
            suffix = f"{suffix}_seed_{selected_seed}"
        use_discovered_widths = len(width_runs) > 1 or (selected_seed is not None and bool(width_runs))
        if width_runs:
            data_dir = width_runs[0][1][0]
        else:
            data_dir = run_dir

        steps = np.load(os.path.join(data_dir, "steps.npy"))
        out_features_path = _run_sibling_path(data_dir, "out_features")
        out_features = int(np.load(out_features_path)) if os.path.isfile(out_features_path) else 1

        if not use_discovered_widths:
            # Some runs/scripts may not produce theory energies; handle that gracefully.
            te_path = _run_sibling_path(data_dir, "theory_energies")
            ee_path = _run_sibling_path(data_dir, "experiment_energies")
            r_path = _run_sibling_path(data_dir, "rescalings")

            theory_energies = np.load(te_path) if os.path.isfile(te_path) else None
            experiment_energies = np.load(ee_path) if os.path.isfile(ee_path) else None
            rescalings = np.load(r_path) if os.path.isfile(r_path) else None

        widths_path = os.path.join(data_dir, "widths.npy")
        if os.path.isfile(widths_path) and not use_discovered_widths:
            widths = list(np.load(widths_path))
            if plot_widths_set is not None:
                keep_idx = [i for i, w in enumerate(widths) if int(w) in plot_widths_set]
                if not keep_idx:
                    raise ValueError(
                        f"--plot_widths={sorted(plot_widths_set)} selected no widths from widths.npy={widths}."
                    )
                widths = [widths[i] for i in keep_idx]
            n_widths = len(widths)
            colors = plt.get_cmap(colormap_name)(np.linspace(0.2, 0.9, n_widths))
            has_widths = True
        elif use_discovered_widths:
            widths = [w for w, _ in width_runs]
            n_widths = len(widths)
            colors = [plt.get_cmap(colormap_name)(_get_color_val(i, n_widths, colormap_name)) for i in range(n_widths)]
            has_widths = True
        else:
            widths = [None]
            n_widths = 1
            colors = ["#4A90E2"]
            has_widths = False

        def get_curve(arr, i):
            return arr[i] if arr.ndim > 1 else arr

        def filter_width_axis0(arr):
            """If arr has a leading width axis, filter it to keep_idx."""
            if plot_widths_set is None:
                return arr
            if arr is None:
                return None
            if not os.path.isfile(widths_path) or use_discovered_widths:
                return arr
            if arr.ndim <= 1:
                return arr
            return arr[keep_idx]

        # Energy plot: numerical vs theory for all widths in one figure
        plt.figure(figsize=FIG_SIZE)
        colormap = plt.get_cmap(colormap_name)
        if use_discovered_widths:
            n_widths = len(width_runs)
            for idx, (width, w_run_dirs) in enumerate(width_runs):
                exp_curves = []
                th_curves = []
                steps_list = []
                for w_run_dir in w_run_dirs:
                    exp = load_numerical_energy(w_run_dir)
                    if exp is None:
                        continue
                    exp_curves.append(exp)
                    steps_list.append(np.load(os.path.join(w_run_dir, "steps.npy")).flatten())
                    th = load_theory_energy(w_run_dir)
                    if th is not None:
                        th_curves.append(th)
                if not exp_curves:
                    continue
                exp_mean, exp_sem, _, n_t = aggregate_seed_curves(exp_curves)
                steps_w = min(steps_list, key=len)[:n_t] if steps_list else np.arange(n_t)
                color = colormap(_get_color_val(idx, n_widths, colormap_name))
                lbl = _legend_n_label(width)
                plot_mean_with_sem(steps_w, exp_mean, exp_sem, color=color, label=lbl, linestyle="-", alpha_band=0.14)
                if th_curves:
                    th_mean, th_sem, _, n_t_th = aggregate_seed_curves(th_curves)
                    steps_th = steps_w[: min(len(steps_w), n_t_th)]
                    plot_mean_with_sem(steps_th, th_mean[: len(steps_th)], th_sem[: len(steps_th)], color=color, label=None, linestyle="--", alpha_line=0.8, alpha_band=0.10)
            if width_runs:
                _setup_plot("$t$", r"$\mathcal{F}(\boldsymbol{\theta}_t)$", log_scale=True)
                ax = plt.gca()
                formatter = ax.yaxis.get_major_formatter()
                if hasattr(formatter, "set_useOffset"):
                    formatter.set_useOffset(False)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"theory_vs_numerical_energy{suffix}.pdf"), bbox_inches="tight")
                plt.close()
        else:
            plt.figure(figsize=FIG_SIZE)
            has_theory = theory_energies is not None
            if experiment_energies is not None:
                theory_energies_f = filter_width_axis0(theory_energies)
                experiment_energies_f = filter_width_axis0(experiment_energies)
                for i in range(n_widths):
                    ee = get_curve(experiment_energies_f, i)
                    w = widths[i]
                    lbl = _legend_n_label(w) if has_widths else "Numerical"
                    plt.plot(steps, ee, "-", color=colors[i], linewidth=LINE_WIDTH, alpha=ALPHA, label=lbl)
                    if theory_energies_f is not None:
                        te = get_curve(theory_energies_f, i)
                        plt.plot(steps, te, "--", color=colors[i] if has_widths else "black", linewidth=LINE_WIDTH, alpha=0.8)
                _setup_plot("$t$", r"$\mathcal{F}(\boldsymbol{\theta}_t)$", log_scale=True)
                ax = plt.gca()
                formatter = ax.yaxis.get_major_formatter()
                if hasattr(formatter, "set_useOffset"):
                    formatter.set_useOffset(False)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"theory_vs_numerical_energy{suffix}.pdf"), bbox_inches="tight")
                plt.close()

        # PC (numerical) energy vs t only (all widths)
        plt.figure(figsize=FIG_SIZE)
        colormap = plt.get_cmap(colormap_name)
        if use_discovered_widths:
            n_widths_energy = len(width_runs)
            for idx, (width, w_run_dirs) in enumerate(width_runs):
                exp_curves = []
                steps_list = []
                for w_run_dir in w_run_dirs:
                    exp = load_numerical_energy(w_run_dir)
                    if exp is None:
                        continue
                    exp_curves.append(exp)
                    steps_list.append(np.load(os.path.join(w_run_dir, "steps.npy")).flatten())
                if not exp_curves:
                    continue
                exp_mean, exp_sem, _, n_t = aggregate_seed_curves(exp_curves)
                steps_w = min(steps_list, key=len)[:n_t] if steps_list else np.arange(n_t)
                color = colormap(_get_color_val(idx, n_widths_energy, colormap_name))
                plot_mean_with_sem(
                    steps_w, exp_mean, exp_sem,
                    color=color, label=_legend_n_label(width), linestyle="-", alpha_band=0.14,
                )
        else:
            exp = load_numerical_energy(data_dir)
            if experiment_energies is not None:
                experiment_energies_f = filter_width_axis0(experiment_energies)
                for i in range(n_widths):
                    ee = get_curve(experiment_energies_f, i)
                    w = widths[i]
                    lbl = _legend_n_label(w) if has_widths else "PC energy"
                    plt.plot(steps, ee, "-", color=colors[i], linewidth=LINE_WIDTH, alpha=ALPHA, label=lbl)
            elif exp is not None:
                n = min(len(steps), len(exp))
                plt.plot(
                    steps[:n], exp[:n], "-", color="#4A90E2",
                    linewidth=LINE_WIDTH, alpha=ALPHA, label="PC energy",
                )
        _setup_plot("$t$", r"$\mathcal{F}(\boldsymbol{\theta}_t)$", log_scale=True)
        ax = plt.gca()
        formatter = ax.yaxis.get_major_formatter()
        if hasattr(formatter, "set_useOffset"):
            formatter.set_useOffset(False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"experiment_energy{suffix}.pdf"), bbox_inches="tight")
        plt.close()

        # BP loss vs t only (all widths, if present)
        plt.figure(figsize=FIG_SIZE)
        if use_discovered_widths:
            n_widths_bp = len(width_runs)
            for idx, (width, w_run_dirs) in enumerate(width_runs):
                bp_curves = []
                steps_list = []
                for w_run_dir in w_run_dirs:
                    bp = load_bp_losses(w_run_dir)
                    if bp is None:
                        continue
                    bp_curves.append(bp)
                    steps_list.append(np.load(os.path.join(w_run_dir, "steps.npy")).flatten())
                if not bp_curves:
                    continue
                bp_mean, bp_sem, _, n_t = aggregate_seed_curves(bp_curves)
                steps_w = min(steps_list, key=len)[:n_t] if steps_list else np.arange(n_t)
                color = colormap(_get_color_val(idx, n_widths_bp, colormap_name))
                plot_mean_with_sem(
                    steps_w, bp_mean, bp_sem,
                    color=color, label=_legend_n_label(width), linestyle="-", alpha_band=0.14,
                )
        else:
            bp = load_bp_losses(data_dir)
            if bp is not None:
                n = min(len(steps), len(bp))
                plt.plot(
                    steps[:n], bp[:n], "-", color="#4A90E2",
                    linewidth=LINE_WIDTH, alpha=ALPHA, label="BP loss",
                )
        _setup_plot("$t$", r"$\mathcal{L}(\boldsymbol{\theta}_t)$", log_scale=True)
        ax = plt.gca()
        formatter = ax.yaxis.get_major_formatter()
        if hasattr(formatter, "set_useOffset"):
            formatter.set_useOffset(False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"bp_loss{suffix}.pdf"), bbox_inches="tight")
        plt.close()

        # Losses & energies overlay (toy-results style)
        # - Energies (numerical/PC) for all widths in Blues (no per-line labels)
        # - BP loss for the widest width in a single red curve
        # - Custom legend: PC, BP, then grayscale width markers
        if use_discovered_widths and width_runs:
            plt.figure(figsize=(12.5, 6))
            blues_cmap = plt.get_cmap("Blues")
            gray_cmap = plt.get_cmap("Greys")
            pc_legend_color = "#4A90E2"
            bp_color = "#DC143C"

            widths_present = []
            max_width = max(w for (w, _) in width_runs)

            # Plot PC/numerical energies (mean +/- SEM band) for all widths
            for idx, (width, w_run_dirs) in enumerate(width_runs):
                exp_curves = []
                steps_list = []
                for w_run_dir in w_run_dirs:
                    exp = load_numerical_energy(w_run_dir)
                    if exp is None:
                        continue
                    exp_curves.append(exp)
                    steps_list.append(np.load(os.path.join(w_run_dir, "steps.npy")).flatten())
                if not exp_curves:
                    continue
                mean, sem, _, n_t = aggregate_seed_curves(exp_curves)
                steps_w = min(steps_list, key=len)[:n_t] if steps_list else np.arange(n_t) + 1
                color = blues_cmap(_get_color_val(idx, len(width_runs), "Blues"))
                plot_mean_with_sem(steps_w, mean, sem, color=color, label=None, linestyle="-", alpha_band=0.10)
                widths_present.append(width)

            # Plot BP loss (widest width only, mean +/- SEM band)
            bp_mean = bp_sem = bp_steps = None
            for width, w_run_dirs in width_runs:
                if width != max_width:
                    continue
                bp_curves = []
                steps_list = []
                for w_run_dir in w_run_dirs:
                    bp = load_bp_losses(w_run_dir)
                    if bp is None:
                        continue
                    bp_curves.append(bp)
                    steps_list.append(np.load(os.path.join(w_run_dir, "steps.npy")).flatten())
                if bp_curves:
                    bp_mean, bp_sem, _, n_t = aggregate_seed_curves(bp_curves)
                    bp_steps = min(steps_list, key=len)[:n_t] if steps_list else np.arange(n_t) + 1
                    plot_mean_with_sem(bp_steps, bp_mean, bp_sem, color=bp_color, label=None, linestyle="-", alpha_line=ALPHA, alpha_band=0.10)
                    widths_present.append(max_width)

            # Build custom legend (same idea as plot_toy_results.py)
            legend_handles = []
            legend_labels = []
            if widths_present:
                legend_handles.append(plt.Line2D([0], [0], color=pc_legend_color, linewidth=LINE_WIDTH, alpha=ALPHA))
                legend_labels.append(r"$\mathcal{F}^*(\boldsymbol{\theta})$ (PC)")
            if bp_mean is not None:
                legend_handles.append(plt.Line2D([0], [0], color=bp_color, linewidth=LINE_WIDTH, alpha=ALPHA))
                legend_labels.append(r"$\mathcal{L}(\boldsymbol{\theta})$ (BP)")

            all_widths = sorted(set(widths_present))
            n_all_widths = len(all_widths)
            for i, w in enumerate(all_widths):
                gray_val = 0.3 + (i / max(n_all_widths - 1, 1)) * 0.5 if n_all_widths > 1 else 0.5
                legend_handles.append(plt.Line2D([0], [0], color=gray_cmap(gray_val), linewidth=LINE_WIDTH))
                legend_labels.append(_legend_n_label(w))

            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.xlabel("$t$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
            plt.ylabel(r"$l(\boldsymbol{\theta}_t)$", fontsize=FONT_SIZES["label"], labelpad=LABEL_PAD)
            if legend_handles:
                plt.legend(handles=legend_handles, labels=legend_labels, fontsize=FONT_SIZES["legend"], bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, which="both", ls="-", alpha=0.4)
            plt.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"losses_and_energies{suffix}.pdf"), bbox_inches="tight")
            plt.close()

        # Energy delta plot (only when theory energies are available)
        if use_discovered_widths:
            plt.figure(figsize=FIG_SIZE)
            colormap = plt.get_cmap(colormap_name)
            n_widths = len(width_runs)
            any_delta_plotted = False
            for idx, (width, w_run_dirs) in enumerate(width_runs):
                delta_curves = []
                steps_list = []
                for w_run_dir in w_run_dirs:
                    th = load_theory_energy(w_run_dir)
                    ex = load_numerical_energy(w_run_dir)
                    if th is None or ex is None:
                        continue
                    n = min(len(th), len(ex))
                    delta_curves.append(ex[:n] - th[:n])
                    steps_list.append(np.load(os.path.join(w_run_dir, "steps.npy")).flatten()[:n])
                if not delta_curves:
                    continue
                delta_mean, delta_sem, _, n_t = aggregate_seed_curves(delta_curves)
                steps_w = min(steps_list, key=len)[:n_t] if steps_list else np.arange(n_t)
                color = colormap(_get_color_val(idx, n_widths, colormap_name))
                lbl = _legend_n_label(width)
                plot_mean_with_sem(steps_w, delta_mean, delta_sem, color=color, label=lbl, linestyle="-", alpha_band=0.14)
                any_delta_plotted = True
            if any_delta_plotted:
                _setup_plot(
                    "$t$",
                    r"$\Delta\mathcal{F}(\boldsymbol{\theta}_t)$",
                    log_scale=True,
                )
                ax = plt.gca()
                formatter = ax.yaxis.get_major_formatter()
                if hasattr(formatter, "set_useOffset"):
                    formatter.set_useOffset(False)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"theory_vs_numerical_energy_delta{suffix}.pdf"), bbox_inches="tight")
                plt.close()
        else:
            if theory_energies is not None and experiment_energies is not None:
                plt.figure(figsize=FIG_SIZE)
                theory_energies_f = filter_width_axis0(theory_energies)
                experiment_energies_f = filter_width_axis0(experiment_energies)
                for i in range(n_widths):
                    te = get_curve(theory_energies_f, i)
                    ee = get_curve(experiment_energies_f, i)
                    w = widths[i]
                    delta = ee - te
                    lbl = _legend_n_label(w) if has_widths else "Delta"
                    plt.plot(steps, delta, "-", color=colors[i], linewidth=LINE_WIDTH, alpha=ALPHA, label=lbl)
                _setup_plot(
                    "$t$",
                    r"$\Delta\mathcal{F}(\boldsymbol{\theta}_t)$",
                    log_scale=True,
                )
                ax = plt.gca()
                formatter = ax.yaxis.get_major_formatter()
                if hasattr(formatter, "set_useOffset"):
                    formatter.set_useOffset(False)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"theory_vs_numerical_energy_delta{suffix}.pdf"), bbox_inches="tight")
                plt.close()

        # Energy rescaling
        rescaling_ylabel = (
            r"$\|\mathbf{S}(\boldsymbol{\theta}_t)\|_F$"
            if out_features > 1
            else r"$s(\boldsymbol{\theta}_t)$"
        )
        plt.figure(figsize=FIG_SIZE)
        if use_discovered_widths:
            colormap = plt.get_cmap(colormap_name)
            any_rescaling_plotted = False
            for idx, (width, w_run_dirs) in enumerate(width_runs):
                r_curves = []
                steps_list = []
                for w_run_dir in w_run_dirs:
                    r_path = _run_sibling_path(w_run_dir, "rescalings")
                    if not os.path.isfile(r_path):
                        continue
                    r_curves.append(np.load(r_path).flatten())
                    steps_list.append(np.load(os.path.join(w_run_dir, "steps.npy")).flatten())
                if not r_curves:
                    continue
                r_mean, r_sem, _, n_t = aggregate_seed_curves(r_curves)
                steps_w = min(steps_list, key=len)[:n_t] if steps_list else np.arange(n_t)
                color = colormap(_get_color_val(idx, len(width_runs), colormap_name))
                plot_mean_with_sem(steps_w, r_mean, r_sem, color=color, label=_legend_n_label(width), linestyle="-", alpha_band=0.14)
                any_rescaling_plotted = True
            if any_rescaling_plotted:
                _setup_plot("$t$", rescaling_ylabel, log_scale=False)
                ax = plt.gca()
                ax.yaxis.get_major_formatter().set_useOffset(False)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"theory_rescaling{suffix}.pdf"), bbox_inches="tight")
                plt.close()
            else:
                plt.close()
        else:
            if rescalings is not None:
                rescalings_f = filter_width_axis0(rescalings)
                for i in range(n_widths):
                    r = get_curve(rescalings_f, i)
                    w = widths[i]
                    lbl = _legend_n_label(w) if has_widths else None
                    plt.plot(steps, r, "-", color=colors[i], linewidth=LINE_WIDTH, alpha=ALPHA, label=lbl)
                _setup_plot("$t$", rescaling_ylabel, log_scale=False)
                ax = plt.gca()
                ax.yaxis.get_major_formatter().set_useOffset(False)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"theory_rescaling{suffix}.pdf"), bbox_inches="tight")
                plt.close()
            else:
                plt.close()

        # Hessian condition number vs width (when multiple widths are discovered)
        if use_discovered_widths and width_runs:
            width_cond_mean = []
            width_cond_sem = []
            widths_cond = []
            for w, w_run_dirs in width_runs:
                vals = []
                for w_run_dir in w_run_dirs:
                    p = os.path.join(w_run_dir, "hessian_condition_number.npy")
                    if os.path.isfile(p):
                        vals.append(float(np.load(p).item()))
                if not vals:
                    continue
                vals = np.asarray(vals, dtype=float)
                widths_cond.append(w)
                width_cond_mean.append(vals.mean())
                if len(vals) <= 1:
                    width_cond_sem.append(0.0)
                else:
                    width_cond_sem.append(vals.std(ddof=1) / np.sqrt(len(vals)))
            if widths_cond:
                plt.figure(figsize=FIG_SIZE)
                plt.errorbar(
                    widths_cond,
                    width_cond_mean,
                    yerr=width_cond_sem,
                    fmt="o-",
                    color="#4A90E2",
                    linewidth=LINE_WIDTH,
                    markersize=12,
                    alpha=ALPHA,
                    capsize=6,
                )
                _setup_plot("Width $N$", r"$\kappa$", log_scale=True)
                ax = plt.gca()
                ax.set_xscale("log", base=2)
                ax.set_xticks(widths_cond)
                ax.set_xticklabels(widths_cond)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"hessian_condition_vs_width{suffix}.pdf"), bbox_inches="tight")
                plt.close()

        # Cosine similarities
        if use_discovered_widths:
            cos_data = []
            cos_data_theory_bp = []
            for width, w_run_dirs in width_runs:
                overall_curves = []
                per_layer_curves = []
                overall_th_curves = []
                per_layer_th_curves = []
                names = None
                any_dir_for_steps = None
                for w_run_dir in w_run_dirs:
                    overall, per_layer, names_i, overall_theory_bp, per_layer_theory_bp = load_cos_sims(w_run_dir)
                    if any_dir_for_steps is None:
                        any_dir_for_steps = w_run_dir
                    if names is None:
                        names = names_i
                    if overall is not None:
                        overall_curves.append(overall)
                        per_layer_curves.append(per_layer)
                    if overall_theory_bp is not None:
                        overall_th_curves.append(overall_theory_bp)
                        per_layer_th_curves.append(per_layer_theory_bp)
                if overall_curves:
                    overall_mean, overall_sem, _, n_t = aggregate_seed_curves(overall_curves)
                    per_layer_mean, per_layer_sem, _, n_t2 = aggregate_seed_arrays_2d(per_layer_curves)
                    if per_layer_mean is None:
                        n_t_use = n_t
                        cos_data.append(
                            (width, any_dir_for_steps, overall_mean[:n_t_use], overall_sem[:n_t_use], None, None, names)
                        )
                    else:
                        n_t_use = min(n_t, n_t2) if (n_t and n_t2) else (n_t or n_t2)
                        cos_data.append(
                            (
                                width,
                                any_dir_for_steps,
                                overall_mean[:n_t_use],
                                overall_sem[:n_t_use],
                                per_layer_mean[:n_t_use],
                                per_layer_sem[:n_t_use],
                                names,
                            )
                        )
                if overall_th_curves:
                    overall_th_mean, overall_th_sem, _, n_t = aggregate_seed_curves(overall_th_curves)
                    per_layer_th_mean, per_layer_th_sem, _, n_t2 = aggregate_seed_arrays_2d(per_layer_th_curves)
                    if per_layer_th_mean is None:
                        n_t_use = n_t
                        cos_data_theory_bp.append(
                            (width, any_dir_for_steps, overall_th_mean[:n_t_use], overall_th_sem[:n_t_use], None, None, names)
                        )
                    else:
                        n_t_use = min(n_t, n_t2) if (n_t and n_t2) else (n_t or n_t2)
                        cos_data_theory_bp.append(
                            (
                                width,
                                any_dir_for_steps,
                                overall_th_mean[:n_t_use],
                                overall_th_sem[:n_t_use],
                                per_layer_th_mean[:n_t_use],
                                per_layer_th_sem[:n_t_use],
                                names,
                            )
                        )
            if cos_data:
                steps_cos = np.load(os.path.join(cos_data[0][1], "steps.npy"))
                layer_names = cos_data[0][6]
            elif cos_data_theory_bp:
                steps_cos = np.load(os.path.join(cos_data_theory_bp[0][1], "steps.npy"))
                layer_names = cos_data_theory_bp[0][6]
        else:
            overall, per_layer, layer_names, overall_theory_bp, per_layer_theory_bp = load_cos_sims(data_dir)
            if overall is not None:
                overall = np.asarray(overall).flatten()
                overall_sem = np.zeros_like(overall)
                if per_layer is None:
                    per_layer_mean = None
                    per_layer_sem = None
                else:
                    per_layer_mean = np.asarray(per_layer)
                    if per_layer_mean.ndim == 1:
                        per_layer_mean = per_layer_mean.reshape(-1, 1)
                    per_layer_sem = np.zeros_like(per_layer_mean)
                cos_data = [(widths[0] if has_widths else None, data_dir, overall, overall_sem, per_layer_mean, per_layer_sem, layer_names)]
                steps_cos = steps
            else:
                cos_data = []

            if overall_theory_bp is not None:
                overall_theory_bp = np.asarray(overall_theory_bp).flatten()
                overall_theory_bp_sem = np.zeros_like(overall_theory_bp)
                if per_layer_theory_bp is None:
                    per_layer_theory_bp_mean = None
                    per_layer_theory_bp_sem = None
                else:
                    per_layer_theory_bp_mean = np.asarray(per_layer_theory_bp)
                    if per_layer_theory_bp_mean.ndim == 1:
                        per_layer_theory_bp_mean = per_layer_theory_bp_mean.reshape(-1, 1)
                    per_layer_theory_bp_sem = np.zeros_like(per_layer_theory_bp_mean)
                cos_data_theory_bp = [
                    (
                        widths[0] if has_widths else None,
                        data_dir,
                        overall_theory_bp,
                        overall_theory_bp_sem,
                        per_layer_theory_bp_mean,
                        per_layer_theory_bp_sem,
                        layer_names,
                    )
                ]
            else:
                cos_data_theory_bp = []
            if not cos_data and cos_data_theory_bp:
                steps_cos = steps

        if cos_data:
            colormap = plt.get_cmap(colormap_name)
            n_w = len(cos_data)

            # Overall grad cos sim vs t (widths in legend, like plot_toy_results)
            cos_sim_ylabel = r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, \nabla_{\boldsymbol{\theta}} \mathcal{F}\right)$"
            plt.figure(figsize=FIG_SIZE)
            for idx, (width, _, overall_mean, overall_sem, _, _, _) in enumerate(cos_data):
                n = min(len(steps_cos), len(overall_mean))
                color = colormap(_get_color_val(idx, n_w, colormap_name))
                lbl = _legend_n_label(width) if width is not None else "Overall"
                plot_mean_with_sem(steps_cos[:n], overall_mean[:n], overall_sem[:n], color=color, label=lbl, linestyle="-", alpha_band=0.14)
            _setup_plot("$t$", cos_sim_ylabel, log_scale=False)
            ax = plt.gca()
            ax.yaxis.get_major_formatter().set_useOffset(False)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"grad_cosine_similarity_overall{suffix}.pdf"), bbox_inches="tight")
            plt.close()

            # Per-layer cos sim vs t (one separate plot per layer)
            if cos_data[0][4] is not None:
                n_layers = cos_data[0][4].shape[1] if cos_data[0][4].ndim > 1 else 1
                names = cos_data[0][6]
                if isinstance(names, np.ndarray):
                    names = list(names)
                else:
                    names = [f"Layer {j}" for j in range(n_layers)]
                for layer_idx in range(n_layers):
                    layer_name = names[layer_idx] if layer_idx < len(names) else f"layer_{layer_idx}"
                    plt.figure(figsize=FIG_SIZE)
                    for idx, (width, _, _, _, per_layer_mean, per_layer_sem, _) in enumerate(cos_data):
                        if per_layer_mean is None:
                            continue
                        n = min(len(steps_cos), per_layer_mean.shape[0])
                        vals = per_layer_mean[:n, layer_idx] if per_layer_mean.ndim > 1 else per_layer_mean[:n]
                        vals_sem = per_layer_sem[:n, layer_idx] if per_layer_sem.ndim > 1 else per_layer_sem[:n]
                        color = colormap(_get_color_val(idx, n_w, colormap_name))
                        lbl = _legend_n_label(width) if width is not None else "Overall"
                        plot_mean_with_sem(steps_cos[:n], vals, vals_sem, color=color, label=lbl, linestyle="-", alpha_band=0.14)
                    display_name = _layer_cos_sim_display_name(layer_name)
                    layer_ylabel = _layer_cos_sim_ylabel(layer_name)
                    _setup_plot("$t$", layer_ylabel, log_scale=False)
                    ax = plt.gca()
                    ax.yaxis.get_major_formatter().set_useOffset(False)
                    plt.tight_layout()
                    file_token = display_name.replace(" ", "")
                    plt.savefig(
                        os.path.join(args.output_dir, f"grad_cosine_similarity_{file_token}{suffix}.pdf"),
                        bbox_inches="tight",
                    )
                    plt.close()

        # Cosine similarities: BP grads vs theory PC grads (overall and per layer)
        if cos_data_theory_bp:
            colormap = plt.get_cmap(colormap_name)
            n_w = len(cos_data_theory_bp)

            # Overall BP vs theory-PC grad cos sim vs t
            cos_sim_ylabel_theory_bp = r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, \nabla_{\boldsymbol{\theta}} \mathcal{F}_{\text{theory}}\right)$"
            plt.figure(figsize=FIG_SIZE)
            for idx, (width, _, overall_th_mean, overall_th_sem, _, _, _) in enumerate(cos_data_theory_bp):
                n = min(len(steps_cos), len(overall_th_mean))
                color = colormap(_get_color_val(idx, n_w, colormap_name))
                lbl = _legend_n_label(width) if width is not None else "Overall (theory vs BP)"
                plot_mean_with_sem(steps_cos[:n], overall_th_mean[:n], overall_th_sem[:n], color=color, label=lbl, linestyle="-", alpha_band=0.14)
            _setup_plot("$t$", cos_sim_ylabel_theory_bp, log_scale=False)
            ax = plt.gca()
            ax.yaxis.get_major_formatter().set_useOffset(False)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"grad_cosine_similarity_overall_theory_bp{suffix}.pdf"), bbox_inches="tight")
            plt.close()

            # Per-layer BP vs theory-PC cos sim vs t (one separate plot per layer)
            if cos_data_theory_bp[0][4] is not None:
                n_layers_th = cos_data_theory_bp[0][4].shape[1] if cos_data_theory_bp[0][4].ndim > 1 else 1
                names_th = cos_data_theory_bp[0][6]
                if isinstance(names_th, np.ndarray):
                    names_th = list(names_th)
                else:
                    names_th = [f"Layer {j}" for j in range(n_layers_th)]

                for layer_idx in range(n_layers_th):
                    layer_name = names_th[layer_idx] if layer_idx < len(names_th) else f"layer_{layer_idx}"
                    plt.figure(figsize=FIG_SIZE)
                    for idx, (width, _, _, _, per_layer_th_mean, per_layer_th_sem, _) in enumerate(cos_data_theory_bp):
                        if per_layer_th_mean is None:
                            continue
                        n = min(len(steps_cos), per_layer_th_mean.shape[0])
                        vals = per_layer_th_mean[:n, layer_idx] if per_layer_th_mean.ndim > 1 else per_layer_th_mean[:n]
                        vals_sem = per_layer_th_sem[:n, layer_idx] if per_layer_th_sem.ndim > 1 else per_layer_th_sem[:n]
                        color = colormap(_get_color_val(idx, n_w, colormap_name))
                        lbl = _legend_n_label(width) if width is not None else "Overall (theory vs BP)"
                        plot_mean_with_sem(steps_cos[:n], vals[:n], vals_sem[:n], color=color, label=lbl, linestyle="-", alpha_band=0.14)
                    display_name = _layer_cos_sim_display_name(layer_name)
                    layer_ylabel = _layer_cos_sim_ylabel(layer_name)
                    _setup_plot("$t$", layer_ylabel, log_scale=False)
                    ax = plt.gca()
                    ax.yaxis.get_major_formatter().set_useOffset(False)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            args.output_dir,
                            f"grad_cosine_similarity_theory_bp_{display_name.replace(' ', '')}{suffix}.pdf",
                        ),
                        bbox_inches="tight",
                    )
                    plt.close()

    # Cos sim (PC vs BP) at initialisation only vs inference iterations (x), one curve per width (legend)
    # Run once over all groups: aggregate by (activity_lr, use_amortiser), then plot init cos sim vs n_infer_iters per width
    groups_by_alr_amortiser = {}  # (activity_lr, use_amortiser) -> [(n_infer_iters, [(width, run_dir), ...]), ...]
    for (alr, n_infer, use_am), width_runs in activity_lr_groups.items():
        key = (alr, use_am)
        if key not in groups_by_alr_amortiser:
            groups_by_alr_amortiser[key] = []
        groups_by_alr_amortiser[key].append((n_infer, width_runs))
    for (activity_lr, use_amortiser), n_infer_width_runs in groups_by_alr_amortiser.items():
        n_infer_vals = sorted(set(n for n, _ in n_infer_width_runs if n is not None))
        if len(n_infer_vals) < 2:
            continue
        # Build width -> [(n_infer_iters, run_dir), ...]
        by_width = {}  # width -> {n_infer_iters: [run_dir, ...]}
        for n_infer, width_runs in n_infer_width_runs:
            if n_infer is None:
                continue
            for width, w_run_dirs in width_runs:
                by_width.setdefault(width, {}).setdefault(n_infer, []).extend(w_run_dirs)

        # For each width, get cos_sim at step 0 for each n_infer_iters (mean +/- SEM over seeds)
        init_cos_data = []  # (width, n_infer_list, cos_mean_list, cos_sem_list)
        for width in sorted(by_width.keys()):
            n_infer_list = []
            cos_mean = []
            cos_sem = []
            for n_infer in sorted(by_width[width].keys()):
                vals = []
                for w_run_dir in by_width[width][n_infer]:
                    overall, _, _, _, _ = load_cos_sims(w_run_dir)
                    if overall is not None and len(overall) > 0:
                        vals.append(float(overall[0]))
                if not vals:
                    continue
                vals = np.asarray(vals, dtype=float)
                n_infer_list.append(n_infer)
                cos_mean.append(vals.mean())
                if len(vals) <= 1:
                    cos_sem.append(0.0)
                else:
                    cos_sem.append(vals.std(ddof=1) / np.sqrt(len(vals)))
            if n_infer_list:
                init_cos_data.append((width, n_infer_list, cos_mean, cos_sem))
        if not init_cos_data:
            continue
        suffix_init = _file_suffix(activity_lr, None, use_amortiser, n_activity_lr_groups, 1, n_use_amortiser_groups)
        if selected_seed is not None:
            suffix_init = f"{suffix_init}_seed_{selected_seed}"
        cos_sim_ylabel = r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, \nabla_{\boldsymbol{\theta}} \mathcal{F}\right)$"
        plt.figure(figsize=FIG_SIZE)
        colormap = plt.get_cmap(colormap_name)
        n_w = len(init_cos_data)
        for idx, (width, n_infer_list, cos_mean, cos_sem) in enumerate(init_cos_data):
            color = colormap(_get_color_val(idx, n_w, colormap_name))
            lbl = _legend_n_label(width)
            plt.errorbar(
                n_infer_list,
                cos_mean,
                yerr=cos_sem,
                fmt="o-",
                color=color,
                linewidth=LINE_WIDTH,
                markersize=12,
                alpha=ALPHA,
                capsize=6,
                label=lbl,
            )
        _setup_plot("Inference iterations", cos_sim_ylabel, log_scale=False)
        ax = plt.gca()
        ax.yaxis.get_major_formatter().set_useOffset(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"grad_cosine_similarity_overall_init_vs_n_infer_iters{suffix_init}.pdf"),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory (contains steps.npy, theory_energies.npy, etc.); or base dir to discover *_width runs for rescaling plot")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to a .npy file in the run dir (used to infer run dir if --run_dir not set)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for PDFs (default: run_dir or same as results_file dir)")
    parser.add_argument("--colormap", type=str, default="Blues",
                        help="Colormap for multi-width plots (default: Blues, same as plot_toy_results)")
    parser.add_argument(
        "--plot_widths",
        type=int,
        nargs="*",
        default=[8, 16, 32],
        help="Optional list of widths N to include in plots (e.g. --plot_widths 16 32 64). "
             "If omitted, include all discovered widths (and all widths in widths.npy, if present).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Plot metrics from this seed/replication only (run dir basename, e.g. .../0 or .../seed_0). "
             "By default, metrics are averaged across all discovered seeds with SEM bands.",
    )
    args = parser.parse_args()
    if args.run_dir is not None:
        if args.output_dir is None:
            args.output_dir = args.run_dir
    elif args.results_file is not None:
        args.run_dir = os.path.dirname(args.results_file)
        if args.output_dir is None:
            args.output_dir = args.run_dir
    else:
        # Default to a path relative to this script so it works from any CWD.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.run_dir = os.path.join(script_dir, "theory_results")
        if args.output_dir is None:
            args.output_dir = args.run_dir
    main(args)
