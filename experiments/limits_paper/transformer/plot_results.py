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
    sequential = {
        "viridis", "plasma", "inferno", "magma", "cividis",
        "Reds", "Blues", "Greens", "Oranges", "Purples", "Greys",
        "YlOrRd", "YlOrBr", "YlGnBu", "YlGn", "RdPu",
        "BuGn", "BuPu", "GnBu", "PuBu", "PuBuGn", "PuRd", "OrRd",
        "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "cool", "hot",
        "copper", "bone", "pink", "spring", "summer", "autumn", "winter",
    }
    return colormap_name in sequential


def _get_color_val(idx, n_sizes, colormap_name="viridis"):
    if _is_sequential_colormap(colormap_name):
        return 0.15 + (idx / max(n_sizes - 1, 1)) * 0.85 if n_sizes > 1 else 0.15
    return (idx / max(n_sizes - 1, 1)) * 0.85 if n_sizes > 1 else 0


def _layer_cos_sim_ylabel(layer_name):
    """Format layer name for y-axis (embed, block0, block_last, readout)."""
    name = str(layer_name).strip().lower()
    if name == "embed":
        return r"Grad cos. Embed"
    if name == "block0":
        return r"Grad cos. Block 0"
    if name == "block_last":
        return r"Grad cos. Block $L$"
    if name == "readout":
        return r"Grad cos. Readout"
    return "Grad cos. " + (layer_name[:1].upper() + layer_name[1:] if layer_name else "Layer")


def _setup_plot(xlabel, ylabel, log_scale=False):
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


def _parse_activity_lr_from_path(path):
    """Extract activity_lr from path segment '*_activity_lr'."""
    for part in path.split(os.sep):
        if part.endswith("_activity_lr"):
            try:
                return float(part.replace("_activity_lr", ""))
            except ValueError:
                pass
    return None


def _parse_n_infer_iters_from_path(path):
    """Extract n_infer_iters from path segment '*_n_infer_iters'."""
    for part in path.split(os.sep):
        if part.endswith("_n_infer_iters"):
            try:
                return int(part.replace("_n_infer_iters", ""))
            except ValueError:
                pass
    return None


def _parse_d_model_from_path(path):
    """Extract d_model from path segment '*_d_model' (transformer 'width' analog)."""
    for part in path.split(os.sep):
        if part.endswith("_d_model"):
            try:
                return int(part.replace("_d_model", ""))
            except ValueError:
                pass
    return None


def _seed_from_run_dir(run_dir):
    """Extract seed index from run_dir (e.g. .../n_infer_iters/2 -> 2)."""
    base = os.path.basename(os.path.normpath(run_dir))
    if base.isdigit():
        return int(base)
    if base.startswith("seed_"):
        try:
            return int(base.replace("seed_", ""))
        except ValueError:
            pass
    return None


def _filter_d_model_runs_by_seed(d_model_runs, seed):
    """Keep only run dirs matching seed. Raises if any d_model has no matching run."""
    filtered = []
    missing = []
    for d_model, run_dirs in d_model_runs:
        matched = [d for d in run_dirs if _seed_from_run_dir(d) == seed]
        if not matched:
            missing.append(d_model)
        else:
            filtered.append((d_model, matched))
    if missing:
        available = ", ".join(
            f"N={dm}: {sorted({_seed_from_run_dir(d) for d in ds})}"
            for dm, ds in d_model_runs
        )
        raise FileNotFoundError(
            f"No run found for --seed {seed} at d_models {missing}. Available seeds: {available}."
        )
    return filtered


def discover_d_model_run_dirs(results_dir):
    """Discover all (d_model, run_dir) under results_dir where run_dir contains steps.npy.

    Transformer train.py saves under .../seq_len/d_model/n_blocks/.../activity_lr/n_infer_iters/.../seed/.
    Returns a dict: (activity_lr, n_infer_iters) -> [(d_model, [run_dir, ...]), ...], sorted by d_model.
    Multiple run_dir entries for the same d_model are interpreted as different seeds/replications and will
    be aggregated in plots.
    """
    found = []  # (d_model, activity_lr, n_infer_iters, run_dir)
    for root, _, files in os.walk(results_dir):
        if "steps.npy" not in files:
            continue
        d_model = _parse_d_model_from_path(root)
        if d_model is None:
            continue
        activity_lr = _parse_activity_lr_from_path(root)
        n_infer_iters = _parse_n_infer_iters_from_path(root)
        found.append((d_model, activity_lr, n_infer_iters, root))

    by_group = {}
    for dm, alr, n_infer, d in found:
        key = (alr, n_infer)
        if key not in by_group:
            by_group[key] = {}
        by_group[key].setdefault(dm, []).append(d)
    return {
        key: [(dm, by_group[key][dm]) for dm in sorted(by_group[key].keys())]
        for key in sorted(
            by_group.keys(),
            key=lambda k: (k[0] is None, k[0] or 0, k[1] is None, k[1] if k[1] is not None else 0),
        )
    }


def _file_suffix(activity_lr, n_infer_iters, n_activity_lr_groups, n_infer_iters_groups):
    """Filename suffix when multiple (activity_lr, n_infer_iters) groups exist."""
    parts = []
    if n_activity_lr_groups > 1:
        if activity_lr is None:
            parts.append("_no_activity_lr")
        else:
            parts.append(f"_{activity_lr}_activity_lr")
    if n_infer_iters_groups > 1 and n_infer_iters is not None:
        parts.append(f"_{n_infer_iters}_n_infer_iters")
    return "".join(parts)


def main(args):
    run_dir = args.run_dir
    colormap_name = getattr(args, "colormap", "Blues") or "Blues"
    selected_seed = getattr(args, "seed", None)

    activity_lr_groups = discover_d_model_run_dirs(run_dir)
    if selected_seed is not None:
        filtered_groups = {}
        for key, d_model_runs in activity_lr_groups.items():
            filtered_groups[key] = (
                _filter_d_model_runs_by_seed(d_model_runs, selected_seed) if d_model_runs else d_model_runs
            )
        activity_lr_groups = filtered_groups
    if not activity_lr_groups:
        steps_path = os.path.join(run_dir, "steps.npy")
        if not os.path.isfile(steps_path):
            raise FileNotFoundError(
                f"No run data found in {run_dir!r}: missing {steps_path}. "
                "Pass the base results directory under which transformer train.py saved runs "
                "(subdirs with *_d_model and steps.npy)."
            )
        activity_lr_groups = {(None, None): []}
        if selected_seed is not None:
            dir_seed = _seed_from_run_dir(run_dir)
            if dir_seed is not None and dir_seed != selected_seed:
                raise FileNotFoundError(
                    f"--seed {selected_seed} does not match run_dir seed {dir_seed} ({run_dir!r})."
                )

    selected_d_models = getattr(args, "d_models", None)
    if selected_d_models:
        selected_set = set(selected_d_models)
        filtered = {}
        for key, d_model_runs in activity_lr_groups.items():
            kept = [(dm, runs) for dm, runs in d_model_runs if dm in selected_set]
            if kept:
                filtered[key] = kept
        if not filtered:
            available = sorted({
                dm for runs in activity_lr_groups.values() for dm, _ in runs
            })
            raise ValueError(
                f"No runs match --d_models {sorted(selected_set)}. "
                f"Available d_models in {run_dir!r}: {available}."
            )
        activity_lr_groups = filtered

    os.makedirs(args.output_dir, exist_ok=True)

    def aggregate_seed_curves(curves):
        """Aggregate a list of 1D arrays into (mean, std) across seeds.

        Curves can have different lengths; we truncate to the minimum available length.
        Returns (mean, std, n_seeds, n_time).
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
            std = np.zeros_like(mean)
        else:
            std = stacked.std(axis=0, ddof=1)
        return mean, std, stacked.shape[0], n_time

    def aggregate_seed_arrays_2d(arrays_2d):
        """Aggregate list of (T, D) arrays into (mean, std) across seeds, truncating to min T.

        Returns (mean, std, n_seeds, n_time) with shape (T, D) for mean/std.
        """
        if not arrays_2d:
            return None, None, 0, 0
        arrays_2d = [np.asarray(a) for a in arrays_2d if a is not None]
        if not arrays_2d:
            return None, None, 0, 0
        n_time = min(a.shape[0] for a in arrays_2d)
        if n_time <= 0:
            return None, None, 0, 0
        n_dim = min(a.shape[1] if a.ndim > 1 else 1 for a in arrays_2d)
        stacked = np.stack(
            [(a[:n_time, :n_dim] if a.ndim > 1 else a[:n_time].reshape(-1, 1)) for a in arrays_2d],
            axis=0,
        )  # (n_seeds, n_time, n_dim)
        mean = stacked.mean(axis=0)
        if stacked.shape[0] <= 1:
            std = np.zeros_like(mean)
        else:
            std = stacked.std(axis=0, ddof=1)
        return mean, std, stacked.shape[0], n_time

    def plot_mean_with_std(x, mean, std, *, color, label=None, linestyle="-", alpha_line=ALPHA, alpha_band=0.18):
        """Line plot of mean with ±1 std band."""
        plt.plot(x, mean, linestyle, color=color, linewidth=LINE_WIDTH, alpha=alpha_line, label=label)
        if std is not None and np.any(np.isfinite(std)) and np.max(std) > 0:
            plt.fill_between(x, mean - std, mean + std, color=color, alpha=alpha_band, linewidth=0)

    def load_cos_sims(run_dir):
        path_overall = os.path.join(run_dir, "grad_cosine_similarities.npy")
        path_per_layer = os.path.join(run_dir, "grad_cosine_similarities_per_layer.npy")
        path_names = os.path.join(run_dir, "grad_cosine_similarities_layer_names.npy")

        names = None
        if os.path.isfile(path_names):
            names = np.load(path_names, allow_pickle=True)
            if isinstance(names, np.ndarray):
                names = list(names)
        if not names:
            names = ["embed", "block0", "block_last", "readout"]

        overall = None
        per_layer = None
        if os.path.isfile(path_overall) and os.path.isfile(path_per_layer):
            overall = np.load(path_overall).flatten()
            per_layer = np.load(path_per_layer)

        if overall is None:
            return None, None, names
        return overall, per_layer, names

    n_groups = len(activity_lr_groups)
    n_activity_lr_groups = len(set(k[0] for k in activity_lr_groups.keys()))
    n_infer_iters_groups = len(set(k[1] for k in activity_lr_groups.keys()))

    for (activity_lr, n_infer_iters), d_model_runs in activity_lr_groups.items():
        suffix = _file_suffix(
            activity_lr, n_infer_iters, n_activity_lr_groups, n_infer_iters_groups
        )
        if selected_seed is not None:
            suffix = f"{suffix}_seed_{selected_seed}"
        use_discovered = len(d_model_runs) > 1 or (selected_seed is not None and bool(d_model_runs))
        if d_model_runs:
            data_dir = d_model_runs[0][1][0]
        else:
            data_dir = run_dir

        steps = np.load(os.path.join(data_dir, "steps.npy"))

        # --- PC (experiment) energy vs t ---
        plt.figure(figsize=FIG_SIZE)
        colormap = plt.get_cmap(colormap_name)
        if use_discovered:
            for idx, (d_model, w_run_dirs) in enumerate(d_model_runs):
                ee_curves = []
                steps_list = []
                for w_run_dir in w_run_dirs:
                    ee_path = os.path.join(w_run_dir, "experiment_energies.npy")
                    if not os.path.isfile(ee_path):
                        continue
                    ee_curves.append(np.load(ee_path).flatten())
                    steps_list.append(np.load(os.path.join(w_run_dir, "steps.npy")).flatten())
                if not ee_curves:
                    continue
                mean, std, _, n_t = aggregate_seed_curves(ee_curves)
                steps_w = min(steps_list, key=len)[:n_t] if steps_list else np.arange(n_t)
                color = colormap(_get_color_val(idx, len(d_model_runs), colormap_name))
                plot_mean_with_std(
                    steps_w,
                    mean,
                    std,
                    color=color,
                    label=rf"$N = {d_model}$",
                    linestyle="-",
                    alpha_band=0.14,
                )
        else:
            ee_path = os.path.join(data_dir, "experiment_energies.npy")
            if os.path.isfile(ee_path):
                experiment_energies = np.load(ee_path).flatten()
                n = min(len(steps), len(experiment_energies))
                plt.plot(
                    steps[:n],
                    experiment_energies[:n],
                    "-",
                    color="#4A90E2",
                    linewidth=LINE_WIDTH,
                    alpha=ALPHA,
                    label="PC energy",
                )
        _setup_plot("$t$", r"$\mathcal{F}(\boldsymbol{\theta}_t)$", log_scale=True)
        ax = plt.gca()
        if hasattr(ax.yaxis.get_major_formatter(), "set_useOffset"):
            ax.yaxis.get_major_formatter().set_useOffset(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"experiment_energy{suffix}.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        # --- BP loss vs t (if present) ---
        plt.figure(figsize=FIG_SIZE)
        if use_discovered:
            for idx, (d_model, w_run_dirs) in enumerate(d_model_runs):
                bp_curves = []
                steps_list = []
                for w_run_dir in w_run_dirs:
                    bp_path = os.path.join(w_run_dir, "bp_losses.npy")
                    if not os.path.isfile(bp_path):
                        continue
                    bp_curves.append(np.load(bp_path).flatten())
                    steps_list.append(np.load(os.path.join(w_run_dir, "steps.npy")).flatten())
                if not bp_curves:
                    continue
                mean, std, _, n_t = aggregate_seed_curves(bp_curves)
                steps_w = min(steps_list, key=len)[:n_t] if steps_list else np.arange(n_t)
                color = colormap(_get_color_val(idx, len(d_model_runs), colormap_name))
                plot_mean_with_std(
                    steps_w,
                    mean,
                    std,
                    color=color,
                    label=rf"$N = {d_model}$",
                    linestyle="-",
                    alpha_band=0.14,
                )
        else:
            bp_path = os.path.join(data_dir, "bp_losses.npy")
            if os.path.isfile(bp_path):
                bp_losses = np.load(bp_path).flatten()
                n = min(len(steps), len(bp_losses))
                plt.plot(
                    steps[:n],
                    bp_losses[:n],
                    "-",
                    color="#4A90E2",
                    linewidth=LINE_WIDTH,
                    alpha=ALPHA,
                    label="BP loss",
                )
        _setup_plot("$t$", r"$\mathcal{L}(\boldsymbol{\theta}_t)$", log_scale=True)
        ax = plt.gca()
        if hasattr(ax.yaxis.get_major_formatter(), "set_useOffset"):
            ax.yaxis.get_major_formatter().set_useOffset(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"bp_loss{suffix}.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        # --- Hessian condition number vs d_model (when multiple d_model runs exist) ---
        if use_discovered and d_model_runs:
            d_models_cond = []
            cond_means = []
            cond_stds = []
            for d_model, w_run_dirs in d_model_runs:
                vals = []
                for w_run_dir in w_run_dirs:
                    p = os.path.join(w_run_dir, "hessian_condition_number.npy")
                    if os.path.isfile(p):
                        vals.append(float(np.load(p).item()))
                if not vals:
                    continue
                vals = np.asarray(vals, dtype=float)
                d_models_cond.append(d_model)
                cond_means.append(vals.mean())
                cond_stds.append(0.0 if len(vals) <= 1 else vals.std(ddof=1))
            if d_models_cond:
                plt.figure(figsize=FIG_SIZE)
                plt.errorbar(
                    d_models_cond,
                    cond_means,
                    yerr=cond_stds,
                    fmt="o-",
                    color="#4A90E2",
                    linewidth=LINE_WIDTH,
                    markersize=12,
                    alpha=ALPHA,
                    capsize=6,
                )
                _setup_plot(r"$N$", r"$\kappa$", log_scale=True)
                ax = plt.gca()
                ax.set_xscale("log", base=2)
                ax.set_xticks(d_models_cond)
                ax.set_xticklabels([str(dm) for dm in d_models_cond])
                plt.tight_layout()
                plt.savefig(
                    os.path.join(args.output_dir, f"hessian_condition_vs_d_model{suffix}.pdf"),
                    bbox_inches="tight",
                )
                plt.close()

        # --- Cosine similarities (overall and per layer) ---
        if use_discovered:
            cos_data = []
            for d_model, w_run_dirs in d_model_runs:
                overall_curves = []
                per_layer_curves = []
                names = None
                any_dir_for_steps = None
                for w_run_dir in w_run_dirs:
                    overall, per_layer, names_i = load_cos_sims(w_run_dir)
                    if any_dir_for_steps is None:
                        any_dir_for_steps = w_run_dir
                    if names is None:
                        names = names_i
                    if overall is not None:
                        overall_curves.append(overall)
                        per_layer_curves.append(per_layer)
                if not overall_curves:
                    continue
                overall_mean, overall_std, _, n_t = aggregate_seed_curves(overall_curves)
                per_layer_mean, per_layer_std, _, n_t2 = aggregate_seed_arrays_2d(per_layer_curves)
                if per_layer_mean is None:
                    n_t_use = n_t
                    cos_data.append(
                        (d_model, any_dir_for_steps, overall_mean[:n_t_use], overall_std[:n_t_use], None, None, names)
                    )
                else:
                    n_t_use = min(n_t, n_t2) if (n_t and n_t2) else (n_t or n_t2)
                    cos_data.append(
                        (
                            d_model,
                            any_dir_for_steps,
                            overall_mean[:n_t_use],
                            overall_std[:n_t_use],
                            per_layer_mean[:n_t_use],
                            per_layer_std[:n_t_use],
                            names,
                        )
                    )
            if cos_data:
                steps_cos = np.load(os.path.join(cos_data[0][1], "steps.npy"))
                layer_names = cos_data[0][6]
        else:
            overall, per_layer, layer_names = load_cos_sims(data_dir)
            if overall is not None:
                d_model_label = d_model_runs[0][0] if d_model_runs else None
                overall = np.asarray(overall).flatten()
                overall_std = np.zeros_like(overall)
                if per_layer is None:
                    per_layer_mean = None
                    per_layer_std = None
                else:
                    per_layer_mean = np.asarray(per_layer)
                    if per_layer_mean.ndim == 1:
                        per_layer_mean = per_layer_mean.reshape(-1, 1)
                    per_layer_std = np.zeros_like(per_layer_mean)
                cos_data = [
                    (d_model_label, data_dir, overall, overall_std, per_layer_mean, per_layer_std, layer_names)
                ]
                steps_cos = steps
            else:
                cos_data = []

        if cos_data:
            colormap = plt.get_cmap(colormap_name)
            n_w = len(cos_data)

            # Overall grad cos sim vs t
            cos_sim_ylabel = (
                r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, "
                r"\nabla_{\boldsymbol{\theta}} \mathcal{F}\right)$"
            )
            plt.figure(figsize=FIG_SIZE)
            for idx, (d_model, _, overall_mean, overall_std, _, _, _) in enumerate(cos_data):
                n = min(len(steps_cos), len(overall_mean))
                color = colormap(_get_color_val(idx, n_w, colormap_name))
                lbl = rf"$N = {d_model}$" if d_model is not None else "Overall"
                plot_mean_with_std(
                    steps_cos[:n],
                    overall_mean[:n],
                    overall_std[:n],
                    color=color,
                    label=lbl,
                    linestyle="-",
                    alpha_band=0.14,
                )
            _setup_plot("$t$", cos_sim_ylabel, log_scale=False)
            ax = plt.gca()
            if hasattr(ax.yaxis.get_major_formatter(), "set_useOffset"):
                ax.yaxis.get_major_formatter().set_useOffset(False)
            plt.tight_layout()
            plt.savefig(
                os.path.join(args.output_dir, f"grad_cosine_similarity_overall{suffix}.pdf"),
                bbox_inches="tight",
            )
            plt.close()

            # Per-layer cos sim vs t
            if cos_data[0][4] is None:
                continue
            n_layers = cos_data[0][4].shape[1] if cos_data[0][4].ndim > 1 else 1
            names = cos_data[0][6]
            if isinstance(names, np.ndarray):
                names = list(names)
            else:
                names = names or [f"Layer {j}" for j in range(n_layers)]
            for layer_idx in range(n_layers):
                layer_name = (
                    names[layer_idx]
                    if layer_idx < len(names)
                    else f"layer_{layer_idx}"
                )
                plt.figure(figsize=FIG_SIZE)
                for idx, (d_model, _, _, _, per_layer_mean, per_layer_std, _) in enumerate(cos_data):
                    if per_layer_mean is None:
                        continue
                    n = min(len(steps_cos), per_layer_mean.shape[0])
                    vals = per_layer_mean[:n, layer_idx] if per_layer_mean.ndim > 1 else per_layer_mean[:n]
                    vals_std = per_layer_std[:n, layer_idx] if per_layer_std.ndim > 1 else per_layer_std[:n]
                    color = colormap(_get_color_val(idx, n_w, colormap_name))
                    lbl = rf"$N = {d_model}$" if d_model is not None else "Overall"
                    plot_mean_with_std(
                        steps_cos[:n],
                        vals,
                        vals_std,
                        color=color,
                        label=lbl,
                        linestyle="-",
                        alpha_band=0.14,
                    )
                layer_ylabel = _layer_cos_sim_ylabel(layer_name)
                _setup_plot("$t$", layer_ylabel, log_scale=False)
                ax = plt.gca()
                if hasattr(ax.yaxis.get_major_formatter(), "set_useOffset"):
                    ax.yaxis.get_major_formatter().set_useOffset(False)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        args.output_dir,
                        f"grad_cosine_similarity_{layer_name}{suffix}.pdf",
                    ),
                    bbox_inches="tight",
                )
                plt.close()

    # Cos sim (PC vs BP) at initialisation only vs inference iterations (x), one curve per d_model (legend).
    # One figure per activity_lr (matches cnn/plot_theory_results.py init-vs-n_infer plot).
    groups_by_activity_lr = {}
    for (alr, n_infer), d_model_runs in activity_lr_groups.items():
        groups_by_activity_lr.setdefault(alr, []).append((n_infer, d_model_runs))

    for activity_lr in sorted(
        groups_by_activity_lr.keys(),
        key=lambda k: (k is None, float(k) if k is not None else 0.0),
    ):
        n_infer_width_runs = groups_by_activity_lr[activity_lr]
        n_infer_vals = sorted(
            {n for n, _ in n_infer_width_runs if n is not None}
        )
        if len(n_infer_vals) < 2:
            continue

        by_width = {}
        for n_infer, d_model_runs in n_infer_width_runs:
            if n_infer is None:
                continue
            for d_model, w_run_dirs in d_model_runs:
                by_width.setdefault(d_model, {}).setdefault(n_infer, []).extend(w_run_dirs)

        init_cos_data = []  # (d_model, n_infer_list, cos_mean, cos_std)
        for d_model in sorted(by_width.keys()):
            n_infer_list = []
            cos_mean = []
            cos_std = []
            for n_infer in sorted(by_width[d_model].keys()):
                vals = []
                for w_run_dir in by_width[d_model][n_infer]:
                    overall, _, _ = load_cos_sims(w_run_dir)
                    if overall is not None and len(overall) > 0:
                        vals.append(float(overall[0]))
                if not vals:
                    continue
                vals = np.asarray(vals, dtype=float)
                n_infer_list.append(n_infer)
                cos_mean.append(vals.mean())
                cos_std.append(0.0 if len(vals) <= 1 else vals.std(ddof=1))
            if n_infer_list:
                init_cos_data.append((d_model, n_infer_list, cos_mean, cos_std))

        if not init_cos_data:
            continue

        suffix_init = _file_suffix(activity_lr, None, n_activity_lr_groups, 1)
        if selected_seed is not None:
            suffix_init = f"{suffix_init}_seed_{selected_seed}"
        cos_sim_ylabel = (
            r"$\cos\left(\nabla_{\boldsymbol{\theta}} \mathcal{L}, "
            r"\nabla_{\boldsymbol{\theta}} \mathcal{F}\right)$"
        )
        plt.figure(figsize=FIG_SIZE)
        colormap = plt.get_cmap(colormap_name)
        n_w = len(init_cos_data)
        for idx, (d_model, n_infer_list, cos_mean, cos_std) in enumerate(init_cos_data):
            color = colormap(_get_color_val(idx, n_w, colormap_name))
            lbl = rf"$N = {d_model}$"
            plt.errorbar(
                n_infer_list,
                cos_mean,
                yerr=cos_std,
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
        if hasattr(ax.yaxis.get_major_formatter(), "set_useOffset"):
            ax.yaxis.get_major_formatter().set_useOffset(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                args.output_dir,
                f"grad_cosine_similarity_overall_init_vs_n_infer_iters{suffix_init}.pdf",
            ),
            bbox_inches="tight",
        )
        plt.close()

    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Base results directory under which transformer train.py saved runs (subdirs with *_d_model, steps.npy).",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Path to a .npy file in a run dir (used to infer run dir if --run_dir not set).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for PDFs (default: run_dir or same as results_file dir).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="Blues",
        help="Colormap for multi-d_model plots (default: Blues).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help=(
            "Plot metrics from this seed/replication only (run dir basename, e.g. .../0 or .../seed_0). "
            "By default, metrics are averaged across all discovered seeds with ±1 std bands."
        ),
    )
    parser.add_argument(
        "--d_models",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512],
        help=(
            "Optional list of d_model values (widths) to plot, e.g. "
            "`--d_models 8 64 512`. Defaults to all discovered widths."
        ),
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
        args.run_dir = "results"
        if args.output_dir is None:
            args.output_dir = "results"
    main(args)
