"""Plot Bregman PC, standard PC, and backprop test curves (limits-paper style).

Hyperparameter sweeps report each method at its best complementary knobs
(mean over seeds). Classification selects by highest final test accuracy;
generation selects by lowest final test reconstruction loss. Best-config
curves then show every logged metric for those winning runs.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

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
BAND_ALPHA = 0.2

_SERIES = (
    ("bp", "BP"),
    ("std_pc", "Standard PC"),
    ("bregman", "Bregman PC"),
)
_PRED_COL_TITLES = {
    "bp": "BP\n ",
    "std_pc": "Standard\nPC",
    "bregman": "Bregman\nPC",
}
_COLORS = {
    "bp": "#2ca02c",
    "std_pc": "#ff7f0e",
    "bregman": "#1f77b4",
}
_ENERGY_SERIES = _SERIES[1:]
_PC_SERIES = _ENERGY_SERIES
_COS_SERIES = (
    ("bregman", "pc_bp_cos", "Bregman PC"),
    ("std_pc", "bp_cos", "Standard PC"),
)
_DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
_DATASET_SLUGS = {
    "MNIST": "mnist",
    "Fashion-MNIST": "fashion_mnist",
    "CIFAR10": "cifar10",
    "toy": "toy",
}
_DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "toy": "Toy",
}
_DATASET_ORDER = ("mnist", "fashion_mnist", "cifar10", "toy")
_PRED_PLOT_N = 5
_IMAGE_LAYOUT = {
    "MNIST": ("hw", 28, 28, 0.1307, 0.3081),
    "Fashion-MNIST": ("hw", 28, 28, 0.5, 0.5),
    "CIFAR10": ("chw", 32, 32, 0.5, 0.5),
}
_SLUG_TO_DATASET = {slug: name for name, slug in _DATASET_SLUGS.items()}


def _image_dim_std(slug: str) -> tuple[int, float]:
    name = _SLUG_TO_DATASET.get(slug)
    if name not in _IMAGE_LAYOUT:
        return 784, 1.0
    kind, h, w, _mean, std = _IMAGE_LAYOUT[name]
    dim = (3 if kind == "chw" else 1) * h * w
    return dim, float(std)


def logged_loss_to_pixel_mse(mean, sem, slug: str) -> tuple[float, float]:
    """Convert logged ``0.5 * mean_i ||e_i||^2`` to elementwise MSE in ``[0, 1]`` pixels."""
    dim, std = _image_dim_std(slug)
    scale = 2.0 * (std**2) / dim
    if mean is None or not np.isfinite(mean):
        return np.nan, np.nan
    value = float(mean) * scale
    err = float(sem) * scale if sem is not None and np.isfinite(sem) else 0.0
    return value, err


_CONFIG_FIELDS = (
    "param_type",
    "gamma_0",
    "width",
    "param_lr",
    "activity_lr",
    "n_infer_iters",
)
_SWEEP_AXES = (
    ("param_lr", "Parameter learning rate", True, False, _SERIES),
    ("activity_lr", "Activity learning rate", True, False, _PC_SERIES),
    ("n_infer_iters", "Inference iterations", False, True, _PC_SERIES),
    ("width", "Width", True, True, _SERIES),
    ("gamma_0", r"$\gamma_0$", True, False, _SERIES),
    ("param_type", "Parameterisation", False, False, _SERIES),
)


def _even_tick_step(xmax: float) -> int:
    xmax = max(float(xmax), 1.0)
    for step in (2, 4, 6, 8, 10, 20, 50):
        if xmax / step <= 10:
            return step
    return int(np.ceil(xmax / 7 / 2.0) * 2)


def _setup_plot(
    xlabel,
    ylabel,
    log_scale=False,
    log_x=False,
    integer_xticks=True,
    even_xticks=False,
    xmax=None,
):
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
    if even_xticks and xmax is not None and np.isfinite(xmax):
        step = _even_tick_step(xmax)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(step))
    elif integer_xticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if log_x:
        plt.xscale("log")
    if log_scale:
        plt.yscale("log", base=10)


def _save_plot(save_dir, filename):
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
    plt.close()


def load_history(path: str | os.PathLike) -> dict:
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    with open(path, "rb") as f:
        return pickle.load(f)


def _log_every(history: dict) -> str:
    value = history.get("log_every", "epoch")
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return str(value)


def _x_axis(history: dict) -> np.ndarray:
    if history.get("t", np.array([])).size:
        return np.asarray(history["t"])
    if history.get("epoch", np.array([])).size:
        return np.asarray(history["epoch"])
    return np.arange(len(history["eval_step"]))


def _prepare_history(history: dict) -> dict:
    history = {k: np.asarray(v) for k, v in history.items()}
    if history.get("t", np.array([])).size or history.get("epoch", np.array([])).size:
        return history
    if "eval_step" not in history:
        raise KeyError("history must contain 't', 'epoch', or 'eval_step'")
    n = len(history["eval_step"])
    history["epoch"] = np.arange(n)
    history["t"] = history["epoch"]
    step = history.get("step")
    if step is None or len(history.get("bregman_train_energy", [])) != len(step):
        return history
    idx = np.searchsorted(step, history["eval_step"], side="right") - 1
    idx = np.clip(idx, 0, len(step) - 1)
    for key in ("bregman_train_energy", "std_pc_train_energy", "bp_train_loss"):
        if key in history:
            history[key] = history[key][idx]
    return history


def _seed_mean_sem(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    mean = float(arr.mean()) if arr.size else np.nan
    if arr.size < 2:
        return mean, 0.0
    return mean, float(arr.std(ddof=1) / np.sqrt(arr.size))


def _is_generate(task: str) -> bool:
    return task == "generate"


def _infer_task(runs: list[dict]) -> str:
    tasks = {
        str(run.get("task") or run.get("metrics", {}).get("task") or "classify")
        for run in runs
    }
    if "generate" in tasks and "classify" not in tasks:
        return "generate"
    return next(iter(tasks)) if len(tasks) == 1 else "classify"


def _history_task(history: dict) -> str:
    acc = history.get("bregman_test_acc")
    if acc is None:
        return "classify"
    if np.asarray(acc, dtype=float).size and not np.any(
        np.isfinite(np.asarray(acc, dtype=float))
    ):
        return "generate"
    return "classify"


def _score_key(model: str, task: str) -> str:
    suffix = "final_test_loss" if _is_generate(task) else "final_test_acc"
    return f"{model}_{suffix}"


def _model_scores(runs: list[dict], model: str, task: str) -> np.ndarray:
    key = _score_key(model, task)
    return np.asarray([run["metrics"].get(key) for run in runs], dtype=float)


def _finite_curves(curves):
    out = []
    for x, y, std, label, name in curves:
        y = np.asarray(y, dtype=float)
        if np.any(np.isfinite(y)):
            out.append((x, y, std, label, name))
    return out


def _plot_series(
    curves,
    ylabel,
    filename,
    save_dir,
    log_scale=False,
    xlabel="Epoch",
    log_x=False,
    integer_xticks=True,
):
    curves = _finite_curves(curves)
    if not curves:
        return
    plt.figure(figsize=FIG_SIZE)
    even_xticks = xlabel == "Epoch" and not log_x
    xmax = None
    for x, y, std, label, name in curves:
        x = np.asarray(x)
        y = np.asarray(y, dtype=float)
        if log_x:
            keep = x > 0
            x, y = x[keep], y[keep]
            if std is not None:
                std = np.asarray(std)[keep]
        if x.size:
            xmax = float(np.nanmax(x)) if xmax is None else max(xmax, float(np.nanmax(x)))
        color = _COLORS[name]
        plt.plot(x, y, label=label, alpha=ALPHA, linewidth=LINE_WIDTH, color=color)
        if std is not None and np.any(np.isfinite(std)) and np.nanmax(std) > 0:
            plt.fill_between(
                x, y - std, y + std, color=color, alpha=BAND_ALPHA, linewidth=0
            )
    _setup_plot(
        xlabel,
        ylabel,
        log_scale=log_scale,
        log_x=log_x,
        integer_xticks=integer_xticks and not even_xticks,
        even_xticks=even_xticks,
        xmax=xmax,
    )
    _save_plot(save_dir, filename)


def _history_curve(history, std_history, name, value_key, x):
    y = np.asarray(history[f"{name}_{value_key}"])
    n = min(len(x), len(y))
    std = None
    if std_history is not None and f"{name}_{value_key}" in std_history:
        std = np.asarray(std_history[f"{name}_{value_key}"])[:n]
    return x[:n], y[:n], std


def _metric_curves(history, series, value_key, x, std_history=None):
    return [
        (*_history_curve(history, std_history, name, value_key, x), label, name)
        for name, label in series
        if f"{name}_{value_key}" in history
    ]


def _mixed_metric_curves(history, series, x, std_history=None):
    return [
        (*_history_curve(history, std_history, name, value_key, x), label, name)
        for name, value_key, label in series
        if f"{name}_{value_key}" in history
    ]


def plot_metrics(
    history: dict,
    save_dir: str,
    std_history: dict | None = None,
    task: str | None = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    history = _prepare_history(history)
    if std_history is not None:
        std_history = _prepare_history(std_history)
    task = task or _history_task(history)
    generate = _is_generate(task)
    log_loss = generate
    test_x = np.asarray(history["epoch"]) if history.get("epoch", np.array([])).size else _x_axis(history)
    train_x = np.asarray(history["t"]) if history.get("t", np.array([])).size else test_x
    step_log = _log_every(history) == "step"
    train_xlabel = "Step" if step_log else "Epoch"
    train_log_x = step_log
    test_loss_label = "Test reconstruction loss" if generate else "Test loss"
    train_loss_label = "Train reconstruction loss" if generate else "Train loss"

    _plot_series(
        _metric_curves(history, _SERIES, "test_loss", test_x, std_history),
        test_loss_label,
        "test_loss.pdf",
        save_dir,
        xlabel="Epoch",
        log_scale=log_loss,
    )
    if not generate:
        _plot_series(
            _metric_curves(history, _SERIES, "test_acc", test_x, std_history),
            "Test accuracy",
            "test_acc.pdf",
            save_dir,
            xlabel="Epoch",
        )
    if all(f"{name}_train_loss" in history for name, _ in _SERIES):
        _plot_series(
            _metric_curves(history, _SERIES, "train_loss", train_x, std_history),
            train_loss_label,
            "train_loss.pdf",
            save_dir,
            xlabel=train_xlabel,
            log_x=train_log_x,
            integer_xticks=not train_log_x,
            log_scale=log_loss,
        )
    if not generate and all(f"{name}_train_acc" in history for name, _ in _SERIES):
        _plot_series(
            _metric_curves(history, _SERIES, "train_acc", train_x, std_history),
            "Train accuracy",
            "train_acc.pdf",
            save_dir,
            xlabel=train_xlabel,
            log_x=train_log_x,
            integer_xticks=not train_log_x,
        )
    _plot_series(
        _metric_curves(history, _ENERGY_SERIES, "train_energy", train_x, std_history),
        "Energy",
        "energy.pdf",
        save_dir,
        xlabel=train_xlabel,
        log_x=train_log_x,
        integer_xticks=not train_log_x,
        log_scale=log_loss,
    )
    energy_vs_bp = (
        ("bp", "train_loss", "BP"),
        ("std_pc", "train_energy", "Standard PC"),
        ("bregman", "train_energy", "Bregman PC"),
    )
    if all(f"{name}_{key}" in history for name, key, _ in energy_vs_bp):
        _plot_series(
            _mixed_metric_curves(history, energy_vs_bp, train_x, std_history),
            r"Energy / loss",
            "energy_bp_loss.pdf",
            save_dir,
            xlabel=train_xlabel,
            log_x=train_log_x,
            integer_xticks=not train_log_x,
            log_scale=log_loss,
        )
    _plot_series(
        _mixed_metric_curves(history, _COS_SERIES, train_x, std_history),
        "PC–BP gradient cosine",
        "pc_bp_cosine.pdf",
        save_dir,
        xlabel=train_xlabel,
        log_x=train_log_x,
        integer_xticks=not train_log_x,
    )


def load_runs(results_dir: str | os.PathLike) -> list[dict]:
    runs = []
    for metrics_path in sorted(Path(results_dir).rglob("metrics.json")):
        with open(metrics_path) as f:
            metrics = json.load(f)
        runs.append(
            {
                "dir": metrics_path.parent,
                "metrics": metrics,
                "dataset": str(metrics.get("dataset", "")),
                "task": str(metrics.get("task", "classify")),
                "seed": int(metrics.get("seed", 0)),
                "param_type": str(metrics.get("param_type", "sp")),
                "gamma_0": float(metrics.get("gamma_0", 1.0)),
                "width": int(metrics.get("width", 0)),
                "param_lr": float(metrics["param_lr"]),
                "activity_lr": float(metrics["activity_lr"]),
                "n_infer_iters": int(metrics["n_infer_iters"]),
            }
        )
    return runs


def _dataset_slug(run: dict) -> str:
    ds = str(run.get("dataset") or run.get("metrics", {}).get("dataset") or "")
    if ds in _DATASET_SLUGS:
        return _DATASET_SLUGS[ds]
    if ds:
        return ds.lower().replace("-", "_").replace(" ", "_")
    for part in reversed(Path(run["dir"]).parts):
        if part in _DATASET_SLUGS.values():
            return part
    return "unknown"


def plot_group_output_dir(
    results_dir: Path, save_dir: Path, slug: str, task: str, n_slugs: int
) -> Path:
    """Write into ``{dataset}/{task}`` unless the results dir already is that task folder."""
    if results_dir.name == task and n_slugs <= 1:
        return save_dir
    if n_slugs <= 1:
        return save_dir / task
    return save_dir / slug / task


def _config_key(run: dict) -> tuple:
    return tuple(run[field] for field in _CONFIG_FIELDS)


def _group_by_config(runs: list[dict]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for run in runs:
        groups[_config_key(run)].append(run)
    return dict(groups)


def _config_histories(group: list[dict]) -> list[dict]:
    histories = []
    for run in group:
        if run.get("history") is None and run.get("dir") is not None:
            try:
                run["history"] = _load_run_history(run)
            except FileNotFoundError:
                continue
        if run.get("history") is not None:
            histories.append(_prepare_history(run["history"]))
    return histories


def best_config_runs(
    runs: list[dict],
    model: str,
    task: str | None = None,
) -> list[dict]:
    """Runs sharing the hparams with the best mean seed score for ``model``."""
    task = task or _infer_task(runs)
    maximize = not _is_generate(task)
    groups = _group_by_config(runs)
    best_runs = None
    best_mean = -np.inf if maximize else np.inf
    for group in groups.values():
        mean, _ = _seed_mean_sem(_model_scores(group, model, task))
        if not np.isfinite(mean):
            continue
        if maximize and mean > best_mean:
            best_mean = mean
            best_runs = group
        elif not maximize and mean < best_mean:
            best_mean = mean
            best_runs = group
    return best_runs or []


def hparam_sweep_stats(
    runs: list[dict], model: str, axis: str, task: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Along ``axis``, each point is the best complementary-hparam config."""
    task = task or _infer_task(runs)
    xs = np.array(sorted({run[axis] for run in runs}))
    means = np.empty(len(xs), dtype=float)
    sems = np.empty(len(xs), dtype=float)
    for i, x in enumerate(xs):
        subset = [run for run in runs if run[axis] == x]
        best = best_config_runs(subset, model, task=task)
        means[i], sems[i] = _seed_mean_sem(_model_scores(best, model, task))
    return xs, means, sems


def _aggregate_histories(histories: list[dict]) -> tuple[dict, dict]:
    histories = [_prepare_history(h) for h in histories]
    keys = set.intersection(*(set(h.keys()) for h in histories))
    mean_h, std_h = {}, {}
    for key in keys:
        arrays = [np.asarray(h[key]) for h in histories]
        if arrays[0].dtype.kind in "UOSbm" or arrays[0].ndim == 0:
            mean_h[key] = arrays[0]
            std_h[key] = arrays[0]
            continue
        n = min(len(a) for a in arrays)
        stacked = np.stack([a[:n] for a in arrays], axis=0).astype(float)
        if not np.any(np.isfinite(stacked)):
            mean_h[key] = np.full(n, np.nan)
            std_h[key] = np.zeros(n)
            continue
        mean_h[key] = np.nanmean(stacked, axis=0)
        if stacked.shape[0] < 2:
            std_h[key] = np.zeros_like(mean_h[key])
        else:
            std_h[key] = np.nanstd(stacked, axis=0, ddof=1)
    return mean_h, std_h


def _load_run_history(run: dict) -> dict:
    npz_path = run["dir"] / "history.npz"
    pkl_path = run["dir"] / "history.pkl"
    if npz_path.exists():
        return load_history(npz_path)
    if pkl_path.exists():
        return load_history(pkl_path)
    raise FileNotFoundError(f"No history.npz or history.pkl in {run['dir']}")


def _plot_hparam_sweep(
    series,
    xlabel,
    ylabel,
    filename,
    save_dir,
    log_x=False,
    integer_xticks=False,
    log_scale=False,
):
    series = [
        (xs, means, sems, label, name)
        for xs, means, sems, label, name in series
        if np.any(np.isfinite(np.asarray(means, dtype=float)))
    ]
    if not series:
        return
    plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()
    for xs, means, sems, label, name in series:
        color = _COLORS[name]
        ax.errorbar(
            xs,
            means,
            yerr=sems,
            label=label,
            color=color,
            alpha=ALPHA,
            linewidth=LINE_WIDTH,
            capsize=6,
            markersize=10,
            fmt="o-",
        )
    _setup_plot(
        xlabel,
        ylabel,
        log_x=log_x,
        integer_xticks=integer_xticks,
        log_scale=log_scale,
    )
    _save_plot(save_dir, filename)


def _merge_best_histories(runs: list[dict], task: str) -> tuple[dict, dict]:
    mean_h, std_h = {}, {}
    log_every = None
    for name, _ in _SERIES:
        best = best_config_runs(runs, name, task=task)
        if not best:
            continue
        histories = _config_histories(sorted(best, key=lambda r: r["seed"]))
        if not histories:
            continue
        model_mean, model_std = _aggregate_histories(histories)
        for key, value in model_mean.items():
            if key.startswith(f"{name}_") or key in ("t", "epoch", "log_every"):
                mean_h[key] = value
                std_h[key] = model_std[key]
        if log_every is None:
            log_every = _log_every(model_mean)
    if log_every is not None:
        mean_h["log_every"] = np.asarray(log_every)
        std_h["log_every"] = np.asarray(log_every)
    return mean_h, std_h


def _best_summary(runs: list[dict], task: str) -> dict:
    maximize = not _is_generate(task)
    summary = {
        "task": task,
        "select": "final_test_acc" if maximize else "final_test_loss",
        "higher_is_better": maximize,
        "models": {},
    }
    for name, label in _SERIES:
        best = best_config_runs(runs, name, task=task)
        if not best:
            continue
        scores = _model_scores(best, name, task)
        mean, sem = _seed_mean_sem(scores)
        run0 = best[0]
        entry = {
            "label": label,
            "param_type": run0["param_type"],
            "gamma_0": run0["gamma_0"],
            "width": run0["width"],
            "param_lr": run0["param_lr"],
            "activity_lr": run0["activity_lr"],
            "n_infer_iters": run0["n_infer_iters"],
            "n_seeds": len(best),
            "seeds": sorted(int(run["seed"]) for run in best),
            "score_mean": mean,
            "score_sem": sem,
        }
        for metric_name in ("final_test_loss", "final_test_acc"):
            vals = np.asarray(
                [run["metrics"].get(f"{name}_{metric_name}") for run in best],
                dtype=float,
            )
            m, s = _seed_mean_sem(vals)
            if np.isfinite(m):
                entry[f"{metric_name}_mean"] = m
                entry[f"{metric_name}_sem"] = s
        summary["models"][name] = entry
    return summary


def _dataset_label(slug: str) -> str:
    return _DATASET_LABELS.get(slug, slug.replace("_", " ").title())


def _ordered_slugs(slugs) -> list[str]:
    slugs = list(slugs)
    known = [slug for slug in _DATASET_ORDER if slug in slugs]
    extra = sorted(slug for slug in slugs if slug not in _DATASET_ORDER)
    return known + extra


def format_mean_sem(
    mean,
    sem,
    *,
    scale: float = 1.0,
    digits: int = 2,
    tex: bool = False,
) -> str:
    if mean is None or not np.isfinite(mean):
        return r"---" if tex else "---"
    value = float(mean) * scale
    err = float(sem) * scale if sem is not None and np.isfinite(sem) else 0.0
    if tex:
        return rf"${value:.{digits}f} \pm {err:.{digits}f}$"
    return f"{value:.{digits}f} ± {err:.{digits}f}"


def _n_seeds_note(summaries_by_slug: dict[str, dict]) -> str:
    seeds = {
        int(entry["n_seeds"])
        for summary in summaries_by_slug.values()
        for entry in summary.get("models", {}).values()
        if entry.get("n_seeds") is not None
    }
    if len(seeds) == 1:
        n = next(iter(seeds))
        return f"{n} seed" if n == 1 else f"{n} seeds"
    return "seeds"


def performance_table_rows(
    summaries_by_slug: dict[str, dict],
    task: str,
    *,
    tex: bool = False,
) -> tuple[list[str], list[list[str]]]:
    slugs = _ordered_slugs(summaries_by_slug)
    generate = _is_generate(task)
    if generate:
        headers = ["Method", *(_dataset_label(slug) for slug in slugs)]
    else:
        headers = ["Method"]
        for slug in slugs:
            label = _dataset_label(slug)
            acc = rf"{label} acc (\%)" if tex else f"{label} acc (%)"
            headers.extend([acc, f"{label} loss"])
    rows = []
    for name, label in _SERIES:
        row = [label]
        for slug in slugs:
            entry = summaries_by_slug.get(slug, {}).get("models", {}).get(name, {})
            if generate:
                mse, mse_sem = logged_loss_to_pixel_mse(
                    entry.get("final_test_loss_mean"),
                    entry.get("final_test_loss_sem"),
                    slug,
                )
                row.append(format_mean_sem(mse, mse_sem, digits=5, tex=tex))
            else:
                row.append(
                    format_mean_sem(
                        entry.get("final_test_acc_mean"),
                        entry.get("final_test_acc_sem"),
                        scale=100.0,
                        digits=2,
                        tex=tex,
                    )
                )
                row.append(
                    format_mean_sem(
                        entry.get("final_test_loss_mean"),
                        entry.get("final_test_loss_sem"),
                        digits=4,
                        tex=tex,
                    )
                )
        if any(cell not in ("---", r"---") for cell in row[1:]):
            rows.append(row)
    return headers, rows


def hparams_table_rows(
    summaries_by_slug: dict[str, dict],
    *,
    tex: bool = False,
) -> tuple[list[str], list[list[str]]]:
    slugs = _ordered_slugs(summaries_by_slug)
    dash = r"---" if tex else "---"
    headers = [
        "Method",
        "Dataset",
        r"$\eta_\theta$" if tex else "param_lr",
        r"$\eta_z$" if tex else "activity_lr",
        r"$T$" if tex else "n_infer_iters",
    ]
    rows = []
    for name, label in _SERIES:
        for slug in slugs:
            entry = summaries_by_slug.get(slug, {}).get("models", {}).get(name)
            if not entry:
                continue
            is_bp = name == "bp"
            rows.append(
                [
                    label,
                    _dataset_label(slug),
                    f"{entry['param_lr']:g}",
                    dash if is_bp else f"{entry['activity_lr']:g}",
                    dash if is_bp else str(int(entry["n_infer_iters"])),
                ]
            )
    return headers, rows


def latex_performance_table(summaries_by_slug: dict[str, dict], task: str) -> str:
    slugs = _ordered_slugs(summaries_by_slug)
    generate = _is_generate(task)
    n_seed = _n_seeds_note(summaries_by_slug)
    if generate:
        caption = (
            "Generation test MSE in original $[0,1]$ pixel space "
            f"(mean $\\pm$ SEM over {n_seed}) at each method's best hyperparameters."
        )
        label = "tab:generate-performance"
        colspec = "l" + "c" * len(slugs)
        header = "Method " + " ".join(
            f"& {_dataset_label(slug)} " for slug in slugs
        ) + r"\\"
        body_rows = []
        for name, model_label in _SERIES:
            cells = [model_label]
            for slug in slugs:
                entry = summaries_by_slug.get(slug, {}).get("models", {}).get(name, {})
                mse, mse_sem = logged_loss_to_pixel_mse(
                    entry.get("final_test_loss_mean"),
                    entry.get("final_test_loss_sem"),
                    slug,
                )
                cells.append(format_mean_sem(mse, mse_sem, digits=5, tex=True))
            if any(cell != r"---" for cell in cells[1:]):
                body_rows.append(" & ".join(cells) + r" \\")
        header_block = [header]
    else:
        caption = (
            "Classification test performance (mean $\\pm$ SEM over "
            f"{n_seed}) at each method's best hyperparameters."
        )
        label = "tab:classify-performance"
        colspec = "l" + "cc" * len(slugs)
        header1 = "".join(
            rf" & \multicolumn{{2}}{{c}}{{{_dataset_label(slug)}}}" for slug in slugs
        ) + r" \\"
        cmid = "".join(
            rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(slugs))
        )
        header2 = "Method" + r" & Acc (\%) & Loss" * len(slugs) + r" \\"
        header_block = [header1, cmid, header2]
        body_rows = []
        for name, model_label in _SERIES:
            cells = [model_label]
            for slug in slugs:
                entry = summaries_by_slug.get(slug, {}).get("models", {}).get(name, {})
                cells.append(
                    format_mean_sem(
                        entry.get("final_test_acc_mean"),
                        entry.get("final_test_acc_sem"),
                        scale=100.0,
                        digits=2,
                        tex=True,
                    )
                )
                cells.append(
                    format_mean_sem(
                        entry.get("final_test_loss_mean"),
                        entry.get("final_test_loss_sem"),
                        digits=4,
                        tex=True,
                    )
                )
            if any(cell != r"---" for cell in cells[1:]):
                body_rows.append(" & ".join(cells) + r" \\")

    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *header_block,
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def latex_hparams_table(summaries_by_slug: dict[str, dict]) -> str:
    _, rows = hparams_table_rows(summaries_by_slug, tex=True)
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Selected hyperparameters for the performance table. "
        r"BP does not use activity learning rate or inference iterations.}",
        r"\label{tab:best-hparams}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Method & Dataset & $\eta_\theta$ & $\eta_z$ & $T$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def combined_task_table_rows(
    classify_summaries: dict[str, dict],
    generate_summaries: dict[str, dict],
    *,
    tex: bool = False,
) -> tuple[list[str], list[list[str]]]:
    slugs = _ordered_slugs({**classify_summaries, **generate_summaries})
    if tex:
        headers = ["Method", *[rf"{_dataset_label(s)} acc (\%)" for s in slugs]]
        headers += [rf"{_dataset_label(s)} MSE" for s in slugs]
    else:
        headers = ["Method"]
        headers += [f"{_dataset_label(s)} acc (%)" for s in slugs]
        headers += [f"{_dataset_label(s)} MSE [0,1]" for s in slugs]
    rows = []
    for name, label in _SERIES:
        row = [label]
        for slug in slugs:
            entry = classify_summaries.get(slug, {}).get("models", {}).get(name, {})
            row.append(
                format_mean_sem(
                    entry.get("final_test_acc_mean"),
                    entry.get("final_test_acc_sem"),
                    scale=100.0,
                    digits=2,
                    tex=tex,
                )
            )
        for slug in slugs:
            entry = generate_summaries.get(slug, {}).get("models", {}).get(name, {})
            mse, mse_sem = logged_loss_to_pixel_mse(
                entry.get("final_test_loss_mean"),
                entry.get("final_test_loss_sem"),
                slug,
            )
            row.append(format_mean_sem(mse, mse_sem, digits=5, tex=tex))
        if any(cell not in ("---", r"---") for cell in row[1:]):
            rows.append(row)
    return headers, rows


def latex_combined_performance_table(
    classify_summaries: dict[str, dict],
    generate_summaries: dict[str, dict],
) -> str:
    slugs = _ordered_slugs({**classify_summaries, **generate_summaries})
    n_seed = _n_seeds_note({**classify_summaries, **generate_summaries})
    n = len(slugs)
    classify_end = 1 + n
    generate_start = classify_end + 1
    generate_end = classify_end + n
    header1 = (
        rf" & \multicolumn{{{n}}}{{c}}{{Classification acc (\%)}}"
        rf" & \multicolumn{{{n}}}{{c}}{{Generation MSE $[0,1]$}} \\"
    )
    cmid = (
        rf"\cmidrule(lr){{2-{classify_end}}}"
        rf"\cmidrule(lr){{{generate_start}-{generate_end}}}"
    )
    header2 = (
        "Method"
        + "".join(rf" & {_dataset_label(slug)}" for slug in slugs)
        + "".join(rf" & {_dataset_label(slug)}" for slug in slugs)
        + r" \\"
    )
    body_rows = []
    for name, model_label in _SERIES:
        cells = [model_label]
        for slug in slugs:
            entry = classify_summaries.get(slug, {}).get("models", {}).get(name, {})
            cells.append(
                format_mean_sem(
                    entry.get("final_test_acc_mean"),
                    entry.get("final_test_acc_sem"),
                    scale=100.0,
                    digits=2,
                    tex=True,
                )
            )
        for slug in slugs:
            entry = generate_summaries.get(slug, {}).get("models", {}).get(name, {})
            mse, mse_sem = logged_loss_to_pixel_mse(
                entry.get("final_test_loss_mean"),
                entry.get("final_test_loss_sem"),
                slug,
            )
            cells.append(format_mean_sem(mse, mse_sem, digits=5, tex=True))
        if any(cell != r"---" for cell in cells[1:]):
            body_rows.append(" & ".join(cells) + r" \\")
    caption = (
        "Classification test accuracy and generation per-pixel MSE in original "
        f"$[0,1]$ pixel space (mean $\\pm$ SEM over {n_seed}). Each method uses "
        "independently selected hyperparameters per dataset and task."
    )
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:summary-performance}",
        rf"\begin{{tabular}}{{l{'c' * (2 * n)}}}",
        r"\toprule",
        header1,
        cmid,
        header2,
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def write_combined_performance_table(
    classify_summaries: dict[str, dict],
    generate_summaries: dict[str, dict],
    save_dir: str | Path,
) -> None:
    classify_summaries = {
        slug: summary
        for slug, summary in classify_summaries.items()
        if summary and summary.get("models")
    }
    generate_summaries = {
        slug: summary
        for slug, summary in generate_summaries.items()
        if summary and summary.get("models")
    }
    if not classify_summaries or not generate_summaries:
        return
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    n_seed = _n_seeds_note({**classify_summaries, **generate_summaries})
    (save_dir / "summary_performance.tex").write_text(
        latex_combined_performance_table(classify_summaries, generate_summaries)
    )
    headers, rows = combined_task_table_rows(
        classify_summaries, generate_summaries, tex=False
    )
    _plot_table_pdf(
        headers,
        rows,
        "summary_performance.pdf",
        save_dir,
        "Classify acc (%) vs generate MSE [0,1] "
        f"(mean ± SEM over {n_seed})",
    )


def _plot_table_pdf(
    headers: list[str],
    rows: list[list[str]],
    filename: str,
    save_dir: Path,
    title: str,
) -> None:
    if not rows:
        return
    n_cols = len(headers)
    fig_w = max(8.0, 2.0 * n_cols)
    fig_h = max(2.8, 0.55 * (len(rows) + 3))
    with plt.rc_context({"text.usetex": False, "font.family": "serif"}):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        fig.suptitle(title, fontsize=18)
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.0, 1.7)
        for (row_i, col_i), cell in table.get_celld().items():
            cell.set_edgecolor("#888888")
            cell.set_linewidth(0.6)
            if row_i == 0:
                cell.get_text().set_fontweight("bold")
            if col_i == 0:
                cell.get_text().set_ha("left")
        _save_plot(save_dir, filename)


def write_performance_tables(
    summaries_by_slug: dict[str, dict],
    save_dir: str | Path,
    task: str,
    prefix: str = "",
) -> None:
    """Write mean±SEM performance (and selected-hparam) tables as .tex and .pdf."""
    summaries_by_slug = {
        slug: summary
        for slug, summary in summaries_by_slug.items()
        if summary and summary.get("models")
    }
    if not summaries_by_slug:
        return
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    n_seed = _n_seeds_note(summaries_by_slug)
    (save_dir / f"{prefix}performance.tex").write_text(
        latex_performance_table(summaries_by_slug, task)
    )
    metric = "MSE in [0,1] pixels" if _is_generate(task) else "accuracy and loss"
    headers, rows = performance_table_rows(summaries_by_slug, task, tex=False)
    _plot_table_pdf(
        headers,
        rows,
        f"{prefix}performance.pdf",
        save_dir,
        f"Test {metric} (mean ± SEM over {n_seed})",
    )

    (save_dir / f"{prefix}best_hparams.tex").write_text(
        latex_hparams_table(summaries_by_slug)
    )
    hp_headers, hp_rows = hparams_table_rows(summaries_by_slug, tex=False)
    _plot_table_pdf(
        hp_headers,
        hp_rows,
        f"{prefix}best_hparams.pdf",
        save_dir,
        "Best hyperparameters (independently selected per method)",
    )


def _batch_to_images(flat, dataset: str) -> np.ndarray:
    x = np.asarray(flat)
    kind, h, w, mean, std = _IMAGE_LAYOUT[dataset]
    if kind == "chw":
        x = x.reshape(x.shape[0], 3, h, w) * std + mean
        x = np.transpose(x, (0, 2, 3, 1))
    else:
        x = x.reshape(x.shape[0], h, w) * std + mean
    return np.clip(x, 0.0, 1.0)


def write_best_generate_predictions(runs: list[dict], save_dir: str | Path) -> None:
    """Write a target vs BP / Standard PC / Bregman PC grid at each method's best config."""
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    labels = targets = dataset = None
    preds = {}
    for name, _ in _SERIES:
        best = best_config_runs(runs, name, task="generate")
        if not best:
            continue
        run = min(best, key=lambda r: r["seed"])
        npz_path = Path(run["dir"]) / "predictions.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        pred_key = f"{name}_pred"
        if pred_key not in data:
            continue
        if labels is None:
            labels = np.asarray(data["labels"])
            targets = np.asarray(data["targets"])
            dataset = str(run.get("dataset") or run["metrics"].get("dataset") or "")
        preds[name] = np.asarray(data[pred_key])
    if targets is None or dataset not in _IMAGE_LAYOUT or not preds:
        return

    n_plot = min(_PRED_PLOT_N, targets.shape[0])
    series = [(name, label) for name, label in _SERIES if name in preds]
    cols = 1 + len(series)
    target_img = _batch_to_images(targets[:n_plot], dataset)
    pred_imgs = {
        name: _batch_to_images(pred[:n_plot], dataset) for name, pred in preds.items()
    }
    np.savez(
        save_dir / "predictions.npz",
        labels=labels[:n_plot],
        targets=targets[:n_plot],
        **{f"{name}_pred": pred[:n_plot] for name, pred in preds.items()},
    )
    with plt.rc_context({"text.usetex": False, "font.family": "serif"}):
        fig, axes = plt.subplots(
            n_plot, cols, figsize=(4.2 * cols, 3.4 * n_plot), squeeze=False
        )
        titles = ("target\n ", *(_PRED_COL_TITLES.get(name, label) for name, label in series))
        for row in range(n_plot):
            panels = [target_img[row]] + [pred_imgs[name][row] for name, _ in series]
            for col, (ax, img) in enumerate(zip(axes[row], panels)):
                cmap = None if img.ndim == 3 else "gray"
                ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(
                        titles[col], fontsize=FONT_SIZES["label"], pad=LABEL_PAD
                    )
        fig.tight_layout()
        fig.savefig(save_dir / "predictions.pdf", bbox_inches="tight")
        fig.savefig(save_dir / "predictions.png", bbox_inches="tight")
        plt.close(fig)
    print(f"  best-config predictions -> {save_dir / 'predictions.png'}")


def _plot_sweep_runs(
    runs: list[dict],
    save_dir: str | Path,
    task: str,
    make_plots: bool = True,
) -> dict:
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    slug = _dataset_slug(runs[0])
    generate = _is_generate(task)
    if make_plots:
        ylabel = "Test reconstruction loss" if generate else "Test accuracy"
        metric_slug = "test_loss" if generate else "test_acc"
        for axis, xlabel, log_x, integer_xticks, series in _SWEEP_AXES:
            xs_all = {run[axis] for run in runs}
            if len(xs_all) < 2:
                continue
            plotted = []
            for name, label in series:
                xs, means, sems = hparam_sweep_stats(runs, name, axis, task=task)
                plotted.append((xs, means, sems, label, name))
            _plot_hparam_sweep(
                plotted,
                xlabel,
                ylabel,
                f"{metric_slug}_vs_{axis}.pdf",
                save_dir,
                log_x=log_x,
                integer_xticks=integer_xticks,
                log_scale=generate,
            )

    summary = _best_summary(runs, task)
    with open(save_dir / "best_configs.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_performance_tables({slug: summary}, save_dir, task)
    if generate:
        write_best_generate_predictions(runs, save_dir)

    if make_plots:
        mean_h, std_h = _merge_best_histories(runs, task)
        if mean_h:
            plot_metrics(mean_h, str(save_dir), std_history=std_h, task=task)
    return summary


def plot_sweep(
    results_dir: str | os.PathLike,
    save_dir: str | os.PathLike | None = None,
    make_plots: bool = True,
) -> None:
    results_dir = Path(results_dir)
    save_dir = Path(save_dir or results_dir)
    os.makedirs(save_dir, exist_ok=True)
    runs = load_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No metrics.json under {results_dir}")

    groups = defaultdict(list)
    for run in runs:
        groups[(_dataset_slug(run), run["task"])].append(run)
    n_slugs = len({slug for slug, _ in groups})
    by_task = defaultdict(dict)
    for (slug, task), task_runs in groups.items():
        out = plot_group_output_dir(results_dir, save_dir, slug, task, n_slugs)
        action = "plotting" if make_plots else "tabulating"
        print(f"{action} {slug}/{task}: {len(task_runs)} runs -> {out}")
        by_task[task][slug] = _plot_sweep_runs(
            task_runs, out, task, make_plots=make_plots
        )

    if n_slugs > 1:
        for task, summaries in by_task.items():
            print(f"  combined {task} tables -> {save_dir}/{task}_performance.tex")
            write_performance_tables(summaries, save_dir, task, prefix=f"{task}_")
    if "classify" in by_task and "generate" in by_task:
        print(f"  summary table -> {save_dir}/summary_performance.tex")
        write_combined_performance_table(
            by_task["classify"], by_task["generate"], save_dir
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot Bregman PC vs Standard PC vs BP curves.")
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results tree with metrics.json files. Plots every dataset/task found.",
    )
    p.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to history.npz or history.pkl.",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Where to write PDFs. Defaults to the results or history directory.",
    )
    p.add_argument(
        "--tables-only",
        action="store_true",
        help="Write mean±SEM performance tables without regenerating plots.",
    )
    args = p.parse_args()
    if args.results_dir:
        plot_sweep(args.results_dir, args.save_dir, make_plots=not args.tables_only)
    elif args.history:
        history_path = Path(args.history)
        if not history_path.exists() and history_path.suffix == ".npz":
            history_path = history_path.with_suffix(".pkl")
        plot_metrics(load_history(history_path), args.save_dir or str(history_path.parent))
    else:
        plot_sweep(_DEFAULT_RESULTS, args.save_dir, make_plots=not args.tables_only)
