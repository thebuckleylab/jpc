"""Plot Bregman PC, standard PC, and backprop test curves (limits-paper style)."""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

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
_ENERGY_SERIES = _SERIES[1:]
_COS_SERIES = (
    ("bregman", "Bregman PC"),
    ("std", "Standard PC"),
)
_PC_SERIES = _ENERGY_SERIES
_DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "mnist"
_CONFIG_FIELDS = (
    "param_type",
    "gamma_0",
    "width",
    "param_lr",
    "activity_lr",
    "n_infer_iters",
)


def _setup_plot(xlabel, ylabel, log_scale=False, log_x=False, integer_xticks=True):
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
    if integer_xticks:
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


def _seed_mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if arr.size else np.nan
    std = 0.0 if arr.size < 2 else float(arr.std(ddof=1))
    return mean, std


def _plot_series(
    curves,
    ylabel,
    filename,
    save_dir,
    ylim=None,
    skip_colors=0,
    log_scale=False,
    xlabel="Epoch",
    log_x=False,
    integer_xticks=True,
):
    plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()
    for _ in range(skip_colors):
        ax._get_lines.get_next_color()
    for x, y, std, label in curves:
        color = ax._get_lines.get_next_color()
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
        integer_xticks=integer_xticks,
    )
    if ylim is not None:
        plt.ylim(*ylim)
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
        (*_history_curve(history, std_history, name, value_key, x), label)
        for name, label in series
        if f"{name}_{value_key}" in history
    ]


def plot_metrics(history: dict, save_dir: str, std_history: dict | None = None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    history = _prepare_history(history)
    if std_history is not None:
        std_history = _prepare_history(std_history)
    test_x = np.asarray(history["epoch"]) if history.get("epoch", np.array([])).size else _x_axis(history)
    train_x = np.asarray(history["t"]) if history.get("t", np.array([])).size else test_x
    train_xlabel = "Step" if _log_every(history) == "step" else "Epoch"
    _plot_series(
        _metric_curves(history, _SERIES, "test_loss", test_x, std_history),
        "Test loss",
        "test_loss.pdf",
        save_dir,
        log_scale=True,
        xlabel="Epoch",
    )
    _plot_series(
        _metric_curves(history, _SERIES, "test_acc", test_x, std_history),
        "Test accuracy",
        "test_acc.pdf",
        save_dir,
        ylim=(0.0, 1.0),
        xlabel="Epoch",
    )
    if all(f"{name}_train_loss" in history for name, _ in _SERIES):
        _plot_series(
            _metric_curves(history, _SERIES, "train_loss", train_x, std_history),
            "Train loss",
            "train_loss.pdf",
            save_dir,
            log_scale=True,
            xlabel=train_xlabel,
        )
    if all(f"{name}_train_acc" in history for name, _ in _SERIES):
        _plot_series(
            _metric_curves(history, _SERIES, "train_acc", train_x, std_history),
            "Train accuracy",
            "train_acc.pdf",
            save_dir,
            ylim=(0.0, 1.0),
            xlabel=train_xlabel,
        )
    _plot_series(
        _metric_curves(history, _ENERGY_SERIES, "train_energy", train_x, std_history),
        "Energy",
        "energy.pdf",
        save_dir,
        skip_colors=1,
        log_scale=True,
        xlabel=train_xlabel,
    )
    if all(f"{name}_pc_bp_cos" in history for name, _ in _COS_SERIES):
        _plot_series(
            _metric_curves(history, _COS_SERIES, "pc_bp_cos", train_x, std_history),
            "PC--BP grad cosine",
            "grad_cosine.pdf",
            save_dir,
            ylim=(-1.05, 1.05),
            xlabel=train_xlabel,
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


def _config_key(run: dict) -> tuple:
    return tuple(run[field] for field in _CONFIG_FIELDS)


def _group_by_config(runs: list[dict]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for run in runs:
        groups[_config_key(run)].append(run)
    return dict(groups)


def _model_accs(runs: list[dict], model: str) -> np.ndarray:
    return np.asarray(
        [run["metrics"][f"{model}_final_test_acc"] for run in runs], dtype=float
    )


def best_config_runs(runs: list[dict], model: str) -> list[dict]:
    groups = _group_by_config(runs)
    best_runs = None
    best_mean = -np.inf
    for group in groups.values():
        mean, _ = _seed_mean_std(_model_accs(group, model))
        if mean > best_mean:
            best_mean = mean
            best_runs = group
    return best_runs or []


def hparam_sweep_stats(
    runs: list[dict], model: str, axis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array(sorted({run[axis] for run in runs}))
    means = np.empty(len(xs))
    stds = np.empty(len(xs))
    for i, x in enumerate(xs):
        subset = [run for run in runs if run[axis] == x]
        best = best_config_runs(subset, model)
        means[i], stds[i] = _seed_mean_std(_model_accs(best, model))
    return xs, means, stds


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
        stacked = np.stack([a[:n] for a in arrays], axis=0)
        mean_h[key] = stacked.mean(axis=0)
        if stacked.shape[0] < 2:
            std_h[key] = np.zeros_like(mean_h[key])
        else:
            std_h[key] = stacked.std(axis=0, ddof=1)
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
    series, xlabel, filename, save_dir, log_x=False, integer_xticks=False, skip_colors=0
):
    plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()
    for _ in range(skip_colors):
        ax._get_lines.get_next_color()
    for xs, means, stds, label in series:
        color = ax._get_lines.get_next_color()
        ax.errorbar(
            xs,
            means,
            yerr=stds,
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
        "Test accuracy",
        log_x=log_x,
        integer_xticks=integer_xticks,
    )
    plt.ylim(0.0, 1.0)
    _save_plot(save_dir, filename)


def _merge_best_histories(runs: list[dict]) -> tuple[dict, dict]:
    mean_h, std_h = {}, {}
    log_every = None
    for name, _ in _SERIES:
        best = best_config_runs(runs, name)
        if not best:
            continue
        histories = [
            _load_run_history(run) for run in sorted(best, key=lambda r: r["seed"])
        ]
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


def plot_sweep(results_dir: str | os.PathLike, save_dir: str | os.PathLike | None = None) -> None:
    results_dir = Path(results_dir)
    save_dir = str(save_dir or results_dir)
    os.makedirs(save_dir, exist_ok=True)
    runs = load_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No metrics.json under {results_dir}")

    param_series = []
    for name, label in _SERIES:
        xs, means, stds = hparam_sweep_stats(runs, name, "param_lr")
        param_series.append((xs, means, stds, label))
    _plot_hparam_sweep(
        param_series,
        "Parameter learning rate",
        "test_acc_vs_param_lr.pdf",
        save_dir,
        log_x=True,
    )

    activity_series = []
    for name, label in _PC_SERIES:
        xs, means, stds = hparam_sweep_stats(runs, name, "activity_lr")
        activity_series.append((xs, means, stds, label))
    _plot_hparam_sweep(
        activity_series,
        "Activity learning rate",
        "test_acc_vs_activity_lr.pdf",
        save_dir,
        log_x=True,
        skip_colors=1,
    )

    infer_series = []
    for name, label in _PC_SERIES:
        xs, means, stds = hparam_sweep_stats(runs, name, "n_infer_iters")
        infer_series.append((xs, means, stds, label))
    _plot_hparam_sweep(
        infer_series,
        "Inference iterations",
        "test_acc_vs_n_infer_iters.pdf",
        save_dir,
        integer_xticks=True,
        skip_colors=1,
    )

    mean_h, std_h = _merge_best_histories(runs)
    plot_metrics(mean_h, save_dir, std_history=std_h)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot Bregman PC vs Standard PC vs BP curves.")
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results tree with metrics.json files. Writes hparam sweeps and best-config curves.",
    )
    p.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to history.npz or history.pkl. Defaults to results/mnist/history.npz.",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Where to write PDFs. Defaults to the results or history directory.",
    )
    args = p.parse_args()
    if args.results_dir:
        plot_sweep(args.results_dir, args.save_dir)
    elif args.history:
        history_path = Path(args.history)
        if not history_path.exists() and history_path.suffix == ".npz":
            history_path = history_path.with_suffix(".pkl")
        plot_metrics(load_history(history_path), args.save_dir or str(history_path.parent))
    else:
        results_root = _DEFAULT_RESULTS.parent
        candidates = [
            path for path in sorted(results_root.iterdir())
            if path.is_dir() and any(path.rglob("metrics.json"))
        ]
        if _DEFAULT_RESULTS in candidates:
            plot_sweep(_DEFAULT_RESULTS, args.save_dir)
        elif candidates:
            plot_sweep(candidates[0], args.save_dir)
        else:
            history_path = _DEFAULT_RESULTS / "history.npz"
            if not history_path.exists():
                history_path = history_path.with_suffix(".pkl")
            plot_metrics(load_history(history_path), args.save_dir or str(history_path.parent))
