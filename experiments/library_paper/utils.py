import os
import random
import numpy as np
from torch import manual_seed
from diffrax import Euler, Heun, Midpoint, Ralston, Bosh3, Tsit5, Dopri5, Dopri8


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    manual_seed(seed)


def get_ode_solver(name):
    if name == "Euler":
        return Euler()
    elif name == "Heun":
        return Heun()
    elif name == "Midpoint":
        return Midpoint()
    elif name == "Ralston":
        return Ralston()
    elif name == "Bosh3":
        return Bosh3()
    elif name == "Tsit5":
        return Tsit5()
    elif name == "Dopri5":
        return Dopri5()
    elif name == "Dopri8":
        return Dopri8()


def setup_mlp_experiment(
        results_dir,
        dataset,
        width,
        n_hidden,
        act_fn,
        max_t1,
        activity_lr,
        param_lr,
        activity_optim_id,
        seed
):
    print(
        f"""
Starting experiment with configuration:

  Dataset: {dataset}
  Width: {width}
  N hidden: {n_hidden}
  Act fn: {act_fn}
  Max t1: {max_t1}
  Activity step size: {activity_lr}
  Param learning rate: {param_lr}
  Activity optim: {activity_optim_id}
  Seed: {seed}
"""
    )
    return os.path.join(
        results_dir,
        dataset,
        f"width_{width}",
        f"{n_hidden}_n_hidden",
        act_fn,
        f"max_t1_{max_t1}",
        f"activity_lr_{activity_lr}",
        f"param_lr_{param_lr}",
        activity_optim_id,
        str(seed)
    )


def get_min_iter(lists):
    min_iter = 100000
    for i in lists:
        if len(i) < min_iter:
            min_iter = len(i)
    return min_iter


def get_min_iter_metrics(metrics):
    n_seeds = len(metrics)
    min_iter = get_min_iter(lists=metrics)

    min_iter_metrics = np.zeros((n_seeds, min_iter))
    for seed in range(n_seeds):
        min_iter_metrics[seed, :] = metrics[seed][:min_iter]

    return min_iter_metrics


def compute_metric_stats(metric):
    min_iter_metrics = get_min_iter_metrics(metrics=metric)
    metric_means = min_iter_metrics.mean(axis=0)
    metric_stds = min_iter_metrics.std(axis=0)
    return metric_means, metric_stds
