import os
import random
import numpy as np
from torch import manual_seed
import jax.random as jr
import jax.numpy as jnp
from jax.tree_util import tree_leaves
import equinox as eqx
from diffrax import Euler, Heun, Midpoint, Ralston, Bosh3, Tsit5, Dopri5, Dopri8
from jpc import (
    linear_activities_coeff_matrix,
    compute_activity_grad,
    compute_pc_param_grads
)


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

def setup_mlp_experiment_test(
        results_dir,
        dataset,
        n_hidden,
        act_fn,
        weight_init_type,
        activity_init,
        max_t1,
        lr,
        weight_decay,
        activity_optim_id,
        seed
):
    print(
        f"""
Starting experiment with configuration:

  Dataset: {dataset}
  N hidden: {n_hidden}
  Act fn: {act_fn}
  Max t1: {max_t1}
  Activity optim: {activity_optim_id}
  Seed: {seed}
"""
    )
    return os.path.join(
        results_dir,
        dataset,
        f"{n_hidden}_n_hidden",
        act_fn,
        f"max_t1_{max_t1}",
        activity_optim_id,
        str(seed)
    )


def setup_cnn_experiment(
        results_dir,
        dataset,
        use_skips,
        act_fn,
        init_type,
        loss,
        optim_id,
        lr,
        batch_size,
        ode_solver,
        max_t1,
        seed
):
    print(
        f"""
Starting experiment with configuration:

  Dataset: {dataset}
  Use skips: {use_skips}
  Act fn: {act_fn}
  Init type: {init_type}
  Loss: {loss}
  Optim: {optim_id}
  Learning rate: {lr}
  Batch size: {batch_size}
  ODE solver: {ode_solver}
  Max t1: {max_t1}
  Seed: {seed}
"""
    )
    return os.path.join(
        results_dir,
        dataset,
        "skips" if use_skips else "no_skips",
        act_fn,
        f"{init_type}_init",
        f"{init_type}_loss",
        optim_id,
        f"lr_{lr}",
        f"batch_{batch_size}",
        ode_solver,
        f"max_t1_{max_t1}",
        str(seed)
    )


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


def origin_init(weight, std_dev, key):
    if len(weight.shape) == 2:
        out, in_ = weight.shape
        return std_dev * jr.normal(key, shape=(out, in_))
    elif len(weight.shape) == 4:
        out, in_, kh, kw = weight.shape
        return std_dev * jr.normal(key, shape=(out, in_, kh, kw))


def init_weights(model, init_fn, std_dev, key):
    is_linear_or_conv = lambda x: isinstance(x, (eqx.nn.Linear, eqx.nn.Conv2d))
    get_weights = lambda m: [x.weight
                             for x in tree_leaves(m, is_leaf=is_linear_or_conv)
                             if is_linear_or_conv(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, std_dev, subkey)
                   for weight, subkey in zip(weights, jr.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def get_network_weights(network):
    weights = [network[l][0].weight for l in range(len(network))]
    return weights


@eqx.filter_jit
def compute_network_metrics(network):
    weights = get_network_weights(network)
    coeff_matrix = linear_activities_coeff_matrix(weights)
    rank = jnp.linalg.matrix_rank(coeff_matrix)
    cond_num = jnp.linalg.cond(coeff_matrix)
    return {
        "coeff_matrix": coeff_matrix,
        "rank": rank,
        "cond_num": cond_num
    }


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
