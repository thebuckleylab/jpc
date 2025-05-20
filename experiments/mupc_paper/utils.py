import os
import random
import numpy as np
from torch import manual_seed

import jax
import jax.random as jr
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from jaxlib.xla_extension import PjitFunction, ArrayImpl

import jpc
import equinox as eqx


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    manual_seed(seed)


def setup_hessian_analysis(
        results_dir,
        in_out_dims,
        act_fn,
        use_biases,
        mode,
        n_skip,
        weight_init,
        param_type,
        activity_decay,
        width,
        n_hidden,
        seed
):
    print(
        f"""
Starting Hessian analysis with configuration:

  Input output dims: {in_out_dims}
  Act fn: {act_fn}
  Use biases: {use_biases}
  Mode: {mode}
  N skip: {n_skip}
  Weight init: {weight_init}
  Param type: {param_type}
  Activity decay: {activity_decay}
  Width: {width}
  N hidden: {n_hidden}
  Seed: {seed}
"""
    )
    use_biases = "biases" if use_biases else "no_biases"
    activity_decay = "activity_decay" if activity_decay else "activity_decay_0"
    return os.path.join(
        results_dir,
        f"{in_out_dims}_in_out_dims",
        act_fn,
        use_biases,
        mode,
        f"{n_skip}_skip",
        f"{weight_init}_weight_init",
        f"{param_type}_param",
        activity_decay,
        f"width_{width}",
        f"{n_hidden}_n_hidden",
        str(seed)
    )


def setup_experiment(
        results_dir,
        dataset,
        width,
        n_hidden,
        act_fn,
        n_skip,
        weight_init,
        param_type,
        param_optim_id,
        param_lr,
        batch_size,
        max_infer_iters,
        activity_optim_id,
        activity_lr,
        activity_decay,
        weight_decay,
        spectral_penalty,
        max_epochs,
        seed
):
    print(
        f"""
Starting training experiment with configuration:

  Dataset: {dataset}
  Width: {width}
  N hidden: {n_hidden}
  Act fn: {act_fn}
  N skip: {n_skip}
  Weight init: {weight_init}
  Param type: {param_type}
  Param optim: {param_optim_id}
  Param lr: {param_lr}
  Batch size: {batch_size}
  Max infer iters: {max_infer_iters}
  Activity optim: {activity_optim_id}
  Activity lr: {activity_lr}
  Activity decay: {activity_decay}
  Weight decay: {weight_decay}
  Spectral penalty: {spectral_penalty}
  Max epochs: {max_epochs}
  Seed: {seed}
"""
    )
    return os.path.join(
        results_dir,
        dataset,
        f"width_{width}",
        f"{n_hidden}_n_hidden",
        act_fn,
        f"{n_skip}_skip",
        f"{weight_init}_weight_init",
        f"{param_type}_param",
        f"param_optim_{param_optim_id}",
        f"param_lr_{param_lr}",
        f"batch_size_{batch_size}",
        f"{max_infer_iters}_max_infer_iters",
        f"activity_optim_{activity_optim_id}",
        f"activity_lr_{activity_lr}",
        f"activity_decay_{activity_decay}",
        f"weight_decay_{weight_decay}",
        f"spectral_penalty_{spectral_penalty}",
        f"{max_epochs}_epochs",
        str(seed)
    )


def setup_chain_experiment(
        results_dir,
        n_hidden,
        param_type,
        n_skip,
        activity_init,
        act_fn,
        activity_lr,
        n_infer_iters,
        n_train_iters,
        param_lr,
        batch_size,
        seed
):
    print(
        f"""
Starting training chain experiment with configuration:

  N hidden: {n_hidden}
  Param type: {param_type}
  N skip: {n_skip}
  Activity init: {activity_init}
  Act fn: {act_fn}
  Activity lr: {activity_lr}
  N infer iters: {n_infer_iters}
  N train iters: {n_train_iters}
  Param lr: {param_lr}
  Batch size: {batch_size}
  Seed: {seed}
"""
    )
    return os.path.join(
        results_dir,
        f"{n_hidden}_n_hidden",
        f"{param_type}_param",
        f"{n_skip}_skip",
        f"{activity_init}_activity_init",
        act_fn,
        f"activity_lr_{activity_lr}",
        f"{n_infer_iters}_n_infer_iters",
        f"{n_train_iters}_n_train_iters",
        f"param_lr_{param_lr}",
        f"batch_size_{batch_size}",
        str(seed)
    )


def setup_cnn_experiment(
        results_dir,
        dataset,
        loss_id,
        init_width,
        n_hidden,
        act_fn,
        n_skip,
        weight_init,
        param_type,
        param_optim_id,
        param_lr,
        batch_size,
        max_infer_iters,
        activity_optim_id,
        activity_lr,
        weight_decay,
        max_epochs,
        seed
):
    print(
        f"""
Starting training experiment with configuration:

  Dataset: {dataset}
  Loss: {loss_id}
  Init width: {init_width}
  N hidden: {n_hidden}
  Act fn: {act_fn}
  N skip: {n_skip}
  Weight init: {weight_init}
  Param type: {param_type}
  Param optim: {param_optim_id}
  Param lr: {param_lr}
  Batch size: {batch_size}
  Max infer iters: {max_infer_iters}
  Activity optim: {activity_optim_id}
  Activity lr: {activity_lr}
  Weight decay: {weight_decay}
  Max epochs: {max_epochs}
  Seed: {seed}
"""
    )
    return os.path.join(
        results_dir,
        dataset,
        loss_id,
        f"init_width_{init_width}",
        f"{n_hidden}_n_hidden",
        act_fn,
        f"{n_skip}_skip",
        f"{weight_init}_weight_init",
        f"{param_type}_param",
        f"param_optim_{param_optim_id}",
        f"param_lr_{param_lr}",
        f"batch_size_{batch_size}",
        f"{max_infer_iters}_max_infer_iters",
        f"activity_optim_{activity_optim_id}",
        f"activity_lr_{activity_lr}",
        f"weight_decay_{weight_decay}",
        f"{max_epochs}_epochs",
        str(seed)
    )


def init_weights(key, model, init_fn_id, gain=1.0):
    is_linear_or_conv = lambda x: isinstance(x, (eqx.nn.Linear, eqx.nn.Conv2d))
    get_weights = lambda m: [x.weight
                             for x in tree_leaves(m, is_leaf=is_linear_or_conv)
                             if is_linear_or_conv(x)]
    weights = get_weights(model)

    subkeys = jr.split(key, len(weights))
    if init_fn_id == "one_over_N":
        new_weights = [one_over_width_init(subkey, weight)
                       for weight, subkey in zip(weights, subkeys)]
    elif init_fn_id == "standard_gauss":
        new_weights = [standard_gauss_init(subkey, weight)
                       for weight, subkey in zip(weights, subkeys)]
    elif init_fn_id == "orthogonal":
        new_weights = [orthogonal_init(subkey, weight, gain)
                       for weight, subkey in zip(weights, subkeys)]
    elif init_fn_id == "zero":
        new_weights = [zero_init(weight) for weight in weights]

    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def standard_gauss_init(key, weight):
    if len(weight.shape) == 2:
        out, in_ = weight.shape
        return jr.normal(key, shape=(out, in_))
    elif len(weight.shape) == 4:
        out, in_, kh, kw = weight.shape
        return jr.normal(key, shape=(out, in_, kh, kw))


def one_over_width_init(key, weight):
    out, in_ = weight.shape
    return jr.normal(key, shape=(out, in_)) / np.sqrt(in_)


def orthogonal_init(key, weight, gain=1.0):
    """
    Orthogonal initialization using QR decomposition
    """
    out, in_ = weight.shape
    shape = (max(out, in_), min(out, in_))

    M = jr.normal(key, shape=shape)
    Q, R = jnp.linalg.qr(M)

    # make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    Q *= jnp.sign(jnp.diag(R))

    # handle non-square matrices
    if out < in_:
        Q = Q.T

    return gain * Q[:out, :in_]


def zero_init(weight):
    return jnp.zeros_like(weight)


def get_weight_init(weight_init):
    if weight_init == "one_over_N":
        return one_over_width_init
    elif weight_init == "standard_gauss":
        return standard_gauss_init
    elif weight_init == "orthogonal":
        return orthogonal_init
    elif weight_init == "zero":
        return zero_init


def get_network_weights(network):
    weights = [network[l][1].weight for l in range(len(network))]
    return weights


@eqx.filter_jit
def compute_param_l2_norms(model, act_fn, layer_idxs):
    all_params = tree_leaves(model)
    if act_fn != "linear":
        all_params = [p for i, p in enumerate(all_params) if i % 2 == 0]

    selected_params = [
        all_params[idx] if (
                idx < len(all_params) and all_params[idx] is not None
        ) else None
        for idx in layer_idxs
    ]

    return jnp.array([
        jnp.linalg.norm(jnp.ravel(p), ord=2) if (
                p is not None and not isinstance(p, (PjitFunction, ArrayImpl))
        ) else 0.
        for p in selected_params
    ])


def spectral_norm(A):
    s = jnp.linalg.svd(A, compute_uv=False)
    return s[0]


@eqx.filter_jit
def compute_param_spectral_norms(model, act_fn, layer_idxs):
    all_params = tree_leaves(model)
    if act_fn != "linear":
        all_params = [p for i, p in enumerate(all_params) if i % 2 == 0]

    selected_params = [
        all_params[idx] if (
                idx < len(all_params) and all_params[idx] is not None
        ) else None
        for idx in layer_idxs
    ]

    def compute_spectral_norm(param):
        if param is None or isinstance(param, (PjitFunction, ArrayImpl)):
            return 0.
        if param.ndim == 1:  # if 1D, treat as a column vector
            param = param.reshape(-1, 1)

        return spectral_norm(param)

    return jnp.array([
        compute_spectral_norm(p) for p in selected_params
    ])


@eqx.filter_jit
def compute_hessian_eigens(
        params,
        activities,
        y,
        x,
        n_skip,
        param_type,
        activity_decay=0,
        weight_decay=0,
        spectral_penalty=0,
):
    hessian_pytree = jax.hessian(jpc.pc_energy_fn, argnums=1)(
        params,
        activities,
        y,
        x,
        n_skip=n_skip,
        param_type=param_type,
        activity_decay=activity_decay,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
    )
    H = unwrap_hessian_pytree(hessian_pytree, activities)
    eigenvals, eigenvecs = jnp.linalg.eigh(H)
    return eigenvals, eigenvecs


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


def unwrap_hessian_pytree(hessian_pytree, activities):
    activities = activities[:-1]
    hessian_pytree = hessian_pytree[:-1]

    widths = [a.shape[1] for a in activities]
    N = sum(widths)
    hessian_matrix = jnp.zeros((N, N))

    start_row_idx = 0
    for l, pytree_l in enumerate(hessian_pytree):
        import jax

        start_col_idx = 0
        for k, pytree_k in enumerate(pytree_l[:-1]):            
            block = pytree_k[0, :, 0].reshape(widths[l], widths[k]) #.sum(axis=(0, 1))

            hessian_matrix = hessian_matrix.at[
                start_row_idx:start_row_idx + widths[l],
                start_col_idx:start_col_idx + widths[k]
            ].set(block)

            start_col_idx += widths[k]

        start_row_idx += widths[l]

    return hessian_matrix


def compute_cond_num(eigenvals):
    return np.abs(max(eigenvals))/np.abs(min(eigenvals))
