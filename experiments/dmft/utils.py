"""
Shared training utilities live in ``experiments.limits_paper.utils``. This
module overrides the BP ``MLP`` so µPC weights are N(0, 1), matching
``jpc.make_mlp``, and hosts the finite-size PC/BP training loops.
"""

import os
import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import jpc
from experiments.limits_paper.utils import MLP as LimitsMLP
from experiments.limits_paper.utils import configure_param_optim, flatten_grads


def create_toy_dataset(key, D, P):
    X = jr.normal(key, (D, P))
    y = jnp.where(jnp.arange(P) < P//2, 1.0, -1.0)
    return X, y


def cosine_similarity(a, b, axis=None, eps=1e-8):
    """
    Computes cosine similarity between two vectors/matrices/kernels.

    Args:
        a: jnp.ndarray, input array (vector, matrix, or kernel).
        b: jnp.ndarray, same shape as a.
        axis: Axis or axes along which to compute similarity. 
            If None, uses the last axis for vectors and (0, 1) for 2D matrices/kernels (unless a/b are 1D).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Cosine similarity as a float or array, depending on axis.
        Values are in the range [-1, 1].
    """
    # Flatten for 1D vectors
    if a.ndim == 1 or b.ndim == 1:
        num = jnp.dot(a, b)
        denom = jnp.linalg.norm(a) * jnp.linalg.norm(b) + eps
        return num / denom

    # Default to last axis (for batches of vectors), or (0,1) for 2D kernels
    if axis is None:
        axis = -1 if a.ndim == 2 else (0, 1) if a.ndim == 2 else None

    # Compute numerator and denominator
    num = jnp.sum(a * b, axis=axis)
    denom = (
        jnp.linalg.norm(a, axis=axis) * jnp.linalg.norm(b, axis=axis) + eps
    )
    return num / denom


class MLP(LimitsMLP):
    def __init__(
            self,
            key,
            d_in,
            N,
            L,
            d_out,
            act_fn,
            param_type,
            gamma,
            use_bias=False,
            use_skips=False
    ):
        super().__init__(
            key,
            d_in,
            N,
            L,
            d_out,
            act_fn,
            param_type,
            gamma,
            use_bias=use_bias,
            use_skips=use_skips,
        )
        # µPC applies explicit forward scalings, so weights must be N(0, 1)
        # (same convention as jpc.make_mlp). Equinox's default 1/sqrt(fan_in)
        # init would otherwise double-scale and kill learning.
        if param_type != "mupc":
            return
        keys = jr.split(key, L)
        layers = []
        for i, layer in enumerate(self.layers):
            linear = layer[1]
            W = jr.normal(keys[i], linear.weight.shape)
            linear = eqx.tree_at(lambda l: l.weight, linear, W)
            layers.append(eqx.tree_at(lambda s: s[1], layer, linear))
        object.__setattr__(self, "layers", layers)


def get_output_energy_scaling(
    param_type: str, gamma_0: float, width: int, depth: int
) -> float:
    """µPC output precision λ = γ² N L (SP: 1)."""
    return (gamma_0 ** 2) * width * depth if param_type == "mupc" else 1.0


def get_hidden_energy_scaling(param_type: str, depth: int) -> float:
    """µPC hidden precision κ = L (SP: 1)."""
    return float(depth) if param_type == "mupc" else 1.0


def cleanup_experiment_dirs(results_dir: str):
    """Remove finite-sim result trees (``*_input_dim``), keeping plot pngs."""
    removed = []
    root = Path(results_dir)
    if not root.exists():
        return removed
    for path in sorted(root.glob("*_input_dim")):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def train_pcn(
      model,
      use_skips,
      X_input,
      Y_target,
      width,
      gamma_0,
      param_type,
      infer_mode,
      n_infer_iters,
      activity_lr,
      param_optim_id,
      param_lr,
      n_train_iters,
      loss_id,
      save_dir,
      store_grads=False
):
    """Train a PC network.

    Parameter / activity updates follow the finite-size convention used by
    ``get_coord_data``: plain ``param_lr`` with
    ``output_energy_scaling = gamma^2 * width * depth`` and
    ``hidden_energy_scaling = depth`` for µPC (rather than baking the
    width/depth factor into the optimiser learning rate).
    """
    os.makedirs(save_dir, exist_ok=True)

    depth = len(model)
    skip_model = jpc.make_skip_model(depth) if use_skips else None
    output_energy_scaling = get_output_energy_scaling(
        param_type, gamma_0, width, depth
    )
    hidden_energy_scaling = get_hidden_energy_scaling(param_type, depth)

    # Optimisers (plain lr; µPC width/gamma/depth scaling via energy terms)
    batch_size = X_input.shape[0]
    activity_optim = optax.sgd(activity_lr * batch_size)
    if param_optim_id == "gd":
        param_optim = optax.sgd(param_lr)
    elif param_optim_id == "adam":
        param_optim = optax.adam(param_lr)
    else:
        raise ValueError(f"Invalid optimiser: {param_optim_id}")
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )

    num_energies, theory_energies = [], []
    train_losses = []
    loss_rescalings = []
    pc_grads = [] if store_grads else None

    for _ in range(n_train_iters):

        # Record supervised loss on the current feedforward prediction *before*
        # the parameter update, matching get_coord_data / DMFT step indexing.
        activities = jpc.init_activities_with_ffwd(
            model=model,
            input=X_input,
            skip_model=skip_model,
            param_type=param_type,
            gamma=gamma_0
        )
        if loss_id == "mse":
            train_loss = jpc.mse_loss(activities[-1], Y_target)
        else:
            train_loss = jpc.cross_entropy_loss(activities[-1], Y_target)
        train_losses.append(train_loss)

        if infer_mode == "closed_form":
            equilib_energy, S = jpc.linear_equilib_energy(
                params=(model, skip_model),
                x=X_input,
                y=Y_target,
                param_type=param_type,
                gamma=gamma_0,
                return_rescaling=True,
                output_energy_scaling=output_energy_scaling,
                hidden_energy_scaling=hidden_energy_scaling,
            )
            theory_energies.append(equilib_energy)
            loss_rescaling = jnp.linalg.norm(S, ord=2) if Y_target.ndim > 1 else S
            loss_rescalings.append(loss_rescaling)

        # inference
        if infer_mode == "optim":
            activity_opt_state = activity_optim.init(activities)
            for _ in range(n_infer_iters):
                activity_update_result = jpc.update_pc_activities(
                    params=(model, skip_model),
                    activities=activities,
                    optim=activity_optim,
                    opt_state=activity_opt_state,
                    output=Y_target,
                    input=X_input,
                    param_type=param_type,
                    gamma=gamma_0,
                    loss_id=loss_id,
                    output_energy_scaling=output_energy_scaling,
                    hidden_energy_scaling=hidden_energy_scaling,
                )
                activities = activity_update_result["activities"]
                activity_opt_state = activity_update_result["opt_state"]
                energy = activity_update_result["energy"]

            num_energies.append(energy)

            param_update_result = jpc.update_pc_params(
                params=(model, skip_model),
                activities=activities,
                optim=param_optim,
                opt_state=param_opt_state,
                output=Y_target,
                input=X_input,
                param_type=param_type,
                gamma=gamma_0,
                loss_id=loss_id,
                output_energy_scaling=output_energy_scaling,
                hidden_energy_scaling=hidden_energy_scaling,
            )

        else:
            # learning with closed form energy
            param_update_result = jpc.update_linear_equilib_energy_params(
                params=(model, skip_model),
                optim=param_optim,
                opt_state=param_opt_state,
                y=Y_target,
                x=X_input,
                param_type=param_type,
                gamma=gamma_0,
                output_energy_scaling=output_energy_scaling,
                hidden_energy_scaling=hidden_energy_scaling,
            )

        model = param_update_result["model"]
        skip_model = param_update_result["skip_model"]
        param_opt_state = param_update_result["opt_state"]
        grads = param_update_result["grads"]

        if pc_grads is not None:
            flat_grads = flatten_grads(grads)
            # Convert JAX array to numpy immediately to free memory
            pc_grads.append(np.array(flat_grads))
            del flat_grads, grads

    energies = (
        jnp.array(theory_energies)
        if infer_mode == "closed_form"
        else jnp.array(num_energies)
    )
    np.save(f"{save_dir}/energies.npy", energies)
    np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_dir}/loss_rescalings.npy", loss_rescalings)

    return pc_grads


def train_bpn(
      model,
      use_skips,
      X_input,
      Y_target,
      width,
      gamma_0,
      param_type,
      optim_id,
      param_lr,
      n_train_iters,
      loss_id,
      save_dir,
      store_grads=False
):
    os.makedirs(save_dir, exist_ok=True)

    # Optimiser
    optim = configure_param_optim(
        optim_id, param_type, use_skips, param_lr, width, model.L, gamma_0
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    if loss_id == "mse":
        @eqx.filter_jit
        def loss_fn(model, x, y):
            y_pred = jax.vmap(model)(x)
            return 0.5 * jnp.mean(jnp.sum((y - y_pred) ** 2, axis=1))
    else:
        @eqx.filter_jit
        def loss_fn(model, x, y):
            y_pred = jax.vmap(model)(x)
            return jpc.cross_entropy_loss(y_pred, y)

    @eqx.filter_jit
    def make_step(model, optim, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            updates=grads,
            state=opt_state,
            params=eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, grads

    losses = []
    bp_grads = [] if store_grads else None

    for _ in range(n_train_iters):
        # Record loss before the parameter update to match get_Delta / DMFT
        # step indexing (pre-update residual).
        if loss_id == "mse":
            y_pred = jax.vmap(model)(X_input)
            train_loss = float(
                0.5 * jnp.mean(jnp.sum((Y_target - y_pred) ** 2, axis=1))
            )
        else:
            y_pred = jax.vmap(model)(X_input)
            train_loss = float(jpc.cross_entropy_loss(y_pred, Y_target))
        losses.append(train_loss)

        model, opt_state, _, grads = make_step(
            model, optim, opt_state, X_input, Y_target
        )

        if bp_grads is not None:
            flat_grads = flatten_grads(grads)
            # Convert JAX array to numpy immediately to free memory
            bp_grads.append(np.array(flat_grads))
            del flat_grads, grads

    np.save(f"{save_dir}/losses.npy", losses)

    return bp_grads
