"""Theoretical equilibrated energy for a linear CNN."""

import math
from typing import List, Tuple

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import optax

from model import ScaledConv2d, ResNet


def _get_layer_output_shapes(model: ResNet) -> List[Tuple[int, int, int]]:
    """Return (C, H, W) for the output of each layer 0..L-2 (all layers except readout)."""
    layers = model.layers
    in_features = layers[-1].linear.weight.shape[1]
    width = layers[0].conv.weight.shape[0]
    in_channels = layers[0].conv.weight.shape[1]
    spatial_side = int(math.sqrt(in_features // width))
    # 3 pools: input_size -> input_size/2 -> input_size/4 -> input_size/8 = spatial_side
    input_size = spatial_side * (2**3)

    shapes = []
    c, h, w = in_channels, input_size, input_size
    for i in range(len(layers) - 1):  # output shape of each layer except readout
        layer = layers[i]
        if isinstance(layer, ScaledConv2d):
            c = layer.conv.weight.shape[0]
        elif isinstance(layer, nn.AvgPool2d):
            h, w = h // 2, w // 2
        # ResNetBlock: (c, h, w) unchanged
        shapes.append((c, h, w))
    return shapes


def _forward_from_layer(model: ResNet, layer_idx: int, h: jnp.ndarray) -> jnp.ndarray:
    """Map from layer layer_idx's output (flattened h) through the rest of the network to output."""
    for f in model.layers[layer_idx + 1 :]:
        h = f(h)
    return h


def _build_S_batched(
    model: ResNet,
    batch_size: int,
    d_y: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """
    Build S = I + sum_ell G_ell G_ell^T by basis propagation.
    G_ell is the map from layer ell output (flattened) to network output; we add
    G_ell G_ell^T without forming G_ell by iterating over basis vectors in chunks.
    """
    shapes = _get_layer_output_shapes(model)
    S = jnp.eye(d_y, dtype=dtype)

    for layer_idx, shape in enumerate(shapes):
        C, H, W = shape
        D = C * H * W

        def basis_to_col(j: jnp.ndarray) -> jnp.ndarray:
            e = jnp.zeros(D, dtype=dtype).at[j].set(1.0)
            h = e.reshape(shape)
            return _forward_from_layer(model, layer_idx, h)

        num_chunks = (D + batch_size - 1) // batch_size

        def scan_step(S, chunk_i):
            inds = chunk_i * batch_size + jnp.arange(batch_size)
            inds_safe = jnp.minimum(inds, D - 1)
            cols = jax.vmap(basis_to_col)(inds_safe)
            valid = inds < D
            cols_masked = jnp.where(valid[:, None], cols, 0.0)
            return S + cols_masked.T @ cols_masked, None

        S, _ = jax.lax.scan(scan_step, S, jnp.arange(num_chunks))

    return S


@eqx.filter_jit
def linear_cnn_equilib_energy(
    model: ResNet,
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    batch_size: int = 64,
    return_rescaling: bool = False,
):
    """
    Closed-form equilibrated energy for LinearCNN:

      F* = (1/2N) sum_i (y_i - W x_i)^T S^{-1} (y_i - W x_i)

    with S = I + sum_ell G_ell G_ell^T (see module docstring). S is built by
    basis propagation; batch_size controls the chunk size for the scan.

    Returns equilib_energy; if return_rescaling=True, returns (equilib_energy, S).
    """
    d_y = y.shape[1]
    dtype = y.dtype
    S = _build_S_batched(model, batch_size, d_y, dtype)
    residuals = y - jax.vmap(model)(x)
    energies = jax.vmap(lambda r: 0.5 * r.T @ jnp.linalg.solve(S, r))(residuals)
    equilib_energy = jnp.mean(energies)
    if return_rescaling:
        return equilib_energy, S
    return equilib_energy


def _linear_cnn_equilib_energy_fixed_S(
    model: ResNet,
    x: jnp.ndarray,
    y: jnp.ndarray,
    S: jnp.ndarray,
):
    """Equilibrated energy with a fixed rescaling matrix S (no gradients through S)."""
    residuals = y - jax.vmap(model)(x)
    energies = jax.vmap(lambda r: 0.5 * r.T @ jnp.linalg.solve(S, r))(residuals)
    return jnp.mean(energies)


@eqx.filter_jit
def compute_linear_cnn_equilib_energy_grads(
    model: ResNet,
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    batch_size: int = 64,
    include_dSdtheta: bool = True,
):
    """Gradients of the closed-form equilibrated energy for LinearCNN.

    This is the CNN analogue of ``jpc.compute_linear_equilib_energy_grads`` used
    for deep linear MLPs. It differentiates ``linear_cnn_equilib_energy`` with
    respect to the model parameters.

    If ``include_dSdtheta`` is True (default), this returns the exact gradient
    of F* and differentiates through the construction of S, which can be very
    memory-intensive. If False, S is treated as a fixed preconditioner: we
    first build S at the current parameters and then differentiate an energy
    that holds S constant, dropping the ∂S/∂θ terms to save memory.
    """
    if include_dSdtheta:
        return eqx.filter_grad(linear_cnn_equilib_energy)(
            model,
            x,
            y,
            batch_size=batch_size,
            return_rescaling=False,
        )

    d_y = y.shape[1]
    dtype = y.dtype
    S = _build_S_batched(model, batch_size, d_y, dtype)

    def energy_given_model(m):
        return _linear_cnn_equilib_energy_fixed_S(m, x, y, S)

    return eqx.filter_grad(energy_given_model)(model)


@eqx.filter_jit
def update_linear_cnn_equilib_energy_params(
    model: ResNet,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    batch_size: int = 64,
    include_dSdtheta: bool = True,
):
    """Update LinearCNN parameters using gradients of the equilibrated energy.

    Mirrors ``jpc.update_linear_equilib_energy_params`` for the CNN case: it
    computes ∇_θ 𝔽* for ``linear_cnn_equilib_energy`` and applies one optax
    update step.
    """
    grads = compute_linear_cnn_equilib_energy_grads(
        model,
        x,
        y,
        batch_size=batch_size,
        include_dSdtheta=include_dSdtheta,
    )
    updates, opt_state = optim.update(
        grads,
        opt_state,
        params=eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    return {
        "model": model,
        "grads": grads,
        "opt_state": opt_state,
    }
