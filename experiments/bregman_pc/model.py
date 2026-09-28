from __future__ import annotations

import math
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray

import jpc


class ScaledLinear(eqx.Module):
    """``y = scaling * W x`` with ``W`` the learnable matrix (no bias).

    Hidden-to-hidden residual layers add an identity skip: ``y = scaling * W x + x``.
    """

    linear: eqx.nn.Linear
    scaling: float = eqx.field(static=True)
    use_skip: bool = eqx.field(static=True)

    def __call__(self, x: ArrayLike, *, key=None) -> Array:
        y = self.scaling * self.linear(x)
        if self.use_skip:
            y = y + x
        return y

    @property
    def weight(self) -> Array:
        return self.scaling * self.linear.weight


def layer_scalings(
    layer_sizes: Sequence[int],
    param_type: str,
    gamma: float,
    use_skips: bool = False,
) -> list[float]:
    """Per-layer forward scalings. Output scaling includes ``1/gamma``.

    Residual μPC hidden maps use ``1/sqrt(N L)`` instead of ``1/sqrt(N)``, matching
    the limits-paper residual MLP.
    """
    d_ins = list(layer_sizes[:-1])
    L = len(d_ins)
    if param_type == "sp":
        scales = [1.0] * L
    elif param_type == "mupc":
        scales = []
        for i, d_in in enumerate(d_ins):
            if i == 0:
                scales.append(1.0 / math.sqrt(d_in))
            elif i < L - 1:
                hidden_scale = math.sqrt(d_in * L) if use_skips else math.sqrt(d_in)
                scales.append(1.0 / hidden_scale)
            else:
                scales.append(1.0 / d_in)
    else:
        raise ValueError(f"Unknown param_type {param_type!r}. Options are 'sp' and 'mupc'.")
    scales[-1] = scales[-1] / gamma
    return scales


def scaled_param_lr(
    param_type: str,
    param_optim: str,
    param_lr: float,
    width: int,
    depth: int,
    gamma: float = 1.0,
    use_skips: bool = False,
) -> float:
    """Width/depth learning-rate multipliers from the limits-paper μPC setup."""
    if param_type == "sp":
        return param_lr
    if param_optim == "sgd":
        return param_lr * (gamma**2) * width
    if param_optim == "adam":
        if use_skips:
            return param_lr / (math.sqrt(width) * math.sqrt(depth))
        return param_lr / math.sqrt(width)
    raise ValueError(f"Unknown param optim {param_optim!r}.")


class BregmanMLP(eqx.Module):
    """Fully connected net ``x^l = phi(W^l x^{l-1})`` with a linear readout.

    With ``use_skips=True``, hidden-to-hidden maps are residual
    ``x^l = phi(W^l x^{l-1} + x^{l-1})`` (identity skip, matching the
    limits-paper residual MLP; no skip on the stem or readout).
    """

    layers: tuple[ScaledLinear, ...]
    act_fn: str = eqx.field(static=True)
    output_loss: str = eqx.field(static=True)
    param_type: str = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    use_skips: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        layer_sizes: Sequence[int],
        act_fn: str = "tanh",
        output_loss: str = "ce",
        init_scale: float | None = None,
        param_type: str = "sp",
        gamma: float = 1.0,
        use_skips: bool = False,
    ):
        jpc.check_bregman_act_fn(act_fn)
        if output_loss not in ("ce", "mse", "bregman"):
            raise ValueError("output_loss must be 'ce', 'mse', or 'bregman'.")
        if param_type not in ("sp", "mupc"):
            raise ValueError("param_type must be 'sp' or 'mupc'.")
        if len(layer_sizes) < 3:
            raise ValueError("Need input, at least one hidden, and output sizes.")
        if init_scale is not None and init_scale <= 0.0:
            raise ValueError("init_scale must be positive.")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive.")

        keys = jr.split(key, len(layer_sizes) - 1)
        scales = layer_scalings(layer_sizes, param_type, gamma, use_skips=use_skips)
        n_maps = len(layer_sizes) - 1
        layers = []
        for i, (k, d_in, d_out, scale) in enumerate(
            zip(keys, layer_sizes[:-1], layer_sizes[1:], scales)
        ):
            # Identity skips only on hidden-to-hidden maps (not stem or readout).
            skip = bool(use_skips and 0 < i < n_maps - 1)
            if skip and d_in != d_out:
                raise ValueError(
                    "Identity skips require matching hidden widths; "
                    f"got {d_in} → {d_out} at layer {i}."
                )
            k_w, k_lin = jr.split(k)
            linear = eqx.nn.Linear(d_in, d_out, use_bias=False, key=k_lin)
            if param_type == "mupc":
                var = init_scale if init_scale is not None else 1.0
            else:
                var = init_scale if init_scale is not None else 1.0 / d_in
            weight = jnp.sqrt(var) * jr.normal(k_w, linear.weight.shape)
            linear = eqx.tree_at(lambda lin: lin.weight, linear, weight)
            layers.append(ScaledLinear(linear=linear, scaling=scale, use_skip=skip))
        self.layers = tuple(layers)
        self.act_fn = act_fn
        self.output_loss = output_loss
        self.param_type = param_type
        self.gamma = gamma
        self.use_skips = bool(use_skips)

    def jpc_params(self):
        return (self.layers, None)

    def replace_layers(self, layers) -> "BregmanMLP":
        return eqx.tree_at(lambda m: m.layers, self, tuple(layers))

    def forward(self, x0: ArrayLike) -> Array:
        x = jnp.asarray(x0)
        for layer in self.layers[:-1]:
            x = jpc.bregman_phi(self.act_fn, jax.vmap(layer)(x))
        return jax.vmap(self.layers[-1])(x)
