"""Activation-matched Bregman geometry and dual-state helpers.

Hidden layers use ``z^l = phi(W^l z^{l-1})`` with dual state ``u^l`` satisfying
``z^l = phi(u^l)``. The matching potential has ``Psi' = phi^{-1}``, so neither
inference nor learning needs an explicit ``phi'``.
"""

from typing import Callable, Optional, Sequence, Tuple

import jax.nn as jnn
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, ArrayLike, PyTree

_SUPPORTED = ("tanh", "sigmoid")
_EPS = 1e-6
_OUTPUT_LOSSES = ("mse", "ce", "bregman")


def check_bregman_act_fn(name: str) -> str:
    if name not in _SUPPORTED:
        raise ValueError(
            f"Unsupported Bregman activation {name!r}. Options are {_SUPPORTED}."
        )
    return name


def bregman_phi(name: str, preact: ArrayLike) -> Array:
    check_bregman_act_fn(name)
    if name == "tanh":
        return jnp.tanh(preact)
    return jnn.sigmoid(preact)


def clip_to_bregman_range(name: str, activity: ArrayLike) -> Array:
    check_bregman_act_fn(name)
    if name == "tanh":
        return jnp.clip(activity, -1.0 + _EPS, 1.0 - _EPS)
    return jnp.clip(activity, _EPS, 1.0 - _EPS)


def inv_bregman_phi(name: str, activity: ArrayLike) -> Array:
    x = clip_to_bregman_range(name, activity)
    if name == "tanh":
        return jnp.arctanh(x)
    return jnp.log(x) - jnp.log1p(-x)


def bregman_psi(name: str, activity: ArrayLike) -> Array:
    """Convex potential with ``Psi' = phi^{-1}`` (up to an additive constant)."""
    x = clip_to_bregman_range(name, activity)
    if name == "tanh":
        return x * jnp.arctanh(x) + 0.5 * jnp.log1p(-x ** 2)
    return x * jnp.log(x) + (1.0 - x) * jnp.log1p(-x)


def bregman_divergence(name: str, x: ArrayLike, y: ArrayLike) -> Array:
    """Elementwise Bregman divergence ``D_Psi(x, y)``."""
    x = clip_to_bregman_range(name, x)
    y = clip_to_bregman_range(name, y)
    return bregman_psi(name, x) - bregman_psi(name, y) - inv_bregman_phi(name, y) * (x - y)


def bregman_from_preact(name: str, x: ArrayLike, preact: ArrayLike) -> Array:
    """``D_Psi(x, phi(a))`` using the preactivation ``a`` in place of ``inv_phi(phi(a))``."""
    y = bregman_phi(name, preact)
    x = clip_to_bregman_range(name, x)
    y = clip_to_bregman_range(name, y)
    return bregman_psi(name, x) - bregman_psi(name, y) - preact * (x - y)


def get_bregman_phi(name: str) -> Callable[[ArrayLike], Array]:
    check_bregman_act_fn(name)
    return lambda a: bregman_phi(name, a)


def _check_output_loss(loss: str) -> str:
    if loss not in _OUTPUT_LOSSES:
        raise ValueError(
            f"Unsupported Bregman PC output loss {loss!r}. Options are {_OUTPUT_LOSSES}."
        )
    return loss


_MAKE_MLP_MSG = (
    "Bregman PC expects linear layers with a `.weight` (e.g. `eqx.nn.Linear`). "
    "Do not pass models from `jpc.make_mlp()`, which bake `phi` into each layer; "
    "the activation is applied separately via `act_fn`."
)


def _check_bregman_layer(layer, index: Optional[int] = None) -> None:
    if hasattr(layer, "weight"):
        return
    where = f" at index {index}" if index is not None else ""
    raise TypeError(f"{_MAKE_MLP_MSG} Got {type(layer).__name__}{where}.")


def _check_bregman_model(model: Sequence) -> None:
    if len(model) < 2:
        raise ValueError(
            "Bregman PC expects at least one hidden linear layer and a linear readout."
        )
    for i, layer in enumerate(model):
        _check_bregman_layer(layer, i)


def _check_bregman_activities(model: Sequence, activities: Sequence) -> None:
    n_hidden = len(model) - 1
    n_act = len(activities)
    if n_act != n_hidden:
        raise ValueError(
            "Bregman PC expects one dual hidden state per hidden layer "
            f"(len(model) - 1 = {n_hidden}); got {n_act}."
        )


def _bregman_model(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
) -> PyTree[Callable]:
    model, skip_model = params
    if skip_model is not None:
        raise ValueError(
            "Bregman PC does not support skip connections; pass `skip_model=None`."
        )
    return model


def _linear_weight(layer):
    _check_bregman_layer(layer)
    return layer.weight


def _hidden_states(
    model: Sequence,
    activities: Sequence[ArrayLike],
    x: ArrayLike,
    act_fn: str,
) -> Tuple[list[Array], list[Array], Array]:
    _check_bregman_model(model)
    _check_bregman_activities(model, activities)
    zs = [jnp.asarray(x)]
    for u in activities:
        zs.append(bregman_phi(act_fn, u))
    preacts = [vmap(layer)(z) for layer, z in zip(model[:-1], zs)]
    logits = vmap(model[-1])(zs[-1])
    return zs, preacts, logits


def _output_error(logits: ArrayLike, y: ArrayLike, act_fn: str, loss: str) -> Array:
    if loss == "ce":
        return y - jnn.softmax(logits)
    if loss == "mse":
        return y - logits
    return y - bregman_phi(act_fn, logits)


def _output_energy(logits: ArrayLike, y: ArrayLike, act_fn: str, loss: str) -> Array:
    if loss == "ce":
        return jnp.mean(-jnp.sum(y * jnn.log_softmax(logits), axis=-1))
    if loss == "mse":
        return 0.5 * jnp.mean(jnp.sum((logits - y) ** 2, axis=-1))
    y_clip = clip_to_bregman_range(act_fn, y)
    return jnp.mean(jnp.sum(bregman_from_preact(act_fn, y_clip, logits), axis=-1))


def bregman_pc_prediction_errors(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: ArrayLike,
    act_fn: str = "tanh",
    loss: str = "mse",
) -> Tuple[Array, ...]:
    """Layerwise errors ``eps^l = z^l - phi(a^l)`` and ``eps^L = -dL/da^L``."""
    check_bregman_act_fn(act_fn)
    _check_output_loss(loss)
    model = _bregman_model(params)
    zs, preacts, logits = _hidden_states(model, activities, x, act_fn)
    hidden_errs = [
        z - bregman_phi(act_fn, a) for z, a in zip(zs[1:], preacts)
    ]
    return tuple(hidden_errs + [_output_error(logits, y, act_fn, loss)])
