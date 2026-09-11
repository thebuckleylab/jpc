"""CNN-only PC helpers: two depth counts and per-layer hidden precision.

MLP training still uses ``jpc`` with a scalar ``κ = L``. This module is used
only for the ResNet path in ``train_benchmark.py``, so ``jpc`` and
``limits_paper`` stay unchanged.

``L_fwd`` is the residual µP depth (``1/√L`` on ``ResNetBlock``s, Adam
residual LRs). It is realised by passing
``additive_depth_factor = L_fwd - n_res_blocks`` into the existing
``ResNet`` constructor.

``L_energy`` is the PC–BP matching depth: ``λ = γ² N L_energy`` and
``κ = L_energy`` on selected hidden layers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, ArrayLike, PyTree, Scalar
from typing import Callable, Optional, Sequence, Tuple, Union

_CNN_DIR = Path(__file__).resolve().parents[1] / "limits_paper" / "cnn"
if str(_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_DIR))

from model import ResNetBlock, ScaledConv2d  # noqa: E402

RESNET_L_PRESETS = ("n_res_blocks", "n_weight_layers", "n_modules")
HIDDEN_ENERGY_LAYER_CHOICES = ("weight", "all", "residual")

ResnetLSpec = Union[str, int]


def parse_resnet_l_arg(value: str) -> ResnetLSpec:
    """Argparse type: preset name or a positive integer."""
    key = str(value).strip()
    if key in RESNET_L_PRESETS:
        return key
    try:
        parsed = int(key)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ResNet L '{value}'. Use {', '.join(RESNET_L_PRESETS)} "
            "or a positive integer."
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"ResNet L must be a positive integer, got {parsed}."
        )
    return parsed


def _as_int(value) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) < 1:
            raise ValueError("Expected a non-empty list of ints.")
        value = value[0]
    return int(value)


def resolve_resnet_l(spec: ResnetLSpec, n_res_blocks: int) -> int:
    """Map a preset or int to a concrete depth."""
    n = _as_int(n_res_blocks)
    if isinstance(spec, bool):
        raise ValueError("ResNet L spec cannot be a bool.")
    if isinstance(spec, int):
        val = int(spec)
    elif spec == "n_res_blocks":
        val = n
    elif spec == "n_weight_layers":
        val = n + 4
    elif spec == "n_modules":
        val = n + 7
    else:
        raise ValueError(
            f"Unknown ResNet L spec '{spec}'. "
            f"Use {', '.join(RESNET_L_PRESETS)} or a positive int."
        )
    if val <= 0:
        raise ValueError(f"ResNet L must be positive, got {val}.")
    return val


def resolve_cnn_ls(args) -> Tuple[int, int]:
    """Return ``(L_fwd, L_energy)`` from CNN CLI flags.

    ``--additive_depth_factor``, when set, overrides ``--resnet_energy_l``
    with ``n_res_blocks + factor``.
    """
    n = _as_int(args.n_res_blocks)
    fwd_l = resolve_resnet_l(
        getattr(args, "resnet_fwd_l", "n_res_blocks"), n
    )
    if getattr(args, "additive_depth_factor", None) is not None:
        energy_l = n + _as_int(args.additive_depth_factor)
        if energy_l <= 0:
            raise ValueError(
                f"n_res_blocks + additive_depth_factor must be positive, "
                f"got {energy_l}."
            )
    else:
        energy_l = resolve_resnet_l(
            getattr(args, "resnet_energy_l", "n_weight_layers"), n
        )
    return fwd_l, energy_l


def resnet_fwd_additive_depth_factor(fwd_l: int, n_res_blocks: int) -> int:
    """Value to pass to ``ResNet(..., additive_depth_factor=...)``.

    ``ResNet`` uses ``depth = n_res_blocks + additive_depth_factor`` for
    residual ``1/√L``. Choosing ``additive = L_fwd - n_res_blocks`` makes
    that depth equal ``L_fwd`` without changing ``limits_paper``.
    """
    return _as_int(fwd_l) - _as_int(n_res_blocks)


def _hidden_layer_kind(layer) -> str:
    if isinstance(layer, nn.AvgPool2d):
        return "pool"
    if isinstance(layer, ResNetBlock):
        return "residual"
    if isinstance(layer, ScaledConv2d):
        return "stem"
    return "other"


def cnn_hidden_energy_scales(
    model,
    kappa: float,
    hidden_energy_layers: str,
) -> jnp.ndarray:
    """Per-hidden-layer κ for a ResNet (length ``len(model) - 1``).

    The readout is the output term (scaled by ``λ``, not this vector).

    * ``all``: every hidden module, including pools, gets ``κ``.
    * ``weight`` (default): stems and residual blocks get ``κ``; pools get 1.
    * ``residual``: only ``ResNetBlock``s get ``κ``; stems and pools get 1.
    """
    if hidden_energy_layers not in HIDDEN_ENERGY_LAYER_CHOICES:
        raise ValueError(
            f"Unknown --hidden_energy_layers '{hidden_energy_layers}'. "
            f"Use {', '.join(HIDDEN_ENERGY_LAYER_CHOICES)}."
        )
    kappa = float(kappa)
    scales = []
    for layer in model.layers[:-1]:
        kind = _hidden_layer_kind(layer)
        if hidden_energy_layers == "all":
            scales.append(kappa)
        elif kind == "pool":
            scales.append(1.0)
        elif hidden_energy_layers == "residual":
            scales.append(kappa if kind == "residual" else 1.0)
        else:
            scales.append(kappa)
    return jnp.asarray(scales, dtype=jnp.float32)


def _as_hidden_scales(hidden_energy_scaling, n_hidden: int) -> jnp.ndarray:
    if hidden_energy_scaling is None:
        return jnp.ones((n_hidden,), dtype=jnp.float32)
    scales = jnp.asarray(hidden_energy_scaling, dtype=jnp.float32)
    return jnp.broadcast_to(scales, (n_hidden,))


def pc_energy_fn_layered(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: Optional[ArrayLike] = None,
    loss: str = "mse",
    output_energy_scaling: Optional[Scalar] = None,
    hidden_energy_scaling: Optional[Union[Scalar, Array, Sequence[float]]] = None,
) -> Scalar:
    """PC energy with optional per-hidden-layer ``κ``.

    CNN µP already lives inside ``ResNet``, so jpc MLP scalings are not
    applied (same as ``pc_jpc_kwargs`` using ``param_type='sp'``).
    """
    model, skip_model = params
    batch_size = y.shape[0]
    start_activity_l = 1 if x is not None else 2
    n_activity_layers = len(activities) - 1
    n_hidden = len(model) - 1

    if skip_model is None:
        skip_model = [None] * len(model)

    output_scale = (
        1.0 if output_energy_scaling is None else output_energy_scaling
    )
    hidden_scales = _as_hidden_scales(hidden_energy_scaling, n_hidden)

    if loss == "mse":
        eL = y - vmap(model[-1])(activities[-2])
        energies = [0.5 * output_scale * jnp.sum(eL ** 2)]
    elif loss == "ce":
        logits = vmap(model[-1])(activities[-2])
        energies = [-output_scale * jnp.sum(y * jax.nn.log_softmax(logits))]
    else:
        raise ValueError(f"Unknown loss '{loss}'.")

    for act_l, net_l in zip(
        range(start_activity_l, n_activity_layers),
        range(1, n_hidden),
    ):
        err = activities[act_l] - vmap(model[net_l])(activities[act_l - 1])
        if skip_model[net_l] is not None:
            err = err - vmap(skip_model[net_l])(activities[act_l - 1])
        energies.append(0.5 * hidden_scales[net_l] * jnp.sum(err ** 2))

    if x is not None:
        e1 = activities[0] - vmap(model[0])(x)
    else:
        e1 = activities[1] - vmap(model[0])(activities[0])
    energies.append(0.5 * hidden_scales[0] * jnp.sum(e1 ** 2))

    return jnp.sum(jnp.stack(energies)) / batch_size


@eqx.filter_jit
def update_pc_activities(
    params,
    activities,
    optim,
    opt_state,
    output,
    *,
    input=None,
    loss_id="mse",
    output_energy_scaling=None,
    hidden_energy_scaling=None,
):
    energy, grads = jax.value_and_grad(pc_energy_fn_layered, argnums=1)(
        params,
        activities,
        output,
        x=input,
        loss=loss_id,
        output_energy_scaling=output_energy_scaling,
        hidden_energy_scaling=hidden_energy_scaling,
    )
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=activities,
    )
    activities = eqx.apply_updates(model=activities, updates=updates)
    return {
        "energy": energy,
        "activities": activities,
        "grads": grads,
        "opt_state": opt_state,
    }


@eqx.filter_jit
def update_pc_params(
    params,
    activities,
    optim,
    opt_state,
    output,
    *,
    input=None,
    loss_id="mse",
    output_energy_scaling=None,
    hidden_energy_scaling=None,
):
    grads = eqx.filter_grad(pc_energy_fn_layered)(
        params,
        activities,
        output,
        x=input,
        loss=loss_id,
        output_energy_scaling=output_energy_scaling,
        hidden_energy_scaling=hidden_energy_scaling,
    )
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=params,
    )
    model, skip_model = eqx.apply_updates(model=params, updates=updates)
    return {
        "model": model,
        "skip_model": skip_model,
        "grads": grads,
        "opt_state": opt_state,
    }
