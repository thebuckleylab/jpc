"""Training steps that call jpc Bregman PC and standard PC."""

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, PyTree
from optax import GradientTransformation, OptState

import jpc

from .evaluate import feedforward_loss
from .model import BregmanMLP


def jpc_loss_id(output_loss: str) -> str:
    return "ce" if output_loss == "ce" else "mse"


def bregman_mlp_to_jpc(model: BregmanMLP) -> list:
    """Same linear maps as BregmanMLP, with ``phi`` after each hidden ``W``."""
    phi_fn = jpc.get_bregman_phi(model.act_fn)
    layers = []
    for i, lin in enumerate(model.layers):
        if i < len(model.layers) - 1:
            layers.append(nn.Sequential([lin, nn.Lambda(phi_fn)]))
        else:
            layers.append(lin)
    return layers


def init_jpc_opt_state(model: PyTree, optim: GradientTransformation) -> OptState:
    return optim.init((eqx.filter(model, eqx.is_array), None))


def _bregman_infer(
    model: BregmanMLP,
    x0: ArrayLike,
    y: ArrayLike,
    n_iters: int,
    step_size: float,
):
    us = jpc.init_bregman_pc_activities(model.layers, x0, act_fn=model.act_fn)
    act_optim = optax.sgd(step_size)
    act_state = act_optim.init(us)
    params = model.jpc_params()

    def body(carry, _):
        acts, st = carry
        out = jpc.update_bregman_pc_activities(
            params,
            acts,
            act_optim,
            st,
            y,
            input=x0,
            act_fn=model.act_fn,
            loss=model.output_loss,
        )
        return (out["activities"], out["opt_state"]), out["energy"]

    (us, _), energies = jax.lax.scan(body, (us, act_state), xs=None, length=n_iters)
    return us, energies[-1]


def _standard_infer(
    model: PyTree,
    x0: ArrayLike,
    y: ArrayLike,
    n_iters: int,
    step_size: float,
    loss_id: str,
):
    activities = jpc.init_activities_with_ffwd(model=model, input=x0)
    act_optim = optax.sgd(step_size * x0.shape[0])
    act_state = act_optim.init(activities)

    def body(carry, _):
        acts, st = carry
        out = jpc.update_pc_activities(
            params=(model, None),
            activities=acts,
            optim=act_optim,
            opt_state=st,
            output=y,
            input=x0,
            loss_id=loss_id,
        )
        return (out["activities"], out["opt_state"]), out["energy"]

    (activities, _), energies = jax.lax.scan(
        body, (activities, act_state), xs=None, length=n_iters
    )
    return activities, energies[-1]


@eqx.filter_jit
def bregman_pc_energy(
    model: BregmanMLP,
    x0: ArrayLike,
    y: ArrayLike,
    n_iters: int,
    step_size: float,
) -> Array:
    _, energy = _bregman_infer(model, x0, y, n_iters, step_size)
    return energy


@eqx.filter_jit
def standard_pc_energy(
    model: PyTree,
    x0: ArrayLike,
    y: ArrayLike,
    n_iters: int,
    step_size: float,
    loss_id: str,
) -> Array:
    _, energy = _standard_infer(model, x0, y, n_iters, step_size, loss_id)
    return energy


@eqx.filter_jit
def bregman_pc_step(
    model: BregmanMLP,
    x0: ArrayLike,
    y: ArrayLike,
    optim: GradientTransformation,
    opt_state: OptState,
    n_iters: int,
    step_size: float,
) -> tuple[BregmanMLP, OptState, Array]:
    us, energy = _bregman_infer(model, x0, y, n_iters, step_size)
    result = jpc.update_bregman_pc_params(
        model.jpc_params(),
        us,
        optim,
        opt_state,
        y,
        input=x0,
        act_fn=model.act_fn,
        loss=model.output_loss,
    )
    return model.replace_layers(result["model"]), result["opt_state"], energy


@eqx.filter_jit
def standard_pc_step(
    model: PyTree,
    x0: ArrayLike,
    y: ArrayLike,
    optim: GradientTransformation,
    opt_state: OptState,
    n_iters: int,
    step_size: float,
    loss_id: str,
) -> tuple[PyTree, OptState, Array]:
    activities, energy = _standard_infer(
        model, x0, y, n_iters, step_size, loss_id
    )
    result = jpc.update_pc_params(
        params=(model, None),
        activities=activities,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=x0,
        loss_id=loss_id,
    )
    return result["model"], result["opt_state"], energy


def _flat_params(tree: PyTree) -> Array:
    flat, _ = ravel_pytree(eqx.filter(tree, eqx.is_array))
    return flat


def _cosine(a: Array, b: Array) -> Array:
    denom = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    return jnp.where(denom > 1e-12, jnp.dot(a, b) / denom, 0.0)


def _jpc_ff_loss(model: PyTree, x0: ArrayLike, y: ArrayLike, loss_id: str) -> Array:
    acts = jpc.init_activities_with_ffwd(model=model, input=x0)
    if loss_id == "ce":
        return jpc.cross_entropy_loss(acts[-1], y)
    return jpc.mse_loss(acts[-1], y)


@eqx.filter_jit
def bregman_pc_bp_grad_cosine(
    model: BregmanMLP,
    x0: ArrayLike,
    y: ArrayLike,
    n_iters: int,
    step_size: float,
) -> Array:
    """Cosine similarity of Bregman PC and BP parameter grads on the same weights."""
    us, _ = _bregman_infer(model, x0, y, n_iters, step_size)
    pc_grads = jpc.compute_bregman_pc_param_grads(
        model.jpc_params(),
        us,
        y,
        x=x0,
        act_fn=model.act_fn,
        loss=model.output_loss,
    )
    bp_grads = eqx.filter_grad(feedforward_loss)(model, x0, y)
    return _cosine(_flat_params(pc_grads[0]), _flat_params(bp_grads.layers))


@eqx.filter_jit
def standard_pc_bp_grad_cosine(
    model: PyTree,
    x0: ArrayLike,
    y: ArrayLike,
    n_iters: int,
    step_size: float,
    loss_id: str,
) -> Array:
    """Cosine similarity of standard PC and BP parameter grads on the same weights."""
    activities, _ = _standard_infer(model, x0, y, n_iters, step_size, loss_id)
    pc_grads = jpc.compute_pc_param_grads(
        params=(model, None),
        activities=activities,
        y=y,
        x=x0,
        loss_id=loss_id,
    )
    bp_grads = eqx.filter_grad(lambda m: _jpc_ff_loss(m, x0, y, loss_id))(model)
    return _cosine(_flat_params(pc_grads[0]), _flat_params(bp_grads))
