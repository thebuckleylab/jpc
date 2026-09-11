"""Feedforward evaluation of BregmanMLP and jpc standard-PC networks."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Scalar

import jpc

from .model import BregmanMLP


def feedforward_loss(model: BregmanMLP, x0: ArrayLike, y: ArrayLike) -> Scalar:
    logits = model.forward(x0)
    if model.output_loss == "ce":
        return jpc.cross_entropy_loss(logits, y)
    if model.output_loss == "mse":
        return jpc.mse_loss(logits, y)
    y_clip = jpc.clip_to_bregman_range(model.act_fn, y)
    return jnp.mean(
        jnp.sum(jpc.bregman_from_preact(model.act_fn, y_clip, logits), axis=-1)
    )


@eqx.filter_jit
def evaluate_batch(
    model: BregmanMLP,
    x0: ArrayLike,
    y: ArrayLike,
) -> tuple[Scalar, Scalar]:
    logits = model.forward(x0)
    if model.output_loss == "ce":
        loss = jpc.cross_entropy_loss(logits, y)
    elif model.output_loss == "mse":
        loss = jpc.mse_loss(logits, y)
    else:
        y_clip = jpc.clip_to_bregman_range(model.act_fn, y)
        loss = jnp.mean(
            jnp.sum(jpc.bregman_from_preact(model.act_fn, y_clip, logits), axis=-1)
        )
    acc = jpc.compute_accuracy(y, logits) / 100.0
    return loss, acc


@eqx.filter_jit
def evaluate_jpc_batch(
    model,
    x0: ArrayLike,
    y: ArrayLike,
    loss_id: str = "mse",
) -> tuple[Scalar, Scalar]:
    loss, acc = jpc.test_discriminative_pc(
        model=model, output=y, input=x0, loss=loss_id
    )
    return loss, acc / 100.0


def evaluate_models(
    models: dict,
    loader,
    max_batches: int | None = None,
    jpc_loss: str = "mse",
) -> dict[str, tuple[float, float]]:
    totals = {name: jnp.zeros(2) for name in models}
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = jnp.asarray(x.numpy())
        y = jnp.asarray(y.numpy())
        for name, model in models.items():
            if isinstance(model, BregmanMLP):
                loss, acc = evaluate_batch(model, x, y)
            else:
                loss, acc = evaluate_jpc_batch(model, x, y, loss_id=jpc_loss)
            totals[name] = totals[name] + jnp.stack([loss, acc])
        n += 1
    totals = {name: vals / n for name, vals in totals.items()}
    totals = jax.device_get(totals)
    return {name: (float(vals[0]), float(vals[1])) for name, vals in totals.items()}
