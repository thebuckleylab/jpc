"""Feedforward backpropagation on the same BregmanMLP."""

import equinox as eqx
from jaxtyping import Array, ArrayLike, PyTree
from optax import GradientTransformation, OptState

from .evaluate import feedforward_loss
from .model import BregmanMLP


@eqx.filter_jit
def update_bp(
    model: BregmanMLP,
    x0: ArrayLike,
    y: ArrayLike,
    optim: GradientTransformation,
    opt_state: OptState,
) -> tuple[BregmanMLP, OptState, PyTree, Array]:
    """One Adam/SGD step on the feedforward output loss (standard backprop)."""
    loss, grads = eqx.filter_value_and_grad(feedforward_loss)(model, x0, y)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, grads, loss
