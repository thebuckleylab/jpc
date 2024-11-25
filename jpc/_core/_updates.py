"""Functions to update activities and parameters of PC networks."""

import equinox as eqx
from ._grads import compute_activity_grad, compute_pc_param_grads
from jaxtyping import PyTree, ArrayLike
from typing import Tuple, Callable, Optional, Dict
from optax import GradientTransformation, GradientTransformationExtraArgs, OptState


@eqx.filter_jit
def update_activities(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        optim: GradientTransformation | GradientTransformationExtraArgs,
        opt_state: OptState,
        output: ArrayLike,
        input: Optional[ArrayLike] = None
) -> Dict:
    """Updates activities of a predictive coding network with a given Optax optimiser.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `optim`: Optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of Optax optimiser.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Returns:**

    Dictionary with energy, updated activities, activity gradients, optimiser,
    and updated optimiser state.

    """
    energy, activity_grads = compute_activity_grad(
        params=params,
        activities=activities,
        y=output,
        x=input
    )
    activity_updates, activity_opt_state = optim.update(
        updates=activity_grads,
        state=opt_state,
        params=activities
    )
    activities = eqx.apply_updates(
        model=activities,
        updates=activity_updates
    )
    return {
        "energy": energy,
        "activities": activities,
        "activity_grads": activity_grads,
        "activity_optim": optim,
        "activity_opt_state": opt_state
    }


@eqx.filter_jit
def update_params(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        optim: GradientTransformation | GradientTransformationExtraArgs,
        opt_state: OptState,
        output: ArrayLike,
        input: Optional[ArrayLike] = None
) -> Dict:
    """Updates parameters of a predictive coding network with a given Optax optimiser.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `optim`: Optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of Optax optimiser.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Returns:**

    Dictionary with model (and optional skip model) with updated parameters,
    parameter gradients, optimiser, and updated optimiser state.

    """
    param_grads = compute_pc_param_grads(
        params=params,
        activities=activities,
        y=output,
        x=input
    )
    param_updates, param_opt_state = optim.update(
        updates=param_grads,
        state=opt_state,
        params=params
    )
    model, skip_model = eqx.apply_updates(
        model=params,
        updates=param_updates
    )
    return {
        "model": model,
        "skip_model": skip_model,
        "param_grads": param_grads,
        "param_optim": optim,
        "param_opt_state": opt_state
    }
