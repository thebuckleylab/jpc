"""Functions to update activities and parameters of PC networks."""

import equinox as eqx
from ._grads import compute_activity_grad, compute_pc_param_grads
from jaxtyping import PyTree, ArrayLike, Scalar
from typing import Tuple, Callable, Optional, Dict
from optax import GradientTransformation, GradientTransformationExtraArgs, OptState


@eqx.filter_jit
def update_activities(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        optim: GradientTransformation | GradientTransformationExtraArgs,
        opt_state: OptState,
        output: ArrayLike,
        *,
        input: Optional[ArrayLike] = None,
        n_skip: int = 0,
        loss_id: str = "mse",
        param_type: str = "sp",
        weight_decay: Scalar = 0.,
        spectral_penalty: Scalar = 0.,
        activity_decay: Scalar = 0.
) -> Dict:
    """Updates activities of a predictive coding network with a given Optax optimiser.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `optim`: Optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of Optax optimiser.
    - `output`: Observation or target of the generative model.

    **Other arguments:**

    - `input`: Optional prior of the generative model.
    - `n_skip`: Number of layers to skip for the skip connections.
    - `loss_id`: Loss function to use at the output layer (mean squared error
        `mse` vs cross-entropy `ce`).
    - `param_type`: Determines the parameterisation. Options are `sp`, `mup`, or `ntp`.
    - `weight_decay`: Weight decay for the weights.
    - `spectral_penalty`: Spectral penalty for the weights.
    - `activity_decay`: Activity decay for the activities.

    **Returns:**

    Dictionary with energy, updated activities, activity gradients, and 
    optimiser state.

    """
    energy, grads = compute_activity_grad(
        params=params,
        activities=activities,
        y=output,
        x=input,
        n_skip=n_skip,
        loss_id=loss_id,
        param_type=param_type,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay
    )
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=activities
    )
    activities = eqx.apply_updates(
        model=activities,
        updates=updates
    )
    return {
        "energy": energy,
        "activities": activities,
        "grads": grads,
        "opt_state": opt_state
    }


@eqx.filter_jit
def update_params(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        optim: GradientTransformation | GradientTransformationExtraArgs,
        opt_state: OptState,
        output: ArrayLike,
        *,
        input: Optional[ArrayLike] = None,
        n_skip: int = 0,
        loss_id: str = "mse",
        param_type: str = "sp",
        weight_decay: Scalar = 0.,
        spectral_penalty: Scalar = 0.,
        activity_decay: Scalar = 0.
) -> Dict:
    """Updates parameters of a predictive coding network with a given Optax optimiser.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `optim`: Optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of Optax optimiser.
    - `output`: Observation or target of the generative model.

    **Other arguments:**

    - `input`: Optional prior of the generative model.
    - `n_skip`: Number of layers to skip for the skip connections.
    - `loss_id`: Loss function to use at the output layer (mean squared error
        `mse` vs cross-entropy `ce`).
    - `param_type`: Determines the parameterisation. Options are `sp`, `mup`, or `ntp`.
    - `weight_decay`: Weight decay for the weights.
    - `spectral_penalty`: Spectral penalty for the weights.
    - `activity_decay`: Activity decay for the activities.

    **Returns:**

    Dictionary with model (and optional skip model) with updated parameters,
    parameter gradients, and optimiser state.

    """
    grads = compute_pc_param_grads(
        params=params,
        activities=activities,
        y=output,
        x=input,
        n_skip=n_skip,
        loss_id=loss_id,
        param_type=param_type,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay
    )
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=params
    )
    model, skip_model = eqx.apply_updates(
        model=params,
        updates=updates
    )
    return {
        "model": model,
        "skip_model": skip_model,
        "grads": grads,
        "opt_state": opt_state
    }
