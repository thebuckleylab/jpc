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
        loss_id: str = "mse",
        param_type: str = "sp",
        weight_decay: Scalar = 0.,
        spectral_penalty: Scalar = 0.,
        activity_decay: Scalar = 0.
) -> Dict:
    """Updates activities of a predictive coding network with a given 
    [optax](https://github.com/google-deepmind/optax) optimiser.

    !!! warning

        `param_type = "mupc"` ([μPC](https://arxiv.org/abs/2505.13124)) assumes 
        that one is using [`jpc.make_mlp()`](https://thebuckleylab.github.io/jpc/api/Utils/#jpc.make_mlp) 
        to create the model.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `optim`: optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of optax optimiser.
    - `output`: Observation or target of the generative model.

    **Other arguments:**

    - `input`: Optional prior of the generative model.
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `"mse"` (default) or cross-entropy `"ce"`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://arxiv.org/abs/2505.13124)), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `weight_decay`: Weight decay for the weights (0 by default).
    - `spectral_penalty`: Weight spectral penalty of the form 
        $||\mathbf{I} - \mathbf{W}_\ell^T \mathbf{W}_\ell||^2$ (0 by default).
    - `activity_decay`: Activity decay for the activities (0 by default).

    **Returns:**

    Dictionary with energy, updated activities, activity gradients, and 
    optimiser state.

    """
    energy, grads = compute_activity_grad(
        params=params,
        activities=activities,
        y=output,
        x=input,
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
        loss_id: str = "mse",
        param_type: str = "sp",
        weight_decay: Scalar = 0.,
        spectral_penalty: Scalar = 0.,
        activity_decay: Scalar = 0.
) -> Dict:
    """Updates parameters of a predictive coding network with a given 
    [optax](https://github.com/google-deepmind/optax) optimiser.

    !!! warning

        `param_type = "mupc"` ([μPC](https://arxiv.org/abs/2505.13124)) assumes 
        that one is using [`jpc.make_mlp()`](https://thebuckleylab.github.io/jpc/api/Utils/#jpc.make_mlp) 
        to create the model.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `optim`: optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of optax optimiser.
    - `output`: Observation or target of the generative model.

    **Other arguments:**

    - `input`: Optional prior of the generative model.
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `"mse"` (default) or cross-entropy `"ce"`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://arxiv.org/abs/2505.13124)), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `weight_decay`: Weight decay for the weights (0 by default).
    - `spectral_penalty`: Weight spectral penalty of the form 
        $||\mathbf{I} - \mathbf{W}_\ell^T \mathbf{W}_\ell||^2$ (0 by default).
    - `activity_decay`: Activity decay for the activities (0 by default).

    **Returns:**

    Dictionary with model and optional skip model with updated parameters,
    parameter gradients, and optimiser state.

    """
    grads = compute_pc_param_grads(
        params=params,
        activities=activities,
        y=output,
        x=input,
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
