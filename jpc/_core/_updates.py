"""Functions to update activities and parameters of PC networks."""

import equinox as eqx
from jaxtyping import PyTree, ArrayLike, Scalar
from typing import Tuple, Callable, Optional, Dict
from optax import GradientTransformation, GradientTransformationExtraArgs, OptState
from ._grads import (
    compute_pc_activity_grad, 
    compute_pc_param_grads, 
    compute_bpc_activity_grad, 
    compute_bpc_param_grads,
    compute_epc_error_grad,
    compute_epc_param_grads,
    compute_pdm_activity_grad,
    compute_pdm_param_grads
)


@eqx.filter_jit
def update_pc_activities(
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

        `param_type = "mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))) assumes 
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
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
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
    energy, grads = compute_pc_activity_grad(
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
def update_pc_params(
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

        `param_type = "mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))) assumes 
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
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
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


@eqx.filter_jit
def update_bpc_activities(
    top_down_model: PyTree[Callable], 
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    optim: GradientTransformation | GradientTransformationExtraArgs,
    opt_state: OptState,
    output: ArrayLike,
    input: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0
) -> Dict:
    """Updates activities of a bidirectional PC network.

    **Main arguments:**

    - `top_down_model`: List of callable model (e.g. neural network) layers for 
        the forward model.
    - `bottom_up_model`: List of callable model (e.g. neural network) layers for 
        the backward model.
    - `activities`: List of activities for each layer free to vary.
    - `optim`: optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of optax optimiser.
    - `output`: Target of the `top_down_model` and input to the `bottom_up_model`.

    **Other arguments:**

    - `input`: Input to the `top_down_model` and target of the `bottom_up_model`.
    - `skip_model`: Optional skip connection model.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `backward_energy_weight`: Scalar weighting for the backward energy terms. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward energy terms. 
        Defaults to `1.0`.

    **Returns:**

    Dictionary with energy, updated activities, activity gradients, and 
    optimiser state.

    """
    energy, grads = compute_bpc_activity_grad(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=output,
        x=input,
        skip_model=skip_model,
        param_type=param_type,
        backward_energy_weight=backward_energy_weight,
        forward_energy_weight=forward_energy_weight
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
def update_bpc_params(
    top_down_model: PyTree[Callable], 
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    top_down_optim: GradientTransformation | GradientTransformationExtraArgs,
    bottom_up_optim: GradientTransformation | GradientTransformationExtraArgs,
    top_down_opt_state: OptState,
    bottom_up_opt_state: OptState,
    output: ArrayLike,
    input: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0
) -> Dict:
    """Updates parameters of a bidirectional PC network.

    **Main arguments:**

    - `top_down_model`: List of callable model (e.g. neural network) layers for 
        the forward model.
    - `bottom_up_model`: List of callable model (e.g. neural network) layers for 
        the backward model.
    - `activities`: List of activities for each layer free to vary.
    - `top_down_optim`: optax optimiser for the top-down model.
    - `bottom_up_optim`: optax optimiser for the bottom-up model.
    - `top_down_opt_state`: State of the top-down optimiser.
    - `bottom_up_opt_state`: State of the bottom-up optimiser.
    - `output`: Target of the `top_down_model` and input to the `bottom_up_model`.

    **Other arguments:**

    - `input`: Input to the `top_down_model` and target of the `bottom_up_model`.
    - `skip_model`: Optional skip connection model.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `backward_energy_weight`: Scalar weighting for the backward energy terms. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward energy terms. 
        Defaults to `1.0`.

    **Returns:**

    Dictionary with models with updated parameters, parameter gradients, and 
    optimiser states.

    """
    top_down_grads, bottom_up_grads = compute_bpc_param_grads(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=output,
        x=input,
        skip_model=skip_model,
        param_type=param_type,
        backward_energy_weight=backward_energy_weight,
        forward_energy_weight=forward_energy_weight
    )
    top_down_updates, top_down_opt_state = top_down_optim.update(
        updates=top_down_grads,
        state=top_down_opt_state,
        params=top_down_model
    )
    bottom_up_updates, bottom_up_opt_state = bottom_up_optim.update(
        updates=bottom_up_grads,
        state=bottom_up_opt_state,
        params=bottom_up_model
    )
    top_down_model = eqx.apply_updates(
        model=top_down_model,
        updates=top_down_updates
    )
    bottom_up_model = eqx.apply_updates(
        model=bottom_up_model,
        updates=bottom_up_updates
    )
    
    return {
        "models": (top_down_model, bottom_up_model),
        "grads": (top_down_grads, bottom_up_grads),
        "opt_states": (top_down_opt_state, bottom_up_opt_state)
    }


@eqx.filter_jit
def update_pdm_activities(
    top_down_model: PyTree[Callable], 
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    optim: GradientTransformation | GradientTransformationExtraArgs,
    opt_state: OptState,
    output: ArrayLike,
    input: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    include_previous_backward_error: bool = False,
    projection_weights_prev: Optional[PyTree[Callable]] = None,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0,
    bpc_terms_factor: Scalar = 0.0
) -> Dict:
    """Updates activities of a predictive dendrites model (PDM).

    **Main arguments:**

    - `top_down_model`: List of callable model (e.g. neural network) layers for 
        the forward model.
    - `bottom_up_model`: List of callable model (e.g. neural network) layers for 
        the backward model.
    - `activities`: List of activities for each layer free to vary.
    - `optim`: optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of optax optimiser.
    - `output`: Target of the `top_down_model` and input to the `bottom_up_model`.
    - `input`: Input to the `top_down_model` and target of the `bottom_up_model`.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `include_previous_backward_error`: If `True`, includes an extra dendritic term 
        $(z_\ell - \delta_{\ell})^2/2$ that couples each layer with the backward 
        error at the previous layer, where $\delta_{\ell} = z_{\ell-1} - V_{\ell} z_{\ell}$. 
        Defaults to `False`.
    - `backward_energy_weight`: Scalar weighting for the backward energy terms. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward energy terms. 
        Defaults to `1.0`.
    - `bpc_terms_factor`: Scaling factor for bPC terms (prediction errors 
        weighted by their derivatives). When `bpc_terms_factor=0.0`, uses only direct 
        terms (standard PDM). When `bpc_terms_factor=1.0`, uses full bPC gradient. 
        Defaults to `0.0`.

    **Returns:**

    Dictionary with energy, updated activities, activity gradients, and 
    optimiser state.

    """
    energy, grads = compute_pdm_activity_grad(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=output,
        x=input,
        skip_model=skip_model,
        param_type=param_type,
        include_previous_backward_error=include_previous_backward_error,
        projection_weights_prev=projection_weights_prev,
        backward_energy_weight=backward_energy_weight,
        forward_energy_weight=forward_energy_weight,
        bpc_terms_factor=bpc_terms_factor
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
def update_pdm_params(
    top_down_model: PyTree[Callable], 
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    top_down_optim: GradientTransformation | GradientTransformationExtraArgs,
    bottom_up_optim: GradientTransformation | GradientTransformationExtraArgs,
    top_down_opt_state: OptState,
    bottom_up_opt_state: OptState,
    output: ArrayLike,
    input: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    spectral_penalty: Scalar = 0.0,
    include_previous_backward_error: bool = False,
    projection_weights_prev: Optional[PyTree[Callable]] = None,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0
) -> Dict:
    """Updates parameters of a predictive dendrites model (PDM).

    **Main arguments:**

    - `top_down_model`: List of callable model (e.g. neural network) layers for 
        the forward model.
    - `bottom_up_model`: List of callable model (e.g. neural network) layers for 
        the backward model.
    - `activities`: List of activities for each layer free to vary.
    - `top_down_optim`: optax optimiser for the top-down model, e.g. `optax.adam()`.
    - `bottom_up_optim`: optax optimiser for the bottom-up model, e.g. `optax.adam()`.
    - `top_down_opt_state`: State of optax optimiser for top-down model.
    - `bottom_up_opt_state`: State of optax optimiser for bottom-up model.
    - `output`: Target of the `top_down_model` and input to the `bottom_up_model`.
    - `input`: Input to the `top_down_model` and target of the `bottom_up_model`.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `spectral_penalty`: Regularization strength for the penalty that 
        penalizes non-orthogonal forward and backward weights. Defaults to 0.0.
    - `include_previous_backward_error`: If `True`, includes an extra dendritic term 
        $(z_\ell - \delta_{\ell})^2/2$ that couples each layer with the backward 
        error at the previous layer, where $\delta_{\ell} = z_{\ell-1} - V_{\ell} z_{\ell}$. 
        Defaults to `False`.
    - `projection_weights_prev`: Optional list of projection weight matrices for projecting 
        input errors to hidden dimensions. Only used for the first layer when 
        `include_previous_backward_error=True`. If `None`, falls back to using the forward 
        model weight. Defaults to `None`.
    - `backward_energy_weight`: Scalar weighting for the backward energy terms. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward energy terms. 
        Defaults to `1.0`.

    **Returns:**

    Dictionary with updated models, parameter gradients, and optimiser states.

    """
    top_down_grads, bottom_up_grads = compute_pdm_param_grads(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=output,
        x=input,
        skip_model=skip_model,
        param_type=param_type,
        spectral_penalty=spectral_penalty,
        include_previous_backward_error=include_previous_backward_error,
        projection_weights_prev=projection_weights_prev,
        backward_energy_weight=backward_energy_weight,
        forward_energy_weight=forward_energy_weight
    )
    top_down_updates, top_down_opt_state = top_down_optim.update(
        updates=top_down_grads,
        state=top_down_opt_state,
        params=top_down_model
    )
    bottom_up_updates, bottom_up_opt_state = bottom_up_optim.update(
        updates=bottom_up_grads,
        state=bottom_up_opt_state,
        params=bottom_up_model
    )
    top_down_model = eqx.apply_updates(
        model=top_down_model,
        updates=top_down_updates
    )
    bottom_up_model = eqx.apply_updates(
        model=bottom_up_model,
        updates=bottom_up_updates
    )
    
    return {
        "models": (top_down_model, bottom_up_model),
        "grads": (top_down_grads, bottom_up_grads),
        "opt_states": (top_down_opt_state, bottom_up_opt_state)
    }


@eqx.filter_jit
def update_epc_errors(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    errors: PyTree[ArrayLike],
    optim: GradientTransformation | GradientTransformationExtraArgs,
    opt_state: OptState,
    output: ArrayLike,
    *,
    input: Optional[ArrayLike] = None,
    loss_id: str = "mse",
    param_type: str = "sp"
) -> Dict:
    """Updates errors of an error-reparameterised Predictive Coding (ePC) network with a given 
    [optax](https://github.com/google-deepmind/optax) optimiser.

    !!! note

        In ePC, errors are updated during inference rather than activities.

    !!! warning

        `param_type = "mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco_Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))) assumes 
        that one is using [`jpc.make_mlp()`](https://thebuckleylab.github.io/jpc/api/Utils/#jpc.make_mlp) 
        to create the model.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `errors`: List of errors for each layer free to vary.
    - `optim`: optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of optax optimiser.
    - `output`: Observation or target of the generative model.

    **Other arguments:**

    - `input`: Optional prior of the generative model.
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `"mse"` (default) or cross-entropy `"ce"`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco_Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.

    **Returns:**

    Dictionary with energy, updated errors, error gradients, and 
    optimiser state.

    """
    energy, grads = compute_epc_error_grad(
        params=params,
        errors=errors,
        y=output,
        x=input,
        loss_id=loss_id,
        param_type=param_type
    )
    updates, opt_state = optim.update(
        updates=grads,
        state=opt_state,
        params=errors
    )
    errors = eqx.apply_updates(
        model=errors,
        updates=updates
    )
    return {
        "energy": energy,
        "errors": errors,
        "grads": grads,
        "opt_state": opt_state
    }


@eqx.filter_jit
def update_epc_params(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    errors: PyTree[ArrayLike],
    optim: GradientTransformation | GradientTransformationExtraArgs,
    opt_state: OptState,
    output: ArrayLike,
    *,
    input: Optional[ArrayLike] = None,
    loss_id: str = "mse",
    param_type: str = "sp"
) -> Dict:
    """Updates parameters of an error-reparameterised Predictive Coding (ePC) network with a given 
    [optax](https://github.com/google-deepmind/optax) optimiser.

    !!! note

        In ePC, errors are updated during inference rather than activities.

    !!! warning

        `param_type = "mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco_Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))) assumes 
        that one is using [`jpc.make_mlp()`](https://thebuckleylab.github.io/jpc/api/Utils/#jpc.make_mlp) 
        to create the model.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `errors`: List of errors for each layer free to vary.
    - `optim`: optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of optax optimiser.
    - `output`: Observation or target of the generative model.

    **Other arguments:**

    - `input`: Optional prior of the generative model.
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `"mse"` (default) or cross-entropy `"ce"`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco_Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.

    **Returns:**

    Dictionary with model and optional skip model with updated parameters,
    parameter gradients, and optimiser state.

    """
    grads = compute_epc_param_grads(
        params=params,
        errors=errors,
        y=output,
        x=input,
        loss_id=loss_id,
        param_type=param_type
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
