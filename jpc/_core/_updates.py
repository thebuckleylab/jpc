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
from ._energies import pdm_energy_fn


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
    activity_decay: Scalar = 0.,
    gamma: Optional[Scalar] = None
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
    - `gamma`: Optional scaling factor for the output layer. If provided, the output 
        layer scaling is multiplied by `1/gamma`. Defaults to `None` (no additional scaling).

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
        activity_decay=activity_decay,
        gamma=gamma
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
    activity_decay: Scalar = 0.,
    gamma: Optional[Scalar] = None
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
    - `gamma`: Optional scaling factor for the output layer. If provided, the output 
        layer scaling is multiplied by `1/gamma`. Defaults to `None` (no additional scaling).

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
        activity_decay=activity_decay,
        gamma=gamma
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
    lateral_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    include_previous_backward_error: bool = False,
    include_next_forward_error: bool = False,
    projection_matrix_prev: Optional[PyTree[Callable]] = None,
    projection_matrix_next: Optional[PyTree[Callable]] = None,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0,
    bpc_terms_factor: Scalar = 0.0,
    stop_gradient_on_extra_errors: bool = True
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
    - `lateral_model`: Optional list of lateral connection models, one per hidden layer.
        Each lateral model maps layer activities to themselves, adding within-layer
        predictions to the backward error. Defaults to `None`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `include_previous_backward_error`: If `True`, modifies the forward energy term to 
        $||\mathbf{e}_\ell + \mathbf{A}_\ell \boldsymbol{\delta}_{\ell-1}||^2/2$ where 
        $\boldsymbol{\delta}_{\ell-1}$ is the backward error at the previous layer.
        Defaults to `False`.
    - `include_next_forward_error`: If `True`, modifies the backward energy term to 
        $||\boldsymbol{\delta}_{\ell+1} + \mathbf{B}_\ell \mathbf{e}_{\ell+1}||^2/2$ where 
        $\mathbf{e}_{\ell+1}$ is the forward error at the next layer.
        Defaults to `False`.
    - `projection_matrix_prev`: Optional projection matrix $\mathbf{A}_\ell$ for projecting 
        the previous backward error. Only used when `include_previous_backward_error=True`.
        Defaults to `None`.
    - `projection_matrix_next`: Optional projection matrix $\mathbf{B}_\ell$ for projecting 
        the next forward error. Only used when `include_next_forward_error=True`.
        Defaults to `None`.
    - `backward_energy_weight`: Scalar weighting for the backward prediction error 
        (delta) only. Scales $\boldsymbol{\delta}_{\ell+1}$ before computing the energy. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward prediction error 
        (epsilon) only. Scales $\mathbf{e}_\ell$ before computing the energy. 
        Defaults to `1.0`.
    - `bpc_terms_factor`: Scaling factor for bPC terms (prediction errors 
        weighted by their derivatives). When `bpc_terms_factor=0.0`, uses only direct 
        terms (standard PDM). When `bpc_terms_factor=1.0`, uses full bPC gradient. 
        Defaults to `0.0`.
    - `stop_gradient_on_extra_errors`: If `True`, applies `stop_gradient` to the 
        extra error terms ($\boldsymbol{\delta}_{\ell-1}$ and $\mathbf{e}_{\ell+1}$) 
        so that gradients do not flow through them during backpropagation. This means
        gradients only flow through the primary errors ($\mathbf{e}_\ell$ and 
        $\boldsymbol{\delta}_{\ell+1}$). Defaults to `True`.

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
        lateral_model=lateral_model,
        param_type=param_type,
        include_previous_backward_error=include_previous_backward_error,
        include_next_forward_error=include_next_forward_error,
        projection_matrix_prev=projection_matrix_prev,
        projection_matrix_next=projection_matrix_next,
        backward_energy_weight=backward_energy_weight,
        forward_energy_weight=forward_energy_weight,
        bpc_terms_factor=bpc_terms_factor,
        stop_gradient_on_extra_errors=stop_gradient_on_extra_errors
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
    lateral_model: Optional[PyTree[Callable]] = None,
    lateral_optim: Optional[GradientTransformation | GradientTransformationExtraArgs] = None,
    lateral_opt_state: Optional[OptState] = None,
    param_type: str = "sp",
    spectral_penalty: Scalar = 0.0,
    include_previous_backward_error: bool = False,
    include_next_forward_error: bool = False,
    projection_matrix_prev: Optional[PyTree[Callable]] = None,
    projection_matrix_next: Optional[PyTree[Callable]] = None,
    projection_matrix_prev_optim: Optional[GradientTransformation | GradientTransformationExtraArgs] = None,
    projection_matrix_next_optim: Optional[GradientTransformation | GradientTransformationExtraArgs] = None,
    projection_matrix_prev_opt_state: Optional[OptState] = None,
    projection_matrix_next_opt_states: Optional[PyTree[OptState]] = None,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0,
    update_projection_matrix_prev: bool = True,
    update_projection_matrix_next: bool = True,
    stop_gradient_on_extra_errors: bool = True
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
    - `lateral_model`: Optional list of lateral connection models, one per hidden layer.
        Each lateral model maps layer activities to themselves, adding within-layer
        predictions to the backward error. Defaults to `None`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `spectral_penalty`: Regularization strength for the penalty that 
        penalizes non-orthogonal forward and backward weights. Defaults to 0.0.
    - `include_previous_backward_error`: If `True`, modifies the forward energy term to 
        $||\mathbf{e}_\ell + \mathbf{A}_\ell \boldsymbol{\delta}_{\ell-1}||^2/2$ where 
        $\boldsymbol{\delta}_{\ell-1}$ is the backward error at the previous layer.
        Defaults to `False`.
    - `include_next_forward_error`: If `True`, modifies the backward energy term to 
        $||\boldsymbol{\delta}_{\ell+1} + \mathbf{B}_\ell \mathbf{e}_{\ell+1}||^2/2$ where 
        $\mathbf{e}_{\ell+1}$ is the forward error at the next layer.
        Defaults to `False`.
    - `projection_matrix_prev`: Optional projection matrix $\mathbf{A}_\ell$ for projecting 
        the previous backward error. Only used when `include_previous_backward_error=True`.
        Defaults to `None`.
    - `projection_matrix_next`: Optional projection matrix $\mathbf{B}_\ell$ for projecting 
        the next forward error. Only used when `include_next_forward_error=True`.
        Defaults to `None`.
    - `projection_matrix_prev_optim`: Optional optax optimiser for `projection_matrix_prev`.
        If not provided, `top_down_optim` will be used. If provided along with `projection_matrix_prev`, 
        the projection matrix will be updated. Defaults to `None`.
    - `projection_matrix_next_optim`: Optional optax optimiser for `projection_matrix_next`.
        If not provided, `top_down_optim` will be used. If provided along with `projection_matrix_next`, 
        the projection matrices will be updated. Defaults to `None`.
    - `projection_matrix_prev_opt_state`: Optional optimizer state for `projection_matrix_prev_optim`.
        If not provided but `projection_matrix_prev` is provided, `top_down_opt_state` will be used.
        Defaults to `None`.
    - `projection_matrix_next_opt_states`: Optional optimizer states for `projection_matrix_next_optim`.
        Should be a list/tuple of states, one for each projection matrix in `projection_matrix_next`.
        If not provided but `projection_matrix_next` is provided, `top_down_opt_state` will be used
        for all projection matrices. Defaults to `None`.
    - `backward_energy_weight`: Scalar weighting for the backward prediction error 
        (delta) only. Scales $\boldsymbol{\delta}_{\ell+1}$ before computing the energy. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward prediction error 
        (epsilon) only. Scales $\mathbf{e}_\ell$ before computing the energy. 
        Defaults to `1.0`.
    - `update_projection_matrix_prev`: If `True`, updates `projection_matrix_prev` during 
        training. If `False`, the projection matrix is used in the energy but not updated.
        Defaults to `True`.
    - `update_projection_matrix_next`: If `True`, updates `projection_matrix_next` during 
        training. If `False`, the projection matrices are used in the energy but not updated.
        Defaults to `True`.
    - `stop_gradient_on_extra_errors`: If `True`, applies `stop_gradient` to the 
        extra error terms ($\boldsymbol{\delta}_{\ell-1}$ and $\mathbf{e}_{\ell+1}$) 
        so that gradients do not flow through them during backpropagation. This means
        gradients only flow through the primary errors ($\mathbf{e}_\ell$ and 
        $\boldsymbol{\delta}_{\ell+1}$). Defaults to `True`.

    **Returns:**

    Dictionary with updated models, parameter gradients, optimiser states, and optionally
    updated projection matrices and their optimizer states.

    """
    grad_result = compute_pdm_param_grads(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=output,
        x=input,
        skip_model=skip_model,
        lateral_model=lateral_model,
        param_type=param_type,
        spectral_penalty=spectral_penalty,
        include_previous_backward_error=include_previous_backward_error,
        include_next_forward_error=include_next_forward_error,
        projection_matrix_prev=projection_matrix_prev,
        projection_matrix_next=projection_matrix_next,
        backward_energy_weight=backward_energy_weight,
        forward_energy_weight=forward_energy_weight,
        stop_gradient_on_extra_errors=stop_gradient_on_extra_errors
    )
    
    # Handle both cases: with and without lateral gradients
    if len(grad_result) == 3:
        top_down_grads, bottom_up_grads, lateral_grads = grad_result
    else:
        top_down_grads, bottom_up_grads = grad_result
        lateral_grads = None
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
    
    result = {
        "models": (top_down_model, bottom_up_model),
        "grads": (top_down_grads, bottom_up_grads),
        "opt_states": (top_down_opt_state, bottom_up_opt_state)
    }
    
    # Update lateral model if provided
    if lateral_model is not None and lateral_optim is not None and lateral_opt_state is not None and lateral_grads is not None:
        lateral_updates, lateral_opt_state = lateral_optim.update(
            updates=lateral_grads,
            state=lateral_opt_state,
            params=lateral_model
        )
        lateral_model = eqx.apply_updates(lateral_model, lateral_updates)
        result["lateral_model"] = lateral_model
        result["lateral_opt_state"] = lateral_opt_state
        result["lateral_grads"] = lateral_grads
    
    # Update projection matrix for previous backward error if provided and update flag is True
    if (update_projection_matrix_prev and
        include_previous_backward_error and 
        projection_matrix_prev is not None and 
        projection_matrix_prev[0] is not None):
        # Use provided optimizer or fall back to top_down_optim
        proj_prev_optim_to_use = projection_matrix_prev_optim if projection_matrix_prev_optim is not None else top_down_optim
        # Use provided opt_state or fall back to updated top_down_opt_state
        proj_prev_opt_state_to_use = (
            projection_matrix_prev_opt_state 
            if projection_matrix_prev_opt_state is not None 
            else top_down_opt_state  # Use the updated state from top_down_model update
        )
        def proj_prev_energy_fn(proj_prev_weight):
            modified_proj_prev = list(projection_matrix_prev)
            modified_proj_prev[0] = proj_prev_weight
            return pdm_energy_fn(
                top_down_model=top_down_model,
                bottom_up_model=bottom_up_model,
                activities=activities,
                y=output,
                x=input,
                skip_model=skip_model,
                lateral_model=lateral_model,
                spectral_penalty=spectral_penalty,
                include_previous_backward_error=include_previous_backward_error,
                include_next_forward_error=include_next_forward_error,
                projection_matrix_prev=modified_proj_prev,
                projection_matrix_next=projection_matrix_next,
                backward_energy_weight=backward_energy_weight,
                forward_energy_weight=forward_energy_weight,
                stop_gradient_on_extra_errors=stop_gradient_on_extra_errors
            )
        
        proj_prev_grad = eqx.filter_grad(proj_prev_energy_fn)(projection_matrix_prev[0])
        proj_prev_updates, proj_prev_opt_state = proj_prev_optim_to_use.update(
            updates=proj_prev_grad,
            state=proj_prev_opt_state_to_use,
            params=projection_matrix_prev[0]
        )
        projection_matrix_prev[0] = eqx.apply_updates(
            model=projection_matrix_prev[0],
            updates=proj_prev_updates
        )
        result["projection_matrix_prev"] = projection_matrix_prev
        # Only return opt_state if we used a separate optimizer, otherwise update top_down_opt_state
        if projection_matrix_prev_optim is not None:
            result["projection_matrix_prev_opt_state"] = proj_prev_opt_state
        else:
            # If using shared optimizer, update the top_down_opt_state in result
            result["opt_states"] = (proj_prev_opt_state, bottom_up_opt_state)
    
    # Update projection matrices for next forward error if provided and update flag is True
    if (update_projection_matrix_next and
        include_next_forward_error and 
        projection_matrix_next is not None):
        # Use provided optimizer or fall back to top_down_optim
        proj_next_optim_to_use = projection_matrix_next_optim if projection_matrix_next_optim is not None else top_down_optim
        updated_projection_matrix_next = list(projection_matrix_next)
        updated_projection_matrix_next_opt_states = []
        
        for i, proj_mat in enumerate(projection_matrix_next):
            if proj_mat is not None:
                # Use provided opt_state for this layer or fall back to updated top_down_opt_state
                # For shared optimizer, use the state from the previous projection matrix update if available,
                # otherwise use top_down_opt_state
                if (projection_matrix_next_opt_states is not None and 
                    i < len(projection_matrix_next_opt_states) and 
                    projection_matrix_next_opt_states[i] is not None):
                    proj_next_opt_state_to_use = projection_matrix_next_opt_states[i]
                elif projection_matrix_next_optim is None and i > 0 and len(updated_projection_matrix_next_opt_states) > 0:
                    # Use the state from previous projection matrix update when using shared optimizer
                    proj_next_opt_state_to_use = updated_projection_matrix_next_opt_states[-1] if updated_projection_matrix_next_opt_states[-1] is not None else top_down_opt_state
                else:
                    proj_next_opt_state_to_use = top_down_opt_state
                def make_proj_next_energy_fn(layer_idx):
                    def proj_next_energy_fn(proj_next_weight):
                        modified_proj_next = list(updated_projection_matrix_next)
                        modified_proj_next[layer_idx] = proj_next_weight
                        return pdm_energy_fn(
                            top_down_model=top_down_model,
                            bottom_up_model=bottom_up_model,
                            activities=activities,
                            y=output,
                            x=input,
                            skip_model=skip_model,
                            lateral_model=lateral_model,
                            spectral_penalty=spectral_penalty,
                            include_previous_backward_error=include_previous_backward_error,
                            include_next_forward_error=include_next_forward_error,
                            projection_matrix_prev=projection_matrix_prev,
                            projection_matrix_next=modified_proj_next,
                            backward_energy_weight=backward_energy_weight,
                            forward_energy_weight=forward_energy_weight,
                            stop_gradient_on_extra_errors=stop_gradient_on_extra_errors
                        )
                    return proj_next_energy_fn
                
                proj_next_energy_fn = make_proj_next_energy_fn(i)
                proj_next_grad = eqx.filter_grad(proj_next_energy_fn)(proj_mat)
                proj_next_updates, proj_next_opt_state = proj_next_optim_to_use.update(
                    updates=proj_next_grad,
                    state=proj_next_opt_state_to_use,
                    params=proj_mat
                )
                updated_projection_matrix_next[i] = eqx.apply_updates(
                    model=proj_mat,
                    updates=proj_next_updates
                )
                # Only store opt_state if we used a separate optimizer, otherwise it's already in top_down_opt_state
                if projection_matrix_next_optim is not None:
                    updated_projection_matrix_next_opt_states.append(proj_next_opt_state)
                else:
                    updated_projection_matrix_next_opt_states.append(None)
            else:
                updated_projection_matrix_next_opt_states.append(None)
        
        result["projection_matrix_next"] = updated_projection_matrix_next
        # Only return opt_states if we used a separate optimizer, otherwise update top_down_opt_state
        if projection_matrix_next_optim is not None:
            result["projection_matrix_next_opt_states"] = updated_projection_matrix_next_opt_states
        else:
            # If using shared optimizer, update the top_down_opt_state in result with the last updated state
            if updated_projection_matrix_next_opt_states and updated_projection_matrix_next_opt_states[-1] is not None:
                result["opt_states"] = (updated_projection_matrix_next_opt_states[-1], bottom_up_opt_state)
    
    return result


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
