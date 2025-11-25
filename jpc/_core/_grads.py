"""Functions to compute gradients of the predictive coding energies."""

import jax.numpy as jnp
from jax import grad, value_and_grad
from jax.tree_util import tree_map
from equinox import filter_grad
from jaxtyping import PyTree, ArrayLike, Array, Scalar
from typing import Tuple, Callable, Optional
from diffrax import AbstractStepSizeController
from ._energies import (
    pc_energy_fn, 
    hpc_energy_fn, 
    bpc_energy_fn, 
    epc_energy_fn,
    pdm_energy_fn, 
    _pdm_single_layer_energy
)


########################### ACTIVITY GRADIENTS #################################
################################################################################
def neg_pc_activity_grad(
    t: float | int,
    activities: PyTree[ArrayLike],
    args: Tuple[
        Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        ArrayLike,
        Optional[ArrayLike],
        int,
        str,
        str,
        AbstractStepSizeController
    ]
) -> PyTree[Array]:
    """Computes the negative gradient of the [PC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.pc_energy_fn) 
    with respect to the activities $- ∇_{\mathbf{z}} \mathcal{F}$.

    This defines an ODE system to be integrated by [`jpc.solve_pc_inference()`](https://thebuckleylab.github.io/jpc/api/Continuous-time%20Inference/#jpc.solve_inference).

    **Main arguments:**

    - `t`: Time step of the ODE system, used for downstream integration by
        [`diffrax.diffeqsolve()`](https://docs.kidger.site/diffrax/api/diffeqsolve/#diffrax.diffeqsolve).
    - `activities`: List of activities for each layer free to vary.
    - `args`: 5-Tuple with:

        (i) Tuple with callable model layers and optional skip connections,

        (ii) model output (observation),

        (iii) model input (prior),

        (iv) loss specified at the output layer (`"mse"` as default or `"ce"`),

        (v) parameterisation type (`"sp"` as default, `"mupc"`, or `"ntp"`),

        (vi) $\ell^2$ regulariser for the weights (0 by default),

        (vii) spectral penalty for the weights (0 by default),

        (viii) $\ell^2$ regulariser for the activities (0 by default), and

        (ix) diffrax controller for step size integration.

    **Returns:**

    List of negative gradients of the energy with respect to the activities.

    """
    params, y, x, loss_id, param_type, weight_decay, spectral_penalty, activity_decay, _ = args
    dFdzs = grad(pc_energy_fn, argnums=1)(
        params,
        activities,
        y,
        x=x,
        loss=loss_id,
        param_type=param_type,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay
    )
    return tree_map(lambda dFdz: -dFdz, dFdzs)


def compute_pc_activity_grad(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: Optional[ArrayLike],
    loss_id: str = "mse",
    param_type: str = "sp",
    weight_decay: Scalar = 0.,
    spectral_penalty: Scalar = 0.,
    activity_decay: Scalar = 0.
) -> PyTree[Array]:
    """Computes the gradient of the [PC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.pc_energy_fn)
    with respect to the activities $∇_{\mathbf{z}} \mathcal{F}$.

    !!! note

        This function differs from [`jpc.neg_activity_grad()`](https://thebuckleylab.github.io/jpc/api/Gradients/#jpc.neg_activity_grad) 
        only in the sign of the gradient (positive as opposed to negative) and 
        is called in [`jpc.update_activities()`](https://thebuckleylab.github.io/jpc/api/Discrete%20updates/#jpc.update_activities) 
        for use with any [optax](https://github.com/google-deepmind/optax) 
        optimiser.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.

    **Other arguments:**

    - `x`: Optional prior of the generative model.
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `"mse"` (default) or cross-entropy `"ce"`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `weight_decay`: $\ell^2$ regulariser for the weights (0 by default).
    - `spectral_penalty`: Weight spectral penalty of the form 
        $||\mathbf{I} - \mathbf{W}_\ell^T \mathbf{W}_\ell||^2$ (0 by default).
    - `activity_decay`: $\ell^2$ regulariser for the activities (0 by default).

    **Returns:**

    The energy and its gradient with respect to the activities.

    """
    energy, dFdzs = value_and_grad(pc_energy_fn, argnums=1)(
        params,
        activities,
        y,
        x=x,
        loss=loss_id,
        param_type=param_type,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay
    )
    return energy, dFdzs


def compute_bpc_activity_grad(
    top_down_model: PyTree[Callable], 
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    x: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0
) -> PyTree[Array]:
    """Computes the gradient of the [BPC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.bpc_energy_fn)
    with respect to the activities $∇_{\mathbf{z}} \mathcal{F}$.

    **Main arguments:**

    - `top_down_model`: List of callable model (e.g. neural network) layers for 
        the forward model.
    - `bottom_up_model`: List of callable model (e.g. neural network) layers for 
        the backward model.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Target of the `top_down_model` and input to the `bottom_up_model`.
    - `x`: Input to the `top_down_model` and target of the `bottom_up_model`.

    **Other arguments:**

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

    The energy and its gradient with respect to the activities.

    """
    energy, dFdzs = value_and_grad(bpc_energy_fn, argnums=2)(
        top_down_model,
        bottom_up_model,
        activities,
        y,
        x,
        skip_model=skip_model,
        param_type=param_type,
        backward_energy_weight=backward_energy_weight,
        forward_energy_weight=forward_energy_weight
    )
    return energy, dFdzs


def compute_pdm_activity_grad(
    top_down_model: PyTree[Callable],
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    x: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    include_previous_backward_error: bool = False,
    projection_weights_prev: Optional[PyTree[Callable]] = None,
    fixed_delta_0: Optional[ArrayLike] = None,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0
) -> Tuple[Scalar, PyTree[Array]]:
    """Computes the gradient of each layer PDM energy[`_pdm_single_layer_energy()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._pdm_single_layer_energy)
    with respect to the activities $∇_{\mathbf{z}_\ell} \mathcal{F}_\ell$ of 
    respective layers.

    **Main arguments:**

    - `top_down_model`: List of callable model (e.g. neural network) layers for 
        the forward model.
    - `bottom_up_model`: List of callable model (e.g. neural network) layers for 
        the backward model.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Target of the `top_down_model` and input to the `bottom_up_model`.
    - `x`: Input to the `top_down_model` and target of the `bottom_up_model`.

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

    **Returns:**

    The PDMenergy and list of gradients with respect to the activities of each layer.

    """
    energy = pdm_energy_fn(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=y,
        x=x,
        skip_model=skip_model,
        param_type=param_type,
        include_previous_backward_error=include_previous_backward_error,
        projection_weights_prev=projection_weights_prev,
        fixed_delta_0=fixed_delta_0,
        backward_energy_weight=backward_energy_weight,
        forward_energy_weight=forward_energy_weight
    )

    H = len(top_down_model) - 1
    grads = []
    
    if skip_model is None:
        skip_model = [None] * (H + 1)
    
    for l in range(H):
        # Contribution from layer l's own energy only
        def energy_l(acts_l):
            # Create a modified activities list with acts_l at position l
            modified_activities = list(activities)
            modified_activities[l] = acts_l
            return _pdm_single_layer_energy(
                top_down_model=top_down_model,
                bottom_up_model=bottom_up_model,
                activities=modified_activities,
                y=y,
                x=x,
                layer_idx=l,
                skip_model=skip_model,
                param_type=param_type,
                include_previous_backward_error=include_previous_backward_error,
                projection_weights_prev=projection_weights_prev,
                fixed_delta_0=fixed_delta_0,
                backward_energy_weight=backward_energy_weight,
                forward_energy_weight=forward_energy_weight
            )
        
        grad_l = grad(energy_l)(activities[l])
    
        grads.append(grad_l)
    
    # Add zero gradient for output layer (layer H) if activities includes it
    # The output layer is not included in the PDM energy, so its gradient is zero
    if len(activities) > H:
        grads.append(jnp.zeros_like(activities[H]))
    
    return energy, grads


########################### PARAMETER GRADIENTS ################################
################################################################################
def compute_pc_param_grads(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: Optional[ArrayLike] = None,
    loss_id: str = "mse",
    param_type: str = "sp",
    weight_decay: Scalar = 0.,
    spectral_penalty: Scalar = 0.,
    activity_decay: Scalar = 0.
) -> Tuple[PyTree[Array], PyTree[Array]]:
    """Computes the gradient of the [PC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.pc_energy_fn)
    with respect to model parameters $∇_θ \mathcal{F}$.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.

    **Other arguments:**

    - `x`: Optional prior of the generative model.
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `"mse"` (default) or cross-entropy `"ce"`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `weight_decay`: $\ell^2$ regulariser for the weights (0 by default).
    - `spectral_penalty`: Weight spectral penalty of the form 
        $||\mathbf{I} - \mathbf{W}_\ell^T \mathbf{W}_\ell||^2$ (0 by default).
    - `activity_decay`: $\ell^2$ regulariser for the activities (0 by default).

    **Returns:**

    List of parameter gradients for each model layer.

    """
    return filter_grad(pc_energy_fn)(
        params,
        activities,
        y,
        x=x,
        loss=loss_id,
        param_type=param_type,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay
    )


def compute_hpc_param_grads(
    model: PyTree[Callable],
    equilib_activities: PyTree[ArrayLike],
    amort_activities: PyTree[ArrayLike],
    x: ArrayLike,
    y: Optional[ArrayLike] = None
) -> PyTree[Array]:
    """Computes the gradient of the [hybrid PC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.hpc_energy_fn) 
    with respect to the amortiser's parameters $∇_θ \mathcal{F}$.

    !!! warning

        The input $x$ and output $y$ are reversed compared to 
        [`jpc.compute_pc_param_grads()`](https://thebuckleylab.github.io/jpc/api/Gradients/#jpc.compute_pc_param_grads) 
        ($x$ is the generator's target and $y$ is its optional input or prior). 
        Just think of $x$ and $y$ as the actual input and output of the 
        amortiser, respectively.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `equilib_activities`: List of equilibrated activities reached by the
        generator and target for the amortiser.
    - `amort_activities`: List of amortiser's feedforward guesses
        (initialisation) for the model activities.
    - `x`: Input to the amortiser.
    - `y`: Optional target of the amortiser (for supervised training).

    **Returns:**

    List of parameter gradients for each model layer.

    """
    return filter_grad(hpc_energy_fn)(
        model,
        equilib_activities,
        amort_activities,
        x,
        y
    )


def compute_bpc_param_grads(
    top_down_model: PyTree[Callable], 
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    x: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0
) -> Tuple[PyTree[Array], PyTree[Array]]:
    """Computes the gradient of the [BPC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.bpc_energy_fn)
    with respect to all the model parameters $∇_θ \mathcal{F}$.

    **Main arguments:**

    - `top_down_model`: List of callable model (e.g. neural network) layers for 
        the forward model.
    - `bottom_up_model`: List of callable model (e.g. neural network) layers for 
        the backward model.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Target of the `top_down_model` and input to the `bottom_up_model`.
    - `x`: Input to the `top_down_model` and target of the `bottom_up_model`.

    **Other arguments:**

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

    Tuple of parameter gradients for the top-down and bottom-up models.

    """
    def wrapped_energy_fn(models, activities, y, x, skip_model, param_type, backward_energy_weight, forward_energy_weight):
        top_down_model, bottom_up_model = models
        return bpc_energy_fn(
            top_down_model, 
            bottom_up_model, 
            activities, 
            y, 
            x, 
            skip_model=skip_model,
            param_type=param_type,
            backward_energy_weight=backward_energy_weight,
            forward_energy_weight=forward_energy_weight
        )
    
    return filter_grad(wrapped_energy_fn)(
        (top_down_model, bottom_up_model), 
        activities, 
        y, 
        x, 
        skip_model=skip_model,
        param_type=param_type,
        backward_energy_weight=backward_energy_weight,
        forward_energy_weight=forward_energy_weight
    )


def compute_pdm_param_grads(
    top_down_model: PyTree[Callable], 
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    x: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    spectral_penalty: Scalar = 0.0,
    include_previous_backward_error: bool = False,
    projection_weights_prev: Optional[PyTree[Callable]] = None,
    fixed_delta_0: Optional[ArrayLike] = None,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0,
) -> Tuple[PyTree[Array], PyTree[Array]]:
    """Computes the gradient of the [PDM energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.pdm_energy_fn)
    with respect to all the model parameters $∇_θ \mathcal{F}$, which is the 
    same as that of the [BPC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.bpc_energy_fn)
    plus the spectral penalties.
    
    **Main arguments:**

    - `top_down_model`: List of callable model (e.g. neural network) layers for 
        the forward model.
    - `bottom_up_model`: List of callable model (e.g. neural network) layers for 
        the backward model.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Target of the `top_down_model` and input to the `bottom_up_model`.
    - `x`: Input to the `top_down_model` and target of the `bottom_up_model`.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `spectral_penalty`: Regularization strength for the penalty that 
        penalizes non-orthogonal forward weights. Defaults to 0.0.
    - `include_previous_backward_error`: If `True`, includes an extra dendritic term 
        $(z_\ell - \delta_{\ell})^2/2$ that couples each layer with the backward 
        error at the previous layer, where $\delta_{\ell} = z_{\ell-1} - V_{\ell} z_{\ell}$. 
        Defaults to `False`.
    - `backward_energy_weight`: Scalar weighting for the backward energy terms. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward energy terms. 
        Defaults to `1.0`.

    **Returns:**

    Tuple of parameter gradients for the top-down and bottom-up models.
    
    """
    if include_previous_backward_error or backward_energy_weight != 1.0 or forward_energy_weight != 1.0:
        # When including previous backward error or energy weightings, compute gradients directly from pdm_energy_fn
        def wrapped_energy_fn(models, activities, y, x, skip_model, param_type, spectral_penalty, include_previous_backward_error, projection_weights_prev, fixed_delta_0, backward_energy_weight, forward_energy_weight):
            top_down_model, bottom_up_model = models
            return pdm_energy_fn(
                top_down_model=top_down_model,
                bottom_up_model=bottom_up_model,
                activities=activities,
                y=y,
                x=x,
                skip_model=skip_model,
                param_type=param_type,
                spectral_penalty=spectral_penalty,
                include_previous_backward_error=include_previous_backward_error,
                projection_weights_prev=projection_weights_prev,
                fixed_delta_0=fixed_delta_0,
                backward_energy_weight=backward_energy_weight,
                forward_energy_weight=forward_energy_weight
            )
        
        top_down_grads, bottom_up_grads = filter_grad(wrapped_energy_fn)(
            (top_down_model, bottom_up_model),
            activities,
            y,
            x,
            skip_model=skip_model,
            param_type=param_type,
            spectral_penalty=spectral_penalty,
            include_previous_backward_error=include_previous_backward_error,
            projection_weights_prev=projection_weights_prev,
            fixed_delta_0=fixed_delta_0,
            backward_energy_weight=backward_energy_weight,
            forward_energy_weight=forward_energy_weight
        )
    else:
        # Get BPC parameter gradients (same as PDM for the base energy)
        bpc_top_down_grads, bpc_bottom_up_grads = compute_bpc_param_grads(
            top_down_model,
            bottom_up_model,
            activities,
            y,
            x,
            skip_model=skip_model,
            param_type=param_type
        )
        
        # Initialize gradients from BPC gradients
        top_down_grads = bpc_top_down_grads
        bottom_up_grads = bpc_bottom_up_grads
        
        # Add forward weights spectral penalty gradients if specified
        if spectral_penalty > 0.0:
            def fwd_regularizer_fn(top_down_model):
                H = len(top_down_model) - 1
                reg = 0.0
                for i in range(H):
                    W = top_down_model[i][1].weight  # Shape: (out_dim, in_dim)
                    out_dim, in_dim = W.shape
                    
                    # Determine which orthonormality to check (same logic as compute_fwd_orthogonality_diff)
                    check_columns = out_dim >= in_dim
                    
                    if check_columns:
                        # Compute ||W^T @ W - I||^2_F (column orthonormality)
                        WT_W = W.T @ W
                        I = jnp.eye(WT_W.shape[0])
                        reg += jnp.sum((WT_W - I) ** 2)
                    else:
                        # Compute ||W @ W^T - I||^2_F (row orthonormality)
                        W_WT = W @ W.T
                        I = jnp.eye(W_WT.shape[0])
                        reg += jnp.sum((W_WT - I) ** 2)
                return spectral_penalty * reg
            
            # Compute regularizer value for debugging
            fwd_reg_value = fwd_regularizer_fn(top_down_model)
            #jax.debug.print("[DEBUG] fwd_spectral_penalty={x}, fwd_reg_value={y}", x=fwd_spectral_penalty, y=fwd_reg_value)
            
            # Compute gradients with respect to top_down_model only
            fwd_reg_top_down_grads = filter_grad(fwd_regularizer_fn)(top_down_model)
            
            # Combine with existing gradients
            def add_grads(g1, g2):
                """Add two gradient trees, handling None values."""
                if g1 is None:
                    return g2
                if g2 is None:
                    return g1
                # Both are not None - recursively add using tree_map
                return tree_map(lambda x, y: x + y, g1, g2)
            
            top_down_grads = add_grads(top_down_grads, fwd_reg_top_down_grads)
    
    # Note: projection_weights_prev gradients are not returned here as they are handled separately
    # if projection_weights_prev is provided, it should be updated separately in the training loop
    return top_down_grads, bottom_up_grads


def compute_epc_error_grad(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    errors: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: Optional[ArrayLike] = None,
    loss_id: str = "mse",
    param_type: str = "sp"
) -> Tuple[Scalar, PyTree[Array]]:
    """Computes the gradient of the [ePC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.epc_energy_fn)
    with respect to the errors $∇_{\epsilon} \mathcal{F}$.

    !!! note

        In ePC, errors are updated during inference rather than activities.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `errors`: List of errors for each layer free to vary.
    - `y`: Observation or target of the generative model.

    **Other arguments:**

    - `x`: Optional prior of the generative model.
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `"mse"` (default) or cross-entropy `"ce"`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco_Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.

    **Returns:**

    The energy and its gradient with respect to the errors.

    """
    energy, dFdes = value_and_grad(epc_energy_fn, argnums=1)(
        params,
        errors,
        y,
        x=x,
        loss=loss_id,
        param_type=param_type
    )
    return energy, dFdes


def compute_epc_param_grads(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    errors: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: Optional[ArrayLike] = None,
    loss_id: str = "mse",
    param_type: str = "sp"
) -> Tuple[PyTree[Array], PyTree[Array]]:
    """Computes the gradient of the [ePC energy](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.epc_energy_fn)
    with respect to model parameters $∇_θ \mathcal{F}$.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `errors`: List of errors for each layer free to vary.
    - `y`: Observation or target of the generative model.

    **Other arguments:**

    - `x`: Optional prior of the generative model.
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `"mse"` (default) or cross-entropy `"ce"`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco_Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.

    **Returns:**

    List of parameter gradients for each model layer.

    """
    return filter_grad(epc_energy_fn)(
        params,
        errors,
        y,
        x=x,
        loss=loss_id,
        param_type=param_type
    )
