"""Energy functions for PC networks."""

from jax import vmap
from jax.numpy import sum, array, eye, sqrt
from jax.nn import log_softmax
from jaxtyping import PyTree, ArrayLike, Scalar, Array
from typing import Tuple, Callable, Optional
from ._errors import _check_param_type


def pc_energy_fn(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: Optional[ArrayLike] = None,
    loss: str = "mse",
    param_type: str = "sp",
    weight_decay: Scalar = 0.,
    spectral_penalty: Scalar = 0.,
    activity_decay: Scalar = 0.,
    record_layers: bool = False
) -> Scalar | Array:
    r"""Computes the PC energy for a neural network of the form

    $$
    \mathcal{F}(\mathbf{z}; θ) = 1/2N \sum_i^N \sum_{\ell=1}^L || \mathbf{z}_{i, \ell} - f_\ell(\mathbf{z}_{i, \ell-1}; θ_\ell) ||^2
    $$

    given parameters $θ$, activities $\mathbf{z}$, output 
    $\mathbf{z}_L = \mathbf{y}$, and optional input $\mathbf{z}_0 = \mathbf{x}$
    for supervised training. The activity of each layer $\mathbf{z}_\ell$ is
    some function of the previous layer, e.g.
    ReLU$(\mathbf{W}_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell)$ for a fully 
    connected layer with biases and ReLU as activation.

    !!! note

        The input $x$ and output $y$ correspond to the prior and observation of
        the generative model, respectively.

    **Main arguments:**

    - `params`: Tuple with callable model (e.g. neural network) layers and
        optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.

    **Other arguments:**

    - `x`: Optional prior of the generative model (for supervised training).
    - `loss`: Loss function to use at the output layer. Options are mean squared 
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
    - `record_layers`: If `True`, returns the energy of each layer.

    **Returns:**

    The total or layer-wise energy normalised by the batch size.

    """
    _check_param_type(param_type)

    model, skip_model = params
    batch_size = y.shape[0]
    start_activity_l = 1 if x is not None else 2
    n_activity_layers = len(activities) - 1
    n_hidden = len(model) - 1

    if skip_model is None:
        skip_model = [None] * len(model)

    scalings = _get_param_scalings(
        model=model, 
        input=x, 
        skip_model=skip_model, 
        param_type=param_type
    )

    if loss == "mse":
        eL = y - scalings[-1] * vmap(model[-1])(activities[-2])
        energies = [0.5 * sum(eL ** 2)]

    elif loss == "ce":
        logits = scalings[-1] * vmap(model[-1])(activities[-2])
        energies = [- sum(y * log_softmax(logits))]

    for act_l, net_l in zip(
            range(start_activity_l, n_activity_layers),
            range(1, n_hidden)
    ):
        err = activities[act_l] - scalings[net_l] * vmap(model[net_l])(activities[act_l - 1])
        if skip_model[net_l] is not None:
            err -= vmap(skip_model[net_l])(activities[act_l - 1])

        energies.append(0.5 * sum(err ** 2))

    if x is not None:
        e1 = activities[0] - scalings[0] * vmap(model[0])(x)
    else:
        e1 = activities[1] - vmap(model[0])(activities[0])

    energies.append(0.5 * sum(e1 ** 2))

    weight_reg = 0.
    if weight_decay > 0.:
        for layer in model:
            if hasattr(layer, "weight"):
                W = layer.weight
                weight_reg += sum(W ** 2)
        weight_reg *= weight_decay / 2

    spectral_reg = 0.
    if spectral_penalty > 0.:
        for layer in model:
            if hasattr(layer, "weight"):
                W = layer.weight
                WT_W_I = W.T @ W - eye(W.shape[1])
                spectral_reg += sum(WT_W_I ** 2)
        spectral_reg *= spectral_penalty / 2

    activity_reg = 0.
    if activity_decay > 0.:
        for activity in activities:
            mean_squared_l2 = (sum(activity ** 2, axis=1)).mean()
            activity_reg += mean_squared_l2
        activity_reg *= activity_decay / 2

    reg_terms = weight_reg + spectral_reg + activity_reg
    total_energy = (sum(array(energies)) / batch_size) + reg_terms

    if record_layers:
        return array(energies) / batch_size
    else:
        return total_energy


def hpc_energy_fn(
    model: PyTree[Callable],
    equilib_activities: PyTree[ArrayLike],
    amort_activities: PyTree[ArrayLike],
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    record_layers: bool = False
) -> Scalar | Array:
    r"""Computes the energy of an amortised PC network ([Tscshantz et al., 2023](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011280))

    $$
    \mathcal{F}(\mathbf{z}^*, \hat{\mathbf{z}}; θ) = 1/N \sum_i^N \sum_{\ell=1}^L || \mathbf{z}^*_{i, \ell} - f_\ell(\hat{\mathbf{z}}_{i, \ell-1}; θ_\ell) ||^2
    $$

    given the equilibrated activities of the generator $\mathbf{z}^*$ (target
    for the amortiser), the feedforward guesses of the amortiser
    $\hat{\mathbf{z}}$, the amortiser's parameters $θ$, input
    $\mathbf{z}_0 = \mathbf{x}$, and optional output
    $\mathbf{z}_L = \mathbf{y}$ for supervised training.

    !!! note

        The input $x$ and output $y$ are reversed compared to [`pc_energy_fn()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc.pc_energy_fn)
        ($x$ is the generator's target and $y$ is its optional input or prior).
        Just think of $x$ and $y$ as the actual input and output of the
        amortiser, respectively.
    
    ??? cite "Reference"

        ```bibtex
        @article{tscshantz2023hybrid,
            title={Hybrid predictive coding: Inferring, fast and slow},
            author={Tscshantz, Alexander and Millidge, Beren and Seth, Anil K and Buckley, Christopher L},
            journal={PLoS computational biology},
            volume={19},
            number={8},
            pages={e1011280},
            year={2023},
            publisher={Public Library of Science San Francisco, CA USA}
        }
        ```

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `equilib_activities`: List of equilibrated activities reached by the
        generator and target for the amortiser.
    - `amort_activities`: List of amortiser's feedforward guesses
        (initialisation) for the network activities.
    - `x`: Input to the amortiser.
    - `y`: Optional target of the amortiser (for supervised training).

    **Other arguments:**

    - `record_layers`: If `True`, returns energies for each layer.

    **Returns:**

    The total or layer-wise energy normalised by batch size.

    """
    batch_size = x.shape[0]
    start_l = 1
    n_layers = len(model) - 1

    amort_activities = amort_activities[::-1][1:]

    eL = y - vmap(model[-1])(amort_activities[-2]) if (
        y is not None
    ) else equilib_activities[-1] - vmap(model[-1])(amort_activities[-2])
    energies = [sum(eL ** 2)]

    for l in range(start_l, n_layers):
        err = equilib_activities[l] - vmap(model[l])(amort_activities[l-1])
        energies.append(sum(err ** 2))

    e1 = equilib_activities[0] - vmap(model[0])(x)
    energies.append(sum(e1 ** 2))

    if record_layers:
        return array(energies) / batch_size
    else:
        return sum(array(energies)) / batch_size


def bpc_energy_fn(
    top_down_model: PyTree[Callable], 
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    x: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    record_layers: bool = False,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0
    ) -> Scalar | Array:
    r"""Computes the energy of a bidirectional PC network (BPC, [Oliviers et al., 2025](https://arxiv.org/abs/2505.23415)) of the form

    $$
    \mathcal{F}(\mathbf{z}; θ) = 1/N \sum_i^N \left[ \alpha_f \sum_{\ell=1}^L || \mathbf{z}_{i, \ell} - f_\ell(\mathbf{z}_{i, \ell-1}; \mathbf{W}_\ell) ||^2/2 + \alpha_b \sum_{\ell=0}^{L-1} || \mathbf{z}_{i, \ell} - g_{\ell+1}(\mathbf{z}_{i, \ell+1}; \mathbf{V}_{\ell+1}) ||^2/2 \right]
    $$

    where $f_\ell(\cdot)$ and $g_{\ell+1}(\cdot)$ are the forward (top-down) and 
    backward (bottom-up) layer-wise transformations, $\mathbf{W}_\ell$ and 
    $\mathbf{V}_{\ell+1}$ as forward and backward weights, and $(\alpha_f, \alpha_b)$ 
    as weightings of the forward and backward energies, respectively. 
    See the reference below for more details.

    ??? cite "Reference"

        ```bibtex
        @article{oliviers2025bidirectional,
            title={Bidirectional predictive coding},
            author={Oliviers, Gaspard and Tang, Mufeng and Bogacz, Rafal},
            journal={arXiv preprint arXiv:2505.23415},
            year={2025}
        }
        ```

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
    - `record_layers`: If `True`, returns the energy of each layer.
    - `backward_energy_weight`: Scalar weighting for the backward energy terms. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward energy terms. 
        Defaults to `1.0`.

    **Returns:**

    The total or layer-wise BPC energy normalised by batch size.

    """
    _check_param_type(param_type)

    batch_size = activities[0].shape[0]
    H = len(top_down_model) - 1
    energies = []

    if skip_model is None:
        skip_model = [None] * (H + 1)

    # discriminative (bottom-up) loss
    delta_0 = x - vmap(bottom_up_model[0])(activities[0])
    bottom_up_energy = backward_energy_weight * 0.5 * sum(delta_0 ** 2)

    # generative (top-down) loss
    e_L = y - vmap(top_down_model[-1])(activities[-2])
    top_down_energy = forward_energy_weight * 0.5 * sum(e_L ** 2)

    energies.append((top_down_energy, bottom_up_energy))

    for l in range(H):
        act_prev = x if l == 0 else activities[l - 1]
        e_l = activities[l] - vmap(top_down_model[l])(act_prev)
        if skip_model[l] is not None and l > 0:
            e_l -= vmap(skip_model[l])(act_prev)
        
        act_next = y if l == H - 1 else activities[l + 1]
        delta_l = activities[l] - vmap(bottom_up_model[l + 1])(act_next)
        if skip_model[l + 1] is not None and l < H - 1:
            delta_l -= vmap(skip_model[l + 1])(act_next)

        top_down_energy = forward_energy_weight * 0.5 * sum(e_l ** 2)
        bottom_up_energy = backward_energy_weight * 0.5 * sum(delta_l ** 2)

        energies.append((top_down_energy, bottom_up_energy))

    total_energy = (sum(array(energies)) / batch_size)

    if record_layers:
        return array([val for pair in energies for val in pair]) / batch_size
    else:
        return total_energy


def epc_energy_fn(
    params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
    errors: PyTree[ArrayLike],
    y: ArrayLike,
    *,
    x: Optional[ArrayLike] = None,
    loss: str = "mse",
    param_type: str = "sp"
) -> Scalar | Array:
    r"""Computes the error-reparameterised PC (ePC) energy for a neural network ([Goemaere et al., 2025](https://arxiv.org/abs/2505.20137)) of the form

    $$
    \mathcal{F} = \sum_{\ell=1}^{L-1} \frac{1}{2} ||\epsilon_\ell||^2 + \frac{1}{2}||\mathbf{y} - \tilde{\mathbf{f}}(\mathbf{z}_0, \epsilon_1, \dots, \epsilon_{L-1})||^2
    $$

    where $\epsilon_\ell = \mathbf{z}_\ell - f_\ell(\mathbf{W}_\ell \mathbf{z}_{\ell-1})$ 
    are the prediction errors at each layer, and 
    $\tilde{\mathbf{f}}(\mathbf{z}_0, \epsilon)$ is an error-perturbed forward 
    pass. In ePC, errors are the variables updated during inference rather than 
    activities. See the reference below for more details.

    ??? cite "Reference"

        ```bibtex
        @article{goemaere2025error,
            title={Error Optimization: Overcoming Exponential Signal Decay in Deep Predictive Coding Networks},
            author={Goemaere, C{\'e}dric and Oliviers, Gaspard and Bogacz, Rafal and Demeester, Thomas},
            journal={arXiv preprint arXiv:2505.20137},
            year={2025}
        }
        ```

    **Main arguments:**

    - `params`: Tuple with callable model (e.g. neural network) layers and
        optional skip connections.
    - `errors`: List of predictionerrors for each layer.
    - `y`: Observation or target of the generative model.

    **Other arguments:**

    - `x`: Optional prior of the generative model (for supervised training).
    - `loss`: Loss function to use at the output layer. Options are mean squared 
        error `"mse"` (default) or cross-entropy `"ce"`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco_Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.

    **Returns:**

    The total ePC energy normalised by the batch size.

    """
    _check_param_type(param_type)

    model, skip_model = params
    batch_size = y.shape[0]
    n_hidden = len(model) - 1

    if skip_model is None:
        skip_model = [None] * len(model)

    scalings = _get_param_scalings(
        model=model, 
        input=x if x is not None else errors[0], 
        skip_model=skip_model, 
        param_type=param_type
    )

    if x is not None:
        current_activity = x
        error_idx = 0
    else:
        current_activity = errors[0]
        error_idx = 1

    for net_l in range(n_hidden):
        z_l = scalings[net_l] * vmap(model[net_l])(current_activity) + errors[error_idx]
        if skip_model[net_l] is not None:
            z_l += vmap(skip_model[net_l])(current_activity)
        
        current_activity = z_l
        error_idx += 1

    zL = scalings[-1] * vmap(model[-1])(current_activity) - errors[error_idx]

    total_energy = 0.0
    for err_idx in range(n_hidden):
        total_energy += 0.5 * sum(errors[err_idx] ** 2)

    if loss == "mse":
        total_energy += 0.5 * sum((y - zL) ** 2)
    elif loss == "ce":
        total_energy += - sum(y * log_softmax(zL))

    return total_energy / batch_size


def _pdm_single_layer_energy(
    top_down_model: PyTree[Callable],
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    x: ArrayLike,
    layer_idx: int,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    include_previous_backward_error: bool = False,
    projection_weights_prev: Optional[PyTree[Callable]] = None,
    fixed_delta_0: Optional[ArrayLike] = None,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0
) -> Scalar:
    r"""Computes the energy for a single layer of a predictive dendrites model (PDM)

    $$
    \mathcal{F}_\ell = 1/2N \sum_i^N [|| \mathbf{z}_{i, \ell} - f_\ell(\mathbf{z}_{i, \ell-1}; \mathbf{W}_\ell) ||^2/2 + || \mathbf{z}_{i, \ell} - g_{\ell+1}(\mathbf{z}_{i, \ell+1}; \mathbf{V}_{\ell+1}) ||^2/2 + || \mathbf{z}_{i, \ell} - \boldsymbol{\epsilon}_{i, \ell+1} ||^2/2]
    $$
    
    where $f_\ell(\cdot)$ and $g_{\ell+1}(\cdot)$ are the forward (top-down) and 
    backward (bottom-up) layer-wise transformations, $\mathbf{W}_\ell$ and 
    $\mathbf{V}_{\ell+1}$ are the forward and backward weights, respectively, and
    $\boldsymbol{\epsilon}_{\ell+1} = \mathbf{z}_{\ell+1} - f_{\ell+1}(\mathbf{z}_\ell; \mathbf{W}_{\ell+1})$
    is the forward error at the next layer. 

    **Main arguments:**

    - `top_down_model`: List of callable model (e.g. neural network) layers for 
        the forward model.
    - `bottom_up_model`: List of callable model (e.g. neural network) layers for 
        the backward model.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Target of the `top_down_model` and input to the `bottom_up_model`.
    - `x`: Input to the `top_down_model` and target of the `bottom_up_model`.
    - `layer_idx`: Index of the layer to compute the energy for.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `include_previous_backward_error`: If `True`, includes an extra dendritic term 
        $(z_\ell - \delta_{\ell})^2/2$ that couples the layer with the backward 
        error at the previous layer, where $\delta_{\ell} = z_{\ell-1} - V_{\ell} z_{\ell}$. 
        Defaults to `False`.
    - `projection_weights_prev`: Optional list of projection weight matrices for projecting 
        input errors to hidden dimensions. Only used for the first layer when 
        `include_previous_backward_error=True`. If `None`, falls back to using the forward 
        model weight. Defaults to `None`.
    - `backward_energy_weight`: Scalar weighting for the backward energy term. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward energy term. 
        Defaults to `1.0`.

    **Returns:**

    The energy for the single PDM layer normalised by batch size.
    
    """
    _check_param_type(param_type)

    H = len(top_down_model) - 1
    batch_size = activities[0].shape[0]
    
    if skip_model is None:
        skip_model = [None] * (H + 1)
    
    # Forward error: ε_ℓ = z_ℓ - W_ℓ z_{ℓ-1}
    act_prev = x if layer_idx == 0 else activities[layer_idx - 1]
    forward_pred = vmap(top_down_model[layer_idx])(act_prev)
    if skip_model[layer_idx] is not None and layer_idx > 0:
        forward_pred += vmap(skip_model[layer_idx])(act_prev)
    e_l = activities[layer_idx] - forward_pred
    forward_energy = forward_energy_weight * 0.5 * sum(e_l ** 2)
    
    # Backward error: δ_{ℓ+1} = z_ℓ - V_{ℓ+1} z_{ℓ+1}
    act_next = y if layer_idx == H - 1 else activities[layer_idx + 1]
    backward_pred = vmap(bottom_up_model[layer_idx + 1])(act_next)
    if skip_model[layer_idx + 1] is not None and layer_idx < H - 1:
        backward_pred += vmap(skip_model[layer_idx + 1])(act_next)
    delta_l = activities[layer_idx] - backward_pred
    backward_energy = backward_energy_weight * 0.5 * sum(delta_l ** 2)
    
    # Extra dendritic term: (z_ℓ - δ_{ℓ-1})²/2
    # where δ_{ℓ-1} is the backward error at layer ℓ-1 (independent of z_ℓ)
    # This approximates the BPC term x - V_1 z_1 by using the backward error from the previous layer
    previous_backward_energy = 0.0
    if include_previous_backward_error:
        if layer_idx > 0:
            # For non-first layers, use the backward error at layer ℓ-1
            if layer_idx == 1:
                # For layer 1, use δ_0 = x - V_0 z_0 (backward error at layer 0)
                # Use fixed_delta_0 if provided (computed from initial activities), otherwise compute fresh
                if fixed_delta_0 is not None:
                    delta_l_minus_1 = fixed_delta_0
                else:
                    delta_l_minus_1 = x - vmap(bottom_up_model[0])(activities[0])
                    if skip_model[0] is not None:
                        delta_l_minus_1 -= vmap(skip_model[0])(activities[0])
            else:
                # For layer ℓ > 1, use δ_{ℓ-1} = z_{ℓ-2} - V_{ℓ-1} z_{ℓ-1}
                delta_l_minus_1 = activities[layer_idx - 2] - vmap(bottom_up_model[layer_idx - 1])(activities[layer_idx - 1])
                if skip_model[layer_idx - 1] is not None:
                    delta_l_minus_1 -= vmap(skip_model[layer_idx - 1])(activities[layer_idx - 1])
            # Project error to hidden dimension if needed (for layer 1, dimensions might not match)
            if layer_idx == 1 and projection_weights_prev is not None and projection_weights_prev[0] is not None:
                delta_l_minus_1_proj = vmap(projection_weights_prev[0])(delta_l_minus_1)
            else:
                delta_l_minus_1_proj = delta_l_minus_1
            # Extra dendritic term: (z_ℓ - δ_{ℓ-1})²/2
            # Weight by backward_energy_weight since it's based on backward errors
            previous_backward_energy = backward_energy_weight * 0.5 * sum((activities[layer_idx] - delta_l_minus_1_proj) ** 2)
        elif layer_idx == 0:
            # For the first layer, δ_{-1} doesn't exist, so we use δ_0 = x - V_0 z_0
            # and project it (this is the backward error at layer 0 itself)
            delta_0 = x - vmap(bottom_up_model[0])(activities[0])
            if skip_model[0] is not None:
                delta_0 -= vmap(skip_model[0])(activities[0])
            # Project error to hidden dimension using projection weight matrix
            if projection_weights_prev is not None and projection_weights_prev[0] is not None:
                delta_0_proj = vmap(projection_weights_prev[0])(delta_0)
            else:
                # Fallback: use forward model weight if projection weight not provided
                delta_0_proj = vmap(top_down_model[0])(delta_0)
            # Extra dendritic term: (z_0 - P_0 * δ_0)²/2 where P_0 is the projection weight
            # Weight by backward_energy_weight since it's based on backward errors
            previous_backward_energy = backward_energy_weight * 0.5 * sum((activities[0] - delta_0_proj) ** 2)
    
    return (forward_energy + backward_energy + previous_backward_energy) / batch_size


def pdm_energy_fn( 
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
) -> Scalar | Array:
    r"""Computes the energy for a predictive dendrites model (PDM) that sums 
    single-layer energies of [`_pdm_single_layer_energy()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._pdm_single_layer_energy).
    
    $$
    \mathcal{F}(\mathbf{z}; θ) = \sum_{\ell=1}^L \mathcal{F}_\ell
    $$
    
    where the single-layer energies are defined as $\mathcal{F}_\ell = 1/2N \sum_i^N [|| \mathbf{z}_{i, \ell} - f_\ell(\mathbf{z}_{i, \ell-1}; \mathbf{W}_\ell) ||^2/2 + || \mathbf{z}_{i, \ell} - g_{\ell+1}(\mathbf{z}_{i, \ell+1}; \mathbf{V}_{\ell+1}) ||^2/2]$.
    See [`_pdm_single_layer_energy()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._pdm_single_layer_energy) 
    for more details.

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
    - `spectral_penalty`: Weight spectral penalty for forward weights (0 by default).
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

    The total PDM energy normalised by batch size.

    """
    _check_param_type(param_type)
    
    H = len(top_down_model) - 1
    total_energy = 0.
    
    for l in range(H):
        layer_energy = _pdm_single_layer_energy(
            top_down_model=top_down_model,
            bottom_up_model=bottom_up_model,
            activities=activities,
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
        total_energy += layer_energy
    
    # Add forward-only spectral penalty if specified
    if spectral_penalty > 0.0:
        reg = 0.0
        for i in range(H):
            # Extract forward weight W_{i+1}
            W = top_down_model[i][1].weight  # Shape: (out_dim, in_dim)
            out_dim, in_dim = W.shape
            
            # Determine which orthonormality to check (same logic as compute_fwd_orthogonality_diff)
            check_columns = out_dim >= in_dim
            
            if check_columns:
                # Compute ||W^T @ W - I||^2_F (column orthonormality)
                WT_W = W.T @ W
                I = eye(WT_W.shape[0])
                reg += sum((WT_W - I) ** 2)
            else:
                # Compute ||W @ W^T - I||^2_F (row orthonormality)
                W_WT = W @ W.T
                I = eye(W_WT.shape[0])
                reg += sum((W_WT - I) ** 2)
        
        total_energy += spectral_penalty * reg
    
    return total_energy


def _get_param_scalings(
    model: PyTree[Callable], 
    input: ArrayLike, 
    *,
    skip_model: Optional[PyTree[Callable]] = None, 
    param_type: str = "sp"
) -> list[float]:
    """Gets layer scalings for a given parameterisation.

    !!! warning

        `param_type = "mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))) assumes 
        that one is using [`jpc.make_mlp()`](https://thebuckleylab.github.io/jpc/api/Utils/#jpc.make_mlp) 
        to create the model.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `input`: input to the model.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). Defaults to `"sp"`.

    **Returns:**

    List with scalings for each layer.

    """
    L = len(model)

    if param_type == "sp":
        scalings = [1.] + [1] * (L-2) + [1]

    else:
        D = input.shape[1]
        N = model[0][1].weight.shape[0]
        
        a1 = 1 / sqrt(D)
        al = 1 / sqrt(N) if skip_model is None else 1 / sqrt(N * L)
        aL = 1 / N if param_type == "mupc" else 1 / sqrt(N)
        scalings = [a1] + [al] * (L-2) + [aL]

    return scalings
