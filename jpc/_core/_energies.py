"""Energy functions for PC networks."""

from jax import vmap
from jax.lax import stop_gradient
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
    record_layers: bool = False,
    gamma: Optional[Scalar] = None
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
    - `gamma`: Optional scaling factor for the output layer. If provided, the output 
        layer scaling is multiplied by `1/gamma`.  Defaults to `None` (no additional scaling).

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
        param_type=param_type,
        gamma=gamma
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
    lateral_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    include_previous_backward_error: bool = False,
    include_next_forward_error: bool = False,
    projection_matrix_prev: Optional[PyTree[Callable]] = None,
    projection_matrix_next: Optional[PyTree[Callable]] = None,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0,
    stop_gradient_on_extra_errors: bool = True
) -> Scalar:
    r"""Computes the energy for a single layer of a predictive dendrites model (PDM)

    $$
    \mathcal{F}_\ell = 1/2N \sum_i^N [|| \mathbf{e}_{i, \ell} + \mathbf{A}_\ell \boldsymbol{\delta}_{i, \ell-1} ||^2/2 + || \boldsymbol{\delta}_{i, \ell+1} + \mathbf{B}_\ell \mathbf{e}_{i, \ell+1} ||^2/2]
    $$
    
    where $\mathbf{e}_\ell = \mathbf{z}_\ell - f_\ell(\mathbf{z}_{\ell-1}; \mathbf{W}_\ell)$ is the forward error,
    $\boldsymbol{\delta}_\ell = \mathbf{z}_{\ell-1} - g_\ell(\mathbf{z}_\ell; \mathbf{V}_\ell)$ is the backward error,
    $f_\ell(\cdot)$ and $g_\ell(\cdot)$ are the forward (top-down) and backward (bottom-up) 
    layer-wise transformations, $\mathbf{W}_\ell$ and $\mathbf{V}_\ell$ are the forward and 
    backward weights, respectively, and $\mathbf{A}_\ell$ and $\mathbf{B}_\ell$ are optional 
    projection matrices. When `include_previous_backward_error=False` and 
    `include_next_forward_error=False`, this reduces to the standard PDM energy:
    $||\mathbf{e}_\ell||^2/2 + ||\boldsymbol{\delta}_{\ell+1}||^2/2$.

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
    - `lateral_model`: Optional list of lateral connection models, one per hidden layer.
        Each lateral model maps layer activities to themselves, adding within-layer
        predictions to the backward error: $\boldsymbol{\delta}_{\ell+1} = \mathbf{z}_\ell - g_{\ell+1}(\mathbf{z}_{\ell+1}) - \mathbf{W}_{lat,\ell} \mathbf{z}_\ell$.
        Defaults to `None`.
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
        If `None` and `include_previous_backward_error=True`, dimensions must match.
        Defaults to `None`.
    - `projection_matrix_next`: Optional projection matrix $\mathbf{B}_\ell$ for projecting 
        the next forward error. Only used when `include_next_forward_error=True`.
        If `None` and `include_next_forward_error=True`, dimensions must match.
        Defaults to `None`.
    - `backward_energy_weight`: Scalar weighting for the backward prediction error 
        (delta) only. Scales $\boldsymbol{\delta}_{\ell+1}$ before computing the energy. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward prediction error 
        (epsilon) only. Scales $\mathbf{e}_\ell$ before computing the energy. 
        Defaults to `1.0`.
    - `stop_gradient_on_extra_errors`: If `True`, applies `stop_gradient` to the 
        extra error terms ($\boldsymbol{\delta}_{\ell-1}$ and $\mathbf{e}_{\ell+1}$) 
        so that gradients do not flow through them during backpropagation. This means
        gradients only flow through the primary errors ($\mathbf{e}_\ell$ and 
        $\boldsymbol{\delta}_{\ell+1}$). Defaults to `True`.

    **Returns:**

    The energy for the single PDM layer normalised by batch size.
    
    """
    _check_param_type(param_type)

    H = len(top_down_model) - 1
    batch_size = activities[0].shape[0]
    
    if skip_model is None:
        skip_model = [None] * (H + 1)
    
    if lateral_model is None:
        lateral_model = [None] * H
    
    # Forward error: e_ℓ = z_ℓ - W_ℓ z_{ℓ-1}
    act_prev = x if layer_idx == 0 else activities[layer_idx - 1]
    forward_pred = vmap(top_down_model[layer_idx])(act_prev)
    if skip_model[layer_idx] is not None and layer_idx > 0:
        forward_pred += vmap(skip_model[layer_idx])(act_prev)
    e_l = activities[layer_idx] - forward_pred
    
    # Backward error: δ_{ℓ+1} = z_ℓ - V_{ℓ+1} z_{ℓ+1} - W_lat z_ℓ (if lateral connections)
    act_next = y if layer_idx == H - 1 else activities[layer_idx + 1]
    backward_pred = vmap(bottom_up_model[layer_idx + 1])(act_next)
    if skip_model[layer_idx + 1] is not None and layer_idx < H - 1:
        backward_pred += vmap(skip_model[layer_idx + 1])(act_next)
    # Add lateral connections (within-layer predictions)
    if lateral_model[layer_idx] is not None:
        backward_pred += vmap(lateral_model[layer_idx])(activities[layer_idx])
    delta_l_plus_1 = activities[layer_idx] - backward_pred
    
    # Compute previous backward error if needed: δ_{ℓ-1} = z_{ℓ-1} - V_ℓ z_ℓ
    # (or δ_0 = x - V_0 z_0 for layer_idx == 0)
    if include_previous_backward_error:
        if layer_idx == 0:
            # For the first layer, δ_0 = x - V_0 z_0
            delta_l_minus_1 = x - vmap(bottom_up_model[0])(activities[0])
            if skip_model[0] is not None:
                delta_l_minus_1 -= vmap(skip_model[0])(activities[0])
        else:
            # For layer ℓ > 0, δ_{ℓ-1} = z_{ℓ-1} - V_ℓ z_ℓ
            # where z_{ℓ-1} is the previous layer's activity and z_ℓ is the current layer's activity
            act_prev = x if layer_idx == 1 else activities[layer_idx - 1]
            delta_l_minus_1 = act_prev - vmap(bottom_up_model[layer_idx])(activities[layer_idx])
            if skip_model[layer_idx] is not None and layer_idx > 0:
                delta_l_minus_1 -= vmap(skip_model[layer_idx])(activities[layer_idx])
        
        # Project if projection matrix is provided
        if projection_matrix_prev is not None:
            if isinstance(projection_matrix_prev, (list, tuple)) and len(projection_matrix_prev) > layer_idx:
                delta_l_minus_1_proj = vmap(projection_matrix_prev[layer_idx])(delta_l_minus_1)
            elif not isinstance(projection_matrix_prev, (list, tuple)):
                # Single projection matrix for all layers
                delta_l_minus_1_proj = vmap(projection_matrix_prev)(delta_l_minus_1)
            else:
                delta_l_minus_1_proj = delta_l_minus_1
        else:
            delta_l_minus_1_proj = delta_l_minus_1
        
        # Optionally stop gradient on the extra error term
        if stop_gradient_on_extra_errors:
            delta_l_minus_1_proj = stop_gradient(delta_l_minus_1_proj)
        
        # Forward energy: ||α_f e_ℓ + A_ℓ δ_{ℓ-1}||^2/2
        # where α_f is forward_energy_weight scaling only the forward error e_ℓ
        forward_energy = 0.5 * sum((
            forward_energy_weight * e_l + backward_energy_weight * delta_l_minus_1_proj
        ) ** 2)
    else:
        # Standard forward energy: ||α_f e_ℓ||^2/2
        # where α_f is forward_energy_weight scaling only the forward error e_ℓ
        forward_energy = forward_energy_weight * 0.5 * sum((e_l) ** 2)
    
    # Compute next forward error if needed: e_{ℓ+1} = z_{ℓ+1} - W_{ℓ+1} z_ℓ
    # (or e_L = y - W_L z_{L-1} for layer_idx == H - 1)
    if include_next_forward_error:
        if layer_idx == H - 1:
            # For the last hidden layer, e_L = y - W_L z_{L-1} where z_{L-1} is current layer
            e_l_plus_1 = y - vmap(top_down_model[H])(activities[layer_idx])
            if skip_model[H] is not None:
                e_l_plus_1 -= vmap(skip_model[H])(activities[layer_idx])
        else:
            # For layer ℓ < H - 1, e_{ℓ+1} = z_{ℓ+1} - W_{ℓ+1} z_ℓ
            # where z_{ℓ+1} is the next layer's activity and z_ℓ is the current layer's activity
            act_next = y if layer_idx == H - 2 else activities[layer_idx + 1]
            e_l_plus_1 = act_next - vmap(top_down_model[layer_idx + 1])(activities[layer_idx])
            if skip_model[layer_idx + 1] is not None and layer_idx < H - 1:
                e_l_plus_1 -= vmap(skip_model[layer_idx + 1])(activities[layer_idx])
        
        # Project if projection matrix is provided
        if projection_matrix_next is not None:
            if isinstance(projection_matrix_next, (list, tuple)) and len(projection_matrix_next) > layer_idx:
                e_l_plus_1_proj = vmap(projection_matrix_next[layer_idx])(e_l_plus_1)
            elif not isinstance(projection_matrix_next, (list, tuple)):
                # Single projection matrix for all layers
                e_l_plus_1_proj = vmap(projection_matrix_next)(e_l_plus_1)
            else:
                e_l_plus_1_proj = e_l_plus_1
        else:
            e_l_plus_1_proj = e_l_plus_1
        
        # Optionally stop gradient on the extra error term
        if stop_gradient_on_extra_errors:
            e_l_plus_1_proj = stop_gradient(e_l_plus_1_proj)
        
        # Backward energy: ||α_b δ_{ℓ+1} + B_ℓ e_{ℓ+1}||^2/2
        # where α_b is backward_energy_weight scaling only the backward error δ_{ℓ+1}
        backward_energy = 0.5 * sum((
            backward_energy_weight * delta_l_plus_1 + forward_energy_weight * e_l_plus_1_proj
        ) ** 2)
    else:
        # Standard backward energy: ||α_b δ_{ℓ+1}||^2/2
        # where α_b is backward_energy_weight scaling only the backward error δ_{ℓ+1}
        backward_energy = backward_energy_weight * 0.5 * sum((delta_l_plus_1) ** 2)
    
    return (forward_energy + backward_energy) / batch_size


def pdm_energy_fn( 
    top_down_model: PyTree[Callable], 
    bottom_up_model: PyTree[Callable],
    activities: PyTree[ArrayLike],
    y: ArrayLike,
    x: ArrayLike,
    *,
    skip_model: Optional[PyTree[Callable]] = None,
    lateral_model: Optional[PyTree[Callable]] = None,
    param_type: str = "sp",
    spectral_penalty: Scalar = 0.0,
    include_previous_backward_error: bool = False,
    include_next_forward_error: bool = False,
    projection_matrix_prev: Optional[PyTree[Callable]] = None,
    projection_matrix_next: Optional[PyTree[Callable]] = None,
    backward_energy_weight: Scalar = 1.0,
    forward_energy_weight: Scalar = 1.0,
    stop_gradient_on_extra_errors: bool = True,
) -> Scalar | Array:
    r"""Computes the energy for a predictive dendrites model (PDM) that sums 
    single-layer energies of [`_pdm_single_layer_energy()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._pdm_single_layer_energy).
    
    $$
    \mathcal{F}(\mathbf{z}; θ) = \sum_{\ell=1}^L \mathcal{F}_\ell
    $$
    
    where the single-layer energies are defined as $\mathcal{F}_\ell = 1/2N \sum_i^N [|| \mathbf{e}_{i, \ell} + \mathbf{A}_\ell \boldsymbol{\delta}_{i, \ell-1} ||^2/2 + || \boldsymbol{\delta}_{i, \ell+1} + \mathbf{B}_\ell \mathbf{e}_{i, \ell+1} ||^2/2]$.
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
    - `lateral_model`: Optional list of lateral connection models, one per hidden layer.
        Each lateral model maps layer activities to themselves, adding within-layer
        predictions to the backward error: $\boldsymbol{\delta}_{\ell+1} = \mathbf{z}_\ell - g_{\ell+1}(\mathbf{z}_{\ell+1}) - \mathbf{W}_{lat,\ell} \mathbf{z}_\ell$.
        This implements the lateral connections as in the 3Q model. Defaults to `None`.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). 
        See [`_get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.
    - `spectral_penalty`: Weight spectral penalty for forward weights (0 by default).
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
        If `None` and `include_previous_backward_error=True`, dimensions must match.
        Defaults to `None`.
    - `projection_matrix_next`: Optional projection matrix $\mathbf{B}_\ell$ for projecting 
        the next forward error. Only used when `include_next_forward_error=True`.
        If `None` and `include_next_forward_error=True`, dimensions must match.
        Defaults to `None`.
    - `backward_energy_weight`: Scalar weighting for the backward prediction error 
        (delta) only. Scales $\boldsymbol{\delta}_{\ell+1}$ before computing the energy. 
        Defaults to `1.0`.
    - `forward_energy_weight`: Scalar weighting for the forward prediction error 
        (epsilon) only. Scales $\mathbf{e}_\ell$ before computing the energy. 
        Defaults to `1.0`.
    - `stop_gradient_on_extra_errors`: If `True`, applies `stop_gradient` to the 
        extra error terms ($\boldsymbol{\delta}_{\ell-1}$ and $\mathbf{e}_{\ell+1}$) 
        so that gradients do not flow through them during backpropagation. This means
        gradients only flow through the primary errors ($\mathbf{e}_\ell$ and 
        $\boldsymbol{\delta}_{\ell+1}$). Defaults to `True`.

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
            lateral_model=lateral_model,
            param_type=param_type,
            include_previous_backward_error=include_previous_backward_error,
            include_next_forward_error=include_next_forward_error,
            projection_matrix_prev=projection_matrix_prev,
            projection_matrix_next=projection_matrix_next,
            backward_energy_weight=backward_energy_weight,
            forward_energy_weight=forward_energy_weight,
            stop_gradient_on_extra_errors=stop_gradient_on_extra_errors
        )
        total_energy += layer_energy
    
    if spectral_penalty > 0.0:
        reg = 0.0
        for i in range(H):
            W = top_down_model[i][1].weight
            out_dim, in_dim = W.shape
            
            # Determine which orthonormality to check
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
    param_type: str = "sp",
    gamma: Optional[Scalar] = None
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
    - `gamma`: Optional scaling factor for the output layer. If provided, the output 
        layer scaling is multiplied by `1/gamma`. Defaults to `None` (no additional scaling).

    **Returns:**

    List with scalings for each layer.

    """
    L = len(model)
    no_skips = skip_model is None or all(s is None for s in skip_model)

    if param_type == "sp":
        scalings = [1.] + [1] * (L-2) + [1]

    else:
        D = input.shape[1]
        N = model[0][1].weight.shape[0]
        
        a1 = 1 / sqrt(D)
        al = 1 / sqrt(N) if no_skips else 1 / sqrt(N * L)
        aL = 1 / N if param_type == "mupc" else 1 / sqrt(N)
        scalings = [a1] + [al] * (L-2) + [aL]

    # Apply gamma scaling to output layer if provided
    if gamma is not None:
        scalings[-1] = scalings[-1] / gamma

    return scalings
