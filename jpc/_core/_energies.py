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
    """Computes the free energy for a neural network with optional skip 
    connections of the form

    $$
    \mathcal{F}(\mathbf{z}; θ) = 1/N \sum_i^N \sum_{\ell=1}^L || \mathbf{z}_{i, \ell} - f_\ell(\mathbf{z}_{i, \ell-1}; θ) ||^2
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
        (standard parameterisation), `"mupc"` ([μPC](https://arxiv.org/abs/2505.13124)), 
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
    """Computes the free energy of an amortised PC network ([Tscshantz et al., 2023](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011280))

    $$
    \mathcal{F}(\mathbf{z}^*, \hat{\mathbf{z}}; θ) = 1/N \sum_i^N \sum_{\ell=1}^L || \mathbf{z}^*_{i, \ell} - f_\ell(\hat{\mathbf{z}}_{i, \ell-1}; θ) ||^2
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


def _get_param_scalings(
        model: PyTree[Callable], 
        input: ArrayLike, 
        *,
        skip_model: Optional[PyTree[Callable]] = None, 
        param_type: str = "sp"
    ) -> list[float]:
    """Gets layer scalings for a given parameterisation.

    !!! warning

        `param_type = "mupc"` ([μPC](https://arxiv.org/abs/2505.13124)) assumes 
        that one is using [`jpc.make_mlp()`](https://thebuckleylab.github.io/jpc/api/Utils/#jpc.make_mlp) 
        to create the model.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `input`: input to the model.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://arxiv.org/abs/2505.13124)), 
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
