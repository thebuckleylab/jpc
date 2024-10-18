"""Energy functions for predictive coding networks."""

from jax import vmap
from jax.numpy import sum, array, log
from jax.nn import softmax
from jaxtyping import PyTree, ArrayLike, Scalar, Array
from typing import Tuple, Callable, Optional


def pc_energy_fn(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        y: ArrayLike,
        x: Optional[ArrayLike] = None,
        loss: str = "MSE",
        record_layers: bool = False
) -> Scalar | Array:
    """Computes the free energy for a feedforward neural network of the form

    $$
    \mathcal{F}(\mathbf{z}; θ) = 1/N \sum_i^N \sum_{\ell=1}^L || \mathbf{z}_{i, \ell} - f_\ell(\mathbf{z}_{i, \ell-1}; θ) ||^2
    $$

    given parameters $θ$, free activities $\mathbf{z}$, output
    $\mathbf{z}_L = \mathbf{y}$ and optional input $\mathbf{z}_0 = \mathbf{x}$
    for supervised training. The activity of each layer $\mathbf{z}_\ell$ is
    some function of the previous layer, e.g.
    ReLU$(W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell)$ for a fully connected
    layer with biases and ReLU as activation.

    !!! note

        The input $x$ and output $y$ correspond to the prior and observation of
        the generative model, respectively.

    **Main arguments:**

    - `params`: Tuple with callable model (e.g. neural network) layers and
        optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model (for supervised training).

    **Other arguments:**

    - `loss`: Loss function to use at the output layer (mean squared error
        'MSE' vs cross-entropy 'CE').
    - `record_layers`: If `True`, returns energies for each layer.

    **Returns:**

    The total or layer-wise energy normalised by the batch size.

    """
    model, skip_model = params
    batch_size = y.shape[0]
    start_activity_l = 1 if x is not None else 2
    n_activity_layers = len(activities) - 1
    n_layers = len(model) - 1
    if skip_model is None:
        skip_model = [None] * len(model)

    if loss == "MSE":
        eL = y - vmap(model[-1])(activities[-2])
    elif loss == "CE":
        logits = vmap(model[-1])(activities[-2])
        probs = softmax(logits, axis=-1)
        eL = - sum(y * log(probs + 1e-10), axis=-1)

    energies = [sum(eL ** 2)]

    for act_l, net_l in zip(
            range(start_activity_l, n_activity_layers),
            range(1, n_layers)
    ):
        err = activities[act_l] - vmap(model[net_l])(activities[act_l - 1])
        if skip_model[net_l] is not None:
            err -= vmap(skip_model[net_l])(activities[act_l - 1])

        energies.append(sum(err ** 2))

    if x is not None:
        main_pred = vmap(model[0])(x)
        skip_pred = vmap(skip_model[0])(x) if skip_model[0] is not None else 0
        e1 = activities[0] - main_pred - skip_pred
    else:
        main_pred = vmap(model[0])(activities[0])
        skip_pred = (vmap(skip_model[0])(activities[0])
                     if skip_model[0] is not None else 0)
        e1 = activities[1] - main_pred - skip_pred

    energies.append(sum(e1 ** 2))

    if record_layers:
        return array(energies) / batch_size
    else:
        return sum(array(energies)) / batch_size


def hpc_energy_fn(
        model: PyTree[Callable],
        equilib_activities: PyTree[ArrayLike],
        amort_activities: PyTree[ArrayLike],
        x: ArrayLike,
        y: Optional[ArrayLike] = None,
        record_layers: bool = False
) -> Scalar | Array:
    """Computes the free energy of an amortised PC network

    $$
    \mathcal{F}(\mathbf{z}^*, \hat{\mathbf{z}}; θ) = 1/N \sum_i^N \sum_{\ell=1}^L || \mathbf{z}^*_{i, \ell} - f_\ell(\hat{\mathbf{z}}_{i, \ell-1}; θ) ||^2
    $$

    given the equilibrated activities of the generator $\mathbf{z}^*$ (target
    for the amortiser), the feedforward guesses of the amortiser
    $\hat{\mathbf{z}}$, the amortiser's parameters $θ$, input
    $\mathbf{z}_0 = \mathbf{x}$, and optional output
    $\mathbf{z}_L = \mathbf{y}$ for supervised training.

    !!! note

        The input $x$ and output $y$ are reversed compared to `pc_energy_fn`
        ($x$ is the generator's target and $y$ is its optional input or prior).
        Just think of $x$ and $y$ as the actual input and output of the
        amortiser, respectively.

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
