"""Energy functions for predictive coding networks."""

from jax.numpy import sum, array
from jax import vmap
from jaxtyping import PyTree, ArrayLike, Scalar, Array
from typing import Callable, Optional


def pc_energy_fn(
        model: PyTree[Callable],
        activities: PyTree[ArrayLike],
        y: ArrayLike,
        x: Optional[ArrayLike] = None,
        record_layers: bool = False
) -> Scalar | Array:
    """Computes the free energy for a feedforward neural network of the form

    $$
    \mathcal{F}(\mathbf{z}; θ) = 1/N \sum_{\ell=1}^L || \mathbf{z}_\ell - f_\ell(\mathbf{z}_{\ell-1}; θ) ||^2
    $$

    given parameters $θ$, free activities $\mathbf{z}$, output
    $\mathbf{z}_L = \mathbf{y}$ and optionally input $\mathbf{z}_0 = \mathbf{x}$.
    The activity of each layer $\mathbf{z}_\ell$ is some function of the previous
    layer, e.g. $\text{ReLU}(W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell)$
    for a fully connected layer with biases and ReLU as activation.

    !!! note

        The input x and output y correspond to the prior and observation of
        the generative model, respectively.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Other arguments:**

    - `record_layers`: If `True`, returns energies for each layer.

    **Returns:**

    The total or layer-wise energy normalised by batch size.

    """
    batch_size = y.shape[0]
    start_activity_l = 1 if x is not None else 2
    n_activity_layers = len(activities) - 1
    n_layers = len(model) - 1

    eL = y - vmap(model[-1])(activities[-2])
    energies = [sum(eL ** 2)]

    for act_l, net_l in zip(
            range(start_activity_l, n_activity_layers),
            range(1, n_layers)
    ):
        err = activities[act_l] - vmap(model[net_l])(activities[act_l-1])
        energies.append(sum(err ** 2))

    e1 = activities[0] - vmap(model[0])(x) if (
            x is not None
    ) else activities[1] - vmap(model[0])(activities[0])
    energies.append(sum(e1 ** 2))

    if record_layers:
        return array(energies) / batch_size
    else:
        return sum(array(energies)) / batch_size
