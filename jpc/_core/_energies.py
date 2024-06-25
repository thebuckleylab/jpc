"""Energy functions for predictive coding networks."""

from jax.numpy import sum, array
from jax import vmap
from jax.tree_util import tree_map
from jaxtyping import PyTree, ArrayLike, Scalar
from typing import Callable, Optional, Union


def pc_energy_fn(
        network: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: Optional[ArrayLike] = None,
        record_layers: bool = False
) -> Union[Scalar, PyTree[Scalar]]:
    """Computes the free energy for a feedforward neural network of the form

    $$
    \mathcal{F}(\mathbf{z}; θ) = \sum_\ell^L || \mathbf{z}_\ell - f_\ell(\mathbf{z}_{\ell-1}; θ) ||^2
    $$

    given parameters $θ$, free activities $\mathbf{z}$, output
    $\mathbf{z}_L = \mathbf{y}$ and optionally input $\mathbf{z}_0 = \mathbf{x}$.
    The activity of each layer $\mathbf{z}_\ell$ is some function of the previous
    layer, e.g. $f_\ell(W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell)$
    for a fully connected layer.

    !!! note

        The input and output correspond to the prior and observation of
        the generative model, respectively.

    **Main arguments:**

    - `network`: List of callable network layers.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `record_layers`: If `True`, returns energies for each layer. Defaults to
        `False`.

    **Returns:**

    The total or layer-wise energy normalised by batch size.

    """
    batch_size = output.shape[0]
    start_activity_l = 1 if input is not None else 2
    n_activity_layers = len(activities) - 1
    n_layers = len(network) - 1

    eL = output - vmap(network[-1])(activities[-2])
    energies = [sum(eL ** 2)]

    for act_l, net_l in zip(
            range(start_activity_l, n_activity_layers),
            range(1, n_layers)
    ):
        err = activities[act_l] - vmap(network[net_l])(activities[act_l-1])
        energies.append(sum(err ** 2))

    e1 = activities[0] - vmap(network[0])(input) if (
            input is not None
    ) else activities[1] - vmap(network[0])(activities[0])
    energies.append(sum(e1 ** 2))

    if record_layers:
        return tree_map(lambda energy: energy / batch_size, energies)
    else:
        return sum(array(energies)) / batch_size
