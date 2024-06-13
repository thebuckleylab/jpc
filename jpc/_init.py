"""Functions to initialise layer activities of predictive coding networks."""

from jax import vmap, random
from jaxtyping import PyTree, ArrayLike, Array, PRNGKeyArray, Scalar
from typing import Callable


def init_activities_with_ffwd(
        network: PyTree[Callable],
        input: ArrayLike
) -> PyTree[Array]:
    """Initialises layers' activity with a feedforward pass.

    **Main arguments:**

    - `network`: List of callable network layers.
    - `input`: for the network.

    **Returns:**

    List with feedforward values of each layer.

    """
    activities = [vmap(network[0])(input)]
    for l in range(1, len(network)):
        activities.append(vmap(network[l])(activities[l-1]))

    return activities


def init_activities_from_gaussian(
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        mode: str,
        batch_size: int,
        sigma: Scalar = 0.05
) -> PyTree[Array]:
    """Initialises network activities from a zero-mean Gaussian $\sim \mathcal{N}(0, \sigma^2)$.

    **Main arguments:**

    - `key`: `jax.random.PRNGKey` for sampling.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `mode`: If 'supervised', all hidden layers are initialised. If
        'unsupervised' the input layer is also initialised.
    - `batch_size`: Dimension of data batch.
    - `sigma`: Standard deviation for Gaussian to sample activities from.

    **Returns:**

    List of randomly initialised activities for each layer.

    """
    activities = []
    if mode == "unsupervised":
        key, subkey = random.split(key)
        n_input = network[0].in_features
        activities.append(sigma * random.normal(
            subkey,
            shape=(batch_size, n_input)
            )
        )

    n_layers = len(network)
    for l, subkey in zip(
            range(n_layers),
            random.split(key, num=n_layers)
    ):
        n_hidden_nodes = network[l].out_features
        activities.append(sigma * random.normal(
            subkey,
            shape=(batch_size, n_hidden_nodes)
            )
        )
    return activities
