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
        Defaults to 5e-2.

    **Returns:**

    List of randomly initialised activities for each layer.

    """
    start_l = 0 if mode == "unsupervised" else 1
    n_layers = len(layer_sizes) if mode == "unsupervised" else len(layer_sizes)-1
    activities = []
    for l, subkey in zip(
            range(start_l, n_layers+1),
            random.split(key, num=n_layers)
    ):
        activities.append(sigma * random.normal(
            subkey,
            shape=(batch_size, layer_sizes[l])
            )
        )
    return activities


def init_activities_with_amort(
        amortiser: PyTree[Callable],
        generator: PyTree[Callable],
        output: ArrayLike
) -> PyTree[Array]:
    """Initialises layers' activity using an amortised network.

    **Main arguments:**

    - `amortiser`: List of callable layers for network amortising the inference
        of the generative model.
    - `generator`: List of callable layers for the generative model.
    - `output`: Input to the amortiser.

    **Returns:**

    List with amortised initialisation of each layer.

    """
    activities = [vmap(amortiser[0])(output)]
    for l in range(1, len(amortiser)):
        activities.append(vmap(amortiser[l])(activities[l - 1]))

    activities = activities[::-1]
    activities.append(
        vmap(generator[-1])(activities[-1])
    )
    return activities
