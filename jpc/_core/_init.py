"""Functions to initialise layer activities of predictive coding networks."""

from jax import vmap, random
from jaxtyping import PyTree, ArrayLike, Array, PRNGKeyArray, Scalar
from typing import Callable, Optional


def init_activities_with_ffwd(
        model: PyTree[Callable],
        input: ArrayLike,
        skip_model: Optional[PyTree[Callable]] = None
) -> PyTree[Array]:
    """Initialises layers' activity with a feedforward pass
    $\{ f_\ell(\mathbf{z}_{\ell-1}) \}_{\ell=1}^L$ where $\mathbf{z}_0 = \mathbf{x}$ is
    the input.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `input`: input to the model.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.

    **Returns:**

    List with activity values of each layer.

    """
    if skip_model is None:
        skip_model = [None] * len(model)

    first_layer_output = vmap(model[0])(input)
    if skip_model[0] is not None:
        first_layer_output += vmap(skip_model[0])(input)

    activities = [first_layer_output]
    for l in range(1, len(model)):
        layer_output = vmap(model[l])(activities[l - 1])

        if skip_model[l] is not None:
            skip_output = vmap(skip_model[l])(activities[l - 1])
            layer_output += skip_output

        activities.append(layer_output)

    return activities


def init_activities_from_normal(
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        mode: str,
        batch_size: int,
        sigma: Scalar = 0.05
) -> PyTree[Array]:
    """Initialises network activities from a zero-mean Gaussian $\sim \mathcal{N}(0, \sigma^2)$.

    **Main arguments:**

    - `key`: `jax.random.PRNGKey` for sampling.
    - `layer_sizes`: List with dimension of all layers (input, hidden and
        output).
    - `mode`: If `supervised`, all hidden layers are initialised. If
        `unsupervised` the input layer $\mathbf{z}_0$ is also initialised.
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
        input: ArrayLike
) -> PyTree[Array]:
    """Initialises layers' activity with an amortised network
    $\{ f_{L-\ell+1}(\mathbf{z}_{L-\ell}) \}_{\ell=1}^L$ where $\mathbf{z}_0 = \mathbf{y}$ is
    the input or generator's target.

    !!! note

        The output order is reversed for downstream use by the generator.

    **Main arguments:**

    - `amortiser`: List of callable layers for model amortising the inference
        of the `generator`.
    - `generator`: List of callable layers for the generative model.
    - `input`: Input to the amortiser.

    **Returns:**

    List with amortised initialisation of each layer.

    """
    activities = [vmap(amortiser[0])(input)]
    for l in range(1, len(amortiser)):
        activities.append(vmap(amortiser[l])(activities[l - 1]))

    activities = activities[::-1]
    # add generator's target prediction
    activities.append(
        vmap(generator[-1])(activities[-1])
    )
    return activities
