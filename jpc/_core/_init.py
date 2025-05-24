"""Functions to initialise the layer activities of PC networks."""

from jax import vmap, random
import equinox as eqx
from ._energies import _get_param_scalings
from jaxtyping import PyTree, ArrayLike, Array, PRNGKeyArray, Scalar
from typing import Callable, Optional


@eqx.filter_jit
def init_activities_with_ffwd(
        model: PyTree[Callable],
        input: ArrayLike,
        *,
        skip_model: Optional[PyTree[Callable]] = None,
        n_skip: int = 0,
        param_type: str = "sp"
) -> PyTree[Array]:
    """Initialises the layers' activity with a feedforward pass
    $\{ f_\ell(\mathbf{z}_{\ell-1}) \}_{\ell=1}^L$ where $f_\ell(\cdot)$ is some
    callable layer transformation and $\mathbf{z}_0 = \mathbf{x}$ is the input.

    !!! warning

        `param_type = mupc` ([μPC](https://arxiv.org/abs/2505.13124)) assumes 
        that one is using [`jpc.make_mlp()`](https://thebuckleylab.github.io/jpc/api/Utils/#jpc.make_mlp) 
        to create the model.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `input`: input to the model.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.
    - `n_skip`: Number of layers to skip for the skip model (0 by default).
    - `param_type`: Determines the parameterisation. Options are `sp` (standard
        parameterisation), `mupc` ([μPC](https://arxiv.org/abs/2505.13124)), or 
        `ntp` (neural tangent parameterisation). See [`jpc._get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `sp`.

    **Returns:**

    List with activity values of each layer.

    """
    L = len(model)
    if skip_model is None:
        skip_model = [None] * len(model)
        
    scalings = _get_param_scalings(
        model=model, 
        input=input, 
        skip_model=skip_model, 
        param_type=param_type
    )

    z1 = scalings[0] * vmap(model[0])(input)
    if skip_model[0] is not None:
        z1 += vmap(skip_model[0])(input)

    activities = [z1]
    for l in range(1, L):
        zl = scalings[l] * vmap(model[l])(activities[l - 1])

        if skip_model[l] is not None:
            skip_output = vmap(skip_model[l])(activities[l - n_skip])
            zl += skip_output

        activities.append(zl)
    
    return activities


def init_activities_from_normal(
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        mode: str,
        batch_size: int,
        sigma: Scalar = 0.05
) -> PyTree[Array]:
    """Initialises network activities from a zero-mean Gaussian 
    $z_i \sim \mathcal{N}(0, \sigma^2)$.

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

    # NOTE: this dummy activity for the last layer is added in case one is 
    # interested in inspecting the generator's target prediction during inference.
    activities.append(
        vmap(generator[-1])(activities[-1])
    )
    return activities
