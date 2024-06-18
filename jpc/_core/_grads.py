"""Functions to compute gradients of the free energy."""

from jax import grad
from equinox import filter_grad
from jaxtyping import PyTree, ArrayLike, Array
from typing import Union, Tuple, Callable, Optional
from ._energies import pc_energy_fn, hpc_energy_fn


def _neg_activity_grad(
        t: Union[float, int],
        activities: PyTree[ArrayLike],
        args: Tuple[Optional[PyTree[Callable]], ArrayLike, ArrayLike]
) -> PyTree[Array]:
    """Computes the negative gradient of the energy with respect to the activities.

    $$
    - \partial \mathcal{F} / \partial \mathbf{z}
    $$

    This defines an ODE system to be integrated by `solve_pc_activities`.

    **Main arguments:**

    - `t`: Time step of the ODE system, used for downstream integration by
        `diffrax.diffeqsolve`.
    - `activities`: List of activities for each layer free to vary.
    - `args`: 4-Tuple with
        (i) list of callable layers of the generative model,
        (ii) optional list of callable layers of a network to amortise inference,
        (iii) network output (observation), and
        (iv) network input (prior).

    **Returns:**

    List of negative gradients of the energy w.r.t the activities.

    """
    generator, output, input = args
    dFdzs = grad(pc_energy_fn, argnums=1)(
        generator,
        activities,
        output,
        input
    )
    return [-dFdz for dFdz in dFdzs]


def compute_pc_param_grads(
        network: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: Optional[ArrayLike] = None
) -> PyTree[Array]:
    """Computes the gradient of the energy with respect to network parameters $\partial \mathcal{F} / \partial θ$.

    **Main arguments:**

    - `network`: List of callable network layers.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Returns:**

    List of parameter gradients for each network layer.

    """
    return filter_grad(pc_energy_fn)(
        network,
        activities,
        output,
        input
    )


def compute_gen_param_grads(
        generator: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: ArrayLike
) -> PyTree[Array]:
    """Computes the gradient of the energy w.r.t the parameters of a generative model $\partial \mathcal{F} / \partial θ$.

    !!! note

        This has the same functionality as `compute_pc_param_grads` but can be
        used together with `compute_amort_param_grads` for a more user-friendly
        API when training hybrid predictive coding networks.

    **Main arguments:**

    - `generator`: List of callable layers for the generative model.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Prior of the generative model.

    **Returns:**

    List of parameter gradients for each layer of the generative network.

    """
    return filter_grad(pc_energy_fn)(
        generator,
        activities,
        output,
        input
    )


def compute_amort_param_grads(
        amortiser: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: ArrayLike
) -> PyTree[Array]:
    """Computes the gradient of the energy w.r.t the parameters of an amortised model $\partial \mathcal{F} / \partial \phi$.

    **Main arguments:**

    - `amortiser`: List of callable layers for a network amortising the
        inference of the generative model.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Returns:**

    List of parameter gradients for each layer of the amortiser.

    """
    return filter_grad(hpc_energy_fn)(
        amortiser,
        activities,
        output,
        input
    )