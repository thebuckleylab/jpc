"""Functions to compute gradients of the free energy."""

from jax import grad
from jax.tree_util import tree_map
from equinox import filter_grad
from jaxtyping import PyTree, ArrayLike, Array
from typing import Tuple, Callable, Optional
from ._energies import pc_energy_fn


def _neg_activity_grad(
        t: float | int,
        activities: PyTree[ArrayLike],
        args: Tuple[Optional[PyTree[Callable]], ArrayLike, ArrayLike],
        energy_fn: Callable = pc_energy_fn,
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
    - `args`: 3-Tuple with
        (i) list of callable layers of the generative model,
        (ii) network output (observation), and
        (iii) network input (prior).
    - `pc_energy_fn`: Free energy to take the gradient of.

    **Returns:**

    List of negative gradients of the energy w.r.t the activities.

    """
    generator, output, input = args
    dFdzs = grad(energy_fn, argnums=1)(
        generator,
        activities,
        output,
        input
    )
    return tree_map(lambda dFdz: -dFdz, dFdzs)


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
