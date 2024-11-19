"""Functions to compute gradients of the PC energy."""

from jax import grad, value_and_grad
from jax.tree_util import tree_map
from equinox import filter_grad
from jaxtyping import PyTree, ArrayLike, Array
from typing import Tuple, Callable, Optional
from diffrax import AbstractStepSizeController
from ._energies import pc_energy_fn, hpc_energy_fn


def neg_activity_grad(
        t: float | int,
        activities: PyTree[ArrayLike],
        args: Tuple[
            Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
            ArrayLike,
            Optional[ArrayLike],
            str,
            AbstractStepSizeController
        ],
        energy_fn: Callable = pc_energy_fn
) -> PyTree[Array]:
    """Computes the negative gradient of the energy with respect to the activities $- \partial \mathcal{F} / \partial \mathbf{z}$.

    This defines an ODE system to be integrated by `solve_pc_inference`.

    **Main arguments:**

    - `t`: Time step of the ODE system, used for downstream integration by
        `diffrax.diffeqsolve`.
    - `activities`: List of activities for each layer free to vary.
    - `args`: 5-Tuple with
        (i) Tuple with callable model layers and optional skip connections,
        (ii) network output (observation),
        (iii) network input (prior),
        (iv) Loss specified at the output layer (MSE vs cross-entropy), and
        (v) diffrax controller for step size integration.

    **Other arguments:**

    - `energy_fn`: Free energy to take the gradient of.

    **Returns:**

    List of negative gradients of the energy w.r.t. the activities.

    """
    params, y, x, loss_id, _ = args
    dFdzs = grad(energy_fn, argnums=1)(
        params,
        activities,
        y,
        x,
        loss_id
    )
    return tree_map(lambda dFdz: -dFdz, dFdzs)


def compute_activity_grad(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        y: ArrayLike,
        x: Optional[ArrayLike],
        loss_id: str = "MSE",
        energy_fn: Callable = pc_energy_fn
) -> PyTree[Array]:
    """Computes the gradient of the energy with respect to the activities $\partial \mathcal{F} / \partial \mathbf{z}$.

    !!! note

        This function differs from `neg_activity_grad`, which computes the
        negative gradients, and is called in `update_activities` for use of
        any Optax optimiser.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Other arguments:**

    - `loss_id`: Loss function for the output layer (mean squared error 'MSE'
        vs cross-entropy 'CE').
    - `energy_fn`: Free energy to take the gradient of.

    **Returns:**

    List of negative gradients of the energy w.r.t. the activities.

    """
    energy, dFdzs = value_and_grad(energy_fn, argnums=1)(
        params,
        activities,
        y,
        x,
        loss_id
    )
    return energy, dFdzs


def compute_pc_param_grads(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        y: ArrayLike,
        x: Optional[ArrayLike] = None,
        loss_id: str = "MSE"
) -> Tuple[PyTree[Array], PyTree[Array]]:
    """Computes the gradient of the PC energy with respect to model parameters $\partial \mathcal{F} / \partial θ$.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Other arguments:**

    - `loss_id`: Loss function for the output layer (mean squared error 'MSE'
        vs cross-entropy 'CE').

    **Returns:**

    List of parameter gradients for each network layer.

    """
    return filter_grad(pc_energy_fn)(
        params,
        activities,
        y,
        x,
        loss_id
    )


def compute_hpc_param_grads(
        model: PyTree[Callable],
        equilib_activities: PyTree[ArrayLike],
        amort_activities: PyTree[ArrayLike],
        x: ArrayLike,
        y: Optional[ArrayLike] = None
) -> PyTree[Array]:
    """Computes the gradient of the hybrid energy with respect to an amortiser's parameters $\partial \mathcal{F} / \partial θ$.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `equilib_activities`: List of equilibrated activities reached by the
        generator and target for the amortiser.
    - `amort_activities`: List of amortiser's feedforward guesses
        (initialisation) for the network activities.
    - `x`: Input to the amortiser.
    - `y`: Optional target of the amortiser (for supervised training).

    !!! note

        The input $x$ and output $y$ are reversed compared to `compute_pc_param_grads`
        ($x$ is the generator's target and $y$ is its optional input or prior).
        Just think of $x$ and $y$ as the actual input and output of the
        amortiser, respectively.

    **Returns:**

    List of parameter gradients for each network layer.

    """
    return filter_grad(hpc_energy_fn)(
        model,
        equilib_activities,
        amort_activities,
        x,
        y
    )
