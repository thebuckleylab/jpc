"""Function to solve activity dynamics of predictive coding networks."""

from jaxtyping import PyTree, ArrayLike, Array
from typing import Callable, Optional
from ._grads import _neg_activity_grad
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Euler,
    ConstantStepSize,
    diffeqsolve,
    ODETerm,
    SaveAt
)


def solve_pc_activities(
        network: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: Optional[ArrayLike] = None,
        solver: AbstractSolver = Euler(),
        dt: float | int = 1,
        n_iters: int = 20,
        stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
        record_iters: bool = False
) -> PyTree[Array]:
    """Solves the activity (inference) dynamics of a predictive coding network.

    This is a wrapper around `diffrax.diffeqsolve` to integrate the gradient
    ODE system `_neg_activity_grad` defining the PC activity dynamics

    $$
    \partial \mathbf{z} / \partial t = - \partial \mathcal{F} / \partial \mathbf{z}
    $$

    where $\mathcal{F}$ is the free energy, $\mathbf{z}$ are the activities,
    with $\mathbf{z}_L$ clamped to some target and $\mathbf{z}_0$ optionally
    equal to some prior.

    **Main arguments:**

    - `network`: List of callable layers for the generative model.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `solver`: Diffrax (ODE) solver to be used. Default is Euler.
    - `dt`: Integration step size. Defaults to 1.
    - `n_iters`: Number of integration steps (20 as default).
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `ConstantStepSize`.
    - `record_iters`: If `True`, returns all integration steps. `False` by
        default.

    **Returns:**

    List with solution of the activity dynamics for each layer.

    """
    sol = diffeqsolve(
        terms=ODETerm(_neg_activity_grad),
        solver=solver,
        t0=0,
        t1=n_iters,
        dt0=dt,
        y0=activities,
        args=(network, output, input),
        stepsize_controller=stepsize_controller,
        saveat=SaveAt(t1=True, steps=record_iters)
    )
    return sol.ys
