"""Function to solve the activity (inference) dynamics of PC networks."""

from jaxtyping import PyTree, ArrayLike, Array
from typing import Callable, Optional
from ._grads import neg_activity_grad
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Heun,
    PIDController,
    diffeqsolve,
    ODETerm,
    SaveAt
)


def solve_pc_activities(
        model: PyTree[Callable],
        activities: PyTree[ArrayLike],
        y: ArrayLike,
        x: Optional[ArrayLike] = None,
        solver: AbstractSolver = Heun(),
        t1: int = 20,
        dt: float | int = None,
        stepsize_controller: AbstractStepSizeController = PIDController(
            rtol=1e-3, atol=1e-3
        ),
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

    - `model`: List of callable model (e.g. neural network) layers.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Other arguments:**

    - `solver`: Diffrax (ODE) solver to be used. Default is Heun, a 2nd order
        explicit Runge--Kutta method.
    - `t1`: Maximum end of integration region (20 by default).
    - `dt`: Integration step size. Defaults to None since the default
        `stepsize_controller` will automatically determine it.
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `PIDController`.
    - `record_iters`: If `True`, returns all integration steps.

    **Returns:**

    List with solution of the activity dynamics for each layer.

    """
    sol = diffeqsolve(
        terms=ODETerm(neg_activity_grad),
        solver=solver,
        t0=0,
        t1=t1,
        dt0=dt,
        y0=activities,
        args=(model, y, x),
        stepsize_controller=stepsize_controller,
        saveat=SaveAt(t1=True, steps=record_iters)
    )
    return sol.ys
