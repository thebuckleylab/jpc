"""Function to solve the inference (activity) dynamics of PC networks."""

from jaxtyping import PyTree, ArrayLike, Array
from jax.numpy import arange
from typing import Tuple, Callable, Optional
from ._grads import neg_activity_grad
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Heun,
    PIDController,
    diffeqsolve,
    ODETerm,
    Event,
    steady_state_event,
    SaveAt
)


def solve_pc_inference(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        y: ArrayLike,
        x: Optional[ArrayLike] = None,
        loss: str = "MSE",
        solver: AbstractSolver = Heun(),
        max_t1: int = 500,
        dt: float | int = None,
        stepsize_controller: AbstractStepSizeController = PIDController(
            rtol=1e-3, atol=1e-3
        ),
        steady_state_tols: Tuple[Optional[float], Optional[float]] = (None, None),
        record_iters: bool = False,
        record_every: int = 1
) -> PyTree[Array]:
    """Solves the inference (activity) dynamics of a predictive coding network.

    This is a wrapper around `diffrax.diffeqsolve` to integrate the gradient
    ODE system `_neg_activity_grad` defining the PC inference dynamics

    $$
    \partial \mathbf{z} / \partial t = - \partial \mathcal{F} / \partial \mathbf{z}
    $$

    where $\mathcal{F}$ is the free energy, $\mathbf{z}$ are the activities,
    with $\mathbf{z}_L$ clamped to some target and $\mathbf{z}_0$ optionally
    equal to some prior.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Other arguments:**

    - `loss`: Loss function to use at the output layer (mean squared error
        'MSE' vs cross-entropy 'CE').
    - `solver`: Diffrax (ODE) solver to be used. Default is Heun, a 2nd order
        explicit Runge--Kutta method.
    - `max_t1`: Maximum end of integration region (500 by default).
    - `dt`: Integration step size. Defaults to `None` since the default
        `stepsize_controller` will automatically determine it.
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `PIDController`.
    - `steady_state_tols`: Optional relative and absolute tolerances for
        determining a steady state to terminate the solver. Defaults to the
        tolerances of the `stepsize_controller`.
    - `record_iters`: If `True`, returns all integration steps.
    - `record_every`: int determining the sampling frequency the integration
        steps if `record_iters=True`. Defaults to 1.

    **Returns:**

    List with solution of the activity dynamics for each layer.

    """
    if record_every is not None:
        ts = arange(0, max_t1, record_every)
        saveat = SaveAt(t1=True, ts=ts)
    else:
        saveat = SaveAt(t1=True, steps=record_iters)

    sol = diffeqsolve(
        terms=ODETerm(neg_activity_grad),
        solver=solver,
        t0=0,
        t1=max_t1,
        dt0=dt,
        y0=activities,
        args=(params, y, x, loss),
        stepsize_controller=stepsize_controller,
        event=Event(
            steady_state_event(
                rtol=steady_state_tols[0],
                atol=steady_state_tols[1]
            )
        ),
        saveat=saveat
    )
    return sol.ys
