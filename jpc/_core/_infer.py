"""Function to solve the inference (activity) dynamics of PC networks."""

from jaxtyping import PyTree, ArrayLike, Array
import jax.numpy as jnp
from typing import Tuple, Callable, Optional
from ._grads import neg_activity_grad
from optimistix import rms_norm
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Heun,
    PIDController,
    diffeqsolve,
    ODETerm,
    Event,
    SaveAt
)


def solve_inference(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: Optional[ArrayLike] = None,
        loss_id: str = "MSE",
        solver: AbstractSolver = Heun(),
        max_t1: int = 20,
        dt: float | int = None,
        stepsize_controller: AbstractStepSizeController = PIDController(
            rtol=1e-3, atol=1e-3
        ),
        record_iters: bool = False,
        record_every: int = None
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
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `loss`: Loss function to use at the output layer (mean squared error
        'MSE' vs cross-entropy 'CE').
    - `solver`: Diffrax (ODE) solver to be used. Default is Heun, a 2nd order
        explicit Runge--Kutta method.
    - `max_t1`: Maximum end of integration region (500 by default).
    - `dt`: Integration step size. Defaults to `None` since the default
        `stepsize_controller` will automatically determine it.
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `PIDController`. Note that the relative and absolute
        tolerances of the controller will also determine the steady state to
        terminate the solver.
    - `record_iters`: If `True`, returns all integration steps.
    - `record_every`: int determining the sampling frequency the integration
        steps.

    **Returns:**

    List with solution of the activity dynamics for each layer.

    """
    if record_every is not None:
        ts = jnp.arange(0, max_t1, record_every)
        saveat = SaveAt(t1=True, ts=ts)
    else:
        saveat = SaveAt(t1=True, steps=record_iters)

    solution = diffeqsolve(
        terms=ODETerm(neg_activity_grad),
        solver=solver,
        t0=0,
        t1=max_t1,
        dt0=dt,
        y0=activities,
        args=(params, output, input, loss_id, stepsize_controller),
        stepsize_controller=stepsize_controller,
        event=Event(steady_state_event_with_timeout),
        saveat=saveat
    )
    return solution.ys


def steady_state_event_with_timeout(t, y, args, **kwargs):
    _stepsize_controller = args[-1]
    try:
        _atol = _stepsize_controller.atol
        _rtol = _stepsize_controller.rtol
    except:
        _atol, _rtol = 1e-3, 1e-3
    steady_state_reached = rms_norm(y) < _atol + _rtol * rms_norm(y)
    timeout_reached = jnp.array(t >= 4096, dtype=jnp.bool_)
    return jnp.logical_or(steady_state_reached, timeout_reached)
