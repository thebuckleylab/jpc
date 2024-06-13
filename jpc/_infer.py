"""Function to solve activity dynamics of predictive coding networks."""

from jaxtyping import PyTree, ArrayLike, Array
from typing import Callable, Optional, Union
from ._grads import _neg_activity_grad, _neg_lateral_activity_grad
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Dopri5,
    PIDController,
    diffeqsolve,
    ODETerm,
    SaveAt
)


def solve_pc_activities(
        generator: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: Optional[ArrayLike] = None,
        amortiser: Optional[PyTree[Callable]] = None,
        solver: AbstractSolver = Dopri5(),
        n_iters: int = 300,
        stepsize_controller: AbstractStepSizeController = PIDController(
            rtol=1e-5,
            atol=1e-5
        ),
        dt: Union[float, int] = None,
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

    - `generator`: List of callable layers for the generative model.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `amortiser`: Optional list of callable layers for a network amortising
        the inference of the generative model.
    - `solver`: Diffrax (ODE) solver to be used. Default is Dopri5.
    - `n_iters`: Number of integration steps (300 as default).
    - `stepsize_controller`: diffrax controllers for step size integration.
        Defaults to `PIDController`.
    - `dt`: Integration step size. Defaults to None, since step size is
        automatically determined by the default `PIDController`.
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
        args=(amortiser, generator, output, input),
        stepsize_controller=stepsize_controller,
        saveat=SaveAt(t1=True, steps=record_iters)
    )
    return sol.ys if record_iters else [activity[0] for activity in sol.ys]


def solve_lateral_pc_activities(
        amortiser: PyTree[Callable],
        activities: PyTree[ArrayLike],
        outputs: PyTree[ArrayLike],
        solver: AbstractSolver = Dopri5(),
        n_iters: int = 300,
        stepsize_controller: AbstractStepSizeController = PIDController(
            rtol=1e-5,
            atol=1e-5
        ),
        dt: Union[float, int] = None,
        record_iters: bool = False
) -> PyTree[Array]:
    """Same as `solve_pc_activities` but for a network with lateral connections.

    **Main arguments:**

    - `amortiser`: List of callable layers for an amortised network.
    - `activities`: List of activities for each layer free to vary, one list
        per branch (n=2).
    - `outputs`: List of two inputs to the amortiser, one for each branch.

    **Other arguments:**

    - `solver`: Diffrax (ODE) solver to be used. Default is Dopri5.
    - `n_iters`: Number of integration steps (300 as default).
    - `stepsize_controller`: diffrax controllers for step size integration.
        Defaults to `PIDController`.
    - `dt`: Integration step size. Defaults to None, since step size is
        automatically determined by the default `PIDController`.
    - `record_iters`: If `True`, returns all integration steps. `False` by
        default.

    **Returns:**

    List with solution of the activity dynamics for each branch and layer.

    """
    sol = diffeqsolve(
        terms=ODETerm(_neg_lateral_activity_grad),
        solver=solver,
        t0=0,
        t1=n_iters,
        dt0=dt,
        y0=activities,
        args=(amortiser, outputs),
        stepsize_controller=stepsize_controller,
        saveat=SaveAt(t1=True, steps=record_iters)
    )
    return sol.ys if record_iters else [
        [activity[0] for activity in branch] for branch in sol.ys
    ]
