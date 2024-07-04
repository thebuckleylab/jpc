"""Utility functions to test predictive coding networks."""

import equinox as eqx
from jpc import (
    init_activities_from_gaussian,
    init_activities_with_ffwd,
    init_activities_with_amort,
    solve_pc_activities
)
from ._utils import compute_accuracy
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Heun,
    PIDController
)
from jaxtyping import PRNGKeyArray, PyTree, ArrayLike, Array, Scalar
from typing import Callable, Tuple


@eqx.filter_jit
def test_discriminative_pc(
        model: PyTree[Callable],
        y: ArrayLike,
        x: ArrayLike,
) -> Scalar:
    """Computes prediction accuracy of a discriminative predictive coding network.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Returns:**

    Accuracy of output predictions.

    """
    preds = init_activities_with_ffwd(model=model, x=x)[-1]
    return compute_accuracy(y, preds)


@eqx.filter_jit
def test_generative_pc(
        model: PyTree[Callable],
        y: ArrayLike,
        x: ArrayLike,
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        batch_size: int,
        sigma: Scalar = 0.05,
        ode_solver: AbstractSolver = Heun(),
        t1: int = 500,
        dt: float | int = None,
        stepsize_controller: AbstractStepSizeController = PIDController(
            rtol=1e-3, atol=1e-3
        ),
) -> Tuple[Scalar, Array]:
    """Computes test metrics for a generative predictive coding network.

    Gets output predictions (e.g. of an image given a label) with a feedforward
    pass and calculates accuracy of inferred input (e.g. of a label given an
    image).

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.
    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for activity initialisation.

    **Other arguments:**

    - `sigma`: Standard deviation for Gaussian to sample activities from.
        Defaults to 5e-2.
    - `ode_solver`: Diffrax ODE solver to be used. Default is Heun, a 2nd order
        explicit Runge--Kutta method.
    - `t1`: Maximum end of integration region. Default is 500, the (default)
        adaptive solver should converge much faster (depending on the problem).
    - `dt`: Integration step size. Defaults to None since the default
        `stepsize_controller` will automatically determine it.
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `PIDController`.

    **Returns:**

    Tuple with accuracy and output predictions.

    """
    activities = init_activities_from_gaussian(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=sigma
    )
    input_preds = solve_pc_activities(
        model=model,
        activities=activities,
        y=y,
        solver=ode_solver,
        t1=t1,
        dt=dt,
        stepsize_controller=stepsize_controller
    )[0][0]
    input_acc = compute_accuracy(x, input_preds)
    output_preds = init_activities_with_ffwd(model=model, x=x)[-1]
    return input_acc, output_preds


@eqx.filter_jit
def test_hpc(
      generator: PyTree[Callable],
      amortiser: PyTree[Callable],
      y: ArrayLike,
      x: ArrayLike,
      key: PRNGKeyArray,
      layer_sizes: PyTree[int],
      batch_size: int,
      sigma: Scalar = 0.05,
      ode_solver: AbstractSolver = Heun(),
      t1: int = 500,
      dt: float | int = None,
      stepsize_controller: AbstractStepSizeController = PIDController(
          rtol=1e-3, atol=1e-3
      ),
) -> Tuple[Scalar, Scalar, Scalar, Array]:
    """Computes test metrics for hybrid predictive coding.

    Calculates input accuracy of (i) amortiser, (ii) generator, and (iii)
    hybrid (amortiser + generator). Also returns output predictions (e.g. of
    an image given a label) with a feedforward pass of the generator.

    **Main arguments:**

    - `generator`: List of callable layers for the generative model.
    - `amortiser`: List of callable layers for model amortising the inference
        of the `generator`.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.
    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for initialisation of activities.

    **Other arguments:**

    - `sigma`: Standard deviation for Gaussian to sample activities from.
        Defaults to 5e-2.
    - `ode_solver`: Diffrax ODE solver to be used. Default is Heun, a 2nd order
        explicit Runge--Kutta method.
    - `t1`: Maximum end of integration region. Default is 500, the (default)
        adaptive solver should converge much faster (depending on the problem).
    - `dt`: Integration step size. Defaults to None since the default
        `stepsize_controller` will automatically determine it.
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `PIDController`.

    **Returns:**

    Accuracies of all models and output predictions.

    """
    amort_activities = init_activities_with_amort(
        amortiser=amortiser,
        generator=generator,
        y=y
    )
    amort_preds = amort_activities[0]
    hpc_preds = solve_pc_activities(
        model=generator,
        activities=amort_activities,
        y=y,
        solver=ode_solver,
        t1=t1,
        dt=dt,
        stepsize_controller=stepsize_controller
    )[0][0]
    activities = init_activities_from_gaussian(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=sigma
    )
    gen_preds = solve_pc_activities(
        model=generator,
        activities=activities,
        y=y,
        solver=ode_solver,
        t1=t1,
        dt=dt,
        stepsize_controller=stepsize_controller
    )[0][0]
    amort_acc = compute_accuracy(x, amort_preds)
    hpc_acc = compute_accuracy(x, hpc_preds)
    gen_acc = compute_accuracy(x, gen_preds)
    output_preds = init_activities_with_ffwd(model=generator, x=x)[-1]
    return amort_acc, hpc_acc, gen_acc, output_preds
