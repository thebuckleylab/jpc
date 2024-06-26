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
    Euler,
    ConstantStepSize
)
from jaxtyping import PRNGKeyArray, PyTree, ArrayLike, Array, Scalar
from typing import Callable, Tuple


@eqx.filter_jit
def test_discriminative_pc(
        network: PyTree[Callable],
        y: ArrayLike,
        x: ArrayLike,
) -> Scalar:
    """Computes prediction accuracy of a discriminative predictive coding network.

    **Main arguments:**

    - `network`: List of callable network layers.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Returns:**

    Accuracy of output predictions.

    """
    preds = init_activities_with_ffwd(network=network, x=x)[-1]
    return compute_accuracy(y, preds)


@eqx.filter_jit
def test_generative_pc(
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        batch_size: int,
        network: PyTree[Callable],
        y: ArrayLike,
        x: ArrayLike,
        sigma: Scalar = 0.05,
        solver: AbstractSolver = Euler(),
        dt: float | int = 1,
        n_iters: int = 20,
        stepsize_controller: AbstractStepSizeController = ConstantStepSize()
) -> Tuple[Scalar, Array]:
    """Computes test metrics for a generative predictive coding network.

    Gets output predictions (e.g. of an image given a label) with a feedforward
    pass and calculates accuracy of inferred input (e.g. of a label given an
    image).

    **Main arguments:**

    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for activity initialisation.
    - `network`: List of callable network layers.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Other arguments:**

    - `sigma`: Standard deviation for Gaussian to sample activities from.
        Defaults to 5e-2.
    - `solver`: Diffrax (ODE) solver to be used. Default is Euler.
    - `dt`: Integration step size. Defaults to 1.
    - `n_iters`: Number of integration steps (20 as default).
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `ConstantStepSize`.

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
        network=network,
        activities=activities,
        y=y,
        solver=solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt
    )[0][0]
    input_acc = compute_accuracy(x, input_preds)
    output_preds = init_activities_with_ffwd(network=network, x=x)[-1]
    return input_acc, output_preds


@eqx.filter_jit
def test_hpc(
      key: PRNGKeyArray,
      layer_sizes: PyTree[int],
      batch_size: int,
      generator: PyTree[Callable],
      amortiser: PyTree[Callable],
      y: ArrayLike,
      x: ArrayLike,
      sigma: Scalar = 0.05,
      solver: AbstractSolver = Euler(),
      dt: float | int = 1,
      n_iters: int = 20,
      stepsize_controller: AbstractStepSizeController = ConstantStepSize()
) -> Tuple[Scalar, Scalar, Scalar, Array]:
    """Computes test metrics for hybrid predictive coding.

    Calculates input accuracy of (i) amortiser, (ii) generative, and (ii)
    hybrid (amortiser + generative). Also returns output predictions (e.g. of
    an image given a label) with a feedforward pass of the generator.

    **Main arguments:**

    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for initialisation of activities.
    - `generator`: List of callable layers for the generative network.
    - `amortiser`: List of callable layers for network amortising the inference
        of the generative model.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Other arguments:**

    - `sigma`: Standard deviation for Gaussian to sample activities from.
        Defaults to 5e-2.
    - `solver`: Diffrax (ODE) solver to be used. Default is Euler.
    - `dt`: Integration step size. Defaults to 1.
    - `n_iters`: Number of integration steps (20 as default).
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `ConstantStepSize`.

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
        network=generator,
        activities=amort_activities,
        y=y,
        solver=solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt
    )[0][0]
    activities = init_activities_from_gaussian(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=sigma
    )
    gen_preds = solve_pc_activities(
        network=generator,
        activities=activities,
        y=y,
        solver=solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt
    )[0][0]
    amort_acc = compute_accuracy(x, amort_preds)
    hpc_acc = compute_accuracy(x, hpc_preds)
    gen_acc = compute_accuracy(x, gen_preds)
    output_preds = init_activities_with_ffwd(network=generator, x=x)[-1]
    return amort_acc, hpc_acc, gen_acc, output_preds
