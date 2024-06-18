"""High-level functions to test predictive coding networks."""

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
    Dopri5,
    PIDController
)
from jaxtyping import PRNGKeyArray, PyTree, ArrayLike, Array, Scalar
from typing import Callable, Union, Tuple


@eqx.filter_jit
def test_generative_pc(
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        batch_size: int,
        network: PyTree[Callable],
        output: ArrayLike,
        input: ArrayLike,
        sigma: Scalar = 0.05,
        solver: AbstractSolver = Dopri5(),
        n_iters: int = 300,
        stepsize_controller: AbstractStepSizeController = PIDController(
            rtol=1e-3,
            atol=1e-3
        ),
        dt: Union[float, int] = None
) -> Tuple[Scalar, Array]:
    """Computes test metrics for a generative predictive coding network.

    Gets output predictions (e.g. of an image given a label) with a feedforward
    pass and calculates accuracy of inferred input (e.g. of a label given an
    image).

    **Main arguments:**

    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for random initialisation of
        activities.
    - `network`: List of callable network layers.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `sigma`: Standard deviation for Gaussian to sample activities from.
        Defaults to 5e-2.
    - `solver`: Diffrax (ODE) solver to be used. Default is Dopri5.
    - `n_iters`: Number of integration steps for inference (300 as default).
    - `stepsize_controller`: diffrax controllers for inference integration.
        Defaults to `PIDController`.
    - `dt`: Integration step size. Defaults to None, since step size is
        automatically determined by the default `PIDController`.

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
    equilib_activities = solve_pc_activities(
        network=network,
        activities=activities,
        output=output,
        solver=solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt
    )
    input_acc = compute_accuracy(input, equilib_activities[0])
    output_preds = init_activities_with_ffwd(network=network, input=input)[-1]
    return input_acc, output_preds


@eqx.filter_jit
def test_hpc(
      key: PRNGKeyArray,
      layer_sizes: PyTree[int],
      batch_size: int,
      generator: PyTree[Callable],
      amortiser: PyTree[Callable],
      output: ArrayLike,
      input: ArrayLike,
      sigma: Scalar = 0.05,
      solver: AbstractSolver = Dopri5(),
      n_iters: int = 300,
      stepsize_controller: AbstractStepSizeController = PIDController(
          rtol=1e-3,
          atol=1e-3
      ),
      dt: Union[float, int] = None
):
    """Computes test metrics for hybrid predictive coding.

    Calculates input accuracy of (i) amortiser, (ii) generative, and (ii)
    hybrid (amortiser + generative). Also returns output predictions (e.g. of
    an image given a label) with a feedforward pass of the generator.

    **Main arguments:**

    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for random initialisation of
        activities.
    - `generator`: List of callable layers for the generative network..
    - `amortiser`: List of callable layers for network amortising the inference
        of the generative model.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `sigma`: Standard deviation for Gaussian to sample activities from.
        Defaults to 5e-2.
    - `solver`: Diffrax (ODE) solver to be used. Default is Dopri5.
    - `n_iters`: Number of integration steps for inference (300 as default).
    - `stepsize_controller`: diffrax controllers for inference integration.
        Defaults to `PIDController`.
    - `dt`: Integration step size. Defaults to None, since step size is
        automatically determined by the default `PIDController`.

    **Returns:**

    Accuracies of all models and output predictions.

    """
    amort_activities = init_activities_with_amort(
        amortiser=amortiser,
        generator=generator,
        output=output
    )
    amort_preds = amort_activities[0]
    hpc_preds = solve_pc_activities(
        network=generator,
        activities=amort_activities[1:],
        output=output,
        solver=solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt
    )[0]
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
        output=output,
        solver=solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt
    )[0]
    amort_acc = compute_accuracy(input, amort_preds)
    hpc_acc = compute_accuracy(input, hpc_preds)
    gen_acc = compute_accuracy(input, gen_preds)
    output_preds = init_activities_with_ffwd(network=generator, input=input)[-1]
    return amort_acc, hpc_acc, gen_acc, output_preds
