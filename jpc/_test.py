"""Utility functions to test predictive coding networks."""

import equinox as eqx
from jpc import (
    init_activities_from_normal,
    init_activities_with_ffwd,
    init_activities_with_amort,
    mse_loss,
    cross_entropy_loss,
    compute_accuracy,
    solve_inference
)
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Heun,
    PIDController
)
from jaxtyping import PRNGKeyArray, PyTree, ArrayLike, Array, Scalar
from typing import Callable, Tuple, Optional


@eqx.filter_jit
def test_discriminative_pc(
        model: PyTree[Callable],
        output: ArrayLike,
        input: ArrayLike,
        *,
        skip_model: Optional[PyTree[Callable]] = None,
        n_skip: int = 0,
        loss: str = "mse",
        param_type: str = "sp"
) -> Tuple[Scalar, Scalar]:
    """Computes test metrics for a discriminative predictive coding network.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.
    - `n_skip`: Number of layers to skip for the skip connections (0 by default).
    - `loss`: Loss function to use at the output layer. Options are mean squared 
        error `mse` (default) or cross-entropy `ce`.
    - `param_type`: Determines the parameterisation. Options are `sp` (standard
        parameterisation), `mupc` ([μPC](https://arxiv.org/abs/2505.13124)), or 
        `ntp` (neural tangent parameterisation). See [`_get_param_scalings()`](http://127.0.0.1:8000/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `sp`.

    **Returns:**

    Test loss and accuracy of output predictions.

    """
    preds = init_activities_with_ffwd(
        model=model,
        input=input,
        skip_model=skip_model,
        n_skip=n_skip,
        param_type=param_type
    )[-1]

    if loss == "mse":
        loss = mse_loss(preds, output)
    elif loss == "ce":
        loss = cross_entropy_loss(preds, output)

    acc = compute_accuracy(output, preds)
    return loss, acc


@eqx.filter_jit
def test_generative_pc(
        model: PyTree[Callable],
        output: ArrayLike,
        input: ArrayLike,
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        batch_size: int,
        *,
        skip_model: Optional[PyTree[Callable]] = None,
        n_skip: int = 0,
        loss_id: str = "mse",
        param_type: str = "sp",
        sigma: Scalar = 0.05,
        ode_solver: AbstractSolver = Heun(),
        max_t1: int = 500,
        dt: Scalar | int = None,
        stepsize_controller: AbstractStepSizeController = PIDController(
            rtol=1e-3, atol=1e-3
        ),
        weight_decay: Scalar = 0.,
        spectral_penalty: Scalar = 0.,
        activity_decay: Scalar = 0.
) -> Tuple[Scalar, Array]:
    """Computes test metrics for a generative predictive coding network.

    Gets output predictions (e.g. of an image given a label) with a feedforward
    pass and calculates accuracy of inferred input (e.g. of a label given an
    image).

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `output`: Observation or target of the generative model.
    - `input`: Prior of the generative model.
    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for activity initialisation.

    **Other arguments:**

    - `skip_model`: Optional skip connection model.
    - `n_skip`: Number of layers to skip for the skip connections (0 by default).
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `mse` (default) or cross-entropy `ce`.
    - `param_type`: Determines the parameterisation. Options are `sp` (standard
        parameterisation), `mupc` ([μPC](https://arxiv.org/abs/2505.13124)), or 
        `ntp` (neural tangent parameterisation). See [`_get_param_scalings()`](http://127.0.0.1:8000/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `sp`.
    - `sigma`: Standard deviation for Gaussian to sample activities from.
        Defaults to 5e-2.
    - `ode_solver`: Diffrax ODE solver to be used. Default is Heun, a 2nd order
        explicit Runge--Kutta method.
    - `max_t1`: Maximum end of integration region (500 by default).
    - `dt`: Integration step size. Defaults to None since the default
        `stepsize_controller` will automatically determine it.
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `PIDController`. Note that the relative and absolute
        tolerances of the controller will also determine the steady state to
        terminate the solver.
    - `weight_decay`: Weight decay for the weights (0 by default).
    - `spectral_penalty`: Weight spectral penalty of the form 
        $||\mathbf{I} - \mathbf{W}_\ell^T \mathbf{W}_\ell||^2$ (0 by default).
    - `activity_decay`: Activity decay for the activities (0 by default).

    **Returns:**

    Accuracy and output predictions.

    """
    params = model, skip_model
    activities = init_activities_from_normal(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=sigma
    )
    input_preds = solve_inference(
        params=params,
        activities=activities,
        output=output,
        n_skip=n_skip,
        loss_id=loss_id,
        param_type=param_type,
        solver=ode_solver,
        max_t1=max_t1,
        dt=dt,
        stepsize_controller=stepsize_controller,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay
    )[0][0]
    input_acc = compute_accuracy(input, input_preds)
    output_preds = init_activities_with_ffwd(
        model=model,
        input=input,
        skip_model=skip_model,
        n_skip=n_skip,
        param_type=param_type
    )[-1]
    return input_acc, output_preds


@eqx.filter_jit
def test_hpc(
      generator: PyTree[Callable],
      amortiser: PyTree[Callable],
      output: ArrayLike,
      input: ArrayLike,
      key: PRNGKeyArray,
      layer_sizes: PyTree[int],
      batch_size: int,
      sigma: Scalar = 0.05,
      ode_solver: AbstractSolver = Heun(),
      max_t1: int = 500,
      dt: Scalar | int = None,
      stepsize_controller: AbstractStepSizeController = PIDController(
          rtol=1e-3, atol=1e-3
      )
) -> Tuple[Scalar, Scalar, Scalar, Array]:
    """Computes test metrics for hybrid predictive coding trained in a supervised manner.

    Calculates input accuracy of (i) amortiser, (ii) generator, and (iii)
    hybrid (amortiser + generator). Also returns output predictions (e.g. of
    an image given a label) with a feedforward pass of the generator.

    !!! note

        The input and output of the generator are the output and input of the
        amortiser, respectively.

    **Main arguments:**

    - `generator`: List of callable layers for the generative model.
    - `amortiser`: List of callable layers for model amortising the inference
        of the `generator`.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generator, target for the amortiser.
    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for initialisation of activities.

    **Other arguments:**

    - `sigma`: Standard deviation for Gaussian to sample activities from.
        Defaults to 5e-2.
    - `ode_solver`: Diffrax ODE solver to be used. Default is Heun, a 2nd order
        explicit Runge--Kutta method.
    - `max_t1`: Maximum end of integration region (500 by default).
    - `dt`: Integration step size. Defaults to None since the default
        `stepsize_controller` will automatically determine it.
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `PIDController`. Note that the relative and absolute
        tolerances of the controller will also determine the steady state to
        terminate the solver.

    **Returns:**

    Accuracies of all models and output predictions.

    """
    gen_params = (generator, None)
    amort_activities = init_activities_with_amort(
        amortiser=amortiser,
        generator=generator,
        input=output
    )
    amort_preds = amort_activities[0]
    hpc_preds = solve_inference(
        params=gen_params,
        activities=amort_activities,
        output=output,
        solver=ode_solver,
        max_t1=max_t1,
        dt=dt,
        stepsize_controller=stepsize_controller
    )[0][0]
    activities = init_activities_from_normal(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=sigma
    )
    gen_preds = solve_inference(
        params=gen_params,
        activities=activities,
        output=output,
        solver=ode_solver,
        max_t1=max_t1,
        dt=dt,
        stepsize_controller=stepsize_controller
    )[0][0]
    amort_acc = compute_accuracy(input, amort_preds)
    hpc_acc = compute_accuracy(input, hpc_preds)
    gen_acc = compute_accuracy(input, gen_preds)
    output_preds = init_activities_with_ffwd(
        model=generator,
        input=input,
        skip_model=None
    )[-1]
    return amort_acc, hpc_acc, gen_acc, output_preds
