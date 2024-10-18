"""High-level API to train neural networks with predictive coding."""

import equinox as eqx
from jax.tree_util import tree_map
from jax.numpy import mean, array
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Heun,
    PIDController
)
from jpc import (
    init_activities_with_ffwd,
    init_activities_from_normal,
    init_activities_with_amort,
    mse_loss,
    cross_entropy_loss,
    solve_pc_inference,
    get_t_max,
    compute_activity_norms,
    pc_energy_fn,
    compute_infer_energies,
    compute_pc_param_grads,
    compute_grad_norms,
    compute_accuracy,
    hpc_energy_fn,
    compute_hpc_param_grads
)
from optax import GradientTransformation, GradientTransformationExtraArgs, OptState
from jaxtyping import PyTree, ArrayLike, Scalar, PRNGKeyArray
from typing import Callable, Optional, Tuple, Dict


@eqx.filter_jit
def make_pc_step(
      model: PyTree[Callable],
      optim: GradientTransformation | GradientTransformationExtraArgs,
      opt_state: OptState,
      output: ArrayLike,
      input: Optional[ArrayLike] = None,
      ode_solver: AbstractSolver = Heun(),
      max_t1: int = 500,
      dt: float | int = None,
      stepsize_controller: AbstractStepSizeController = PIDController(
          rtol=1e-3, atol=1e-3
      ),
      steady_state_tols: Optional[Tuple[float]] = (None, None),
      skip_model: Optional[PyTree[Callable]] = None,
      key: Optional[PRNGKeyArray] = None,
      layer_sizes: Optional[PyTree[int]] = None,
      batch_size: Optional[int] = None,
      sigma: Scalar = 0.05,
      loss_id: str = "MSE",
      record_activities: bool = False,
      record_energies: bool = False,
      record_every: int = None,
      activity_norms: bool = False,
      grad_norms: bool = False,
      calculate_accuracy: bool = False
) -> Dict:
    """Updates network parameters with predictive coding.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `optim`: Optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of Optax optimiser.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    !!! note

        `key`, `layer_sizes` and `batch_size` must be passed if `input` is
        `None`, since unsupervised training will be assumed and activities need
        to be initialised randomly.

    **Other arguments:**

    - `ode_solver`: Diffrax ODE solver to be used. Default is Heun, a 2nd order
        explicit Runge--Kutta method.
    - `max_t1`: Maximum end of integration region (500 by default).
    - `dt`: Integration step size. Defaults to None since the default
        `stepsize_controller` will automatically determine it.
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `PIDController`.
    - `steady_state_tols`: Optional relative and absolute tolerances for
        determining a steady state to terminate the inference solver. Defaults
        to the tolerances of the `stepsize_controller`.
    - `skip_model`: Optional list of callable skip connection functions.
    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for activity initialisation.
    - `sigma`: Standard deviation for Gaussian to sample activities from for
        random initialisation. Defaults to 5e-2.
    - `record_activities`: If `True`, returns activities at every inference
        iteration.
    - `record_energies`: If `True`, returns layer-wise energies at every
        inference iteration.
    - `record_every`: int determining the sampling frequency the integration
        steps if `record_activities` or `record_energies` is True`. Defaults to
        1.
    - `activity_norms`: If `True`, computes norm of the activities.
    - `grad_norms`: If `True`, computes norm of parameter gradients.
    - `calculate_accuracy`: If `True`, computes the training accuracy.

    **Returns:**

    Dict including model with updated parameters, optimiser, updated optimiser
    state, loss, energies, equilibrated activities, and optionally other
    metrics.

    **Raises:**

    - `ValueError` for inconsistent inputs and invalid losses.

    """
    if input is None and any(arg is None for arg in (key, layer_sizes, batch_size)):
        raise ValueError("""
            If there is no input (i.e. `x` is None), then unsupervised training 
            is assumed, and `key`, `layer_sizes` and `batch_size` must be 
            passed for random initialisation of activities.
        """)

    if record_energies:
        record_activities = True

    activities = init_activities_from_normal(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=sigma
    ) if input is None else init_activities_with_ffwd(
        params=(model, skip_model),
        input=input
    )

    if loss_id == "MSE":
        loss = mse_loss(activities[-1], output) if input is not None else None
    elif loss_id == "CE":
        loss = cross_entropy_loss(activities[-1], output) if input is not None else None
    else:
        raise ValueError("'MSE' and 'CE' are the only valid losses.")

    params = (model, skip_model)
    equilib_activities = solve_pc_inference(
        params=params,
        activities=activities,
        y=output,
        x=input,
        loss=loss_id,
        solver=ode_solver,
        max_t1=max_t1,
        dt=dt,
        stepsize_controller=stepsize_controller,
        steady_state_tols=steady_state_tols,
        record_iters=record_activities,
        record_every=record_every
    )
    activity_norms = (compute_activity_norms(equilib_activities)
                      if activity_norms else None)
    t_max = get_t_max(equilib_activities) if record_activities else None
    energies = compute_infer_energies(
        params=params,
        activities_iters=equilib_activities,
        t_max=t_max,
        y=output,
        x=input,
        loss=loss_id
    ) if record_energies else pc_energy_fn(
        params=params,
        activities=tree_map(
            lambda act: act[t_max if record_activities else array(0)],
            equilib_activities
        ),
        y=output,
        x=input,
        loss=loss_id,
        record_layers=True
    )

    param_grads = compute_pc_param_grads(
        params=params,
        activities=tree_map(
            lambda act: act[t_max if record_activities else array(0)],
            equilib_activities
        ),
        y=output,
        x=input,
        loss=loss_id
    )
    grad_norms = compute_grad_norms(param_grads) if grad_norms else (None, None)
    updates, opt_state = optim.update(
        updates=param_grads,
        state=opt_state,
        params=params
    )
    params = eqx.apply_updates(model=params, updates=updates)

    acc = compute_accuracy(
        output,
        init_activities_with_ffwd(params=params, input=input)[-1]
    ) if calculate_accuracy else None

    return {
        "model": params[0],
        "skip_model": params[1],
        "optim": optim,
        "opt_state": opt_state,
        "loss": loss,
        "acc": acc,
        "activities": equilib_activities,
        "t_max": t_max,
        "energies": energies,
        "activity_norms": activity_norms,
        "model_grad_norms": grad_norms[0],
        "skip_model_grad_norms": grad_norms[1]
    }


@eqx.filter_jit
def make_hpc_step(
      generator: PyTree[Callable],
      amortiser: PyTree[Callable],
      optims: Tuple[GradientTransformationExtraArgs],
      opt_states: Tuple[OptState],
      output: ArrayLike,
      input: Optional[ArrayLike] = None,
      ode_solver: AbstractSolver = Heun(),
      max_t1: int = 500,
      dt: float | int = None,
      stepsize_controller: AbstractStepSizeController = PIDController(
          rtol=1e-3, atol=1e-3
      ),
      steady_state_tols: Optional[Tuple[float]] = (None, None),
      record_activities: bool = False,
      record_energies: bool = False
) -> Dict:
    """Updates parameters of a hybrid predictive coding network.

    ??? cite "Reference"

        ```bibtex
        @article{tscshantz2023hybrid,
          title={Hybrid predictive coding: Inferring, fast and slow},
          author={Tscshantz, Alexander and Millidge, Beren and Seth, Anil K and Buckley, Christopher L},
          journal={PLoS Computational Biology},
          volume={19},
          number={8},
          pages={e1011280},
          year={2023},
          publisher={Public Library of Science San Francisco, CA USA}
        }
        ```

    !!! note

        The input and output of the generator are the output and input of the
        amortiser, respectively.

    **Main arguments:**

    - `generator`: List of callable layers for the generative model.
    - `amortiser`: List of callable layers for model amortising the inference
        of the `generator`.
    - `optims`: Optax optimisers (e.g. `optax.sgd()`), one for each model.
    - `opt_states`: State of Optax optimisers, one for each model.
    - `output`: Observation of the generator, input to the amortiser.
    - `input`: Optional prior of the generator, target for the amortiser.

    **Other arguments:**

    - `ode_solver`: Diffrax ODE solver to be used. Default is Heun, a 2nd order
        explicit Runge--Kutta method..
    - `max_t1`: Maximum end of integration region (20 by default).
    - `dt`: Integration step size. Defaults to None since the default
        `stepsize_controller` will automatically determine it.
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `PIDController`.
    - `steady_state_tols`: Optional relative and absolute tolerances for
        determining a steady state to terminate the inference solver. Defaults
        to the tolerances of the `stepsize_controller`.
    - `record_activities`: If `True`, returns activities at every inference
        iteration.
     - `record_energies`: If `True`, returns layer-wise energies at every
        inference iteration.

    **Returns:**

    Dict including models with updated parameters, optimiser and state for each
    model, model activities, last inference step for the generator, MSE losses,
    and energies.

    """
    if record_energies:
        record_activities = True

    gen_params = (generator, None)
    gen_optim, amort_optim = optims
    gen_opt_state, amort_opt_state = opt_states

    if input is not None:
        gen_activities = init_activities_with_ffwd(
            params=(generator, None),
            input=input
        )
        gen_loss = mean((output - gen_activities[-1]) ** 2)
    else:
        gen_loss = None

    amort_activities = init_activities_with_amort(
        amortiser=amortiser,
        generator=generator,
        input=output
    )
    equilib_activities = solve_pc_inference(
        params=gen_params,
        activities=amort_activities[1:] if input is not None else amort_activities,
        y=output,
        x=input,
        solver=ode_solver,
        max_t1=max_t1,
        dt=dt,
        stepsize_controller=stepsize_controller,
        steady_state_tols=steady_state_tols,
        record_iters=record_activities
    )
    t_max = get_t_max(equilib_activities) if record_activities else None

    gen_energies = compute_infer_energies(
        params=gen_params,
        activities_iters=equilib_activities,
        t_max=t_max,
        y=output,
        x=input
    ) if record_energies else None
    # remove target prediction of the generator
    equilib_activities_for_amort = tree_map(
        lambda act: act[t_max if record_activities else array(0)],
        equilib_activities[::-1][1:]
    )
    amort_loss = mean((input - amort_activities[0]) ** 2) if (
        input is not None
    ) else mean((equilib_activities_for_amort[-1] - amort_activities[0]) ** 2)

    amort_energies = hpc_energy_fn(
        model=amortiser,
        equilib_activities=equilib_activities_for_amort,
        amort_activities=amort_activities,
        x=output,
        y=input
    ) if record_energies else None

    gen_param_grads = compute_pc_param_grads(
        params=gen_params,
        activities=tree_map(
            lambda act: act[t_max if record_activities else array(0)],
            equilib_activities
        ),
        x=input,
        y=output
    )
    amort_param_grads = compute_hpc_param_grads(
        model=amortiser,
        equilib_activities=equilib_activities_for_amort,
        amort_activities=amort_activities,
        x=output,
        y=input
    )

    gen_updates, gen_opt_state = gen_optim.update(
        updates=gen_param_grads,
        state=gen_opt_state,
        params=generator
    )
    amort_updates, amort_opt_state = amort_optim.update(
        updates=amort_param_grads,
        state=amort_opt_state,
        params=amortiser
    )
    generator = eqx.apply_updates(model=generator, updates=gen_updates)
    amortiser = eqx.apply_updates(model=amortiser, updates=amort_updates)
    return {
        "models": (generator, amortiser),
        "optims": (gen_optim, amort_optim),
        "opt_states": (gen_opt_state, amort_opt_state),
        "activities": (amort_activities, equilib_activities),
        "t_max": t_max,
        "losses": (gen_loss, amort_loss),
        "energies": (gen_energies, amort_energies)
    }
