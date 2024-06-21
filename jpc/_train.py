"""High-level API to train neural networks with predictive coding."""

import equinox as eqx
from jax import vmap
from jax.numpy import mean
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Dopri5,
    PIDController
)
from jpc import (
    init_activities_with_ffwd,
    init_activities_with_amort,
    solve_pc_activities,
    compute_pc_param_grads
)
from optax import GradientTransformationExtraArgs, OptState
from jaxtyping import PyTree, ArrayLike, Scalar, Array
from typing import Callable, Optional, Union, Tuple


@eqx.filter_jit
def make_pc_step(
      network: PyTree[Callable],
      optim: GradientTransformationExtraArgs,
      opt_state: OptState,
      output: ArrayLike,
      input: Optional[ArrayLike] = None,
      solver: AbstractSolver = Dopri5(),
      n_iters: Optional[int] = 300,
      stepsize_controller: AbstractStepSizeController = PIDController(
          rtol=1e-3,
          atol=1e-3
      ),
      dt: Union[float, int] = None,
      record_activities: bool = False
) -> Union[
         Tuple[
             PyTree[Callable],
             GradientTransformationExtraArgs,
             OptState,
             Scalar,
             PyTree[Array]
         ],
         Tuple[
             PyTree[Callable],
             GradientTransformationExtraArgs,
             OptState,
             Scalar
         ]
    ]:
    """Updates network parameters with predictive coding.

    **Main arguments:**

    - `network`: List of callable network layers.
    - `optim`: Optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of Optax optimiser.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `solver`: Diffrax (ODE) solver to be used. Default is Dopri5.
    - `n_iters`: Number of integration steps for inference (300 as default).
    - `stepsize_controller`: diffrax controllers for inference integration.
        Defaults to `PIDController`.
    - `dt`: Integration step size. Defaults to None, since step size is
        automatically determined by the default `PIDController`.
    - `record_activities`: If `True`, returns activities at every inference
        iteration.

    **Returns:**

    Network with updated weights, optimiser, optimiser state, training loss and
    optionally activities during inference.

    """
    activities = init_activities_with_ffwd(network=network, input=input)
    train_mse_loss = mean((output - activities[-1])**2)
    equilib_activities = solve_pc_activities(
        network=network,
        activities=activities,
        output=output,
        input=input,
        solver=solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt,
        record_iters=record_activities
    )
    param_grads = compute_pc_param_grads(
        network=network,
        activities=[act[-1] for act in equilib_activities],
        output=output,
        input=input
    )
    updates, opt_state = optim.update(
        updates=param_grads,
        state=opt_state,
        params=network
    )
    network = eqx.apply_updates(model=network, updates=updates)
    if record_activities:
        return (
            network,
            optim,
            opt_state,
            train_mse_loss,
            equilib_activities
        )
    else:
        return (
            network,
            optim,
            opt_state,
            train_mse_loss
        )


@eqx.filter_jit
def make_hpc_step(
      generator: PyTree[Callable],
      amortiser: PyTree[Callable],
      optims: Tuple[GradientTransformationExtraArgs],
      opt_states: Tuple[OptState],
      output: ArrayLike,
      input: ArrayLike,
      solver: AbstractSolver = Dopri5(),
      n_iters: Optional[int] = 300,
      stepsize_controller: AbstractStepSizeController = PIDController(
          rtol=1e-3,
          atol=1e-3
      ),
      dt: Union[float, int] = None
) -> Tuple[
        PyTree[Callable],
        PyTree[Callable],
        Tuple[GradientTransformationExtraArgs],
        Tuple[OptState],
        Scalar
]:
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

    **Main arguments:**

    - `generator`: List of callable layers for the generative network.
    - `amortiser`: List of callable layers for network amortising the inference
        of the generative model.
    - `optims`: Optax optimisers (e.g. `optax.sgd()`), one for each model.
    - `opt_states`: State of Optax optimisers, one for each model.
    - `output`: Observation of the generator, input to the amortiser.
    - `input`: Prior of the generator, target for the amortiser.

    **Other arguments:**

    - `solver`: Diffrax (ODE) solver to be used. Default is Dopri5.
    - `n_iters`: Number of integration steps (300 as default).
    - `stepsize_controller`: diffrax controllers for step size integration.
        Defaults to `PIDController`.
    - `dt`: Integration step size. Defaults to None, since step size is
        automatically determined by the default `PIDController`.

    **Returns:**

    Generator and amortiser with updated weights, optimiser and state for each
    model and training loss.

    """
    gen_optim, amort_optim = optims
    gen_opt_state, amort_opt_state = opt_states
    amort_activities = init_activities_with_amort(
        amortiser=amortiser,
        generator=generator,
        output=output
    )
    train_mse_loss = mean((input - amort_activities[0])**2)
    equilib_activities = solve_pc_activities(
        network=generator,
        activities=amort_activities[1:],
        output=output,
        input=input,
        solver=solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt
    )
    gen_param_grads = compute_pc_param_grads(
        network=generator,
        activities=equilib_activities,
        output=output,
        input=input
    )
    activities_for_amort = equilib_activities[::-1][1:]
    activities_for_amort.append(
        vmap(amortiser[-1])(equilib_activities[0])
    )
    amort_param_grads = compute_pc_param_grads(
        network=amortiser,
        activities=activities_for_amort,
        output=input,
        input=output,
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
    return (
        generator,
        amortiser,
        (gen_optim, amort_optim),
        (gen_opt_state, amort_opt_state),
        train_mse_loss
    )
