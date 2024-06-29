"""High-level API to train neural networks with predictive coding."""

import equinox as eqx
from jax import vmap
from jax.tree_util import tree_map
from jax.numpy import mean, array
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Euler,
    ConstantStepSize
)
from jpc import (
    init_activities_with_ffwd,
    init_activities_from_gaussian,
    init_activities_with_amort,
    pc_energy_fn,
    solve_pc_activities,
    get_t_max,
    compute_infer_energies,
    compute_pc_param_grads
)
from optax import GradientTransformationExtraArgs, OptState
from jaxtyping import PyTree, ArrayLike, Scalar, PRNGKeyArray
from typing import Callable, Optional, Tuple, Dict


@eqx.filter_jit
def make_pc_step(
      model: PyTree[Callable],
      optim: GradientTransformationExtraArgs,
      opt_state: OptState,
      y: ArrayLike,
      x: Optional[ArrayLike] = None,
      ode_solver: AbstractSolver = Euler(),
      dt: float | int = 1,
      n_iters: Optional[int] = 20,
      stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
      key: Optional[PRNGKeyArray] = None,
      layer_sizes: Optional[PyTree[int]] = None,
      batch_size: Optional[int] = None,
      sigma: Scalar = 0.05,
      record_activities: bool = False,
      record_energies: bool = False
) -> Dict:
    """Updates network parameters with predictive coding.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `optim`: Optax optimiser, e.g. `optax.sgd()`.
    - `opt_state`: State of Optax optimiser.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    !!! note

        `key`, `layer_sizes` and `batch_size` must be passed if `input` is
        `None`, since unsupervised training will be assumed and activities need
        to be initialised randomly.

    **Other arguments:**

    - `ode_solver`: Diffrax ODE solver to be used. Default is Euler.
    - `dt`: Integration step size. Defaults to 1.
    - `n_iters`: Number of integration steps (20 as default).
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `ConstantStepSize`.
    - `key`: `jax.random.PRNGKey` for random initialisation of activities.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
    - `batch_size`: Dimension of data batch for activity initialisation.
    - `sigma`: Standard deviation for Gaussian to sample activities from for
        random initialisation. Defaults to 5e-2.
    - `record_activities`: If `True`, returns activities at every inference
        iteration.
    - `record_energies`: If `True`, returns layer-wise energies at every
        inference iteration.

    **Returns:**

    Dict including model with updated parameters, optimiser, updated optimiser
    state, equilibrated activities, last inference step, MSE loss, and energies.

    **Raises:**

    - `ValueError` for inconsistent inputs.

    """
    if x is None and any(arg is None for arg in (key, layer_sizes, batch_size)):
        raise ValueError("""
            If there is no input (i.e. `x` is None), then unsupervised training 
            is assumed, and `key`, `layer_sizes` and `batch_size` must be 
            passed for random initialisation of activities.
        """)

    if record_energies and not record_activities:
        raise ValueError("""
            `record_energies` = `True` requires `record_activities` = `True`. 
        """)

    activities = init_activities_from_gaussian(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=sigma
    ) if x is None else init_activities_with_ffwd(model=model, x=x)

    mse_loss = mean((y - activities[-1])**2) if x is not None else None
    equilib_activities = solve_pc_activities(
        model=model,
        activities=activities,
        y=y,
        x=x,
        solver=ode_solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt,
        record_iters=record_activities
    )
    t_max = get_t_max(equilib_activities) if record_activities else None
    energies = compute_infer_energies(
        model=model,
        activities_iters=equilib_activities,
        t_max=t_max,
        y=y,
        x=x
    ) if record_energies else None

    param_grads = compute_pc_param_grads(
        model=model,
        activities=tree_map(
            lambda act: act[t_max if record_activities else array(0)],
            equilib_activities
        ),
        y=y,
        x=x
    )
    updates, opt_state = optim.update(
        updates=param_grads,
        state=opt_state,
        params=model
    )
    model = eqx.apply_updates(model=model, updates=updates)
    return {
        "model": model,
        "optim": optim,
        "opt_state": opt_state,
        "activities": equilib_activities,
        "t_max": t_max,
        "loss": mse_loss,
        "energies": energies
    }


@eqx.filter_jit
def make_hpc_step(
      generator: PyTree[Callable],
      amortiser: PyTree[Callable],
      optims: Tuple[GradientTransformationExtraArgs],
      opt_states: Tuple[OptState],
      y: ArrayLike,
      x: ArrayLike,
      ode_solver: AbstractSolver = Euler(),
      dt: float | int = 1,
      n_iters: Optional[int] = 20,
      stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
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

    **Main arguments:**

    - `generator`: List of callable layers for the generative model.
    - `amortiser`: List of callable layers for model amortising the inference
        of the `generator`.
    - `optims`: Optax optimisers (e.g. `optax.sgd()`), one for each model.
    - `opt_states`: State of Optax optimisers, one for each model.
    - `y`: Observation of the generator, input to the amortiser.
    - `x`: Prior of the generator, target for the amortiser.

    **Other arguments:**

    - `ode_solver`: Diffrax ODE solver to be used. Default is Euler.
    - `dt`: Integration step size. Defaults to 1.
    - `n_iters`: Number of integration steps (20 as default).
    - `stepsize_controller`: diffrax controller for step size integration.
        Defaults to `ConstantStepSize`.
    - `record_activities`: If `True`, returns activities at every inference
        iteration.
     - `record_energies`: If `True`, returns layer-wise energies at every
        inference iteration.

    **Returns:**

    Dict including models with updated parameters, optimiser and state for each
    model, model activities, last inference step for the generator, MSE losses,
    and energies.

    **Raises:**

    - `ValueError` for inconsistent inputs.

    """
    if record_energies and not record_activities:
        raise ValueError("""
            `record_energies` = `True` requires `record_activities` = `True`. 
        """)

    gen_optim, amort_optim = optims
    gen_opt_state, amort_opt_state = opt_states

    gen_activities = init_activities_with_ffwd(model=generator, x=x)
    amort_activities = init_activities_with_amort(
        amortiser=amortiser,
        generator=generator,
        y=y
    )
    gen_loss = mean((y - gen_activities[-1]) ** 2)
    amort_loss = mean((x - amort_activities[0]) ** 2)

    equilib_activities = solve_pc_activities(
        model=generator,
        activities=amort_activities[1:],
        y=y,
        x=x,
        solver=ode_solver,
        n_iters=n_iters,
        stepsize_controller=stepsize_controller,
        dt=dt,
        record_iters=record_activities
    )
    t_max = get_t_max(equilib_activities) if record_activities else None

    gen_energies = compute_infer_energies(
        model=generator,
        activities_iters=equilib_activities,
        t_max=t_max,
        y=y,
        x=x
    ) if record_energies else None
    equilib_activities_for_amort = tree_map(
        lambda act: act[t_max if record_activities else array(0)],
        equilib_activities[::-1][1:]
    )
    equilib_activities_for_amort.append(
        vmap(amortiser[-1])(equilib_activities_for_amort[0])
    )
    amort_energies = pc_energy_fn(
        model=amortiser,
        activities=equilib_activities_for_amort,
        y=x,
        x=y
    ) if record_energies else None

    gen_param_grads = compute_pc_param_grads(
        model=generator,
        activities=tree_map(
            lambda act: act[t_max if record_activities else array(0)],
            equilib_activities
        ),
        y=y,
        x=x
    )
    amort_param_grads = compute_pc_param_grads(
        model=amortiser,
        activities=equilib_activities_for_amort,
        y=x,
        x=y
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
