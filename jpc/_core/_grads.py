"""Functions to compute gradients of the PC energy."""

from jax import grad, value_and_grad
from jax.tree_util import tree_map
from equinox import filter_grad
from jaxtyping import PyTree, ArrayLike, Array, Scalar
from typing import Tuple, Callable, Optional
from diffrax import AbstractStepSizeController
from ._energies import pc_energy_fn, hpc_energy_fn


def neg_activity_grad(
        t: float | int,
        activities: PyTree[ArrayLike],
        args: Tuple[
            Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
            ArrayLike,
            Optional[ArrayLike],
            int,
            str,
            str,
            AbstractStepSizeController
        ]
    ) -> PyTree[Array]:
    """Computes the negative gradient of the [PC energy](http://127.0.0.1:8000/api/Energy%20functions/#jpc.pc_energy_fn) 
    with respect to the activities $- ∇_{\mathbf{z}} \mathcal{F}$.

    This defines an ODE system to be integrated by [`solve_pc_inference()`](http://127.0.0.1:8000/api/Continuous%20Inference/#jpc.solve_inference).

    **Main arguments:**

    - `t`: Time step of the ODE system, used for downstream integration by
        [`diffrax.diffeqsolve()`](https://docs.kidger.site/diffrax/api/diffeqsolve/#diffrax.diffeqsolve).
    - `activities`: List of activities for each layer free to vary.
    - `args`: 5-Tuple with:

        (i) Tuple with callable model layers and optional skip connections,

        (ii) model output (observation),

        (iii) model input (prior),

        (iv) number of layers to skip for the skip connections (0 by default),

        (v) loss specified at the output layer (MSE as default or cross-entropy),

        (vi) parameterisation type (`sp` as default, `mupc`, or `ntp`),

        (vii) $\ell^2$ regulariser for the weights (0 by default),

        (viii) spectral penalty for the weights (0 by default),

        (ix) $\ell^2$ regulariser for the activities (0 by default), and

        (x) diffrax controller for step size integration.

    **Returns:**

    List of negative gradients of the energy with respect to the activities.

    """
    params, y, x, n_skip, loss_id, param_type, weight_decay, spectral_penalty, activity_decay, _ = args
    dFdzs = grad(pc_energy_fn, argnums=1)(
        params,
        activities,
        y,
        x=x,
        n_skip=n_skip,
        loss=loss_id,
        param_type=param_type,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay
    )
    return tree_map(lambda dFdz: -dFdz, dFdzs)


def compute_activity_grad(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        y: ArrayLike,
        *,
        x: Optional[ArrayLike],
        n_skip: int = 0,
        loss_id: str = "mse",
        param_type: str = "sp",
        weight_decay: Scalar = 0.,
        spectral_penalty: Scalar = 0.,
        activity_decay: Scalar = 0.
) -> PyTree[Array]:
    """Computes the gradient of the [PC energy](http://127.0.0.1:8000/api/Energy%20functions/#jpc.pc_energy_fn)
    with respect to the activities $∇_{\mathbf{z}} \mathcal{F}$.

    !!! note

        This function differs from [`jpc.neg_activity_grad()`](http://127.0.0.1:8000/api/Gradients/#jpc.neg_activity_grad) 
        only in the sign of the gradient (positive as opposed to negative) and 
        is called in [`update_activities()`](http://127.0.0.1:8000/api/Discrete%20updates/#jpc.update_activities) 
        for use with any [optax](https://github.com/google-deepmind/optax) 
        optimiser.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.

    **Other arguments:**

    - `x`: Optional prior of the generative model.
    - `n_skip`: Number of layers to skip for the skip connections (0 by default).
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `mse` (default) or cross-entropy `ce`.
    - `param_type`: Determines the parameterisation. Options are `sp` (default), 
        `mupc`, or `ntp`.
    - `weight_decay`: $\ell^2$ regulariser for the weights (0 by default).
    - `spectral_penalty`: Weight spectral penalty of the form 
        $||\mathbf{I} - \mathbf{W}_\ell^T \mathbf{W}_\ell||^2$ (0 by default).
    - `activity_decay`: $\ell^2$ regulariser for the activities (0 by default).

    **Returns:**

    List of negative gradients of the energy with respect to the activities.

    """
    energy, dFdzs = value_and_grad(pc_energy_fn, argnums=1)(
        params,
        activities,
        y,
        x=x,
        n_skip=n_skip,
        loss=loss_id,
        param_type=param_type,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay
    )
    return energy, dFdzs


def compute_pc_param_grads(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities: PyTree[ArrayLike],
        y: ArrayLike,
        *,
        x: Optional[ArrayLike] = None,
        n_skip: int = 0,
        loss_id: str = "mse",
        param_type: str = "sp",
        weight_decay: Scalar = 0.,
        spectral_penalty: Scalar = 0.,
        activity_decay: Scalar = 0.
) -> Tuple[PyTree[Array], PyTree[Array]]:
    """Computes the gradient of the [PC energy](http://127.0.0.1:8000/api/Energy%20functions/#jpc.pc_energy_fn)
    with respect to model parameters $∇_θ \mathcal{F}$.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities`: List of activities for each layer free to vary.
    - `y`: Observation or target of the generative model.

    **Other arguments:**

    - `x`: Optional prior of the generative model.
    - `n_skip`: Number of layers to skip for the skip connections (0 by default).
    - `loss_id`: Loss function to use at the output layer. Options are mean squared 
        error `mse` (default) or cross-entropy `ce`.
    - `param_type`: Determines the parameterisation. Options are `sp` (default), 
        `mupc`, or `ntp`.
    - `weight_decay`: $\ell^2$ regulariser for the weights (0 by default).
    - `spectral_penalty`: Weight spectral penalty of the form 
        $||\mathbf{I} - \mathbf{W}_\ell^T \mathbf{W}_\ell||^2$ (0 by default).
    - `activity_decay`: $\ell^2$ regulariser for the activities (0 by default).

    **Returns:**

    List of parameter gradients for each model layer.

    """
    return filter_grad(pc_energy_fn)(
        params,
        activities,
        y,
        x=x,
        n_skip=n_skip,
        loss=loss_id,
        param_type=param_type,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay
    )


def compute_hpc_param_grads(
        model: PyTree[Callable],
        equilib_activities: PyTree[ArrayLike],
        amort_activities: PyTree[ArrayLike],
        x: ArrayLike,
        y: Optional[ArrayLike] = None
) -> PyTree[Array]:
    """Computes the gradient of the [hybrid PC energy](http://127.0.0.1:8000/api/Energy%20functions/#jpc.hpc_energy_fn) 
    with respect to an amortiser's parameters $∇_θ \mathcal{F}$.

    !!! warning

        The input $x$ and output $y$ are reversed compared to 
        `compute_pc_param_grads()` ($x$ is the generator's target and $y$ is its 
        optional input or prior). Just think of $x$ and $y$ as the actual input 
        and output of the amortiser, respectively.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `equilib_activities`: List of equilibrated activities reached by the
        generator and target for the amortiser.
    - `amort_activities`: List of amortiser's feedforward guesses
        (initialisation) for the model activities.
    - `x`: Input to the amortiser.
    - `y`: Optional target of the amortiser (for supervised training).

    **Returns:**

    List of parameter gradients for each model layer.

    """
    return filter_grad(hpc_energy_fn)(
        model,
        equilib_activities,
        amort_activities,
        x,
        y
    )
