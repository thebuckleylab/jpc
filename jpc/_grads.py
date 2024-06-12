"""Functions to compute gradients of the free energy."""

from jax import grad
from equinox import filter_grad
from jaxtyping import PyTree, ArrayLike, Array
from typing import Union, Tuple, Callable, Optional
from ._energies import _energy_fn, pc_energy_fn, _lateral_energy_fn


def _neg_activity_grad(
        t: Union[float, int],
        activities: PyTree[ArrayLike],
        args: Tuple[PyTree[Callable], Optional[PyTree[Callable]], ArrayLike, ArrayLike]
) -> PyTree[Array]:
    """Computes the negative gradient of the energy with respect to the activities.

    $$
    - \partial \mathcal{F} / \partial \mathbf{z}
    $$

    This defines an ODE system to be integrated by `solve_pc_activities`.

    **Main arguments:**

    - `t`: time step of the ODE system, used for downstream integration by
        `diffrax.diffeqsolve`.
    - `activities`: List of activities for each layer free to vary.
    - `args`: 4-Tuple with
        (i) list of callable layers of the generative model,
        (ii) optional list of callable layers of a network to amortise inference,
        (iii) network output (observation), and
        (iv) network input (prior).

    **Returns:**

    List of negative gradients of the energy w.r.t the activities.

    """
    amortiser, generator, output, input = args
    dFdzs = grad(_energy_fn)(
        activities=activities,
        generator=generator,
        output=output,
        input=input,
        amortiser=amortiser
    )
    return [-dFdz for dFdz in dFdzs]


def _neg_lateral_activity_grad(
    t: Union[float, int],
    activities: PyTree[ArrayLike],
    args: Tuple[PyTree[Callable], PyTree[ArrayLike], PyTree[ArrayLike]]
) -> PyTree[Array]:
    """Same as `_neg_activity_grad` but for a network with lateral connections.

    **Main arguments:**

    - `t`: time step of the ODE system, used for downstream integration by
        `diffrax.diffeqsolve`.
    - `activities`: List of activities for each layer free to vary, one list
        per branch (n=2).
    - `args`: 2-Tuple with
        (i) list of callable layers for amortised network, and
        (ii) network outpus (observations), one for each branch.

    **Returns:**

    List of negative gradients of the energy w.r.t the activities.

    """
    amortiser, outputs = args
    dFdzs = grad(_lateral_energy_fn)(
        activities=activities,
        amortiser=amortiser,
        outputs=outputs
    )
    for branch in range(2):
        dFdzs[branch] = [-dFdz for dFdz in dFdzs[branch]]
    return dFdzs


def compute_pc_param_grads(
        network: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: Optional[ArrayLike] = None
) -> PyTree[Array]:
    """Computes the gradient of the energy with respect to network parameters $\partial \mathcal{F} / \partial θ$.

    **Main arguments:**

    - `network`: List of callable network layers.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Returns:**

    List of parameter gradients for each network layer.

    """
    return filter_grad(pc_energy_fn)(
        network=network,
        activities=activities,
        output=output,
        input=input
    )


def compute_gen_param_grads(
        amortiser: PyTree[Callable],
        generator: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: ArrayLike
) -> PyTree[Array]:
    """Computes the gradient of the energy w.r.t the parameters of a generative model $\partial \mathcal{F} / \partial θ$.

    !!! note

        This has the same functionality as `compute_pc_param_grads` but can be
        used together with `compute_amort_param_grads` for a more user-friendly
        API when training hybrid predictive coding networks.

    **Main arguments:**

    - `amortiser`: List of callable layers for the network amortising the
        inference of the generative model.
    - `generator`: List of callable layers for the generative model.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Prior of the generative model.

    **Returns:**

    List of parameter gradients for each layer of the generative network.

    """
    return filter_grad(_energy_fn)(
        generator=generator,
        amortiser=amortiser,
        activities=activities,
        output=output,
        input=input
    )


def compute_amort_param_grads(
        amortiser: PyTree[Callable],
        generator: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: ArrayLike
) -> PyTree[Array]:
    """Computes the gradient of the energy w.r.t the parameters of an amortised model $\partial \mathcal{F} / \partial \phi$.

    **Main arguments:**

    - `amortiser`: List of callable layers for a network amortising the
        inference of the generative model.
    - `generator`: List of callable layers for the generative network.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Returns:**

    List of parameter gradients for each layer of the amortiser.

    """
    return filter_grad(_energy_fn)(
        amortiser=amortiser,
        generator=generator,
        activities=activities,
        output=output,
        input=input
    )


def compute_lateral_pc_param_grads(
        amortiser: PyTree[Callable],
        activities: PyTree[ArrayLike],
        outputs: PyTree[ArrayLike],
) -> PyTree[Array]:
    """Same as `compute_pc_param_grads` but for a network with lateral connections.

    **Main arguments:**

    - `amortiser`: List of callable layers for an amortised network.
    - `activities`: List of activities for each layer free to vary, one list
        per branch (n=2).
    - `outputs`: List of two inputs to the amortiser, one for each branch.

    **Returns:**

    List of parameter gradients for each network layer.

    """
    return filter_grad(_lateral_energy_fn)(
        amortiser=amortiser,
        activities=activities,
        outputs=outputs
    )
