"""Energy functions for predictive coding networks."""

from jax.numpy import sum, log, exp
from jax import vmap
from jaxtyping import PyTree, ArrayLike, Scalar
from typing import Callable, Optional


def _energy_fn(
        generator: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: Optional[ArrayLike] = None,
        amortiser: Optional[PyTree[Callable]] = None
) -> Scalar:
    """Computes the free energy for a 'hybrid' or standard predictive coding network.

    **Main arguments:**

    - `generator`: List of callable layers for the generative model.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `amortiser`: Optional list of callable layers for a network amortising
        the inference of the generative model.

    **Returns:**

    The total energy normalised by batch size.

    """
    batch_size = output.shape[0]
    start_activity_l = 1 if input is not None else 2
    n_activity_layers = len(activities)-1
    n_layers = len(generator)-1

    gen_eL = output - vmap(generator[-1])(activities[-2])
    energy = 0.5 * sum(gen_eL ** 2)
    if amortiser is not None:
        amort_eL = input - vmap(amortiser[-1])(activities[0])
        energy += 0.5 * sum(amort_eL ** 2)

    for act_l, gen_l, amort_l in zip(
            range(start_activity_l, n_activity_layers),
            range(1, n_layers),
            reversed(range(1, n_layers))
    ):
        gen_err = activities[act_l] - vmap(generator[gen_l])(activities[act_l-1])
        energy += 0.5 * sum(gen_err ** 2)
        if amortiser is not None:
            amort_err = activities[amort_l-1] - vmap(amortiser[gen_l])(activities[amort_l])
            energy += 0.5 * sum(amort_err ** 2)

    gen_e1 = activities[0] - vmap(generator[0])(input) if (
            input is not None
    ) else activities[1] - vmap(generator[0])(activities[0])
    energy += 0.5 * sum(gen_e1 ** 2)
    if amortiser is not None:
        amort_e1 = activities[-1] - vmap(amortiser[0])(output)
        energy += 0.5 * sum(amort_e1 ** 2)

    return energy / batch_size


def pc_energy_fn(
        network: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: Optional[ArrayLike] = None
) -> Scalar:
    """Computes the free energy for a feedforward neural network of the form

    $$
    \mathcal{F}(\mathbf{z}; θ) = \sum_\ell^L || \mathbf{z}_\ell - f_\ell(\mathbf{z}_{\ell-1}; θ) ||^2
    $$

    given parameters $θ$, free activities $\mathbf{z}$, output
    $\mathbf{z}_L = \mathbf{y}$ and optionally input $\mathbf{z}_0 = \mathbf{x}$.
    The activity of each layer $\mathbf{z}_\ell$ is some function of the previous
    layer, e.g. $f_\ell(W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell)$
    for a fully connected layer.

    !!! note

        The input and output correspond to the prior and observation of
        the generative model, respectively.

    **Main arguments:**

    - `network`: List of callable network layers.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Returns:**

    The total energy normalised by batch size.

    """
    batch_size = output.shape[0]
    start_activity_l = 1 if input is not None else 2
    n_activity_layers = len(activities) - 1
    n_layers = len(network) - 1

    gen_eL = output - vmap(network[-1])(activities[-2])
    energy = 0.5 * sum(gen_eL ** 2)

    for act_l, gen_l in zip(
            range(start_activity_l, n_activity_layers),
            range(1, n_layers)
    ):
        gen_err = activities[act_l] - vmap(network[gen_l])(activities[act_l - 1])
        energy += 0.5 * sum(gen_err ** 2)

    gen_e1 = activities[0] - vmap(network[0])(input) if (
            input is not None
    ) else activities[1] - vmap(network[0])(activities[0])
    energy += 0.5 * sum(gen_e1 ** 2)

    return energy / batch_size


def hpc_energy_fn(
        amortiser: PyTree[Callable],
        generator: PyTree[Callable],
        activities: PyTree[ArrayLike],
        output: ArrayLike,
        input: ArrayLike
) -> Scalar:
    """Computes the free energy for a 'hybrid' predictive coding network.

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

        Input is required so currently this only supports supervised training.

    **Main arguments:**

    - `amortiser`: List of callable layers for network amortising the inference
        of the generative model.
    - `generator`: List of callable layers for the generative model.
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation of the generative model (or input of the amortiser).
    - `input`: Prior of the generative model (or output of the amortiser).

    **Returns:**

    The total energy normalised by batch size.

    """
    batch_size = output.shape[0]
    n_hidden = len(generator) - 1

    gen_eL = output - vmap(generator[-1])(activities[-2])
    amort_eL = input - vmap(amortiser[-1])(activities[0])
    energy = 0.5 * sum(amort_eL ** 2) + 0.5 * sum(gen_eL ** 2)

    for l, rev_l in zip(range(1, n_hidden), reversed(range(1, n_hidden))):
        gen_err = activities[l] - vmap(generator[l])(activities[l-1])
        amort_err = activities[rev_l-1] - vmap(amortiser[l])(activities[rev_l])
        energy += 0.5 * sum(gen_err ** 2) + 0.5 * sum(amort_err ** 2)

    gen_e1 = activities[0] - vmap(generator[0])(input)
    amort_e1 = activities[-1] - vmap(amortiser[0])(output)
    energy += 0.5 * sum(amort_e1 ** 2) + 0.5 * sum(gen_e1 ** 2)

    return energy / batch_size


def _lateral_energy_fn(
        amortiser: PyTree[Callable],
        activities: PyTree[ArrayLike],
        outputs: PyTree[ArrayLike],
) -> Scalar:
    """Computes the free energy for a predictive coding network with lateral connections.

    !!! note

        This is currently experimental.

    **Main arguments:**

    - `amortiser`: List of callable layers for an amortised network.
    - `activities`: List of activities for each layer free to vary, one list
        per branch (n=2).
    - `outputs`: List of two inputs to the amortiser, one for each branch.

    **Returns:**

    The total energy normalised by batch size.

    """
    activities1, activities2 = activities
    output1, output2 = outputs
    batch_size = output1.shape[0]
    n_layers = len(amortiser)

    amort_e1 = activities1[-1] - vmap(amortiser[0])(output1)
    amort_e12 = activities2[-1] - vmap(amortiser[0])(output2)
    energy = 0.5 * sum(amort_e1 ** 2) + 0.5 * sum(amort_e12 ** 2)

    lateral1 = activities1[-1] - activities2[-1]
    lateralL = activities1[0] - activities2[0]
    energy += 0.5 * sum(lateral1 ** 2) + 0.5 * sum(lateralL ** 2)

    for l, rev_l in zip(range(1, n_layers), reversed(range(1, n_layers))):
        amort_err = activities1[rev_l-1] - vmap(amortiser[l])(activities1[rev_l])
        amort_err2 = activities2[rev_l-1] - vmap(amortiser[l])(activities2[rev_l])
        energy += 0.5 * sum(amort_err ** 2) + 0.5 * sum(amort_err2 ** 2)

        lateral_err = activities2[l] - activities2[l]
        energy += 0.5 * sum(lateral_err ** 2)

        logsumexp = log(sum(exp(-amort_err**2), axis=0))
        logsumexp2 = log(sum(exp(-amort_err2**2), axis=0))
        energy += 0.5 * sum(logsumexp) + 0.5 * sum(logsumexp2)

    return energy / batch_size
