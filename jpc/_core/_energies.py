"""Energy functions for predictive coding networks."""

from jax.numpy import sum
from jax import vmap
from jaxtyping import PyTree, ArrayLike, Scalar
from typing import Callable, Optional


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
    n_activity_layers = len(activities)-1
    n_layers = len(network)-1

    gen_eL = output - vmap(network[-1])(activities[-2])
    energy = 0.5 * sum(gen_eL ** 2)

    for act_l, gen_l in zip(
            range(start_activity_l, n_activity_layers),
            range(1, n_layers)
    ):
        gen_err = activities[act_l] - vmap(network[gen_l])(activities[act_l-1])
        energy += 0.5 * sum(gen_err ** 2)

    gen_e1 = activities[0] - vmap(network[0])(input) if (
            input is not None
    ) else activities[1] - vmap(network[0])(activities[0])
    energy += 0.5 * sum(gen_e1 ** 2)

    return energy / batch_size


def hpc_energy_fn(
        amortiser: PyTree[Callable],
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
    - `activities`: List of activities for each layer free to vary.
    - `output`: Observation of the generative model (or input of the amortiser).
    - `input`: Prior of the generative model (or output of the amortiser).

    **Returns:**

    The total energy normalised by batch size.

    """
    batch_size = output.shape[0]
    n_hidden = len(amortiser) - 1

    amort_eL = input - vmap(amortiser[-1])(activities[0])
    energy = 0.5 * sum(amort_eL ** 2)

    for l, rev_l in zip(range(1, n_hidden), reversed(range(1, n_hidden))):
        amort_err = activities[rev_l-1] - vmap(amortiser[l])(activities[rev_l])
        energy += 0.5 * sum(amort_err ** 2)

    amort_e1 = activities[-2] - vmap(amortiser[0])(output)
    energy += 0.5 * sum(amort_e1 ** 2)

    return energy / batch_size
