import jax
from jax.numpy import tanh, mean, argmax
import equinox.nn as nn
from jpc import pc_energy_fn
from jaxtyping import PRNGKeyArray, PyTree, ArrayLike, Scalar, Array
from typing import Callable, Optional


def get_fc_network(
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        act_fn: str,
        use_bias: bool = True
) -> PyTree[Callable]:
    """Defines a fully connected network compatible with predictive coding updates.

    **Main arguments:**

    - `key`: `jax.random.PRNGKey` for parameter initialisation.
    - `layer_sizes`: Dimension of all layers (input, hidden and output).
        Options are `linear`, `tanh` and `relu`.
    - `act_fn`: Activation function for all layers except the output.
    - `use_bias`: `True` by default.

    **Returns:**

    List of callable fully connected layers.

    """
    layers = []
    for i, subkey in enumerate(jax.random.split(key, len(layer_sizes)-1)):
        is_last = i+1 == len(layer_sizes)-1
        if act_fn == "linear" or is_last:
            hidden_layer = nn.Linear(
                layer_sizes[i],
                layer_sizes[i+1],
                use_bias=use_bias,
                key=subkey
            )
        elif act_fn == "tanh" and not is_last:
            hidden_layer = nn.Sequential(
                [
                    nn.Linear(
                        layer_sizes[i],
                        layer_sizes[i+1],
                        use_bias=use_bias,
                        key=subkey
                    ),
                    nn.Lambda(tanh)
                ]
            )
        elif act_fn == "relu" and not is_last:
            hidden_layer = nn.Sequential(
                [
                    nn.Linear(
                        layer_sizes[i],
                        layer_sizes[i+1],
                        use_bias=use_bias,
                        key=subkey
                    ),
                    nn.Lambda(jax.nn.relu)
                ]
            )
        else:
            raise ValueError("""
                Invalid activation function ID. Options are 'linear', 'tanh'
                and 'relu'.
            """)
        layers.append(hidden_layer)

    return layers


def compute_accuracy(truths: ArrayLike, preds: ArrayLike) -> Scalar:
    return mean(
        argmax(truths, axis=1) == argmax(preds, axis=1)
    )


def get_t_max(activities_iters: PyTree[Array]) -> int:
    t_max = argmax(activities_iters[0][:, 0, 0])-1
    return int(t_max)


def compute_pc_infer_energies(
        network: PyTree[Callable],
        activities_iters: PyTree[Array],
        t_max: int,
        output: ArrayLike,
        input: Optional[ArrayLike] = None,
        compute_every: int = 1
) -> PyTree[Scalar]:
    """Calculates layer energies during predictive coding inference.

    **Main arguments:**

    - `network`: List of callable network layers.
    - `activities_iters`: Layer-wise activities at every inference iteration.
        Note that each set of activities will have 4096 steps as first
        dimension by diffrax default.
    - `t_max`: Maximum number of inference iterations to compute energies for.
    - `output`: Observation or target of the generative model.
    - `input`: Optional prior of the generative model.

    **Other arguments:**

    - `compute_every`: Defaults to 1, calculating the energies at every
        inference iteration.

    **Returns:**

    List of layer-wise energies for selected inference iterations.

    """
    energies_iters = [[] for _ in range(len(network))]
    for t in range(t_max):
        if t % compute_every == 0:
            energies = pc_energy_fn(
                network=network,
                activities=[act[t] for act in activities_iters],
                output=output,
                input=input,
                record_layers=True
            )
            for l in range(len(network)):
                energies_iters[l].append(energies[l])

    return energies_iters
