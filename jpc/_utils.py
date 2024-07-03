import jax
from jax.numpy import tanh, mean, argmax, zeros
from jax.tree_util import tree_map
import equinox.nn as nn
from jpc import pc_energy_fn
from jaxtyping import PRNGKeyArray, PyTree, ArrayLike, Scalar, Array
from typing import Callable, Optional


def make_mlp(
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        act_fn: str,
        use_bias: bool = True
) -> PyTree[Callable]:
    """Creates a multi-layer perceptron compatible with predictive coding updates.

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
        elif act_fn == "leaky_relu" and not is_last:
            hidden_layer = nn.Sequential(
                [
                    nn.Linear(
                        layer_sizes[i],
                        layer_sizes[i+1],
                        use_bias=use_bias,
                        key=subkey
                    ),
                    nn.Lambda(jax.nn.leaky_relu)
                ]
            )
        elif act_fn == "gelu" and not is_last:
            hidden_layer = nn.Sequential(
                [
                    nn.Linear(
                        layer_sizes[i],
                        layer_sizes[i+1],
                        use_bias=use_bias,
                        key=subkey
                    ),
                    nn.Lambda(jax.nn.gelu)
                ]
            )
        else:
            raise ValueError("""
                Invalid activation function ID. Options are 'linear', 'tanh'
                and 'relu'.
            """)
        layers.append(hidden_layer)

    return layers


def get_t_max(activities_iters: PyTree[Array]) -> Array:
    return argmax(activities_iters[0][:, 0, 0]) - 1


def compute_infer_energies(
        model: PyTree[Callable],
        activities_iters: PyTree[Array],
        t_max: Array,
        y: ArrayLike,
        x: Optional[ArrayLike] = None
) -> PyTree[Scalar]:
    """Calculates layer energies during predictive coding inference.

    **Main arguments:**

    - `model`: List of callable model (e.g. neural network) layers.
    - `activities_iters`: Layer-wise activities at every inference iteration.
        Note that each set of activities will have 4096 steps as first
        dimension by diffrax default.
    - `t_max`: Maximum number of inference iterations to compute energies for.
    - `y`: Observation or target of the generative model.
    - `x`: Optional prior of the generative model.

    **Returns:**

    List of layer-wise energies for selected inference iterations.

    """
    def loop_body(state):
        t, energies_iters = state

        energies = pc_energy_fn(
            model=model,
            activities=tree_map(lambda act: act[t], activities_iters),
            y=y,
            x=x,
            record_layers=True
        )
        energies_iters = energies_iters.at[:, t].set(energies)
        return t + 1, energies_iters

    # 4096 is the max number of steps set in diffrax
    energies_iters = zeros((len(model), 4096))
    _, energies_iters = jax.lax.while_loop(
        lambda state: state[0] < t_max,
        loop_body,
        (0, energies_iters)
    )
    return energies_iters[::-1, :]


def compute_accuracy(truths: ArrayLike, preds: ArrayLike) -> Scalar:
    return mean(
        argmax(truths, axis=1) == argmax(preds, axis=1)
    )
