import jax
import jax.numpy as jnp
import equinox.nn as nn
from jaxtyping import PRNGKeyArray, PyTree
from typing import Callable


def get_fc_network(
        key: PRNGKeyArray,
        layer_sizes: PyTree[int],
        act_fn: str,
        use_bias: bool = True
) -> PyTree[Callable]:
    """Defines a fully connected network compatible with predictive coding updates.

    **Main arguments:**

    - `key`: `jax.random.PRNGKey` for parameter initialisation.
    - `layer_sizes`: dimension of all layers (input, hidden and output).
        Options are `linear`, `tanh` and `relu`.
    - `act_fn`: activation function for all layers except the output.
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
                    nn.Lambda(jnp.tanh)
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
                and 'relu'
            """)
        layers.append(hidden_layer)

    return layers
