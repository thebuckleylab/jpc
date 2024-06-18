<h1 align='center'>JPC</h1>
<h2 align='center'>Predictive coding networks in JAX.</h2>

JPC is a [JAX](https://github.com/google/jax) library for predictive 
coding networks (PCNs). It is built on top of two main libraries:

* [Equinox](https://github.com/patrick-kidger/equinox), to define neural 
networks with PyTorch-like syntax, and
* [Diffrax](https://github.com/patrick-kidger/diffrax), to solve the PC 
activity (inference) dynamics.

JPC provides a simple but flexible API for research of PCNs compatible with
useful JAX transforms such as `vmap` and `jit`.

## Installation

```
pip install jpc
```

Requires Python 3.9+, JAX 0.4.23+, [Equinox](https://github.com/patrick-kidger/equinox) 
0.11.2+, [Diffrax](https://github.com/patrick-kidger/diffrax) 0.5.1+, 
[Optax](https://github.com/google-deepmind/optax) 0.2.2+, and 
[Jaxtyping](https://github.com/patrick-kidger/jaxtyping) 0.2.24+.

## Documentation
Available at X.

## Quickstart

Given a neural network with callable layers defined with
[Equinox](https://github.com/patrick-kidger/equinox)
```py
import jax
import jax.numpy as jnp
from equinox import nn as nn

# some data
x = jnp.array([1., 1., 1.])
y = -x

# network
key = jax.random.key(0)
_, *subkeys = jax.random.split(key)
network = [
    nn.Sequential(
        [
            nn.Linear(3, 100, key=subkeys[0]),
            nn.Lambda(jax.nn.relu)
        ],
    ),
    nn.Linear(100, 3, key=subkeys[1]),
]
```
We can train it with predictive coding in a few lines of code 
```py
import jpc

# initialise layer activities with a feedforward pass
activities = jpc.init_activities_with_ffwd(network, x)

# run the inference dynamics to equilibrium
equilib_activities = jpc.solve_pc_activities(network, activities, y, x)

# compute the PC parameter gradients
pc_param_grads = jpc.compute_pc_param_grads(
    network, 
    equilib_activities, 
    y, 
    x
)
```
The gradients can then be fed to your favourite optimiser (e.g. gradient
descent) to update the network parameters.
