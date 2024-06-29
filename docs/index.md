# Getting started

JPC is a [JAX](https://github.com/google/jax) library to train neural networks 
with predictive coding. It is built on top of three main libraries:

* [Equinox](https://github.com/patrick-kidger/equinox), to define neural 
networks with PyTorch-like syntax,
* [Diffrax](https://github.com/patrick-kidger/diffrax), to solve the PC 
activity (inference) dynamics, and
* [Optax](https://github.com/google-deepmind/optax), for parameter optimisation.

JPC provides a simple but flexible API for research of PCNs compatible with
useful JAX transforms such as `vmap` and `jit`.

## 💻 Installation

```
pip install jpc
```

Requires Python 3.9+, JAX 0.4.23+, [Equinox](https://github.com/patrick-kidger/equinox) 
0.11.2+, [Diffrax](https://github.com/patrick-kidger/diffrax) 0.5.1+, 
[Optax](https://github.com/google-deepmind/optax) 0.2.2+, and 
[Jaxtyping](https://github.com/patrick-kidger/jaxtyping) 0.2.24+.

## ⚡️ Quick example

Given a neural network with callable layers
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
network = [nn.Sequential(
    [
        nn.Linear(3, 100, key=subkeys[0]), 
        nn.Lambda(jax.nn.relu)],
    ),
    nn.Linear(100, 3, key=subkeys[1]),
]
```
we can perform a PC parameter update with a single function call
```py
import jpc
import optax
import equinox as eqx

# optimiser
optim = optax.adam(1e-3)
opt_state = optim.init(eqx.filter(network, eqx.is_array))

# PC parameter update
result = jpc.make_pc_step(
      model=network,
      optim=optim,
      opt_state=opt_state,
      y=y,
      x=x
)

```

## 📄 Citation

If you found this library useful in your work, please cite (arXiv link):

```bibtex
@article{innocenti2024jpc,
    title={JPC: Predictive Coding Networks in JAX},
    author={Innocenti, Francesco and Kinghorn, Paul and Singh, Ryan and 
    De Llanza Varona, Miguel and Buckley, Christopher},
    journal={arXiv preprint},
    year={2024}
}
```
Also consider starring the project [on GitHub](https://github.com/thebuckleylab/jpc)! ⭐️ 

## ⏭️ Next steps

