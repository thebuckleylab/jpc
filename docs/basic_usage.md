!!! info 
    JPC provides two types of API depending on the use case:
* a simple, basic API that allows to train and test models with predictive 
coding with a few lines of code 
  * a more advanced and flexible API allowing for 

Describe purposes/use cases of both basic and advanced.

# Basic usage

JPC provides a single convenience function `jpc.make_pc_step()` to train 
predictive coding networks (PCNs) on classification and generation tasks, in a 
supervised as well as unsupervised manner.
```py
import jpc

relu_net = jpc.get_fc_network(key, [10, 100, 100, 10], "relu")
result = jpc.make_pc_step(
      model=relu_net,
      optim=optim,
      opt_state=opt_state,
      y=y,
      x=x
)
```
At a minimum, `jpc.make_pc_step()` takes a model, an optax optimiser and its 
state, and an output target. Under the hood, `jpc.make_pc_step()` uses diffrax
to solve the activity (inference) dynamics of PC. The arguments can be changed
```py
import jpc

result = jpc.make_pc_step(
      model=network,
      optim=optim,
      opt_state=opt_state,
      y=y,
      x=x,
      solver=other_solver,
      dt=1e-1,
)
```
Moreover, 

JPC provides a similar function for training a hybrid PCN
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
