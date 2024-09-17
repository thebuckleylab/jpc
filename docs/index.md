# Getting started

JPC is a [**J**AX](https://github.com/google/jax) library to train neural networks 
with **P**redictive **C**oding (PC). It is built on top of three main libraries:

* [Equinox](https://github.com/patrick-kidger/equinox), to define neural 
networks with PyTorch-like syntax,
* [Diffrax](https://github.com/patrick-kidger/diffrax), to solve the PC 
activity (inference) dynamics, and
* [Optax](https://github.com/google-deepmind/optax), for parameter optimisation.

JPC provides a simple, fast and flexible API for research on PCNs compatible 
with all of JAX and leveraging ODE solvers to integrate the PC inference
dynamics.

## 💻 Installation

```
pip install jpc
```

Requires Python 3.9+, JAX 0.4.23+, [Equinox](https://github.com/patrick-kidger/equinox) 
0.11.2+, [Diffrax](https://github.com/patrick-kidger/diffrax) 0.5.1+, 
[Optax](https://github.com/google-deepmind/optax) 0.2.2+, and 
[Jaxtyping](https://github.com/patrick-kidger/jaxtyping) 0.2.24+.

## ⚡️ Quick example
Use `jpc.make_pc_step` to update the parameters of essentially any neural 
network with PC
```py
import jpc
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# toy data
x = jnp.array([1., 1., 1.])
y = -x

# define model and optimiser
key = jax.random.PRNGKey(0)
model = jpc.make_mlp(key, layer_sizes=[3, 5, 5, 3], act_fn="relu")
optim = optax.adam(1e-3)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

# update model parameters with PC
result = jpc.make_pc_step(
    model,
    optim,
    opt_state,
    y,
    x
)
```
Under the hood, `jpc.make_pc_step`:

1. integrates the activity (inference) dynamics using a [Diffrax](https://github.com/patrick-kidger/diffrax) ODE solver (Euler by default), 
2. computes the PC gradient w.r.t. the model parameters at the numerical solution of the activities, and 
3. updates the parameters with the provided [Optax](https://github.com/google-deepmind/optax) optimiser.

> **NOTE**: All convenience training and test functions including `make_pc_step` 
> are already "jitted" (for increased performance) for the user's convenience.

## 🧠️ Predictive coding primer
TODO

## 🚀 Advanced usage
More advanced users can access the functionality used by `jpc.make_pc_step`.

```py
import jpc

# 1. initialise activities with a feedforward pass
activities0 = jpc.init_activities_with_ffwd(model, x)

# 2. run the inference dynamics to equilibrium
equilib_activities = jpc.solve_pc_inference(model, activities0, y, x)

# 3. compute PC parameter gradients
param_grads = jpc.compute_pc_param_grads(
    model,
    equilib_activities,
    y,
    x
)

# 4. update parameters
updates, opt_states = optim.update(
    param_grads,
    opt_state,
    model
)
model = eqx.apply_updates(model, updates)
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

