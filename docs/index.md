# Getting started
JPC is a [**J**AX](https://github.com/google/jax) library to train neural networks 
with **P**redictive **C**oding (PC). It is built on top of three main libraries:

* [Equinox](https://github.com/patrick-kidger/equinox), to define neural 
networks with PyTorch-like syntax,
* [Diffrax](https://github.com/patrick-kidger/diffrax), to solve the PC inference (activity) dynamics, and
* [Optax](https://github.com/google-deepmind/optax), for parameter optimisation.

Unlike existing PC libraries, JPC leverages ordinary differential equation solvers
to integrate the inference (activity) dynamics of PC networks, which we find
can provide significant speed-ups compared to standard optimisers, especially
for deeper models. 

JPC provides a **simple**, **relatively fast** and **flexible** API.
1. It is simple in that, like JAX, JPC follows a fully functional paradigm, 
and the core library is <1000 lines of code. 
2. It is relatively fast in that higher-order solvers can provide speed-ups 
compared to standard optimisers, especially on deeper models. 
3. And it is flexible in that it allows training a variety of PC networks 
including discriminative, generative and hybrid models.

## 💻 Installation
```
pip install jpc
```

Requires Python 3.9+, JAX 0.4.23+, [Equinox](https://github.com/patrick-kidger/equinox) 
0.11.2+, [Diffrax](https://github.com/patrick-kidger/diffrax) 0.6.0+, and 
[Optax](https://github.com/google-deepmind/optax) 0.2.4+. 

For GPU usage, upgrade jax to the appropriate cuda version (12 as an example 
here).

```
pip install --upgrade "jax[cuda12]"
```

## ⚡️ Quick example
Use `jpc.make_pc_step` to update the parameters of any neural network compatible
with PC updates (see examples)
```py
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import jpc

# toy data
x = jnp.array([1., 1., 1.])
y = -x

# define model and optimiser
key = jr.PRNGKey(0)
model = jpc.make_mlp(key, layer_sizes=[3, 5, 5, 3], act_fn="relu")
optim = optax.adam(1e-3)
opt_state = optim.init(
    (eqx.filter(model, eqx.is_array), None)
)

# perform one training step with PC
result = jpc.make_pc_step(
    model=model,
    optim=optim,
    opt_state=opt_state,
    output=y,
    input=x
)
```
Under the hood, `jpc.make_pc_step`
1. integrates the inference (activity) dynamics using a [Diffrax](https://github.com/patrick-kidger/diffrax) ODE solver, and
2. updates model parameters at the numerical solution of the activities with a given [Optax](https://github.com/google-deepmind/optax) optimiser.

> **NOTE**: All convenience training and test functions including `make_pc_step` 
> are already "jitted" (for increased performance) for the user's convenience.

## 🧠️ Predictive coding primer
...

## 🚀 Advanced usage
More advanced users can access any of the functionality used by `jpc.make_pc_step`.

```py
import jpc

# 1. initialise activities with a feedforward pass
activities0 = jpc.init_activities_with_ffwd(model=model, input=x)

# 2. run the inference dynamics to equilibrium
equilibrated_activities = jpc.solve_pc_inference(
    params=(model, None), 
    activities=activities0, 
    output=y, 
    input=x
)

# 3. update parameters with PC
step_result = jpc.update_params(
    params=(model, None), 
    activities=equilibrated_activities,
    optim=optim,
    opt_state=opt_state,
    output=y, 
    input=x
)
```

## 📄 Citation
If you found this library useful in your work, please cite (arXiv link):

```bibtex
@article{innocenti2024jpc,
    title={JPC: Flexible Inference for Predictive Coding Networks in JAX},
    author={Innocenti, Francesco and Kinghorn, Paul and Yun-Farmbrough, Will 
    and Singh, Ryan and De Llanza Varona, Miguel and Buckley, Christopher},
    journal={arXiv preprint},
    year={2024}
}
```
Also consider starring the project [on GitHub](https://github.com/thebuckleylab/jpc)! ⭐️ 

## ⏭️ Next steps
