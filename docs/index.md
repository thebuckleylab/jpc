# Getting started
JPC is a [**J**AX](https://github.com/google/jax) library for training neural 
networks with **P**redictive **C**oding (PC). It is built on top of three main 
libraries:

* [Equinox](https://github.com/patrick-kidger/equinox), to define neural 
networks with PyTorch-like syntax,
* [Diffrax](https://github.com/patrick-kidger/diffrax), to solve the PC inference (activity) dynamics, and
* [Optax](https://github.com/google-deepmind/optax), for parameter optimisation.

JPC provides a **simple**, **relatively fast** and **flexible** API for 
training of a variety of PCNs including discriminative, generative and hybrid 
models.

* Like JAX, JPC is completely functional, and the core library is <1000 lines 
of code. 
* Unlike existing implementations, JPC leverages ordinary differential 
equation (ODE) solvers to integrate the inference dynamics of PC networks 
(PCNs), which we find can provide significant speed-ups compared to standard 
optimisers, especially for deeper models. 
* JPC also provides some analytical tools that can be used to study and 
diagnose issues with PCNs.

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

# updated model and optimiser
model = result["model"]
optim, opt_state = result["optim"], result["opt_state"]
```
Under the hood, `jpc.make_pc_step`

1. integrates the inference (activity) dynamics using a [Diffrax](https://github.com/patrick-kidger/diffrax) ODE solver, and
2. updates model parameters at the numerical solution of the activities with a given [Optax](https://github.com/google-deepmind/optax) optimiser.

> **NOTE**: All convenience training and test functions including `make_pc_step` 
> are already "jitted" (for increased performance) for the user's convenience.

## 🧠️ Predictive coding primer
...

## 🚀 Advanced usage
Advanced users can access all the underlying functions of `jpc.make_pc_step` as 
well as additional features. A custom PC training step looks like the following:
```py
import jpc

# 1. initialise activities with a feedforward pass
activities = jpc.init_activities_with_ffwd(model=model, input=x)

# 2. run inference to equilibrium
equilibrated_activities = jpc.solve_inference(
    params=(model, None), 
    activities=activities, 
    output=y, 
    input=x
)

# 3. update parameters at the activities' solution with PC
result = jpc.update_params(
    params=(model, None), 
    activities=equilibrated_activities,
    optim=optim,
    opt_state=opt_state,
    output=y, 
    input=x
)
```
which can be embedded in a jitted function with any other additional 
computations.

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
