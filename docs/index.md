# Getting started
JPC is a [**J**AX](https://github.com/google/jax) library for training neural 
networks with **P**redictive **C**oding (PC). 

JPC provides a **simple**, **fast** and **flexible** API for 
training of a variety of PCNs including discriminative, generative and hybrid 
models. 

* Like JAX, JPC is completely functional in design, and the core library code is 
<1000 lines of code. 

* Unlike existing implementations, JPC provides a wide range of optimisers, both 
discrete and continuous, to solve the inference dynamics of PC, including 
ordinary differential equation (ODE) solvers.

* JPC also provides some analytical tools that can be used to study and
potentially diagnose issues with PCNs.

If you're new to JPC, we recommend starting from the [
example notebooks](https://thebuckleylab.github.io/jpc/examples/discriminative_pc/)
and checking the [documentation](https://thebuckleylab.github.io/jpc/).

## 💻 Installation
Clone the repo and in the project's directory run
```
pip install .
```

Requires Python 3.10+ and JAX 0.4.38–0.5.2 (inclusive). For GPU usage, upgrade 
jax to the appropriate cuda version (12 as an example here).

```
pip install --upgrade "jax[cuda12]"
```

## ⚡️ Quick example
Use `jpc.make_pc_step()` to update the parameters of any neural network 
compatible with PC updates (see [examples](https://thebuckleylab.github.io/jpc/examples/discriminative_pc/))
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
model = jpc.make_mlp(
    key, 
    input_dim=3,
    width=50,
    depth=5,
    output_dim=3
    act_fn="relu"
)
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
model, opt_state = result["model"], result["opt_state"]
```
Under the hood, `jpc.make_pc_step()`

1. integrates the inference (activity) dynamics using a [diffrax](https://github.com/patrick-kidger/diffrax) ODE solver, and
2. updates model parameters at the numerical solution of the activities with a given [optax](https://github.com/google-deepmind/optax) optimiser.

> **NOTE**: All convenience training and test functions such as `make_pc_step()` 
> are already "jitted" (for optimised performance) for the user's convenience.

## 🚀 Advanced usage
Advanced users can access all the underlying functions of `jpc.make_pc_step()` 
as well as additional features. A custom PC training step looks like the 
following:
```py
import jpc

# 1. initialise activities with a feedforward pass
activities = jpc.init_activities_with_ffwd(model=model, input=x)

# 2. perform inference (state optimisation)
activity_opt_state = activity_optim.init(activities)
for _ in range(len(model)):
    activity_update_result = jpc.update_pc_activities(
        params=(model, None),
        activities=activities,
        optim=activity_optim,
        opt_state=activity_opt_state,
        output=y,
        input=x
    )
    activities = activity_update_result["activities"]
    activity_opt_state = activity_update_result["opt_state"]

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
If you found this library useful in your work, please cite ([paper link](https://arxiv.org/abs/2412.03676)):

```bibtex
@article{innocenti2024jpc,
  title={JPC: Flexible Inference for Predictive Coding Networks in JAX},
  author={Innocenti, Francesco and Kinghorn, Paul and Yun-Farmbrough, Will and Varona, Miguel De Llanza and Singh, Ryan and Buckley, Christopher L},
  journal={arXiv preprint arXiv:2412.03676},
  year={2024}
}
```
Also consider starring the project [on GitHub](https://github.com/thebuckleylab/jpc)! ⭐️ 

## 🙏 Acknowledgements
We are grateful to Patrick Kidger for early advice on how to use Diffrax.

## See also: other PC libraries
* [ngc-learn](https://github.com/NACLab/ngc-learn) (jax & pytorch)
* [pcx](https://github.com/liukidar/pcx) (jax)
* [pyhgf](https://github.com/ComputationalPsychiatry/pyhgf) (jax)
* [Torch2PC](https://github.com/RobertRosenbaum/Torch2PC) (pytorch)
* [pypc](https://github.com/infer-actively/pypc) (pytorch)
* [pybrid](https://github.com/alec-tschantz/pybrid) (pytorch)
