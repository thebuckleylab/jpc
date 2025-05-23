JPC provides two types of API depending on the use case:

* a simple, high-level API that allows to train and test models with predictive 
coding in a few lines of code, and
* a more advanced API offering greater flexibility as well as additional features.

# Basic usage
At a high level, JPC provides a single convenience function `jpc.make_pc_step()` 
to update the parameters of a neural network with PC.
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
update_result = jpc.make_pc_step(
    model=model,
    optim=optim,
    opt_state=opt_state,
    output=y,
    input=x
)

# updated model and optimiser
model, opt_state = update_result["model"], update_result["opt_state"]
```
As shown above, at a minimum `jpc.make_pc_step()` takes a model, an [optax
](https://github.com/google-deepmind/optax) optimiser and its 
state, and some data. The model needs to be compatible with PC updates in the 
sense that it's split into callable layers (see the 
[example notebooks
](https://thebuckleylab.github.io/jpc/examples/discriminative_pc/)). Also note 
that the `input` is actually not needed for unsupervised training. In fact, 
`jpc.make_pc_step()` can be used for classification and generation tasks, for 
supervised as well as unsupervised training (again see the [example notebooks
](https://thebuckleylab.github.io/jpc/examples/discriminative_pc/)). 

Under the hood, `jpc.make_pc_step()` uses [diffrax
](https://github.com/patrick-kidger/diffrax) to solve the activity (inference) 
dynamics of PC. Many default arguments, for example related to the ODE solver,
can be changed, including the ODE solver, and there is an option to record a 
variety of metrics such as loss, accuracy, and energies. See the [docs
](https://thebuckleylab.github.io/jpc/api/Training/#jpc.make_pc_step) for more 
details.

A similar convenience function `jpc.make_hpc_step()` is provided for updating the
parameters of a hybrid PCN ([Tschantz et al., 2023
](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011280)).
```py
import jax.random as jr
import equinox as eqx
import optax
import jpc

# models
key = jr.PRNGKey(0)
subkeys = jr.split(key, 2)

input_dim, output_dim = 10, 3
width, depth = 100, 5
generator = jpc.make_mlp(
    subkeys[0], 
    input_dim=input_dim,
    width=width,
    depth=depth,
    output_dim=output_dim
    act_fn="tanh"
)
# NOTE that the input and output of the amortiser are reversed
amortiser = jpc.make_mlp(
    subkeys[0], 
    input_dim=output_dim,
    width=width,
    depth=depth,
    output_dim=input_dim
    act_fn="tanh"
)

# optimisers
gen_optim = optax.adam(1e-3)
amort_optim = optax.adam(1e-3)
gen_pt_state = gen_optim.init(
    (eqx.filter(generator, eqx.is_array), None)
)
amort_opt_state = amort_optim.init(
    eqx.filter(amortiser, eqx.is_array)
)

update_result = jpc.make_hpc_step(
    generator=generator,
    amortiser=amortiser,
    optims=[gen_optim, amort_optim],
    opt_states=[gen_opt_state, amort_opt_state],
    output=y,
    input=x
)
generator, amortiser = update_result["generator"], update_result["amortiser"]
opt_states = update_result["opt_states"]
gen_loss, amort_loss = update_result["losses"]
```
See the [docs
](https://thebuckleylab.github.io/jpc/api/Training/#jpc.make_hpc_step) and the
[example notebook
](https://thebuckleylab.github.io/jpc/examples/hybrid_pc/) for more details.
