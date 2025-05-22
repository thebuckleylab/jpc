# Advanced usage

Advanced users can access all the underlying functions of `jpc.make_pc_step()` 
as well as additional features. A custom PC training step looks like the 
following:
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
param_update_result = jpc.update_params(
    params=(model, None), 
    activities=equilibrated_activities,
    optim=param_optim,
    opt_state=param_opt_state,
    output=y, 
    input=x
)

# updated model and optimiser
model = param_update_result["model"]
param_opt_state = param_update_result["opt_state"]
```
which can be embedded in a jitted function with any other additional 
computations. One can also use any [optax
](https://optax.readthedocs.io/en/latest/api/optimizers.html) optimiser to 
equilibrate the inference dynamics by replacing the function in step 2, as 
shown below.
```py
activity_optim = optax.adam(1e-3)

# 1. initialise activities
...

# 2. infer with adam
activity_opt_state = activity_optim.init(activities)

for t in range(T):
    activity_update_result = jpc.update_activities(
        params=(model, None),
        activities=activities,
        optim=activity_optim,
        opt_state=activity_opt_state,
        output=y,
        input=x
    )
    # updated activities and optimiser
    activities = activity_update_result["activities"]
    activity_opt_state = activity_update_result["opt_state"]

# 3. update parameters at the activities' solution with PC
...
```
See the [updates docs
](https://thebuckleylab.github.io/jpc/api/Updates/) for more details. JPC also 
comes with some analytical tools that can be used to study and potentially 
diagnose issues with PCNs 
(see [docs
](https://thebuckleylab.github.io/jpc/api/Analytical%20tools/) 
and [example notebook
](https://thebuckleylab.github.io/jpc/examples/theoretical_energy_with_linear_net/)).
