import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves
from equinox import tree_at
import equinox.nn as nn
from jpc import pc_energy_fn, _check_param_type
from jaxtyping import PRNGKeyArray, PyTree, ArrayLike, Scalar, Array
from jaxlib.xla_extension import PjitFunction
from typing import Callable, Optional, Tuple
from dataclasses import dataclass


_ACT_FNS = [
    "linear", "tanh", "hard_tanh", "relu", "leaky_relu", "gelu", "selu", "silu"
]


def get_act_fn(name: str) -> Callable:
    if name == "linear":
        return nn.Identity()
    elif name == "tanh":
        return jnp.tanh
    elif name == "hard_tanh":
        return jax.nn.hard_tanh
    elif name == "relu":
        return jax.nn.relu
    elif name == "leaky_relu":
        return jax.nn.leaky_relu
    elif name == "gelu":
        return jax.nn.gelu
    elif name == "selu":
        return jax.nn.selu
    elif name == "silu":
        return jax.nn.silu
    else:
        raise ValueError(f"""
                Invalid activation function ID. Options are {_ACT_FNS}.
        """)


def make_mlp(
        key: PRNGKeyArray, 
        input_dim: int, 
        width: int,
        depth: int, 
        output_dim: int, 
        act_fn: str, 
        use_bias: bool = False,
        param_type: str = "sp"
    ) -> PyTree[Callable]:
    """Creates a multi-layer perceptron compatible with predictive coding updates.

    !!! note

        This implementation places the activation function before the linear 
        transformation, $\mathbf{W}_\ell \phi(\mathbf{z}_{\ell-1})$, for 
        compatibility with the [μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1)) 
        scalings when `param_type = "mupc"` in functions including 
        [`jpc.init_activities_with_ffwd()`](https://thebuckleylab.github.io/jpc/api/Initialisation/#jpc.init_activities_with_ffwd), 
        [`jpc.update_activities()`](https://thebuckleylab.github.io/jpc/api/Discrete%20updates/#jpc.update_activities), 
        and [`jpc.update_params()`](https://thebuckleylab.github.io/jpc/api/Discrete%20updates/#jpc.update_params).

    **Main arguments:**

    - `key`: `jax.random.PRNGKey` for parameter initialisation.
    - `input_dim`: Input dimension.
    - `width`: Network width.
    - `depth`: Network depth.
    - `output_dim`: Output dimension.
    - `act_fn`: Activation function (for all layers except the output).
    - `use_bias`: `False` by default.
    - `param_type`: Determines the parameterisation. Options are `"sp"` 
        (standard parameterisation), `"mupc"` ([μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1))), 
        or `"ntp"` (neural tangent parameterisation). See [`jpc._get_param_scalings()`](https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) 
        for the specific scalings of these different parameterisations. Defaults
        to `"sp"`.

    **Returns:**

    List of callable fully connected layers.

    """
    _check_param_type(param_type)
    
    subkeys = jax.random.split(key, depth)
    layers = []
    for i in range(depth):
        act_fn_l = nn.Identity() if i == 0 else get_act_fn(act_fn)
        _in = input_dim if i == 0 else width
        _out = output_dim if (i + 1) == depth else width

        linear = nn.Linear(
            _in,
            _out,
            use_bias=use_bias,
            key=subkeys[i]
        )
        if param_type == "mupc":
            W = jax.random.normal(subkeys[i], linear.weight.shape)
            linear = tree_at(lambda l: l.weight, linear, W)

        layers.append(
            nn.Sequential(
                [nn.Lambda(act_fn_l), linear]
            )
        )

    return layers


def make_skip_model(depth: int) -> PyTree[Callable]:
    """Creates a residual network with one-layer skip connections at every layer 
    except from the input to the next layer and from the penultimate layer to 
    the output.

    This is used for compatibility with the [μPC](https://openreview.net/forum?id=lSLSzYuyfX&referrer=%5Bthe%20profile%20of%20Francesco%20Innocenti%5D(%2Fprofile%3Fid%3D~Francesco_Innocenti1)) 
    parameterisation when `param_type = "mupc"` in functions including 
    [`jpc.init_activities_with_ffwd()`](https://thebuckleylab.github.io/jpc/api/Initialisation/#jpc.init_activities_with_ffwd), 
    [`jpc.update_activities()`](https://thebuckleylab.github.io/jpc/api/Discrete%20updates/#jpc.update_activities), 
    and [`jpc.update_params()`](https://thebuckleylab.github.io/jpc/api/Discrete%20updates/#jpc.update_params).
    """
    skips = [None] * depth
    for l in range(1, depth-1):
        skips[l] = nn.Lambda(nn.Identity())
        
    return skips


def mse_loss(preds: ArrayLike, labels: ArrayLike) -> Scalar:
    return 0.5 * jnp.mean(jnp.sum((labels - preds) ** 2, axis=1))


def cross_entropy_loss(logits: ArrayLike, labels: ArrayLike) -> Scalar:
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jnp.log(probs)
    return - jnp.mean(jnp.sum(labels * log_probs, axis=-1))


def compute_accuracy(truths: ArrayLike, preds: ArrayLike) -> Scalar:
    return jnp.mean(
        jnp.argmax(truths, axis=1) == jnp.argmax(preds, axis=1)
    ) * 100


def get_t_max(activities_iters: PyTree[Array]) -> Array:
    return jnp.argmax(activities_iters[0][:, 0, 0]) - 1


def compute_infer_energies(
        params: Tuple[PyTree[Callable], Optional[PyTree[Callable]]],
        activities_iters: PyTree[Array],
        t_max: Array,
        y: ArrayLike,
        *,
        x: Optional[ArrayLike] = None,
        loss: str = "mse",
        param_type: str = "sp",
        weight_decay: Scalar = 0.,
        spectral_penalty: Scalar = 0.,
        activity_decay: Scalar = 0.
) -> PyTree[Scalar]:
    """Calculates layer energies during predictive coding inference.

    **Main arguments:**

    - `params`: Tuple with callable model layers and optional skip connections.
    - `activities_iters`: Layer-wise activities at every inference iteration.
        Note that each set of activities will have 4096 steps as first
        dimension by diffrax default.
    - `t_max`: Maximum number of inference iterations to compute energies for.
    - `y`: Observation or target of the generative model.

    **Other arguments:**
    
    - `x`: Optional prior of the generative model.
    - `loss`: Loss function to use at the output layer (mean squared error
        `"mse"` vs cross-entropy `"ce"`).
    - `param_type`: Determines the parameterisation. Options are `"sp"`, 
        `"mupc"`, or `"ntp"`.
    - `weight_decay`: Weight decay for the weights.
    - `spectral_penalty`: Spectral penalty for the weights.
    - `activity_decay`: Activity decay for the activities.

    **Returns:**

    List of layer-wise energies at every inference iteration.

    """
    model, _ = params

    def loop_body(state):
        t, energies_iters = state

        energies = pc_energy_fn(
            params=params,
            activities=tree_map(lambda act: act[t], activities_iters),
            y=y,
            x=x,
            loss=loss,
            param_type=param_type,
            weight_decay=weight_decay,
            spectral_penalty=spectral_penalty,
            activity_decay=activity_decay,
            record_layers=True
        )
        energies_iters = energies_iters.at[:, t].set(energies)
        return t + 1, energies_iters

    # for memory reasons, we set 500 as the max iters to record
    energies_iters = jnp.zeros((len(model), 500))
    _, energies_iters = jax.lax.while_loop(
        lambda state: state[0] < t_max,
        loop_body,
        (0, energies_iters)
    )
    return energies_iters[::-1, :]


def compute_activity_norms(activities: PyTree[Array]) -> Array:
    """Calculates $\ell^2$ norm of activities at each layer."""
    return jnp.array([
        jnp.mean(
            jnp.linalg.norm(
                a,
                axis=-1,
                ord=2
            )
        ) for a in tree_leaves(activities)
    ])


def compute_param_norms(params):
    """Calculates $\ell^2$ norm of all model parameters."""
    def process_model_params(model_params):
        norms = []
        for p in tree_leaves(model_params):
            if p is None or isinstance(p, PjitFunction):
                norms.append(0.)
            elif callable(p) and not hasattr(p, 'shape'):
                # Skip callable functions (like Lambda-wrapped activations) that don't have shape
                # But keep arrays which might be callable in some JAX contexts
                norms.append(0.)
            else:
                try:
                    # Check if p is a JAX array-like object
                    if hasattr(p, 'shape') and hasattr(p, 'dtype'):
                        norms.append(jnp.linalg.norm(jnp.ravel(p), ord=2))
                    else:
                        norms.append(0.)
                except (TypeError, AttributeError):
                    # If ravel fails, it's not an array
                    norms.append(0.)
        return jnp.array(norms)

    model_params, skip_model_params = params
    model_norms = process_model_params(model_params)
    skip_model_norms = (process_model_params(skip_model_params) if
                        skip_model_params is not None else None)

    return model_norms, skip_model_norms
