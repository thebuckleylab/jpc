"""Tests for update functions."""

import pytest
import jax
import jax.numpy as jnp
import optax
from jpc import (
    update_pc_activities,
    update_pc_params,
    update_bpc_activities,
    update_bpc_params
)


def test_update_pc_activities(simple_model, x, y):
    """Test PC activity updates."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init(activities)
    
    result = update_pc_activities(
        params=(simple_model, None),
        activities=activities,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp"
    )
    
    assert "energy" in result
    assert "activities" in result
    assert "grads" in result
    assert "opt_state" in result
    assert len(result["activities"]) == len(activities)
    assert jnp.isfinite(result["energy"])


def test_update_pc_activities_unsupervised(simple_model, y, key, layer_sizes, batch_size):
    """Test PC activity updates in unsupervised mode."""
    from jpc import init_activities_from_normal
    
    activities = init_activities_from_normal(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=0.05
    )
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init(activities)
    
    result = update_pc_activities(
        params=(simple_model, None),
        activities=activities,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=None,
        loss_id="mse",
        param_type="sp"
    )
    
    assert "energy" in result
    assert len(result["activities"]) == len(activities)


def test_update_pc_params(simple_model, x, y):
    """Test PC parameter updates."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init((simple_model, None))
    
    result = update_pc_params(
        params=(simple_model, None),
        activities=activities,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp"
    )
    
    assert "model" in result
    assert "skip_model" in result
    assert "grads" in result
    assert "opt_state" in result
    assert len(result["model"]) == len(simple_model)


def test_update_pc_params_with_regularization(simple_model, x, y):
    """Test PC parameter updates with regularization."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init((simple_model, None))
    
    result = update_pc_params(
        params=(simple_model, None),
        activities=activities,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp",
        weight_decay=0.01,
        spectral_penalty=0.01,
        activity_decay=0.01
    )
    
    assert "model" in result
    assert len(result["model"]) == len(simple_model)


def test_update_bpc_activities(key, x, y, input_dim, hidden_dim, output_dim, depth):
    """Test BPC activity updates."""
    from jpc import make_mlp, init_activities_from_normal
    
    top_down_model = make_mlp(
        key=key,
        input_dim=input_dim,
        width=hidden_dim,
        depth=depth,
        output_dim=output_dim,
        act_fn="relu",
        use_bias=False,
        param_type="sp"
    )
    
    # Bottom-up model structure for BPC (see test_energies.py for details)
    import equinox.nn as nn
    subkeys = jax.random.split(jax.random.PRNGKey(123), depth + 1)
    bottom_up_model = []
    bottom_up_model.append(nn.Sequential([nn.Lambda(jax.nn.relu), nn.Linear(hidden_dim, input_dim, use_bias=False, key=subkeys[0])]))
    for i in range(1, depth - 1):
        bottom_up_model.append(nn.Sequential([nn.Lambda(jax.nn.relu), nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=subkeys[i])]))
    bottom_up_model.append(nn.Sequential([nn.Lambda(jax.nn.relu), nn.Linear(output_dim, hidden_dim, use_bias=False, key=subkeys[depth-1])]))
    
    # Activities should only include hidden layers (not input/output)
    hidden_layer_sizes = [hidden_dim] * (depth - 1)
    activities = init_activities_from_normal(
        key=jax.random.PRNGKey(456),
        layer_sizes=hidden_layer_sizes,
        mode="unsupervised",
        batch_size=x.shape[0],
        sigma=0.05
    )
    
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init(activities)
    
    result = update_bpc_activities(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=x,
        param_type="sp"
    )
    
    assert "energy" in result
    assert "activities" in result
    assert "grads" in result
    assert "opt_state" in result
    assert len(result["activities"]) == len(activities)


def test_update_bpc_params(key, x, y, input_dim, hidden_dim, output_dim, depth):
    """Test BPC parameter updates."""
    from jpc import make_mlp, init_activities_from_normal
    
    top_down_model = make_mlp(
        key=key,
        input_dim=input_dim,
        width=hidden_dim,
        depth=depth,
        output_dim=output_dim,
        act_fn="relu",
        use_bias=False,
        param_type="sp"
    )
    
    # Bottom-up model structure for BPC (see test_energies.py for details)
    import equinox.nn as nn
    subkeys = jax.random.split(jax.random.PRNGKey(123), depth + 1)
    bottom_up_model = []
    bottom_up_model.append(nn.Sequential([nn.Lambda(jax.nn.relu), nn.Linear(hidden_dim, input_dim, use_bias=False, key=subkeys[0])]))
    for i in range(1, depth - 1):
        bottom_up_model.append(nn.Sequential([nn.Lambda(jax.nn.relu), nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=subkeys[i])]))
    bottom_up_model.append(nn.Sequential([nn.Lambda(jax.nn.relu), nn.Linear(output_dim, hidden_dim, use_bias=False, key=subkeys[depth-1])]))
    
    # Activities should only include hidden layers (not input/output)
    hidden_layer_sizes = [hidden_dim] * (depth - 1)
    activities = init_activities_from_normal(
        key=jax.random.PRNGKey(456),
        layer_sizes=hidden_layer_sizes,
        mode="unsupervised",
        batch_size=x.shape[0],
        sigma=0.05
    )
    
    top_down_optim = optax.sgd(learning_rate=0.01)
    bottom_up_optim = optax.sgd(learning_rate=0.01)
    top_down_opt_state = top_down_optim.init(top_down_model)
    bottom_up_opt_state = bottom_up_optim.init(bottom_up_model)
    
    result = update_bpc_params(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        top_down_optim=top_down_optim,
        bottom_up_optim=bottom_up_optim,
        top_down_opt_state=top_down_opt_state,
        bottom_up_opt_state=bottom_up_opt_state,
        output=y,
        input=x,
        param_type="sp"
    )
    
    assert "models" in result
    assert "grads" in result
    assert "opt_states" in result
    assert len(result["models"]) == 2
    assert len(result["models"][0]) == len(top_down_model)
    assert len(result["models"][1]) == len(bottom_up_model)

