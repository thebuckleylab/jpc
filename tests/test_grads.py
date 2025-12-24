"""Tests for gradient computation functions."""

import pytest
import jax
import jax.numpy as jnp
from jpc import (
    neg_pc_activity_grad,
    compute_pc_activity_grad,
    compute_pc_param_grads,
    compute_hpc_param_grads,
    compute_bpc_activity_grad,
    compute_bpc_param_grads,
    compute_epc_error_grad,
    compute_epc_param_grads
)


def test_neg_pc_activity_grad(simple_model, x, y):
    """Test negative PC activity gradient."""
    from jpc import init_activities_with_ffwd
    from diffrax import PIDController
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    args = (
        (simple_model, None),
        y,
        x,
        "mse",
        "sp",
        0.0,
        0.0,
        0.0,
        None,
        PIDController(rtol=1e-3, atol=1e-3)
    )
    
    grads = neg_pc_activity_grad(0.0, activities, args)
    
    assert len(grads) == len(activities)
    for grad, act in zip(grads, activities):
        assert grad.shape == act.shape


def test_compute_pc_activity_grad(simple_model, x, y):
    """Test PC activity gradient computation."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    energy, grads = compute_pc_activity_grad(
        params=(simple_model, None),
        activities=activities,
        y=y,
        x=x,
        loss_id="mse",
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)
    assert len(grads) == len(activities)
    for grad, act in zip(grads, activities):
        assert grad.shape == act.shape
        assert jnp.all(jnp.isfinite(grad))


def test_compute_pc_activity_grad_unsupervised(simple_model, y, key, layer_sizes, batch_size):
    """Test PC activity gradient in unsupervised mode."""
    from jpc import init_activities_from_normal
    
    activities = init_activities_from_normal(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=0.05
    )
    
    energy, grads = compute_pc_activity_grad(
        params=(simple_model, None),
        activities=activities,
        y=y,
        x=None,
        loss_id="mse",
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)
    assert len(grads) == len(activities)


def test_compute_pc_param_grads(simple_model, x, y):
    """Test PC parameter gradient computation."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    grads = compute_pc_param_grads(
        params=(simple_model, None),
        activities=activities,
        y=y,
        x=x,
        loss_id="mse",
        param_type="sp"
    )
    
    model_grads, skip_grads = grads
    assert len(model_grads) == len(simple_model)
    assert skip_grads is None or len(skip_grads) == len(simple_model)


def test_compute_pc_param_grads_with_regularization(simple_model, x, y):
    """Test PC parameter gradients with regularization."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    grads = compute_pc_param_grads(
        params=(simple_model, None),
        activities=activities,
        y=y,
        x=x,
        loss_id="mse",
        param_type="sp",
        weight_decay=0.01,
        spectral_penalty=0.01,
        activity_decay=0.01
    )
    
    model_grads, skip_grads = grads
    assert len(model_grads) == len(simple_model)


def test_compute_hpc_param_grads(key, simple_model, x, y, output_dim, hidden_dim, input_dim, depth):
    """Test HPC parameter gradient computation."""
    from jpc import make_mlp, init_activities_with_ffwd, init_activities_with_amort
    
    generator = simple_model
    amortiser = make_mlp(
        key=key,
        input_dim=output_dim,
        width=hidden_dim,
        depth=depth,
        output_dim=input_dim,
        act_fn="relu",
        use_bias=False,
        param_type="sp"
    )
    
    equilib_activities = init_activities_with_ffwd(generator, x, param_type="sp")
    amort_activities = init_activities_with_amort(amortiser, generator, y)
    
    grads = compute_hpc_param_grads(
        model=amortiser,
        equilib_activities=equilib_activities,
        amort_activities=amort_activities,
        x=y,
        y=x
    )
    
    assert len(grads) == len(amortiser)


def test_compute_bpc_activity_grad(key, x, y, input_dim, hidden_dim, output_dim, depth):
    """Test BPC activity gradient computation."""
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
    
    energy, grads = compute_bpc_activity_grad(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=y,
        x=x,
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)
    assert len(grads) == len(activities)


def test_compute_bpc_activity_grad_pdm_mode(key, x, y, input_dim, hidden_dim, output_dim, depth):
    """Test BPC activity gradient in PDM mode."""
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
    
    energy, grads = compute_bpc_activity_grad(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=y,
        x=x,
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)
    assert len(grads) == len(activities)


def test_compute_bpc_param_grads(key, x, y, input_dim, hidden_dim, output_dim, depth):
    """Test BPC parameter gradient computation."""
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
    
    top_down_grads, bottom_up_grads = compute_bpc_param_grads(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=y,
        x=x,
        param_type="sp"
    )
    
    assert len(top_down_grads) == len(top_down_model)
    assert len(bottom_up_grads) == len(bottom_up_model)


def test_compute_epc_error_grad_supervised(simple_model, x, y, layer_sizes):
    """Test EPC error gradient computation in supervised mode."""
    from jpc import init_epc_errors
    
    errors = init_epc_errors(
        layer_sizes=layer_sizes,
        batch_size=x.shape[0],
        mode="supervised"
    )
    
    energy, grads = compute_epc_error_grad(
        params=(simple_model, None),
        errors=errors,
        y=y,
        x=x,
        loss_id="mse",
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)
    assert len(grads) == len(errors)
    for grad, err in zip(grads, errors):
        assert grad.shape == err.shape
        assert jnp.all(jnp.isfinite(grad))


def test_compute_epc_error_grad_cross_entropy(simple_model, x, y_onehot, layer_sizes):
    """Test EPC error gradient with cross-entropy loss."""
    from jpc import init_epc_errors
    
    errors = init_epc_errors(
        layer_sizes=layer_sizes,
        batch_size=x.shape[0],
        mode="supervised"
    )
    
    energy, grads = compute_epc_error_grad(
        params=(simple_model, None),
        errors=errors,
        y=y_onehot,
        x=x,
        loss_id="ce",
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)
    assert len(grads) == len(errors)


def test_compute_epc_param_grads_supervised(simple_model, x, y, layer_sizes):
    """Test EPC parameter gradient computation in supervised mode."""
    from jpc import init_epc_errors
    
    errors = init_epc_errors(
        layer_sizes=layer_sizes,
        batch_size=x.shape[0],
        mode="supervised"
    )
    
    grads = compute_epc_param_grads(
        params=(simple_model, None),
        errors=errors,
        y=y,
        x=x,
        loss_id="mse",
        param_type="sp"
    )
    
    model_grads, skip_grads = grads
    assert len(model_grads) == len(simple_model)
    assert skip_grads is None or len(skip_grads) == len(simple_model)


def test_compute_epc_param_grads_different_param_types(key, x, y, input_dim, hidden_dim, output_dim, depth):
    """Test EPC parameter gradients with different parameter types."""
    from jpc import make_mlp, init_epc_errors
    
    for param_type in ["sp", "mupc", "ntp"]:
        model = make_mlp(
            key=key,
            input_dim=input_dim,
            width=hidden_dim,
            depth=depth,
            output_dim=output_dim,
            act_fn="relu",
            use_bias=False,
            param_type=param_type
        )
        
        layer_sizes = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
        errors = init_epc_errors(
            layer_sizes=layer_sizes,
            batch_size=x.shape[0],
            mode="supervised"
        )
        
        grads = compute_epc_param_grads(
            params=(model, None),
            errors=errors,
            y=y,
            x=x,
            loss_id="mse",
            param_type=param_type
        )
        
        model_grads, skip_grads = grads
        assert len(model_grads) == len(model)

