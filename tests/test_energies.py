"""Tests for energy functions."""

import pytest
import jax
import jax.numpy as jnp
from jpc import (
    pc_energy_fn,
    hpc_energy_fn,
    bpc_energy_fn,
    _get_param_scalings
)


def test_pc_energy_fn_supervised(simple_model, x, y):
    """Test PC energy function in supervised mode."""
    from jpc import init_activities_with_ffwd
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    energy = pc_energy_fn(
        params=(simple_model, None),
        activities=activities,
        y=y,
        x=x,
        loss="mse",
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)
    assert energy >= 0


def test_pc_energy_fn_unsupervised(simple_model, y, key, layer_sizes, batch_size):
    """Test PC energy function in unsupervised mode."""
    from jpc import init_activities_from_normal
    
    activities = init_activities_from_normal(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=0.05
    )
    
    energy = pc_energy_fn(
        params=(simple_model, None),
        activities=activities,
        y=y,
        x=None,
        loss="mse",
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)
    assert energy >= 0


def test_pc_energy_fn_cross_entropy(simple_model, x, y_onehot):
    """Test PC energy function with cross-entropy loss."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    energy = pc_energy_fn(
        params=(simple_model, None),
        activities=activities,
        y=y_onehot,
        x=x,
        loss="ce",
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)


def test_pc_energy_fn_with_regularization(simple_model, x, y):
    """Test PC energy function with regularization terms."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    energy = pc_energy_fn(
        params=(simple_model, None),
        activities=activities,
        y=y,
        x=x,
        loss="mse",
        param_type="sp",
        weight_decay=0.01,
        spectral_penalty=0.01,
        activity_decay=0.01
    )
    
    assert jnp.isfinite(energy)
    assert energy >= 0


def test_pc_energy_fn_record_layers(simple_model, x, y):
    """Test PC energy function with layer-wise recording."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    energies = pc_energy_fn(
        params=(simple_model, None),
        activities=activities,
        y=y,
        x=x,
        loss="mse",
        param_type="sp",
        record_layers=True
    )
    
    assert len(energies) == len(simple_model)
    assert all(jnp.isfinite(e) for e in energies)


def test_pc_energy_fn_different_param_types(key, x, y, input_dim, hidden_dim, output_dim, depth):
    """Test PC energy function with different parameter types."""
    from jpc import make_mlp, init_activities_with_ffwd
    
    for param_type in ["sp", "mupc", "ntk"]:  # Note: library uses "ntk" not "ntp"
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
        
        activities = init_activities_with_ffwd(model, x, param_type=param_type)
        
        energy = pc_energy_fn(
            params=(model, None),
            activities=activities,
            y=y,
            x=x,
            loss="mse",
            param_type=param_type
        )
        
        assert jnp.isfinite(energy)


def test_hpc_energy_fn(key, simple_model, x, y, output_dim, hidden_dim, input_dim, depth):
    """Test HPC energy function."""
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
    
    energy = hpc_energy_fn(
        model=amortiser,
        equilib_activities=equilib_activities,
        amort_activities=amort_activities,
        x=y,
        y=x
    )
    
    assert jnp.isfinite(energy)
    assert energy >= 0


def test_bpc_energy_fn(key, x, y, input_dim, hidden_dim, output_dim, depth):
    """Test BPC energy function."""
    from jpc import make_mlp, init_activities_from_normal
    
    # For BPC, we need matching dimensions between top-down and bottom-up
    # Top-down: input_dim -> hidden_dim -> ... -> output_dim
    # Bottom-up: output_dim -> hidden_dim -> ... -> input_dim
    # Activities should be for hidden layers only (excluding input and output)
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
    
    # Bottom-up model structure for BPC:
    # bottom_up_model[0]: hidden_dim -> input_dim
    # bottom_up_model[1] to [H-1]: hidden_dim -> hidden_dim  
    # bottom_up_model[H]: output_dim -> hidden_dim
    # We need to create this manually since make_mlp doesn't support this structure
    import equinox.nn as nn
    subkeys = jax.random.split(jax.random.PRNGKey(123), depth + 1)
    bottom_up_model = []
    # First layer: hidden_dim -> input_dim
    bottom_up_model.append(nn.Sequential([nn.Lambda(jax.nn.relu), nn.Linear(hidden_dim, input_dim, use_bias=False, key=subkeys[0])]))
    # Middle layers: hidden_dim -> hidden_dim
    for i in range(1, depth - 1):
        bottom_up_model.append(nn.Sequential([nn.Lambda(jax.nn.relu), nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=subkeys[i])]))
    # Last layer: output_dim -> hidden_dim
    bottom_up_model.append(nn.Sequential([nn.Lambda(jax.nn.relu), nn.Linear(output_dim, hidden_dim, use_bias=False, key=subkeys[depth-1])]))
    
    # Activities should only include hidden layers (not input/output)
    # For depth=3: input, hidden1, hidden2, output -> activities = [hidden1, hidden2]
    hidden_layer_sizes = [hidden_dim] * (depth - 1)
    activities = init_activities_from_normal(
        key=jax.random.PRNGKey(456),
        layer_sizes=hidden_layer_sizes,
        mode="unsupervised",
        batch_size=x.shape[0],
        sigma=0.05
    )
    
    energy = bpc_energy_fn(
        top_down_model=top_down_model,
        bottom_up_model=bottom_up_model,
        activities=activities,
        y=y,
        x=x,
        param_type="sp"
    )
    
    assert jnp.isfinite(energy)
    assert energy >= 0


def test_get_param_scalings(simple_model, x):
    """Test parameter scaling function."""
    scalings = _get_param_scalings(
        model=simple_model,
        input=x,
        skip_model=None,
        param_type="sp"
    )
    
    assert len(scalings) == len(simple_model)
    assert all(isinstance(s, (int, float)) for s in scalings)


def test_get_param_scalings_with_skip(key, x, input_dim, hidden_dim, output_dim, depth):
    """Test parameter scaling with skip connections."""
    from jpc import make_mlp, make_skip_model
    
    model = make_mlp(
        key=key,
        input_dim=input_dim,
        width=hidden_dim,
        depth=depth,
        output_dim=output_dim,
        act_fn="relu",
        use_bias=False,
        param_type="sp"
    )
    
    skip_model = make_skip_model(len(model))
    
    scalings = _get_param_scalings(
        model=model,
        input=x,
        skip_model=skip_model,
        param_type="sp"
    )
    
    assert len(scalings) == len(model)
