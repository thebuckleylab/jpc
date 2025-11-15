"""Tests for initialization functions."""

import pytest
import jax
import jax.numpy as jnp
from jpc import (
    init_activities_with_ffwd,
    init_activities_from_normal,
    init_activities_with_amort
)


def test_init_activities_with_ffwd(simple_model, x):
    """Test feedforward initialization."""
    activities = init_activities_with_ffwd(
        model=simple_model,
        input=x,
        param_type="sp"
    )
    
    assert len(activities) == len(simple_model)
    assert activities[0].shape == (x.shape[0], simple_model[0][1].weight.shape[0])
    assert activities[-1].shape == (x.shape[0], simple_model[-1][1].weight.shape[0])


def test_init_activities_with_ffwd_mupc(key, x, input_dim, hidden_dim, output_dim, depth):
    """Test feedforward initialization with mupc parameterization."""
    from jpc import make_mlp
    
    model = make_mlp(
        key=key,
        input_dim=input_dim,
        width=hidden_dim,
        depth=depth,
        output_dim=output_dim,
        act_fn="relu",
        use_bias=False,
        param_type="mupc"
    )
    
    activities = init_activities_with_ffwd(
        model=model,
        input=x,
        param_type="mupc"
    )
    
    assert len(activities) == len(model)


def test_init_activities_with_ffwd_skip_connections(simple_model, x):
    """Test feedforward initialization with skip connections."""
    from jpc import make_skip_model
    
    skip_model = make_skip_model(len(simple_model))
    activities = init_activities_with_ffwd(
        model=simple_model,
        input=x,
        skip_model=skip_model,
        param_type="sp"
    )
    
    assert len(activities) == len(simple_model)


def test_init_activities_from_normal_supervised(key, layer_sizes, batch_size):
    """Test random initialization in supervised mode."""
    activities = init_activities_from_normal(
        key=key,
        layer_sizes=layer_sizes,
        mode="supervised",
        batch_size=batch_size,
        sigma=0.05
    )
    
    # In supervised mode, input layer is not initialized
    assert len(activities) == len(layer_sizes) - 1
    for i, (act, size) in enumerate(zip(activities, layer_sizes[1:]), 1):
        assert act.shape == (batch_size, size)


def test_init_activities_from_normal_unsupervised(key, layer_sizes, batch_size):
    """Test random initialization in unsupervised mode."""
    activities = init_activities_from_normal(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=0.05
    )
    
    # In unsupervised mode, all layers including input are initialized
    assert len(activities) == len(layer_sizes)
    for act, size in zip(activities, layer_sizes):
        assert act.shape == (batch_size, size)


def test_init_activities_with_amort(key, simple_model, y, input_dim, hidden_dim, output_dim, depth):
    """Test amortized initialization."""
    from jpc import make_mlp
    
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
    
    activities = init_activities_with_amort(
        amortiser=amortiser,
        generator=simple_model,
        input=y
    )
    
    # Should return reversed activities plus dummy target prediction
    assert len(activities) == len(amortiser) + 1

