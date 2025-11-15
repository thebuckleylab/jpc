"""Tests for test utility functions."""

import pytest
import jax
import jax.numpy as jnp
from diffrax import Heun, PIDController
from jpc import test_discriminative_pc, test_generative_pc, test_hpc


def test_test_discriminative_pc(simple_model, x, y_onehot):
    """Test discriminative PC testing function."""
    # For accuracy computation, we need one-hot encoded targets
    loss, acc = test_discriminative_pc(
        model=simple_model,
        output=y_onehot,
        input=x,
        loss="ce",  # Use cross-entropy for one-hot targets
        param_type="sp"
    )
    
    assert jnp.isfinite(loss)
    assert jnp.isfinite(acc)
    assert 0 <= acc <= 100


def test_test_discriminative_pc_cross_entropy(simple_model, x, y_onehot):
    """Test discriminative PC testing with cross-entropy."""
    loss, acc = test_discriminative_pc(
        model=simple_model,
        output=y_onehot,
        input=x,
        loss="ce",
        param_type="sp"
    )
    
    assert jnp.isfinite(loss)
    assert jnp.isfinite(acc)


def test_test_discriminative_pc_with_skip(simple_model, x, y_onehot):
    """Test discriminative PC testing with skip connections."""
    from jpc import make_skip_model
    
    skip_model = make_skip_model(len(simple_model))
    
    # compute_accuracy requires one-hot encoded targets
    loss, acc = test_discriminative_pc(
        model=simple_model,
        output=y_onehot,
        input=x,
        skip_model=skip_model,
        loss="ce",  # Use cross-entropy for one-hot targets
        param_type="sp"
    )
    
    assert jnp.isfinite(loss)
    assert jnp.isfinite(acc)


def test_test_generative_pc(simple_model, x, y_onehot, key, layer_sizes, batch_size, input_dim):
    """Test generative PC testing function."""
    # For compute_accuracy, input needs to be one-hot encoded
    key1, key2 = jax.random.split(key)
    input_onehot = jax.nn.one_hot(
        jax.random.randint(key1, (batch_size,), 0, input_dim),
        input_dim
    )
    input_acc, output_preds = test_generative_pc(
        model=simple_model,
        output=y_onehot,
        input=input_onehot,
        key=key2,
        layer_sizes=layer_sizes,
        batch_size=batch_size,
        loss_id="ce",
        param_type="sp",
        sigma=0.05,
        ode_solver=Heun(),
        max_t1=10,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert jnp.isfinite(input_acc)
    assert 0 <= input_acc <= 100
    assert output_preds.shape == y_onehot.shape


def test_test_generative_pc_cross_entropy(simple_model, x, y_onehot, key, layer_sizes, batch_size, input_dim):
    """Test generative PC testing with cross-entropy."""
    # For compute_accuracy, input needs to be one-hot encoded
    key1, key2 = jax.random.split(key)
    input_onehot = jax.nn.one_hot(
        jax.random.randint(key1, (batch_size,), 0, input_dim),
        input_dim
    )
    input_acc, output_preds = test_generative_pc(
        model=simple_model,
        output=y_onehot,
        input=input_onehot,
        key=key2,
        layer_sizes=layer_sizes,
        batch_size=batch_size,
        loss_id="ce",
        param_type="sp",
        sigma=0.05,
        ode_solver=Heun(),
        max_t1=10,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert jnp.isfinite(input_acc)
    assert output_preds.shape == y_onehot.shape


def test_test_hpc(key, simple_model, x, y_onehot, output_dim, hidden_dim, input_dim, depth, layer_sizes, batch_size):
    """Test HPC testing function."""
    from jpc import make_mlp
    
    # Split keys first
    key1, key2, key3 = jax.random.split(key, 3)
    
    generator = simple_model
    amortiser = make_mlp(
        key=key2,
        input_dim=output_dim,
        width=hidden_dim,
        depth=depth,
        output_dim=input_dim,
        act_fn="relu",
        use_bias=False,
        param_type="sp"
    )
    
    # For compute_accuracy, input needs to be one-hot encoded
    input_onehot = jax.nn.one_hot(
        jax.random.randint(key1, (batch_size,), 0, input_dim),
        input_dim
    )
    amort_acc, hpc_acc, gen_acc, output_preds = test_hpc(
        generator=generator,
        amortiser=amortiser,
        output=y_onehot,
        input=input_onehot,
        key=key3,
        layer_sizes=layer_sizes,
        batch_size=batch_size,
        sigma=0.05,
        ode_solver=Heun(),
        max_t1=10,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert jnp.isfinite(amort_acc)
    assert jnp.isfinite(hpc_acc)
    assert jnp.isfinite(gen_acc)
    assert 0 <= amort_acc <= 100
    assert 0 <= hpc_acc <= 100
    assert 0 <= gen_acc <= 100
    assert output_preds.shape == y_onehot.shape

