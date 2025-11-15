"""Tests for inference solving functions."""

import pytest
import jax
import jax.numpy as jnp
from diffrax import Heun, PIDController
from jpc import solve_inference


def test_solve_inference_supervised(simple_model, x, y):
    """Test inference solving in supervised mode."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    solution = solve_inference(
        params=(simple_model, None),
        activities=activities,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp",
        solver=Heun(),
        max_t1=10,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert len(solution) == len(activities)
    # Solution shape depends on whether recording is enabled
    # Without record_iters, it's just the final state
    for sol, act in zip(solution, activities):
        # Solution may have time dimension if recorded, or match activity shape
        assert sol.shape[-2:] == act.shape or sol.shape == act.shape


def test_solve_inference_unsupervised(simple_model, y, key, layer_sizes, batch_size):
    """Test inference solving in unsupervised mode."""
    from jpc import init_activities_from_normal
    
    activities = init_activities_from_normal(
        key=key,
        layer_sizes=layer_sizes,
        mode="unsupervised",
        batch_size=batch_size,
        sigma=0.05
    )
    
    solution = solve_inference(
        params=(simple_model, None),
        activities=activities,
        output=y,
        input=None,
        loss_id="mse",
        param_type="sp",
        solver=Heun(),
        max_t1=10,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert len(solution) == len(activities)


def test_solve_inference_cross_entropy(simple_model, x, y_onehot):
    """Test inference solving with cross-entropy loss."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    solution = solve_inference(
        params=(simple_model, None),
        activities=activities,
        output=y_onehot,
        input=x,
        loss_id="ce",
        param_type="sp",
        solver=Heun(),
        max_t1=10,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert len(solution) == len(activities)


def test_solve_inference_with_regularization(simple_model, x, y):
    """Test inference solving with regularization."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    solution = solve_inference(
        params=(simple_model, None),
        activities=activities,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp",
        solver=Heun(),
        max_t1=10,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
        weight_decay=0.01,
        spectral_penalty=0.01,
        activity_decay=0.01
    )
    
    assert len(solution) == len(activities)


def test_solve_inference_record_iters(simple_model, x, y):
    """Test inference solving with iteration recording."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    solution = solve_inference(
        params=(simple_model, None),
        activities=activities,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp",
        solver=Heun(),
        max_t1=10,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
        record_iters=True
    )
    
    assert len(solution) == len(activities)
    # When recording, solution should have an extra dimension for time
    assert solution[0].ndim == 3  # (time, batch, features)


def test_solve_inference_record_every(simple_model, x, y):
    """Test inference solving with record_every parameter."""
    from jpc import init_activities_with_ffwd
    
    activities = init_activities_with_ffwd(simple_model, x, param_type="sp")
    
    solution = solve_inference(
        params=(simple_model, None),
        activities=activities,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp",
        solver=Heun(),
        max_t1=10,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
        record_every=2
    )
    
    assert len(solution) == len(activities)

