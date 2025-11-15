"""Tests for training functions."""

import pytest
import jax
import jax.numpy as jnp
import optax
from diffrax import Heun, PIDController
from jpc import make_pc_step, make_hpc_step


def test_make_pc_step_supervised(simple_model, x, y):
    """Test PC training step in supervised mode."""
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init((simple_model, None))
    
    result = make_pc_step(
        model=simple_model,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp",
        ode_solver=Heun(),
        max_t1=5,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert "model" in result
    assert "skip_model" in result
    assert "opt_state" in result
    assert "loss" in result
    assert len(result["model"]) == len(simple_model)


def test_make_pc_step_unsupervised(simple_model, y, key, layer_sizes, batch_size):
    """Test PC training step in unsupervised mode."""
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init((simple_model, None))
    
    result = make_pc_step(
        model=simple_model,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=None,
        loss_id="mse",
        param_type="sp",
        ode_solver=Heun(),
        max_t1=5,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
        key=key,
        layer_sizes=layer_sizes,
        batch_size=batch_size,
        sigma=0.05
    )
    
    assert "model" in result
    assert "opt_state" in result


def test_make_pc_step_cross_entropy(simple_model, x, y_onehot):
    """Test PC training step with cross-entropy loss."""
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init((simple_model, None))
    
    result = make_pc_step(
        model=simple_model,
        optim=optim,
        opt_state=opt_state,
        output=y_onehot,
        input=x,
        loss_id="ce",
        param_type="sp",
        ode_solver=Heun(),
        max_t1=5,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert "model" in result
    assert "loss" in result


def test_make_pc_step_with_regularization(simple_model, x, y):
    """Test PC training step with regularization."""
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init((simple_model, None))
    
    result = make_pc_step(
        model=simple_model,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp",
        ode_solver=Heun(),
        max_t1=5,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
        weight_decay=0.01,
        spectral_penalty=0.01,
        activity_decay=0.01
    )
    
    assert "model" in result


def test_make_pc_step_with_metrics(simple_model, x, y):
    """Test PC training step with metrics recording."""
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init((simple_model, None))
    
    result = make_pc_step(
        model=simple_model,
        optim=optim,
        opt_state=opt_state,
        output=y,
        input=x,
        loss_id="mse",
        param_type="sp",
        ode_solver=Heun(),
        max_t1=5,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
        record_activities=True,
        record_energies=True,
        activity_norms=True,
        param_norms=True,
        grad_norms=True,
        calculate_accuracy=True
    )
    
    assert "model" in result
    assert "activities" in result
    assert "energies" in result
    assert "activity_norms" in result
    assert "model_param_norms" in result
    assert "acc" in result


def test_make_pc_step_invalid_input():
    """Test PC training step with invalid input (missing required args for unsupervised)."""
    import jax
    from jpc import make_mlp
    import optax
    from diffrax import Heun, PIDController
    
    key = jax.random.PRNGKey(42)
    model = make_mlp(key, 10, 20, 5, 3, "relu", False, "sp")
    optim = optax.sgd(learning_rate=0.01)
    opt_state = optim.init((model, None))
    y = jax.random.normal(key, (4, 5))
    
    with pytest.raises(ValueError):
        make_pc_step(
            model=model,
            optim=optim,
            opt_state=opt_state,
            output=y,
            input=None,
            loss_id="mse",
            param_type="sp",
            ode_solver=Heun(),
            max_t1=5,
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
        )


def test_make_hpc_step(key, simple_model, x, y, output_dim, hidden_dim, input_dim, depth):
    """Test HPC training step."""
    from jpc import make_mlp
    
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
    
    gen_optim = optax.sgd(learning_rate=0.01)
    amort_optim = optax.sgd(learning_rate=0.01)
    gen_opt_state = gen_optim.init((generator, None))
    amort_opt_state = amort_optim.init(amortiser)
    
    result = make_hpc_step(
        generator=generator,
        amortiser=amortiser,
        optims=(gen_optim, amort_optim),
        opt_states=(gen_opt_state, amort_opt_state),
        output=y,
        input=x,
        ode_solver=Heun(),
        max_t1=5,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert "generator" in result
    assert "amortiser" in result
    assert "opt_states" in result
    assert "losses" in result
    assert len(result["opt_states"]) == 2


def test_make_hpc_step_unsupervised(key, simple_model, y, output_dim, hidden_dim, input_dim, depth):
    """Test HPC training step in unsupervised mode."""
    from jpc import make_mlp
    
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
    
    gen_optim = optax.sgd(learning_rate=0.01)
    amort_optim = optax.sgd(learning_rate=0.01)
    gen_opt_state = gen_optim.init((generator, None))
    amort_opt_state = amort_optim.init(amortiser)
    
    result = make_hpc_step(
        generator=generator,
        amortiser=amortiser,
        optims=(gen_optim, amort_optim),
        opt_states=(gen_opt_state, amort_opt_state),
        output=y,
        input=None,
        ode_solver=Heun(),
        max_t1=5,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3)
    )
    
    assert "generator" in result
    assert "amortiser" in result


def test_make_hpc_step_with_recording(key, simple_model, x, y, output_dim, hidden_dim, input_dim, depth):
    """Test HPC training step with activity/energy recording."""
    from jpc import make_mlp
    
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
    
    gen_optim = optax.sgd(learning_rate=0.01)
    amort_optim = optax.sgd(learning_rate=0.01)
    gen_opt_state = gen_optim.init((generator, None))
    amort_opt_state = amort_optim.init(amortiser)
    
    result = make_hpc_step(
        generator=generator,
        amortiser=amortiser,
        optims=(gen_optim, amort_optim),
        opt_states=(gen_opt_state, amort_opt_state),
        output=y,
        input=x,
        ode_solver=Heun(),
        max_t1=5,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
        record_activities=True,
        record_energies=True
    )
    
    assert "activities" in result
    assert "energies" in result

