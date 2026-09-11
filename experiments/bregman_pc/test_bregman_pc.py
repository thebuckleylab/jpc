"""Unit tests for the Bregman PC experiment wiring (core maths is in tests/test_bregman.py)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jpc
import numpy as np
import optax

from experiments.bregman_pc.bp import update_bp
from experiments.bregman_pc.evaluate import evaluate_jpc_batch, feedforward_loss
from experiments.bregman_pc.model import BregmanMLP, layer_scalings, scaled_param_lr
from experiments.bregman_pc.steps import (
    bregman_mlp_to_jpc,
    bregman_pc_bp_grad_cosine,
    bregman_pc_step,
    init_jpc_opt_state,
    jpc_loss_id,
    standard_pc_bp_grad_cosine,
    standard_pc_step,
)


def test_init_scale_none_uses_fan_in_variance():
    key = jax.random.PRNGKey(0)
    sizes = (64, 32, 16)
    model = BregmanMLP(key, layer_sizes=sizes, act_fn="tanh", init_scale=None)
    expected_var = 1.0 / sizes[0]
    np.testing.assert_allclose(float(jnp.var(model.layers[0].linear.weight)), expected_var, rtol=0.25)


def test_init_scale_is_weight_variance():
    key = jax.random.PRNGKey(0)
    sizes = (64, 32, 16)
    model = BregmanMLP(key, layer_sizes=sizes, act_fn="tanh", init_scale=0.04)
    np.testing.assert_allclose(float(jnp.var(model.layers[0].linear.weight)), 0.04, rtol=0.25)
    np.testing.assert_allclose(float(jnp.var(model.layers[1].linear.weight)), 0.04, rtol=0.25)


def test_mupc_scalings_and_unit_init():
    sizes = (64, 32, 32, 16)
    scales = layer_scalings(sizes, "mupc", gamma=2.0)
    np.testing.assert_allclose(scales[0], 1.0 / np.sqrt(64))
    np.testing.assert_allclose(scales[1], 1.0 / np.sqrt(32))
    np.testing.assert_allclose(scales[2], 1.0 / 32 / 2.0)
    key = jax.random.PRNGKey(0)
    model = BregmanMLP(key, layer_sizes=sizes, act_fn="tanh", param_type="mupc")
    np.testing.assert_allclose(model.layers[0].scaling, 1.0 / np.sqrt(64))
    np.testing.assert_allclose(model.layers[1].scaling, 1.0 / np.sqrt(32))
    np.testing.assert_allclose(model.layers[-1].scaling, 1.0 / 32)
    np.testing.assert_allclose(float(jnp.var(model.layers[0].linear.weight)), 1.0, rtol=0.35)
    x = jax.random.normal(jax.random.PRNGKey(1), (64,))
    y = model.layers[0](x)
    y_explicit = (1.0 / np.sqrt(64)) * (model.layers[0].linear.weight @ x)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_explicit), atol=1e-5)


def test_scaled_param_lr_mupc_sgd():
    assert scaled_param_lr("sp", "sgd", 0.1, width=100, depth=4) == 0.1
    np.testing.assert_allclose(
        scaled_param_lr("mupc", "sgd", 0.1, width=100, depth=4, gamma=2.0),
        0.1 * 4.0 * 100,
    )
    np.testing.assert_allclose(
        scaled_param_lr("mupc", "adam", 0.1, width=100, depth=4), 0.1 / 10.0
    )


def test_bregman_pc_step_runs():
    key = jax.random.PRNGKey(8)
    model = BregmanMLP(key, layer_sizes=(8, 6, 3), act_fn="sigmoid", output_loss="ce")
    x0 = jax.random.normal(jax.random.PRNGKey(9), (4, 8))
    y = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)
    optim = optax.adam(1e-3)
    opt_state = init_jpc_opt_state(model.layers, optim)
    model, _, energy = bregman_pc_step(
        model, x0, y, optim, opt_state, n_iters=8, step_size=0.2
    )
    assert np.isfinite(float(energy))
    assert model.forward(x0).shape == (4, 3)


def test_bp_decreases_loss():
    key = jax.random.PRNGKey(10)
    model = BregmanMLP(key, layer_sizes=(8, 6, 3), act_fn="tanh", output_loss="ce")
    x0 = jax.random.normal(jax.random.PRNGKey(11), (4, 8))
    y = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)
    loss0 = float(feedforward_loss(model, x0, y))
    optim = optax.sgd(0.5)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    _, _, _, loss1 = update_bp(model, x0, y, optim, opt_state)
    assert float(loss1) < loss0


def test_standard_pc_step_shares_weights_and_runs():
    key = jax.random.PRNGKey(16)
    model = BregmanMLP(key, layer_sizes=(8, 6, 3), act_fn="tanh", output_loss="ce")
    jpc_model = bregman_mlp_to_jpc(model)
    np.testing.assert_allclose(jpc_model[0].layers[0].linear.weight, model.layers[0].linear.weight)
    np.testing.assert_allclose(jpc_model[-1].linear.weight, model.layers[-1].linear.weight)

    x0 = jax.random.normal(jax.random.PRNGKey(17), (4, 8))
    y = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)
    activities = jpc.init_activities_with_ffwd(model=jpc_model, input=x0)
    energies = jpc.pc_energy_fn(
        (jpc_model, None), activities, y, x=x0, loss="ce", record_layers=True
    )
    np.testing.assert_allclose(np.asarray(energies[1:]), 0.0, atol=1e-5)

    optim = optax.sgd(1e-2)
    opt_state = init_jpc_opt_state(jpc_model, optim)
    jpc_model, _, energy = standard_pc_step(
        jpc_model, x0, y, optim, opt_state, n_iters=8, step_size=0.05, loss_id=jpc_loss_id("ce")
    )
    assert np.isfinite(float(energy))
    loss, acc = evaluate_jpc_batch(jpc_model, x0, y, loss_id="ce")
    assert np.isfinite(float(loss))
    assert 0.0 <= float(acc) <= 1.0


def test_mupc_pc_bp_cosine_is_finite():
    key = jax.random.PRNGKey(20)
    model = BregmanMLP(
        key, layer_sizes=(16, 32, 32, 4), act_fn="tanh", output_loss="mse", param_type="mupc"
    )
    x0 = jax.random.normal(jax.random.PRNGKey(21), (8, 16))
    y = jax.nn.one_hot(jnp.array([0, 1, 2, 3, 0, 1, 2, 3]), 4)
    cos_b = float(bregman_pc_bp_grad_cosine(model, x0, y, n_iters=8, step_size=5e-3))
    jpc_model = bregman_mlp_to_jpc(model)
    cos_s = float(
        standard_pc_bp_grad_cosine(jpc_model, x0, y, n_iters=8, step_size=5e-3, loss_id="mse")
    )
    assert -1.0 <= cos_b <= 1.0
    assert -1.0 <= cos_s <= 1.0
    assert np.isfinite(cos_b) and np.isfinite(cos_s)


def _fake_run(param_lr, activity_lr, n_infer, seed, bregman, std_pc, bp):
    return {
        "dir": None,
        "seed": seed,
        "param_type": "sp",
        "gamma_0": 1.0,
        "width": 256,
        "param_lr": param_lr,
        "activity_lr": activity_lr,
        "n_infer_iters": n_infer,
        "metrics": {
            "bregman_final_test_acc": bregman,
            "std_pc_final_test_acc": std_pc,
            "bp_final_test_acc": bp,
        },
    }


def test_best_config_and_hparam_sweep_stats():
    from experiments.bregman_pc.plot import best_config_runs, hparam_sweep_stats

    runs = []
    for seed, bregman in enumerate((0.80, 0.82, 0.84)):
        runs.append(_fake_run(1e-3, 1e-2, 20, seed, bregman, 0.70, 0.90))
    for seed, bregman in enumerate((0.50, 0.51, 0.52)):
        runs.append(_fake_run(1e-3, 1e-3, 5, seed, bregman, 0.95, 0.90))
    for seed, bregman in enumerate((0.60, 0.61, 0.62)):
        runs.append(_fake_run(1e-2, 1e-2, 20, seed, bregman, 0.60, 0.70))

    best_bregman = best_config_runs(runs, "bregman")
    assert best_bregman[0]["param_lr"] == 1e-3
    assert best_bregman[0]["activity_lr"] == 1e-2
    assert best_bregman[0]["n_infer_iters"] == 20
    assert len(best_bregman) == 3

    best_std = best_config_runs(runs, "std_pc")
    assert best_std[0]["activity_lr"] == 1e-3
    assert best_std[0]["n_infer_iters"] == 5

    xs, means, stds = hparam_sweep_stats(runs, "bregman", "param_lr")
    np.testing.assert_allclose(xs, [1e-3, 1e-2])
    np.testing.assert_allclose(means, [0.82, 0.61])
    np.testing.assert_allclose(stds[0], np.std([0.80, 0.82, 0.84], ddof=1))
    np.testing.assert_allclose(stds[1], np.std([0.60, 0.61, 0.62], ddof=1))

    xs, means, _ = hparam_sweep_stats(runs, "bregman", "activity_lr")
    np.testing.assert_allclose(xs, [1e-3, 1e-2])
    np.testing.assert_allclose(means, [0.51, 0.82])

