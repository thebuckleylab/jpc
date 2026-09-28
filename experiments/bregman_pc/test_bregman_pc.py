"""Unit tests for the Bregman PC experiment wiring (core maths is in tests/test_bregman.py)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jpc
import numpy as np
import optax

from experiments.bregman_pc.bp import update_bp
from experiments.bregman_pc.evaluate import (
    _batch_accuracy,
    evaluate_batch,
    evaluate_jpc_batch,
    feedforward_loss,
    feedforward_preds,
    predict_models,
)
from experiments.bregman_pc.model import BregmanMLP, layer_scalings, scaled_param_lr
from experiments.bregman_pc.steps import (
    bregman_mlp_to_jpc,
    bregman_pc_bp_grad_cosine,
    bregman_pc_step,
    bregman_pc_step_with_bp_cosine,
    init_jpc_opt_state,
    jpc_loss_id,
    standard_pc_bp_grad_cosine,
    standard_pc_step,
    standard_pc_step_with_bp_cosine,
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
    np.testing.assert_allclose(
        scaled_param_lr("mupc", "adam", 0.1, width=100, depth=4, use_skips=True),
        0.1 / (10.0 * 2.0),
    )


def test_mupc_residual_hidden_scaling():
    sizes = (64, 32, 32, 16)
    scales = layer_scalings(sizes, "mupc", gamma=1.0, use_skips=True)
    np.testing.assert_allclose(scales[0], 1.0 / np.sqrt(64))
    np.testing.assert_allclose(scales[1], 1.0 / np.sqrt(32 * 3))
    np.testing.assert_allclose(scales[2], 1.0 / 32)
    model = BregmanMLP(
        jax.random.PRNGKey(0),
        layer_sizes=sizes,
        act_fn="tanh",
        param_type="mupc",
        use_skips=True,
    )
    assert not model.layers[0].use_skip
    assert model.layers[1].use_skip
    assert not model.layers[-1].use_skip
    np.testing.assert_allclose(model.layers[1].scaling, 1.0 / np.sqrt(32 * 3))


def test_residual_forward_differs_from_plain():
    key = jax.random.PRNGKey(0)
    sizes = (8, 12, 12, 4)
    skipped = BregmanMLP(
        key, layer_sizes=sizes, act_fn="tanh", output_loss="mse", use_skips=True
    )
    plain = BregmanMLP(
        key, layer_sizes=sizes, act_fn="tanh", output_loss="mse", use_skips=False
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
    y_skip = np.asarray(skipped.forward(x))
    y_plain = np.asarray(plain.forward(x))
    assert not np.allclose(y_skip, y_plain, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(skipped.layers[1].linear.weight),
        np.asarray(plain.layers[1].linear.weight),
    )


def test_binary_sign_accuracy():
    y = jnp.array([[1.0], [-1.0], [1.0]])
    preds = jnp.array([[0.5], [-0.2], [-0.1]])
    np.testing.assert_allclose(float(_batch_accuracy(y, preds)), 2.0 / 3.0)
    one_hot = jax.nn.one_hot(jnp.array([0, 1, 2]), 3)
    logits = jnp.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(float(_batch_accuracy(one_hot, logits)), 1.0)


def test_binary_label_pc_step_runs():
    key = jax.random.PRNGKey(8)
    model = BregmanMLP(key, layer_sizes=(8, 6, 1), act_fn="tanh", output_loss="mse")
    x0 = jax.random.normal(jax.random.PRNGKey(9), (4, 8))
    y = jnp.array([[1.0], [-1.0], [1.0], [-1.0]])
    optim = optax.adam(1e-3)
    opt_state = init_jpc_opt_state(model.layers, optim)
    model, _, energy = bregman_pc_step(
        model, x0, y, optim, opt_state, n_iters=8, step_size=0.2
    )
    assert np.isfinite(float(energy))
    assert model.forward(x0).shape == (4, 1)
    loss, acc = evaluate_batch(model, x0, y)
    assert np.isfinite(float(loss))
    assert 0.0 <= float(acc) <= 1.0


def test_generate_label_to_image_step():
    key = jax.random.PRNGKey(12)
    model = BregmanMLP(key, layer_sizes=(3, 6, 8), act_fn="tanh", output_loss="mse")
    labels = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)
    images = jax.random.normal(jax.random.PRNGKey(13), (4, 8))
    optim = optax.sgd(1e-2)
    opt_state = init_jpc_opt_state(model.layers, optim)
    model, _, energy = bregman_pc_step(
        model, labels, images, optim, opt_state, n_iters=8, step_size=0.2
    )
    assert np.isfinite(float(energy))
    assert model.forward(labels).shape == (4, 8)
    loss, acc = evaluate_batch(model, labels, images, task="generate")
    assert np.isfinite(float(loss))
    assert np.isnan(float(acc))
    jpc_model = bregman_mlp_to_jpc(model)
    preds = predict_models({"bregman": model, "std_pc": jpc_model, "bp": model}, labels)
    for name, arr in preds.items():
        assert arr.shape == (4, 8), name
    np.testing.assert_allclose(
        np.asarray(feedforward_preds(model, labels)),
        np.asarray(preds["bregman"]),
        atol=1e-5,
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


def test_step_with_bp_cosine_matches_standalone_cosine():
    key = jax.random.PRNGKey(22)
    model = BregmanMLP(
        key, layer_sizes=(8, 12, 3), act_fn="tanh", output_loss="mse", param_type="mupc"
    )
    x0 = jax.random.normal(jax.random.PRNGKey(23), (4, 8))
    y = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)
    n_iters, step_size = 6, 5e-3
    cos_b = float(bregman_pc_bp_grad_cosine(model, x0, y, n_iters, step_size))
    optim = optax.sgd(1e-3)
    opt_state = init_jpc_opt_state(model.layers, optim)
    _, _, energy, cos_step = bregman_pc_step_with_bp_cosine(
        model, x0, y, optim, opt_state, n_iters, step_size
    )
    np.testing.assert_allclose(float(cos_step), cos_b, atol=1e-4)
    assert np.isfinite(float(energy))

    jpc_model = bregman_mlp_to_jpc(model)
    cos_s = float(
        standard_pc_bp_grad_cosine(
            jpc_model, x0, y, n_iters, step_size, loss_id="mse"
        )
    )
    std_opt_state = init_jpc_opt_state(jpc_model, optim)
    _, _, _, cos_std_step = standard_pc_step_with_bp_cosine(
        jpc_model, x0, y, optim, std_opt_state, n_iters, step_size, "mse"
    )
    np.testing.assert_allclose(float(cos_std_step), cos_s, atol=1e-4)


def test_residual_activity_grad_matches_autodiff():
    key = jax.random.PRNGKey(30)
    model = BregmanMLP(
        key,
        layer_sizes=(8, 6, 6, 3),
        act_fn="tanh",
        output_loss="mse",
        use_skips=True,
    )
    x = jax.random.normal(jax.random.PRNGKey(31), (4, 8))
    y = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)
    us = tuple(
        u + 0.2
        for u in jpc.init_bregman_pc_activities(model.layers, x, act_fn="tanh")
    )
    _, explicit = jpc.compute_bregman_pc_activity_grad(
        model.jpc_params(), us, y, x=x, act_fn="tanh", loss="mse"
    )
    dFdu = jax.grad(
        lambda u: jpc.bregman_pc_energy_fn(
            model.jpc_params(), u, y, x=x, act_fn="tanh", loss="mse"
        )
    )(us)
    for e, g, u in zip(explicit, dFdu, us):
        phi_p = 1.0 - jnp.tanh(u) ** 2
        np.testing.assert_allclose(
            np.asarray(g), np.asarray(phi_p * e / x.shape[0]), atol=1e-4, rtol=1e-4
        )


def test_mupc_width_bp_convergence_run_is_finite(tmp_path):
    from experiments.bregman_pc.mupc_bp_convergence import train_width

    key = jax.random.PRNGKey(24)
    x = jax.random.normal(jax.random.PRNGKey(25), (6, 8))
    y = jnp.where(jnp.arange(6) < 3, 1.0, -1.0)[:, None]
    result = train_width(
        key,
        x,
        y,
        width=8,
        n_hidden=1,
        act_fn="tanh",
        output_loss="mse",
        param_type="mupc",
        gamma_0=1.0,
        param_optim_id="sgd",
        param_lr=0.1,
        activity_lr=0.2,
        n_infer_iters=4,
        n_train_iters=2,
        save_dir=tmp_path,
        log_every=1,
    )
    for name in (
        "bregman_grad_cosine_similarities",
        "std_pc_grad_cosine_similarities",
        "bregman_train_losses",
        "bp_losses",
    ):
        arr = result[name]
        assert arr.shape == (2,)
        assert np.all(np.isfinite(arr))
    assert np.all(result["bregman_grad_cosine_similarities"] >= -1.0)
    assert np.all(result["bregman_grad_cosine_similarities"] <= 1.0)


def test_residual_mupc_bp_convergence_run_is_finite(tmp_path):
    from experiments.bregman_pc.mupc_bp_convergence import train_width

    key = jax.random.PRNGKey(26)
    x = jax.random.normal(jax.random.PRNGKey(27), (6, 8))
    y = jnp.where(jnp.arange(6) < 3, 1.0, -1.0)[:, None]
    result = train_width(
        key,
        x,
        y,
        width=8,
        n_hidden=2,
        act_fn="tanh",
        output_loss="mse",
        param_type="mupc",
        gamma_0=1.0,
        param_optim_id="adam",
        param_lr=1e-3,
        activity_lr=0.2,
        n_infer_iters=3,
        n_train_iters=2,
        save_dir=tmp_path / "residual",
        log_every=1,
        use_skips=True,
    )
    assert result["bregman_grad_cosine_similarities"].shape == (2,)
    assert np.all(np.isfinite(result["bregman_grad_cosine_similarities"]))
    assert np.all(np.isfinite(result["bp_losses"]))


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
