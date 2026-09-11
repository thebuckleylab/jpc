"""Tests for activation-matched Bregman PC."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import grad

from jpc import (
    bregman_from_preact,
    bregman_pc_energy_fn,
    bregman_phi,
    compute_bregman_pc_activity_grad,
    compute_bregman_pc_param_grads,
    init_bregman_pc_activities,
    update_bregman_pc_activities,
    update_bregman_pc_params,
)


def _linear_mlp(key, sizes):
    import equinox.nn as nn

    keys = jax.random.split(key, len(sizes) - 1)
    return [
        nn.Linear(d_in, d_out, use_bias=False, key=k)
        for k, d_in, d_out in zip(keys, sizes[:-1], sizes[1:])
    ]


def test_bregman_from_preact_cancels_phi_prime():
    x = jnp.array(0.2)
    a = jnp.array(0.5)
    dda = grad(lambda a_: jnp.sum(bregman_from_preact("tanh", x, a_)))(a)
    assert jnp.allclose(dda, bregman_phi("tanh", a) - x, atol=1e-5)


def test_bregman_pc_energy_zero_hidden_at_ffwd(key):
    model = _linear_mlp(key, (8, 6, 3))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    us = init_bregman_pc_activities(model, x, act_fn="tanh")
    hidden = bregman_pc_energy_fn(
        (model, None), us, y=None, x=x, act_fn="tanh", loss="mse"
    )
    assert jnp.allclose(hidden, 0.0, atol=1e-5)


def test_bregman_pc_activity_grad_matches_mirror_autodiff(key):
    """Unnormalized residual; autodiff of the mean energy is φ' * residual / B."""
    model = _linear_mlp(key, (8, 6, 4, 3))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    y = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)
    us = tuple(u + 0.2 for u in init_bregman_pc_activities(model, x, act_fn="tanh"))
    _, explicit = compute_bregman_pc_activity_grad(
        (model, None), us, y, x=x, act_fn="tanh", loss="mse"
    )
    dFdu = grad(
        lambda u: bregman_pc_energy_fn(
            (model, None), u, y, x=x, act_fn="tanh", loss="mse"
        )
    )(us)
    for e, g, u in zip(explicit, dFdu, us):
        phi_p = 1.0 - jnp.tanh(u) ** 2
        assert jnp.allclose(g, phi_p * e / x.shape[0], atol=1e-4, rtol=1e-4)


def test_bregman_pc_activity_grad_shape(key):
    model = _linear_mlp(key, (8, 6, 3))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    y = jax.random.normal(jax.random.PRNGKey(2), (4, 3))
    us = tuple(u + 0.1 for u in init_bregman_pc_activities(model, x, act_fn="tanh"))
    energy, grads = compute_bregman_pc_activity_grad(
        (model, None), us, y, x=x, act_fn="tanh", loss="mse"
    )
    assert jnp.isfinite(energy)
    assert len(grads) == len(us)
    for g, u in zip(grads, us):
        assert g.shape == u.shape


def test_bregman_pc_param_grads_match_autodiff(key):
    model = _linear_mlp(key, (6, 5, 4))
    x = jax.random.normal(jax.random.PRNGKey(3), (3, 6))
    y = jax.nn.one_hot(jnp.array([0, 1, 2]), 4)
    us = tuple(u + 0.15 for u in init_bregman_pc_activities(model, x, act_fn="tanh"))
    explicit = compute_bregman_pc_param_grads(
        (model, None), us, y, x=x, act_fn="tanh", loss="ce"
    )[0]
    auto = eqx.filter_grad(bregman_pc_energy_fn)(
        (model, None), us, y, x=x, act_fn="tanh", loss="ce"
    )[0]
    for e, a in zip(explicit, auto):
        assert jnp.allclose(e.weight, a.weight, atol=1e-4, rtol=1e-4)


def test_update_bregman_pc_activities_decreases_energy(key):
    model = _linear_mlp(key, (8, 6, 3))
    x = jax.random.normal(jax.random.PRNGKey(4), (4, 8))
    y = jax.nn.one_hot(jnp.array([0, 1, 2, 1]), 3)
    us = init_bregman_pc_activities(model, x, act_fn="tanh")
    e0 = bregman_pc_energy_fn((model, None), us, y, x=x, act_fn="tanh", loss="ce")
    optim = optax.sgd(0.15)
    opt_state = optim.init(us)
    for _ in range(20):
        out = update_bregman_pc_activities(
            (model, None), us, optim, opt_state, y, input=x, act_fn="tanh", loss="ce"
        )
        us, opt_state = out["activities"], out["opt_state"]
    e1 = bregman_pc_energy_fn((model, None), us, y, x=x, act_fn="tanh", loss="ce")
    assert float(e1) < float(e0)


def test_update_bregman_pc_params_runs(key):
    model = _linear_mlp(key, (8, 6, 3))
    x = jax.random.normal(jax.random.PRNGKey(5), (4, 8))
    y = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)
    us = init_bregman_pc_activities(model, x, act_fn="tanh")
    optim = optax.sgd(1e-2)
    opt_state = optim.init((model, None))
    result = update_bregman_pc_params(
        (model, None), us, optim, opt_state, y, input=x, act_fn="tanh", loss="ce"
    )
    assert result["model"] is not None
    assert jnp.isfinite(
        bregman_pc_energy_fn(
            (result["model"], None), us, y, x=x, act_fn="tanh", loss="ce"
        )
    )
