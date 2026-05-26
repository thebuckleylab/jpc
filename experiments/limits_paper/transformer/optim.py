import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import optax
import optax.tree
from optax import EmptyState, ScaleByAdamState
from optax._src import numerics, utils as optax_utils


def _scale_by_per_param_lr(lr_tree):
    """GradientTransformation: multiply updates by per-param lr and negate for descent."""

    def init(params):
        return {}

    def update(updates, state, params=None):
        new_updates = jtu.tree_map(lambda u, lr: -u * lr, updates, lr_tree)
        return (new_updates, state)

    return optax.GradientTransformation(init, update)


def _add_decayed_weights_pytree(wd_tree):
    """Like ``optax.add_decayed_weights`` but with a PyTree of coefficients (same structure as params)."""

    def init_fn(params):
        del params
        return EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(
                "params must be provided when using add_decayed_weights with per-leaf coefficients."
            )
        new_updates = jax.tree.map(
            lambda g, p, wd: None if g is None else g + wd * p,
            updates,
            params,
            wd_tree,
            is_leaf=lambda x: x is None,
        )
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def weight_decay_tree_for_transformer(
    model,
    *,
    param_type: str,
    weight_decay: float,
    width: int,
    params_for_pc: bool,
):
    """Coefficients matching ``configure_transformer_adamw``'s ``add_decayed_weights`` (per leaf).

    When ``params_for_pc`` is True, returns ``(wd_tree, None)`` for optimiser PyTrees
    ``(params, None)``. Returns ``None`` if ``weight_decay == 0`` (no extra term).
    """
    if weight_decay == 0.0:
        return None
    if param_type == "mupc":
        wd_model = _build_mupc_wd_tree(model, weight_decay, width)
    elif param_type == "sp":

        def leaf_wd(p):
            if not eqx.is_inexact_array(p):
                return p
            return jnp.asarray(weight_decay, dtype=jnp.float32)

        wd_model = jtu.tree_map(leaf_wd, eqx.filter(model, eqx.is_array))
    else:
        raise ValueError(f"param_type must be 'sp' or 'mupc', got {param_type!r}")
    return (wd_model, None) if params_for_pc else wd_model


def add_weight_decay_to_grads(grads, params, wd_tree):
    """``grads + wd * params`` in PyTree form, same as Optax ``add_decayed_weights`` before Adam."""
    if wd_tree is None:
        return grads
    return jtu.tree_map(
        lambda g, p, wd: None if g is None else g + wd * p,
        grads,
        params,
        wd_tree,
        is_leaf=lambda x: x is None,
    )


def _is_matrix_leaf(x):
    return eqx.is_inexact_array(x) and x.ndim >= 2


def _scale_by_adam_eps_tree(
    eps_tree,
    b1: float = 0.9,
    b2: float = 0.95,
    eps_root: float = 0.0,
    mu_dtype=None,
):
    """Adam rescaling with a PyTree of ``eps`` (one scalar per parameter tensor), matching optax ``scale_by_adam``."""

    mu_dtype = optax_utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)
        nu = optax.tree.zeros_like(params)
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = optax.tree.update_moment(updates, state.mu, b1, 1)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_increment(state.count)
        mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
        nu_hat = optax.tree.bias_correction(nu, b2, count_inc)
        new_updates = jax.tree.map(
            lambda m, v, e: None
            if m is None
            else m / (jnp.sqrt(v + eps_root) + e),
            mu_hat,
            nu_hat,
            eps_tree,
            is_leaf=lambda x: x is None,
        )
        mu = optax.tree.cast(mu, mu_dtype)
        return new_updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def _build_transformer_lr_tree(model, lr_other, lr_blocks):
    """Build a tree of learning rates matching eqx.filter(model, eqx.is_array)."""
    leaves = []
    for layer in model.layers:
        part = eqx.filter(layer, eqx.is_array)
        flat_part, _ = jtu.tree_flatten(part)
        use_blocks = type(layer).__name__ == "Block"
        lr = lr_blocks if use_blocks else lr_other
        leaves.extend([lr] * len(flat_part))
    full_flat, full_treedef = jtu.tree_flatten(eqx.filter(model, eqx.is_array))
    assert len(full_flat) == len(leaves), "transformer lr tree size mismatch"
    return jtu.tree_unflatten(full_treedef, leaves)


def _mupc_adam_eps_factors(adam_eps: float, width: int, depth: int):
    """Reference: emb/unemb ``eps / mup_width_multiplier``; hidden ``eps * (...) * depth_multiplier**(-α)`` with α=1."""
    mup_w = float(width)
    depth_mult = float(max(int(depth), 1))
    emb_unemb = adam_eps / mup_w
    hidden = adam_eps / (mup_w * depth_mult)
    return emb_unemb, hidden


def _build_mupc_eps_tree(model, adam_eps: float, width, depth: int):
    """Per-tensor Adam ε for μPC (emb+readout vs transformer blocks)."""
    emb_eps, hidden_eps = _mupc_adam_eps_factors(adam_eps, width, depth)

    def map_embed(x):
        if not eqx.is_inexact_array(x):
            return x
        return jnp.asarray(emb_eps, dtype=jnp.float32)

    def map_block(x):
        if not eqx.is_inexact_array(x):
            return x
        return jnp.asarray(hidden_eps, dtype=jnp.float32)

    def map_lm_head(x):
        if not eqx.is_inexact_array(x):
            return x
        return jnp.asarray(emb_eps, dtype=jnp.float32)

    # In the model, the penultimate layer is the final LayerNorm (or Identity).
    # When present, its Adam ε should match the "emb+readout" scaling.
    def map_final_ln(x):
        if not eqx.is_inexact_array(x):
            return x
        return jnp.asarray(emb_eps, dtype=jnp.float32)

    pieces = []
    pieces.append(jtu.tree_map(map_embed, eqx.filter(model.layers[0], eqx.is_array)))
    for i in range(1, len(model.layers) - 1):
        if i == len(model.layers) - 2:
            pieces.append(
                jtu.tree_map(map_final_ln, eqx.filter(model.layers[i], eqx.is_array))
            )
        else:
            pieces.append(
                jtu.tree_map(map_block, eqx.filter(model.layers[i], eqx.is_array))
            )
    pieces.append(jtu.tree_map(map_lm_head, eqx.filter(model.layers[-1], eqx.is_array)))

    flat = []
    for p in pieces:
        flat.extend(jtu.tree_leaves(p))
    full_flat, full_treedef = jtu.tree_flatten(eqx.filter(model, eqx.is_array))
    assert len(flat) == len(full_flat), "μPC eps tree leaf count mismatch"
    return jtu.tree_unflatten(full_treedef, flat)


def _mupc_lr_and_wd_factors(param_lr, width):
    """μPC (CompleteP-style) factors.

    For this μPC setup we treat LayerNorm parameters the same way as 1D "bias"
    parameters for lr/wd. (Only the Adam ε scaling is adjusted when LayerNorm
    is enabled.)

    depth α = 1 ⇒ depth_lr_scaling = 1.

    Uses ``mup_width_multiplier = width`` (no separate base width), so
    ``width_lr_scaling = 1 / width`` as in the reference when base width is 1.
    """
    mup_width_multiplier = float(width)
    width_lr_scaling = 1.0 / mup_width_multiplier
    depth_lr_scaling = 1.0

    lr_emb = param_lr * 1.0
    lr_hidden_w = param_lr * width_lr_scaling * depth_lr_scaling
    lr_hidden_b = param_lr * depth_lr_scaling
    return lr_emb, lr_hidden_w, lr_hidden_b, width_lr_scaling


def _build_mupc_lr_tree(model, param_lr, width):
    """μPC learning-rate multipliers (emb + readout vs hidden matmul vs hidden bias)."""
    lr_emb, lr_hidden_w, lr_hidden_b, _ = _mupc_lr_and_wd_factors(param_lr, width)

    def map_embed(x):
        if not eqx.is_inexact_array(x):
            return x
        return jnp.asarray(lr_emb, dtype=jnp.float32)

    def map_block(x):
        if not eqx.is_inexact_array(x):
            return x
        lr = lr_hidden_w if _is_matrix_leaf(x) else lr_hidden_b
        return jnp.asarray(lr, dtype=jnp.float32)

    def map_lm_head(x):
        if not eqx.is_inexact_array(x):
            return x
        return jnp.asarray(lr_emb, dtype=jnp.float32)

    pieces = []
    pieces.append(jtu.tree_map(map_embed, eqx.filter(model.layers[0], eqx.is_array)))
    for i in range(1, len(model.layers) - 1):
        pieces.append(jtu.tree_map(map_block, eqx.filter(model.layers[i], eqx.is_array)))
    pieces.append(jtu.tree_map(map_lm_head, eqx.filter(model.layers[-1], eqx.is_array)))

    flat = []
    for p in pieces:
        flat.extend(jtu.tree_leaves(p))
    full_flat, full_treedef = jtu.tree_flatten(eqx.filter(model, eqx.is_array))
    assert len(flat) == len(full_flat), "μPC lr tree leaf count mismatch"
    return jtu.tree_unflatten(full_treedef, flat)


def _build_mupc_wd_tree(model, weight_decay, width):
    """μPC weight-decay: emb/readout matrices get ``weight_decay``; hidden matrices ``wd / width_lr_scaling``."""
    _, _, _, width_lr_scaling = _mupc_lr_and_wd_factors(1.0, width)
    wd_emb = weight_decay
    wd_hidden_w = weight_decay / width_lr_scaling
    wd_zero = 0.0

    def map_embed(x):
        if not eqx.is_inexact_array(x):
            return x
        return jnp.asarray(wd_emb if _is_matrix_leaf(x) else wd_zero, dtype=jnp.float32)

    def map_block(x):
        if not eqx.is_inexact_array(x):
            return x
        wd = wd_hidden_w if _is_matrix_leaf(x) else wd_zero
        return jnp.asarray(wd, dtype=jnp.float32)

    def map_lm_head(x):
        if not eqx.is_inexact_array(x):
            return x
        return jnp.asarray(wd_emb if _is_matrix_leaf(x) else wd_zero, dtype=jnp.float32)

    pieces = []
    pieces.append(jtu.tree_map(map_embed, eqx.filter(model.layers[0], eqx.is_array)))
    for i in range(1, len(model.layers) - 1):
        pieces.append(jtu.tree_map(map_block, eqx.filter(model.layers[i], eqx.is_array)))
    pieces.append(jtu.tree_map(map_lm_head, eqx.filter(model.layers[-1], eqx.is_array)))

    flat = []
    for p in pieces:
        flat.extend(jtu.tree_leaves(p))
    full_flat, full_treedef = jtu.tree_flatten(eqx.filter(model, eqx.is_array))
    assert len(flat) == len(full_flat), "μPC wd tree leaf count mismatch"
    return jtu.tree_unflatten(full_treedef, flat)


def configure_transformer_adamw(
    model,
    param_type,
    param_lr,
    width,
    depth: int,
    params_for_pc=False,
    *,
    weight_decay: float = 0.1,
    adam_eps: float = 1e-8,
    b1: float = 0.9,
    b2: float = 0.95,
):
    """Configure AdamW for Transformer.

    Follows scalings of https://arxiv.org/abs/2505.01618.

    **Arguments:**

    - model: Transformer instance.
    - param_type: parameterisation type.
    - param_lr: learning rate.
    - width: width of the model.
    - depth: depth of the model.
    - params_for_pc: whether to use params for PC.
    - weight_decay: weight decay.
    - adam_eps: Adam ε.
    - b1: beta1. Defaults to 0.9.
    - b2: beta2. Defaults to 0.95.

    """
    if param_type == "mupc":
        lr_tree_model = _build_mupc_lr_tree(model, param_lr, width)
        wd_tree_model = _build_mupc_wd_tree(model, weight_decay, width)
        eps_tree_model = _build_mupc_eps_tree(model, adam_eps, width, depth)
        lr_tree = (lr_tree_model, None) if params_for_pc else lr_tree_model
        wd_for_chain = (wd_tree_model, None) if params_for_pc else wd_tree_model
        eps_for_chain = (eps_tree_model, None) if params_for_pc else eps_tree_model
        decay = _add_decayed_weights_pytree(wd_for_chain)
        adam = _scale_by_adam_eps_tree(eps_for_chain, b1=b1, b2=b2)
    
    elif param_type == "sp":
        lr_other = param_lr
        lr_blocks = param_lr
        lr_tree_model = _build_transformer_lr_tree(model, lr_other, lr_blocks)
        lr_tree = (lr_tree_model, None) if params_for_pc else lr_tree_model
        decay = optax.add_decayed_weights(weight_decay)
        adam = optax.scale_by_adam(eps=adam_eps, b1=b1, b2=b2)
    
    else:
        raise ValueError(f"param_type must be 'sp' or 'mupc', got {param_type!r}")

    return optax.chain(
        decay,
        adam,
        _scale_by_per_param_lr(lr_tree),
    )
