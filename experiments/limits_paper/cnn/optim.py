import numpy as np
import jax.tree_util as jtu
import equinox as eqx
import optax

from model import ResNetBlock


def _build_cnn_lr_tree(model, lr_other, lr_res, lr_first=None):
    """Build a tree of learning rates matching eqx.filter(model, eqx.is_array).
    Conv layers inside ResNetBlocks get lr_res; all others (stage convs,
    readout) get lr_other. Optionally override the first stage conv lr.
    """
    leaves = []
    for li, layer in enumerate(model.layers):
        part = eqx.filter(layer, eqx.is_array)
        flat_part, _ = jtu.tree_flatten(part)
        use_res = isinstance(layer, ResNetBlock)
        lr = lr_res if use_res else lr_other
        if lr_first is not None and li == 0:
            lr = lr_first
        leaves.extend([lr] * len(flat_part))
    full_flat, full_treedef = jtu.tree_flatten(eqx.filter(model, eqx.is_array))
    assert len(full_flat) == len(leaves), "lr tree size mismatch"
    return jtu.tree_unflatten(full_treedef, leaves)


def _scale_by_per_param_lr(lr_tree):
    """GradientTransformation: multiply updates by per-param lr and negate for descent."""

    def init(params):
        return {}

    def update(updates, state, params=None):
        new_updates = jtu.tree_map(lambda u, lr: -u * lr, updates, lr_tree)
        return (new_updates, state)

    return optax.GradientTransformation(init, update)


def configure_cnn_param_optim(
    model,
    optim_id,
    param_type,
    param_lr,
    width,
    depth,
    gamma_0=1.0,
    params_for_pc=False,
):
    """Configure parameter optimiser for CNN (ResNet-style).

    Only conv layers inside residual blocks get depth scaling with Adam; stage 
    convs and readout get width-only scaling. Same GD scaling as MLP for both.
    Follows scalings of https://arxiv.org/abs/2309.16620.

    Adam LR convention:
    - MLP with use_skips: lr / sqrt(width * depth); without: lr / sqrt(width).
    - CNN: res-block convs = "with skips" -> lr_res = lr / sqrt(width * depth);
            stage convs + readout = no depth factor -> lr_other = lr / sqrt(width).
    
    **Arguments:**

    - model: ResNet instance.
    - optim_id: optimiser ID.
    - param_type: parameterisation type.
    - param_lr: base learning rate.
    - width: channel width.
    - depth: effective depth for μPC LRs; must match ``ResNet`` forward scaling
        ``n_res_blocks + additive_depth_factor`` (same as ``depth`` in 
        ``model.py``), not ``n_res_blocks`` alone when ``additive_depth_factor`` 
        is non-zero. Defaults to 1.
    - params_for_pc: if True, lr tree matches (model_params, None) for 
        `jpc.update_pc_params`. Defaults to False.
    - gamma_0: scaling factor for GD optimiser.

    """
    if param_type == "sp":
        lr_other = param_lr
        lr_res = param_lr
    else:
        # mupc: res-block convs get depth scaling, others width only
        lr_res = param_lr / np.sqrt(width * depth)
        lr_other = param_lr / np.sqrt(width)

    lr_first = param_lr if param_type != "sp" else None
    lr_tree_model = _build_cnn_lr_tree(model, lr_other, lr_res, lr_first=lr_first)
    if params_for_pc:
        lr_tree = (lr_tree_model, None)
    else:
        lr_tree = lr_tree_model

    if optim_id == "gd":
        # GD: single scaled lr for all params (same as MLP)
        scaled_lr = param_lr * (gamma_0**2) * width if param_type != "sp" else param_lr
        lr_tree_gd = jtu.tree_map(lambda _: scaled_lr, lr_tree_model)
        if params_for_pc:
            lr_tree_gd = (lr_tree_gd, None)
        return _scale_by_per_param_lr(lr_tree_gd)
    
    elif optim_id == "adam":
        return optax.chain(optax.scale_by_adam(), _scale_by_per_param_lr(lr_tree))
    else:
        raise ValueError(f"Invalid optimiser: {optim_id}")
