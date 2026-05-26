import os


def setup_pc_experiment(
        results_dir,
        dataset,
        loss_id,
        width,
        n_blocks,
        act_fn,
        param_type,
        param_optim_id,
        param_lr,
        batch_size,
        max_infer_iters,
        activity_lr,
        max_epochs,
        seed
):
    """Setup save directory for CNN PC training."""
    print(
            f"""
Starting CNN PC training:

  Dataset: {dataset}
  Loss: {loss_id}
  Width: {width}
  N blocks: {n_blocks}
  Act fn: {act_fn}
  Param type: {param_type}
  Param optim id: {param_optim_id}
  Param lr: {param_lr}
  Batch size: {batch_size}
  Max infer iters: {max_infer_iters}
  Activity lr: {activity_lr}
  Max epochs: {max_epochs}
  Seed: {seed}
"""
        )
    return os.path.join(
        results_dir,
        dataset,
        loss_id,
        "pc",
        f"width_{width}",
        f"{n_blocks}_blocks",
        act_fn,
        f"{param_type}_param",
        f"{param_optim_id}_param_optim_id",
        f"param_lr_{param_lr}",
        f"batch_size_{batch_size}",
        f"{max_infer_iters}_max_infer_iters",
        f"activity_lr_{activity_lr}",
        f"{max_epochs}_epochs",
        str(seed)
    )


def setup_bp_experiment(
        results_dir,
        dataset,
        loss_id,
        width,
        n_blocks,
        act_fn,
        param_type,
        param_optim_id,
        param_lr,
        batch_size,
        max_epochs,
        seed
):
    """Setup save directory for CNN BP training (no PC-specific args)."""
    print(
            f"""
Starting CNN BP training:

  Dataset: {dataset}
  Loss: {loss_id}
  Width: {width}
  N blocks: {n_blocks}
  Act fn: {act_fn}
  Param type: {param_type}
  Param optim id: {param_optim_id}
  Param lr: {param_lr}
  Batch size: {batch_size}
  Max epochs: {max_epochs}
  Seed: {seed}
"""
        )
    return os.path.join(
        results_dir,
        dataset,
        loss_id,
        "bp",
        f"width_{width}",
        f"{n_blocks}_blocks",
        act_fn,
        f"{param_type}_param",
        f"{param_optim_id}_param_optim_id",
        f"param_lr_{param_lr}",
        f"batch_size_{batch_size}",
        f"{max_epochs}_epochs",
        str(seed)
    )
