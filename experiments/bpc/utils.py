import os


def setup_experiment(
        results_dir,
        dataset,
        width,
        n_hidden,
        layer_type,
        init_type,
        activity_lr,
        param_lr,
        batch_size,
        n_train_iters,
        test_every,
        seed
):
    print(
        f"""
Starting training experiment with configuration:

  Dataset: {dataset}
  Width: {width}
  N hidden: {n_hidden}
  Layer type: {layer_type}
  Init type: {init_type}
  Activity lr: {activity_lr}
  Param lr: {param_lr}
  Batch size: {batch_size}
  N train iters: {n_train_iters}
  Test every: {test_every}
  Seed: {seed}
"""
    )
    return os.path.join(
        results_dir,
        dataset,
        f"width_{width}",
        f"{n_hidden}_n_hidden",
        f"{layer_type}_layer",
        f"{init_type}_init",
        f"{activity_lr}_activity_lr",
        f"{param_lr}_param_lr",
        f"batch_size_{batch_size}",
        f"{n_train_iters}_train_iters",
        f"test_every_{test_every}",
        str(seed)
    )


def setup_hessian_analysis(
        results_dir,
        in_out_dims,
        act_fn,
        width,
        n_hidden,
        seed
):
    print(
        f"""
Starting Hessian analysis with configuration:

  Input output dims: {in_out_dims}
  Act fn: {act_fn}
  Width: {width}
  N hidden: {n_hidden}
  Seed: {seed}
"""
    )
    return os.path.join(
        results_dir,
        f"{in_out_dims}_in_out_dims",
        act_fn,
        f"width_{width}",
        f"{n_hidden}_n_hidden",
        str(seed)
    )
