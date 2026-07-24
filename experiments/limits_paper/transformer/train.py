import argparse
import os

import numpy as np
import jax
import jpc
import optax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from experiments.limits_paper.cnn.utils import (
    hessian_vector_product,
    power_iteration,
    inverse_iteration_cg,
)
from utils import (
    load_shakespeare,
    flatten_grads_per_layer_transformer,
    get_tracked_transformer_param_positions_and_names,
    setup_experiment,
    compute_cosine_similarity
)
from model import Transformer
from optim import (
    add_weight_decay_to_grads,
    configure_transformer_adamw,
    weight_decay_tree_for_transformer,
)


def train_bp(args, x, y, model, pc_grads, pc_grads_per_layer, tracked_param_positions):
    optim = configure_transformer_adamw(
        model,
        param_type=args.param_type,
        param_lr=args.param_lr,
        width=args.d_model,
        depth=args.n_blocks,
        params_for_pc=False,
        weight_decay=args.weight_decay,
        adam_eps=args.adam_eps,
        b1=args.beta1,
        b2=args.beta2,
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    wd_tree_bp = weight_decay_tree_for_transformer(
        model,
        param_type=args.param_type,
        weight_decay=args.weight_decay,
        width=args.d_model,
        params_for_pc=False,
    )

    if args.loss_id == "mse":

        @eqx.filter_jit
        def loss_fn(model, x_batch, y_batch):
            pred = jax.vmap(model)(x_batch)
            return 0.5 * jnp.mean(jnp.sum((y_batch - pred) ** 2, axis=1))
    else:

        @eqx.filter_jit
        def loss_fn(model, x_batch, y_batch):
            pred = jax.vmap(model)(x_batch)
            return jpc.cross_entropy_loss(pred, y_batch)

    @eqx.filter_jit
    def step(model, opt_state, x_batch, y_batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_batch, y_batch)
        updates, opt_state = optim.update(
            updates=grads, state=opt_state, params=eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, grads

    losses = []
    cos_pc_bp = []
    cos_pc_bp_per_layer = None

    for t in range(args.n_steps):
        model_pre = model
        model, opt_state, loss, grads = step(model, opt_state, x, y)
        losses.append(float(loss))
        # Same handling for all runs: loss grad + wd·θ (identity when weight_decay=0).
        grads_bp = add_weight_decay_to_grads(
            grads,
            eqx.filter(model_pre, eqx.is_array),
            wd_tree_bp,
        )
        bp_layers_all = flatten_grads_per_layer_transformer(model_pre, grads_bp)
        bp_layers = [bp_layers_all[pos] for pos in tracked_param_positions]
        bp_flat = np.concatenate([np.asarray(g).reshape(-1) for g in bp_layers])
        n_layers = len(bp_layers)

        pc_flat = np.asarray(pc_grads[t])
        cos_pc_bp.append(compute_cosine_similarity(pc_flat, bp_flat))
        pc_layers = pc_grads_per_layer[t]
        if cos_pc_bp_per_layer is None:
            cos_pc_bp_per_layer = np.zeros((args.n_steps, n_layers), dtype=np.float32)
        for li in range(n_layers):
            cos_pc_bp_per_layer[t, li] = compute_cosine_similarity(
                np.asarray(pc_layers[li]), np.asarray(bp_layers[li])
            )

        print(f"  step {t}: loss={loss:.6f}  cos_pc_bp={cos_pc_bp[-1]:.4f}")

    save_dir = setup_experiment(args)
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/bp_losses.npy", np.array(losses))
    cos_pc_bp = np.asarray(cos_pc_bp, dtype=np.float32)
    return cos_pc_bp, cos_pc_bp_per_layer


def train_pcn(args, x, y, model):
    activity_optim = optax.sgd(args.activity_lr * args.batch_size)
    param_optim = configure_transformer_adamw(
        model,
        param_type=args.param_type,
        param_lr=args.param_lr,
        width=args.d_model,
        depth=args.n_blocks,
        params_for_pc=True,
        weight_decay=args.weight_decay,
        adam_eps=args.adam_eps,
        b1=args.beta1,
        b2=args.beta2,
    )
    param_opt_state = param_optim.init((eqx.filter(model, eqx.is_array), None))

    experiment_energies = []
    pc_grads = []
    pc_grads_per_layer = []

    wd_tree_pc = weight_decay_tree_for_transformer(
        model,
        param_type=args.param_type,
        weight_decay=args.weight_decay,
        width=args.d_model,
        params_for_pc=True,
    )

    _, tracked_param_positions, _ = get_tracked_transformer_param_positions_and_names(model)
    n_infer_this = args.n_infer_iters

    for step in range(args.n_steps):
        model_pre = model
        params = (model, None)
        activities = jpc.init_activities_with_ffwd(model=model, input=x)
        activity_opt_state = activity_optim.init(activities)

        for _ in range(n_infer_this):
            result = jpc.update_pc_activities(
                params=params,
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=y,
                input=x,
                loss_id=args.loss_id
            )
            activities = result["activities"]
            activity_opt_state = result["opt_state"]

        experiment_energy = float(
            jpc.pc_energy_fn(
                params=params,
                activities=activities,
                y=y,
                x=x,
                loss=args.loss_id
            )
        )
        if not np.isfinite(experiment_energy):
            raise ValueError(
                f"Numerical (PC) energy is non-finite at step {step}. Halting."
            )
        experiment_energy_per_token = experiment_energy / args.seq_len
        experiment_energies.append(experiment_energy_per_token)
        print(f"  step {step}: experiment_energy={experiment_energy_per_token:.6f}")

        param_result = jpc.update_pc_params(
            params=params,
            activities=activities,
            optim=param_optim,
            opt_state=param_opt_state,
            output=y,
            input=x,
            loss_id=args.loss_id
        )
        model = param_result["model"]
        param_opt_state = param_result["opt_state"]
        grads = param_result["grads"]
        params_frozen = (eqx.filter(model_pre, eqx.is_array), None)
        # Same as BP: energy grad + wd·θ (identity when weight_decay=0).
        grads_pc = add_weight_decay_to_grads(grads, params_frozen, wd_tree_pc)
        all_layers = flatten_grads_per_layer_transformer(model_pre, grads_pc)
        tracked_layers = [all_layers[pos] for pos in tracked_param_positions]
        flat_tracked = np.concatenate(
            [np.asarray(g).reshape(-1) for g in tracked_layers]
        )
        pc_grads.append(np.array(flat_tracked))
        pc_grads_per_layer.append(tracked_layers)

    save_dir = setup_experiment(args)
    os.makedirs(save_dir, exist_ok=True)
    steps = np.arange(args.n_steps, dtype=np.int64)
    np.save(f"{save_dir}/steps.npy", steps)
    np.save(f"{save_dir}/experiment_energies.npy", np.asarray(experiment_energies))
    np.save(f"{save_dir}/vocab_size.npy", np.int64(args.vocab_size))

    # Optionally estimate condition number of activity Hessian at final inferred activities
    if args.estimate_hessian_cond:
        params_final = (model, None)

        def energy_fn(acts):
            return jpc.pc_energy_fn(
                params=params_final,
                activities=acts,
                y=y,
                x=x,
                loss=args.loss_id,
            )

        @jax.jit
        def hvp_fn(v):
            return hessian_vector_product(energy_fn, activities, v)

        hvp_key = jr.PRNGKey(args.seed + 1000)
        key_max, key_min = jr.split(hvp_key, 2)

        print("Estimating activity Hessian condition number...")
        max_eigenval, _ = power_iteration(
            hvp_fn, activities, key_max, n_iters=args.hessian_power_iters
        )
        min_eigenval = inverse_iteration_cg(
            hvp_fn,
            activities,
            key_min,
            n_iters=args.hessian_inverse_iters,
            cg_iters=args.hessian_cg_iters,
        )
        cond = max_eigenval / (float(min_eigenval) + 1e-30)
        print(
            f"  max_eigenval≈{max_eigenval:.6e}  min_eigenval≈{min_eigenval:.6e}  cond≈{cond:.6e}"
        )

        np.save(f"{save_dir}/hessian_max_eigenval.npy", np.float64(max_eigenval))
        np.save(f"{save_dir}/hessian_min_eigenval.npy", np.float64(min_eigenval))
        np.save(f"{save_dir}/hessian_condition_number.npy", np.float64(cond))

    return pc_grads, pc_grads_per_layer


def main(args):
    key = jr.PRNGKey(args.seed)
    x, y, vocab_size = load_shakespeare(args.batch_size, args.seq_len, args.seed)
    args.vocab_size = vocab_size

    # PC training
    model = Transformer(
        key=key,
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        n_embd=args.d_model,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        param_type=args.param_type,
        use_layer_norm=args.use_layer_norm,
        use_softmax=args.use_softmax,
        use_bias=args.use_bias,
        act_fn=args.act_fn,
        init_std=args.init_std
    )
    print(
        f"\nPC training (d_model={args.d_model}, n_blocks={args.n_blocks}, n_heads={args.n_heads}, "
        f"seq_len={args.seq_len}, vocab={vocab_size}, seed={args.seed}, "
        f"activity_lr={args.activity_lr}, n_infer_iters={args.n_infer_iters})\n"
    )
    pc_grads, pc_grads_per_layer = train_pcn(args, x, y, model)
    _, tracked_param_positions, tracked_layer_names = (
        get_tracked_transformer_param_positions_and_names(model)
    )

    # BP training
    model_bp = Transformer(
        key=key,
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        n_embd=args.d_model,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        param_type=args.param_type,
        use_layer_norm=args.use_layer_norm,
        use_softmax=args.use_softmax,
        use_bias=args.use_bias,
        act_fn=args.act_fn,
        init_std=args.init_std
    )
    print("\nBP training (same init, same data)\n")
    cosine_similarities, cosine_similarities_per_layer = train_bp(
        args, x, y, model_bp, pc_grads, pc_grads_per_layer, tracked_param_positions
    )

    # Save results
    save_dir = setup_experiment(args)
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/grad_cosine_similarities.npy", cosine_similarities)
    np.save(
        f"{save_dir}/grad_cosine_similarities_per_layer.npy",
        cosine_similarities_per_layer,
    )
    np.save(
        f"{save_dir}/grad_cosine_similarities_layer_names.npy",
        np.array(tracked_layer_names, dtype=object),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")

    # data parameters
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)

    # model parameters
    parser.add_argument("--d_models", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--n_blocks", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--param_type", type=str, default="mupc", choices=["sp", "mupc"])
    parser.add_argument("--use_layer_norm", action="store_true", default=False)
    parser.add_argument("--use_softmax", action="store_true", default=True)
    parser.add_argument("--use_bias", action="store_true", default=False)
    parser.add_argument("--act_fn", type=str, default="gelu", choices=["linear", "gelu", "relu", "swish"])
    parser.add_argument("--init_std", type=float, default=0.02)

    # training parameters
    parser.add_argument("--param_lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam β₁.")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam β₂.")
    parser.add_argument("--adam_eps", type=float, default=1e-12, help="Adam ε")
    parser.add_argument("--weight_decay", type=float, default=0.)  # 1e-1
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--loss_id", type=str, default="ce", choices=["mse", "ce"])
    parser.add_argument("--n_seeds", type=int, default=3)

    # inference parameters
    parser.add_argument("--activity_lrs", type=float, nargs="+", default=[2e-1, 3e-1])
    parser.add_argument("--n_infer_iters", type=int, nargs="+", default=[100, 200])

    # Hessian conditioning parameters
    parser.add_argument("--estimate_hessian_cond", action="store_true", default=False)
    parser.add_argument(
        "--hessian_power_iters",
        type=int,
        default=25,
        help="Power iteration steps for λ_max.",
    )
    parser.add_argument(
        "--hessian_inverse_iters",
        type=int,
        default=10,
        help="Inverse iteration steps for λ_min.",
    )
    parser.add_argument(
        "--hessian_cg_iters",
        type=int,
        default=25,
        help="CG steps per inverse iteration.",
    )

    args = parser.parse_args()
    for n_infer_iters in args.n_infer_iters:
        args.n_infer_iters = n_infer_iters
        for d_model in args.d_models:
            args.d_model = d_model
            for activity_lr in args.activity_lrs:
                args.activity_lr = activity_lr
                for seed in range(args.n_seeds):

                    if seed == 2:
                        args.seed = seed
                        print(f"Running experiment for seed: {seed}")
                        main(args)
