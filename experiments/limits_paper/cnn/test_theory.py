import argparse
import os

import jpc
import optax
import numpy as np
import equinox as eqx

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from theory_utils import (
    linear_cnn_equilib_energy,
    compute_linear_cnn_equilib_energy_grads
)
from utils import (
    flatten_grads_per_layer_cnn,
    get_tracked_cnn_param_positions_and_names,
    compute_cosine_similarity,
    load_cifar10_batch,
    load_tinyimagenet_batch,
    load_imagenet_batch,
    setup_theory_experiment,
    hessian_vector_product,
    power_iteration,
    inverse_iteration_cg,
)
from optim import configure_cnn_param_optim
from model import ResNet


def train_bp_cnn(
    args,
    x,
    y,
    model,
    pc_grads,
    pc_grads_per_layer,
    tracked_param_positions,
    tracked_layer_names,
):
    depth = args.n_res_blocks + args.additive_depth_factor
    optim = configure_cnn_param_optim(
        model,
        optim_id=args.param_optim,
        param_type=args.param_type,
        param_lr=args.param_lr,
        width=args.width,
        depth=depth,
        params_for_pc=False,
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

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
    # On-the-fly cosine similarities (restricted to tracked layers)
    use_num = args.pc_grad_type == "numerical"
    use_th = args.pc_grad_type == "theoretical"

    def _grad_stats(v: np.ndarray):
        v = np.asarray(v).reshape(-1)
        if v.size == 0:
            return {"n": 0, "norm": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "n": int(v.size),
            "norm": float(np.linalg.norm(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }

    cos_pc_bp = [] if use_num else None
    cos_pc_bp_per_layer = None  # initialised after first step if needed
    cos_theory_bp = [] if use_th else None
    cos_theory_bp_per_layer = None
    n_pc_steps = len(pc_grads)

    if use_num and n_pc_steps < args.n_steps:
        print(
            f"Warning: only {n_pc_steps}/{args.n_steps} PC updates produced valid gradients. "
            "Computing numerical PC/BP cosine similarities for successful PC updates only."
        )

    for t in range(args.n_steps):
        model, opt_state, loss, grads = step(model, opt_state, x, y)
        losses.append(float(loss))
        # Flatten BP grads only for tracked layers
        bp_layers_all = flatten_grads_per_layer_cnn(model, grads)
        bp_layers = [bp_layers_all[pos] for pos in tracked_param_positions]
        bp_flat = np.concatenate([np.asarray(g).reshape(-1) for g in bp_layers])

        n_layers = len(bp_layers)

        # BP vs numerical PC cosine (overall and per layer)
        if use_num and t < n_pc_steps:
            pc_flat = np.asarray(pc_grads[t])
            cos_pc_bp.append(compute_cosine_similarity(pc_flat, bp_flat))

            pc_layers = pc_grads_per_layer[t]
            if cos_pc_bp_per_layer is None:
                cos_pc_bp_per_layer = np.zeros(
                    (n_pc_steps, n_layers), dtype=np.float32
                )
            for li in range(n_layers):
                pc_l = np.asarray(pc_layers[li])
                bp_l = np.asarray(bp_layers[li])
                cos_pc_bp_per_layer[t, li] = compute_cosine_similarity(pc_l, bp_l)

            # Print per-layer gradient magnitudes/stats (helps diagnose cosine fluctuations).
            # Only print for conv layers by name (stage{1,2,3}_conv).
            for li in range(n_layers):
                name = tracked_layer_names[li]
                if "conv" not in name:
                    continue
                pc_s = _grad_stats(pc_layers[li])
                bp_s = _grad_stats(bp_layers[li])
                print(
                    f"    {name} grad stats: "
                    f"pc||g||={pc_s['norm']:.3e} bp||g||={bp_s['norm']:.3e}  "
                    f"pc(mean±std)={pc_s['mean']:.2e}±{pc_s['std']:.2e}  "
                    f"bp(mean±std)={bp_s['mean']:.2e}±{bp_s['std']:.2e}"
                )

        # BP vs theoretical PC cosine (overall and per layer)
        if use_th:
            th_grads = compute_linear_cnn_equilib_energy_grads(
                model,
                x,
                y,
                batch_size=args.batch_size,
                include_dSdtheta=True,
            )
            th_layers_all = flatten_grads_per_layer_cnn(model, th_grads)
            th_layers = [th_layers_all[pos] for pos in tracked_param_positions]
            th_flat = np.concatenate([np.asarray(g).reshape(-1) for g in th_layers])

            cos_theory_bp.append(compute_cosine_similarity(th_flat, bp_flat))

            if cos_theory_bp_per_layer is None:
                cos_theory_bp_per_layer = np.zeros(
                    (args.n_steps, n_layers), dtype=np.float32
                )
            for li in range(n_layers):
                th_l = np.asarray(th_layers[li])
                bp_l = np.asarray(bp_layers[li])
                cos_theory_bp_per_layer[t, li] = compute_cosine_similarity(th_l, bp_l)

        # Print cos sims during training
        cos_parts = [f"loss={loss:.6f}"]
        if use_num and t < n_pc_steps:
            cos_parts.append(f"cos_pc_bp={cos_pc_bp[-1]:.4f}")
        if use_th:
            cos_parts.append(f"cos_theory_bp={cos_theory_bp[-1]:.4f}")
        print(f"  step {t}: " + "  ".join(cos_parts))

        # Also print per-layer cosine similarities for the tracked layers.
        if use_num and t < n_pc_steps and cos_pc_bp_per_layer is not None:
            per_layer_pc_bp = "  ".join(
                f"{tracked_layer_names[li]}={cos_pc_bp_per_layer[t, li]:.4f}"
                for li in range(n_layers)
            )
            print(f"    per-layer cos_pc_bp: {per_layer_pc_bp}")
        if use_th and cos_theory_bp_per_layer is not None:
            per_layer_th_bp = "  ".join(
                f"{tracked_layer_names[li]}={cos_theory_bp_per_layer[t, li]:.4f}"
                for li in range(n_layers)
            )
            print(f"    per-layer cos_theory_bp: {per_layer_th_bp}")

    save_dir = setup_theory_experiment(args)
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/bp_losses.npy", np.array(losses))

    # Convert cosine similarities to numpy arrays before returning
    if use_num:
        cos_pc_bp = np.asarray(cos_pc_bp, dtype=np.float32)
    if use_th:
        cos_theory_bp = np.asarray(cos_theory_bp, dtype=np.float32)

    return (
        cos_pc_bp,
        cos_pc_bp_per_layer,
        cos_theory_bp,
        cos_theory_bp_per_layer,
    )


def train_pcn(args, x, y, model):
    activity_optim = optax.sgd(args.activity_lr * args.batch_size)
    depth = args.n_res_blocks + args.additive_depth_factor
    param_optim = configure_cnn_param_optim(
        model,
        optim_id=args.param_optim,
        param_type=args.param_type,
        param_lr=args.param_lr,
        width=args.width,
        depth=depth,
        params_for_pc=True,
    )
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), None)
    )

    use_amortiser = args.use_amortiser
    amortiser = None
    amort_optim = None
    amort_opt_state = None
    amort_energies = []

    if use_amortiser:
        amort_key = jr.PRNGKey(args.seed + 1)

        amortiser = ResNet(
            key=amort_key,
            width=args.width,
            n_res_blocks=args.n_res_blocks,
            in_channels=args.in_channels,
            input_size=args.input_size,
            out_features=args.out_features,
            param_type=args.param_type,
            act_fn=args.act_fn,
            scale_non_res_layers=args.scale_non_res_layers,
            additive_depth_factor=args.additive_depth_factor,
        )
        amort_optim = optax.adam(args.param_lr)
        amort_opt_state = amort_optim.init(eqx.filter(amortiser, eqx.is_array))

        def _same_dir_amort_energy(am_, x_batch, equilib_activities):
            """
            Same-direction amortiser energy (input x, like generator) with
            layer-local gradients: each layer's error uses stop_gradient on
            the previous layer's prediction so param gradients are local.
            """
            # Forward with stop_gradient between layers -> local gradients.
            acts = []
            h = x_batch
            for li in range(len(am_)):
                h = jax.vmap(am_[li])(h)
                acts.append(h)
                h = jax.lax.stop_gradient(h)
            # Target: generator equilibrium (exclude last slot clamped to y).
            # Amortiser has one output per layer; match hidden layers only (exclude readout).
            target_acts = equilib_activities[:-1]
            pred_acts = acts[:-1]
            assert len(pred_acts) == len(target_acts)
            batch_size = x_batch.shape[0]
            
            layer_energies = []
            for t_act, p_act in zip(target_acts, pred_acts):
                err = t_act - p_act
                layer_energies.append(0.5 * jnp.sum(err ** 2))
            total_energy = jnp.sum(jnp.stack(layer_energies)) / batch_size
            return total_energy

        @eqx.filter_jit
        def amort_step(amortiser, opt_state, x_batch, y_batch, equilib_activities):
            """
            Train same-direction amortiser (input x) with a layer-local energy
            (HPC assumes inverted amortiser from y; we do not use it here).
            """
            def energy_fn(am_):
                return _same_dir_amort_energy(am_, x_batch, equilib_activities)

            energy, grads = eqx.filter_value_and_grad(energy_fn)(amortiser)
            updates, opt_state = amort_optim.update(
                grads, opt_state, params=eqx.filter(amortiser, eqx.is_array)
            )
            amortiser = eqx.apply_updates(amortiser, updates)
            return amortiser, opt_state, energy

    if args.loss_id == "mse":
        theory_energies = []
        rescalings = []
    
    numerical_energies = []
    pc_grads = []
    pc_grads_per_layer = []
    pc_successful_steps = []
    infer_iters_used_per_step = []

    # Decide which parameter layers to track for PC/BP comparisons.
    _, tracked_param_positions, _ = get_tracked_cnn_param_positions_and_names(model)

    n_infer_max = args.n_infer_iters
    infer_energy_thresh = args.infer_energy_thresh

    for step in range(args.n_steps):
        params = (model, None)
        if use_amortiser:
            activities = jpc.init_activities_with_ffwd(model=amortiser, input=x)
        else:
            activities = jpc.init_activities_with_ffwd(model=model, input=x)

        n_infer_this = n_infer_max
        activity_opt_state = activity_optim.init(activities)

        for infer_iter in range(n_infer_this):
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
            # if float(result["energy"]) <= infer_energy_thresh:
            #     break

        infer_iters_used_per_step.append(infer_iter + 1)

        # Theoretical equilibrated energy (closed form); only defined for MSE loss
        if args.loss_id == "mse":
            theory_energy, S = linear_cnn_equilib_energy(
                model,
                x,
                y,
                batch_size=args.batch_size,
                return_rescaling=True,
            )
            rescaling = jnp.linalg.norm(S, ord=2) if args.out_features > 1 else S
            rescalings.append(rescaling)
            theory_energies.append(float(theory_energy))

        # Numerical PC energy at the inferred activities
        numerical_energy = float(
            jpc.pc_energy_fn(
                params=params,
                activities=activities,
                y=y,
                x=x,
                loss=args.loss_id
            )
        )
        if not np.isfinite(numerical_energy):
            print(
                f"Warning: numerical (PC) energy is non-finite at step {step}. Skipping this step."
            )
            continue
        numerical_energies.append(numerical_energy)
        if args.loss_id == "mse":
            print(
                f"  step {step}: theory energy={theory_energy:.6f}  "
                f"numerical energy={numerical_energy:.6f}  "
                f"squared diff={(theory_energy - numerical_energy) ** 2:.2e}"
            )
        else:
            print(f"  step {step}: numerical={numerical_energy:.6f}")

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
        # Flatten only the tracked layers to save memory.
        all_layers = flatten_grads_per_layer_cnn(model, grads)
        tracked_layers = [all_layers[pos] for pos in tracked_param_positions]
        flat_tracked = np.concatenate(
            [np.asarray(g).reshape(-1) for g in tracked_layers]
        )
        pc_grads.append(np.array(flat_tracked))
        pc_grads_per_layer.append(tracked_layers)
        pc_successful_steps.append(step)

        if use_amortiser:
            amortiser, amort_opt_state, amort_energy = amort_step(
                amortiser,
                amort_opt_state,
                x,
                y,
                activities,
            )
            amort_energies.append(float(amort_energy))
            print(
                f"    amort energy={float(amort_energy):.6f}  "
                f"n_infer={infer_iter + 1}  init=amort"
            )

    save_dir = setup_theory_experiment(args)
    os.makedirs(save_dir, exist_ok=True)
    steps = np.arange(args.n_steps, dtype=np.int64)
    np.save(f"{save_dir}/steps.npy", steps)
    np.save(f"{save_dir}/numerical_energies.npy", np.asarray(numerical_energies))
    np.save(f"{save_dir}/pc_successful_steps.npy", np.asarray(pc_successful_steps, dtype=np.int64))
    np.save(
        f"{save_dir}/infer_iters_used_per_step.npy",
        np.asarray(infer_iters_used_per_step, dtype=np.int64),
    )
    if args.loss_id == "mse":
        np.save(f"{save_dir}/theory_energies.npy", np.asarray(theory_energies))
        np.save(
            f"{save_dir}/rescalings.npy",
            np.asarray([float(r) for r in rescalings])
        )
    if use_amortiser:
        np.save(f"{save_dir}/amortiser_energies.npy", np.array(amort_energies))

    # Optionally estimate condition number of activity Hessian at final inferred activities
    if args.estimate_hessian_cond:
        params_final = (model, None)

        def energy_fn(acts):
            return jpc.pc_energy_fn(
                params=params_final,
                activities=acts,
                y=y,
                x=x,
                loss=args.loss_id
            )

        @jax.jit
        def hvp_fn(v):
            return hessian_vector_product(energy_fn, activities, v)

        n_activities = sum(a.size for a in jtu.tree_leaves(activities))
        hvp_key = jr.PRNGKey(args.seed + 1000)
        key_max, key_min = jr.split(hvp_key, 2)
        power_iters = args.hessian_power_iters
        inverse_iters = args.hessian_inverse_iters
        cg_iters = args.hessian_cg_iters

        print(f"  Estimating activity Hessian condition number (n_activities={n_activities})...")
        max_eigenval, _ = power_iteration(hvp_fn, activities, key_max, n_iters=power_iters)
        min_eigenval = inverse_iteration_cg(
            hvp_fn, activities, key_min,
            n_iters=inverse_iters,
            cg_iters=cg_iters,
        )
        cond = max_eigenval / (float(min_eigenval) + 1e-30)
        print(f"  max_eigenval≈{max_eigenval:.6e}  min_eigenval≈{min_eigenval:.6e}  cond≈{cond:.6e}")

        np.save(f"{save_dir}/hessian_max_eigenval.npy", np.float64(max_eigenval))
        np.save(f"{save_dir}/hessian_min_eigenval.npy", np.float64(min_eigenval))
        np.save(f"{save_dir}/hessian_condition_number.npy", np.float64(cond))
        np.save(f"{save_dir}/hessian_n_activities.npy", np.int64(n_activities))

    return pc_grads, pc_grads_per_layer, pc_successful_steps



def main(args):
    key = jr.PRNGKey(args.seed)
    model_key, _ = jr.split(key, 2)

    # One fixed batch for the whole run (same batch for PC and BP)
    if args.dataset == "cifar":
        x, y = load_cifar10_batch(args.batch_size, seed=args.seed)
    elif args.dataset == "tinyimagenet":
        x, y = load_tinyimagenet_batch(args.batch_size, seed=args.seed)
    elif args.dataset == "imagenet":
        x, y = load_imagenet_batch(args.batch_size, seed=args.seed)
    else:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. Expected 'cifar', 'tinyimagenet', or 'imagenet'."
        )

    args.in_channels = x.shape[1]
    args.input_size, args.out_features = x.shape[-1], y.shape[-1]

    model = ResNet(
        key=model_key,
        width=args.width,
        n_res_blocks=args.n_res_blocks,
        in_channels=args.in_channels,
        input_size=args.input_size,
        out_features=args.out_features,
        param_type=args.param_type,
        act_fn=args.act_fn,
        scale_non_res_layers=args.scale_non_res_layers,
        additive_depth_factor=args.additive_depth_factor,
    )

    print(
        f"\nPC training (width={args.width}, n_res_blocks={args.n_res_blocks}, seed={args.seed}, "
        f"act_fn={args.act_fn}, activity_lr={args.activity_lr}, n_infer_iters={args.n_infer_iters})..."
    )
    pc_grads, pc_grads_per_layer, pc_successful_steps = train_pcn(args, x, y, model)

    # Determine which parameter layers are being tracked (same convention for BP).
    _, tracked_param_positions, tracked_layer_names = get_tracked_cnn_param_positions_and_names(model)

    model_bp = ResNet(
        key=model_key,
        width=args.width,
        n_res_blocks=args.n_res_blocks,
        in_channels=args.in_channels,
        input_size=args.input_size,
        out_features=args.out_features,
        param_type=args.param_type,
        act_fn=args.act_fn,
        scale_non_res_layers=args.scale_non_res_layers,
        additive_depth_factor=args.additive_depth_factor,
    )
    print(f"\nBP training (same init, same data)...")
    (
        cosine_similarities,
        cosine_similarities_per_layer,
        cosine_sim_theory_bp,
        cosine_sim_theory_bp_per_layer,
    ) = train_bp_cnn(
        args,
        x,
        y,
        model_bp,
        pc_grads,
        pc_grads_per_layer,
        tracked_param_positions,
        tracked_layer_names,
    )

    save_dir = setup_theory_experiment(args)
    os.makedirs(save_dir, exist_ok=True)
    if cosine_similarities is not None:
        np.save(f"{save_dir}/grad_cosine_similarities.npy", cosine_similarities)
        np.save(
            f"{save_dir}/grad_cosine_similarities_pc_successful_steps.npy",
            np.asarray(pc_successful_steps[: len(cosine_similarities)], dtype=np.int64),
        )
    if cosine_similarities_per_layer is not None:
        np.save(
            f"{save_dir}/grad_cosine_similarities_per_layer.npy",
            cosine_similarities_per_layer,
        )
    if cosine_sim_theory_bp is not None:
        np.save(
            f"{save_dir}/grad_cosine_similarities_theory_bp.npy",
            cosine_sim_theory_bp,
        )
    if cosine_sim_theory_bp_per_layer is not None:
        np.save(
            f"{save_dir}/grad_cosine_similarities_theory_bp_per_layer.npy",
            cosine_sim_theory_bp_per_layer,
        )
    np.save(
        f"{save_dir}/grad_cosine_similarities_layer_names.npy",
        np.array(tracked_layer_names, dtype=object),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir", type=str, default="theory_results")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["cifar", "tinyimagenet", "imagenet"])

    # Model parameters
    parser.add_argument("--widths", type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--n_res_blocks", type=int, default=3)
    parser.add_argument("--param_type", type=str, default="mupc", choices=["sp", "mupc"])
    parser.add_argument("--act_fn", type=str, default="linear")
    parser.add_argument("--scale_non_res_layers", action="store_true", default=False)
    parser.add_argument("--additive_depth_factor", type=int, default=4)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--param_optim", type=str, default="adam", choices=["gd", "adam"])
    parser.add_argument("--param_lr", type=float, default=1e-3)
    parser.add_argument("--loss_id", type=str, default="ce", choices=["mse", "ce"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pc_grad_type", type=str, default="numerical", choices=["numerical", "theoretical"])

    # Inference parameters
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[3e-1]) 
    parser.add_argument("--n_infer_iters", type=int, nargs='+', default=[200])
    parser.add_argument("--infer_energy_thresh", type=float, default=0)
    parser.add_argument("--use_amortiser", action="store_true", default=False)

    # Activity Hessian condition number estimation
    parser.add_argument("--estimate_hessian_cond", action="store_true", default=False)  # ~10^20
    parser.add_argument("--hessian_power_iters", type=int, default=25, help="Power iteration steps for λ_max.")
    parser.add_argument("--hessian_inverse_iters", type=int, default=10, help="Inverse iteration steps for λ_min.")
    parser.add_argument("--hessian_cg_iters", type=int, default=25, help="CG steps per inverse iteration.")

    args = parser.parse_args()

    for n_infer_iters in args.n_infer_iters:
        args.n_infer_iters = n_infer_iters
        for width in args.widths:
            args.width = width
            for activity_lr in args.activity_lrs:
                args.activity_lr = activity_lr
                main(args)
