import jax
import jax.numpy as jnp
import numpy as np

# import jpc
# import equinox as eqx
# import optax

import os
import argparse
from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import set_seed
from utils import create_toy_dataset

# from utils import (
#     setup_pc_experiment, 
#     setup_bp_experiment, 
#     configure_param_optim,
#     create_toy_dataset, 
#     MLP, 
#     flatten_grads,
#     compute_grad_cosine_similarities
# )
from theory_utils import solve_kernels, get_Delta
from theory_pc_utils import solve_pc_kernels
from plot_dmft_results import (
    plot_dmft_kernels_and_loss,
    plot_pc_dmft_kernels_and_loss,
    plot_pc_theory_vs_finite_loss,
)
from test_coord_check import jpc_model, get_coord_data

# def train_pcn(
#       model,
#       use_skips,
#       X_input,
#       Y_target,
#       width,
#       gamma_0,
#       param_type,
#       infer_mode,
#       n_infer_iters,
#       activity_lr,
#       param_optim_id,
#       param_lr,
#       n_train_iters,
#       loss_id,
#       save_dir,
#       store_grads=False
# ):    
#     os.makedirs(save_dir, exist_ok=True)

#     depth = len(model)
#     skip_model = jpc.make_skip_model(depth) if use_skips else None

#     # Optimisers
#     batch_size = X_input.shape[0]
#     activity_optim = optax.sgd(activity_lr * batch_size)
#     param_optim = configure_param_optim(
#         param_optim_id, param_type, use_skips, param_lr, width, depth, gamma_0
#     )
#     param_opt_state = param_optim.init(
#         (eqx.filter(model, eqx.is_array), skip_model)
#     )
    
#     num_energies, theory_energies = [], []
#     train_losses = []
#     loss_rescalings = []
#     pc_grads = [] if store_grads else None 
    
#     # Initialize activities for the first iteration
#     activities = jpc.init_activities_with_ffwd(
#         model=model,
#         input=X_input,
#         skip_model=skip_model,
#         param_type=param_type,
#         gamma=gamma_0
#     )
    
#     for _ in range(n_train_iters):

#         if infer_mode == "closed_form":
#             equilib_energy, S = jpc.linear_equilib_energy(
#                 params=(model, skip_model), 
#                 x=X_input, 
#                 y=Y_target,
#                 param_type=param_type,
#                 gamma=gamma_0,
#                 return_rescaling=True
#             )
#             theory_energies.append(equilib_energy)
#             loss_rescaling = jnp.linalg.norm(S, ord=2) if Y_target.ndim > 1 else S
#             loss_rescalings.append(loss_rescaling)
                
#         # inference
#         if infer_mode == "optim":
#             activities = jpc.init_activities_with_ffwd(
#                 model=model,
#                 input=X_input,
#                 skip_model=skip_model,
#                 param_type=param_type,
#                 gamma=gamma_0
#             )
#             activity_opt_state = activity_optim.init(activities)
#             for _ in range(n_infer_iters):
#                 activity_update_result = jpc.update_pc_activities(
#                     params=(model, skip_model),
#                     activities=activities,
#                     optim=activity_optim,
#                     opt_state=activity_opt_state,
#                     output=Y_target,
#                     input=X_input,
#                     param_type=param_type,
#                     gamma=gamma_0,
#                     loss_id=loss_id
#                 )
#                 activities = activity_update_result["activities"]
#                 activity_opt_state = activity_update_result["opt_state"]
#                 energy = activity_update_result["energy"]
            
#             num_energies.append(energy)

#             param_update_result = jpc.update_pc_params(
#                 params=(model, skip_model),
#                 activities=activities,
#                 optim=param_optim,
#                 opt_state=param_opt_state,
#                 output=Y_target,
#                 input=X_input,
#                 param_type=param_type,
#                 gamma=gamma_0,
#                 loss_id=loss_id
#             )

#         else:
#             # learning with closed form energy
#             param_update_result = jpc.update_linear_equilib_energy_params(
#                 params=(model, skip_model),
#                 optim=param_optim,
#                 opt_state=param_opt_state,
#                 y=Y_target,
#                 x=X_input,
#                 param_type=param_type,
#                 gamma=gamma_0
#             )
        
#         model = param_update_result["model"]
#         skip_model = param_update_result["skip_model"]
#         param_opt_state = param_update_result["opt_state"]
#         grads = param_update_result["grads"]
        
#         if pc_grads is not None:
#             flat_grads = flatten_grads(grads)
#             # Convert JAX array to numpy immediately to free memory
#             pc_grads.append(np.array(flat_grads))
#             del flat_grads, grads

#         activities = jpc.init_activities_with_ffwd(
#             model=model,
#             input=X_input,
#             skip_model=skip_model,
#             param_type=param_type,
#             gamma=gamma_0
#         )
#         if loss_id == "mse":
#             train_loss = jpc.mse_loss(activities[-1], Y_target)
#         else:
#             train_loss = jpc.cross_entropy_loss(activities[-1], Y_target)
#         train_losses.append(train_loss)

#     energies = (
#         jnp.array(theory_energies) 
#         if infer_mode == "closed_form" 
#         else jnp.array(num_energies)
#     )
#     np.save(f"{save_dir}/energies.npy", energies)
#     np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
#     np.save(f"{save_dir}/loss_rescalings.npy", loss_rescalings)
    
#     return pc_grads


# def train_bpn(
#       model,
#       use_skips,
#       X_input,
#       Y_target,
#       width,
#       gamma_0,
#       param_type,
#       optim_id,
#       param_lr,
#       n_train_iters,
#       loss_id,
#       save_dir,
#       store_grads=False
# ):
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Optimiser
#     optim = configure_param_optim(
#         optim_id, param_type, use_skips, param_lr, gamma_0, width, model.L
#     )
#     opt_state = optim.init(eqx.filter(model, eqx.is_array))

#     if loss_id == "mse":
#         @eqx.filter_jit
#         def loss_fn(model, x, y):
#             y_pred = jax.vmap(model)(x)
#             return 0.5 * jnp.mean(jnp.sum((y - y_pred) ** 2, axis=1))
#     else:
#         @eqx.filter_jit
#         def loss_fn(model, x, y):
#             y_pred = jax.vmap(model)(x)
#             return jpc.cross_entropy_loss(y_pred, y)

#     @eqx.filter_jit
#     def make_step(model, optim, opt_state, x, y):
#         loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
#         updates, opt_state = optim.update(
#             updates=grads, 
#             state=opt_state, 
#             params=eqx.filter(model, eqx.is_array)
#         )
#         model = eqx.apply_updates(model, updates)
#         return model, opt_state, loss, grads
        
#     losses = []
#     bp_grads = [] if store_grads else None
    
#     for _ in range(n_train_iters):
#         model, opt_state, loss, grads = make_step(
#             model, optim, opt_state, X_input, Y_target
#         )
#         losses.append(float(loss))
        
#         if bp_grads is not None:
#             flat_grads = flatten_grads(grads)
#             # Convert JAX array to numpy immediately to free memory
#             bp_grads.append(np.array(flat_grads))
#             del flat_grads, grads
    
#     np.save(f"{save_dir}/losses.npy", losses)
    
#     return bp_grads


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "Fashion-MNIST", "CIFAR10"])
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=5) # 20)
    
    # Model parameters
    parser.add_argument("--act_fn", type=str, default="linear", choices=["linear", "tanh", "relu"])
    parser.add_argument("--param_types", type=str, nargs='+', default=["mupc"], choices=["mupc", "sp", "my-mup"])
    parser.add_argument("--use_skips", nargs='+', default=[False])

    # Training parameters
    parser.add_argument("--param_optim", type=str, default="gd")
    parser.add_argument("--param_lr", type=float, default=0.05)
    parser.add_argument("--gamma_0s", type=float, nargs='+', default=[1])
    parser.add_argument("--n_train_iters", type=int, default=20) # 100)
    parser.add_argument("--loss_id", type=str, default="mse", choices=["mse", "ce"])
    parser.add_argument("--n_fixed_point_steps", type=int, default=10)
    
    # Inference parameters
    parser.add_argument("--param_lr_pc", type=float, default=0.5)
    parser.add_argument("--infer_mode", type=str, default="closed_form", choices=["optim", "closed_form"])
    parser.add_argument("--n_infer_iters", type=int, default=5)
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[0.05])
    
    # Loop parameters
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_hiddens", type=int, nargs='+', default=[5])
    parser.add_argument("--widths", type=int, nargs='+', 
        # default=[8, 16, 32, 64, 128]  #256, 512, 1024, 2048 
        default=[64, 512, 2048, 8192]
    )
    
    # PC DMFT parameters
    parser.add_argument("--pc_damping", type=float, default=1.0)
    parser.add_argument("--pc_tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--skip_pc_theory",
        action="store_true",
        default=False,
        help="Skip PC DMFT (matrices are K*T*P dimensional and can be costly).",
    )

    # Other parameters
    parser.add_argument("--compute_cos_sims", action="store_true", default=False)
    args = parser.parse_args()

    # PC DMFT inverts (K*T*P) matrices; float64 helps stability.
    # Also needed for large width & depth computation of s(theta).
    if (
        not args.skip_pc_theory
        or (len(args.n_hiddens) > 1 and len(args.widths) > 1)
    ):
        jax.config.update("jax_enable_x64", True)
    
    os.makedirs(args.results_dir, exist_ok=True)
    for seed in range(args.n_seeds):
        print(f"\nRunning experiment for seed: {seed}")

        # --- Set Seed ---
        set_seed(seed)
        key = jax.random.PRNGKey(seed)
        data_key, model_key = jax.random.split(key)
        
        # --- Setup Dataset ---
        if args.dataset == "toy":
            X, y = create_toy_dataset(
                key=data_key, D=args.input_dim, P=args.n_samples
            )
            input_dim = args.input_dim
            output_dim = 1
        else:
            train_loader, _ = get_dataloaders(args.dataset, args.n_samples)
            img_batch, label_batch = next(iter(train_loader))
            
            input_dim = img_batch.shape[1]
            output_dim = label_batch.shape[1]
            print(f"Input dim: {input_dim}, Output dim: {output_dim}")

            X = img_batch.numpy().T
            y = label_batch.numpy()
        
        Kx = X.T @ X / input_dim

        # In this dataset, we treat the whole P samples as one batch, matching
        # the convention assumed by the (whole-batch) DMFT theory.
        X_input = X.T  # Shape (P, D)
        Y_target = y[:, None] if y.ndim == 1 else y

        for n_hidden in args.n_hiddens:
            print(f"\n\tn hidden H = {n_hidden}")

            for use_skips in args.use_skips:
                print(f"\n\t\tuse_skips = {use_skips}")

                for gamma_0 in args.gamma_0s:
                    print(f"\n\t\t\tgamma_0 = {gamma_0}")

                    for param_type in args.param_types:
                        print(f"\n\t\t\t\tparam_type = {param_type}")

                        for activity_lr in args.activity_lrs:
                            print(f"\n\t\t\t\t\tactivity_lr = {activity_lr}")

                            # --- Calculate theory (BP) ---
                            # if args.param_optim == "gd" and param_type != "sp" and args.n_train_iters <= 100 and n_hidden <= 8 and not use_skips:
                            print("\t\t\t\t\tCalculating BP Theory...\n")
                            all_H, all_G, _, _ = solve_kernels(
                                Kx=Kx, 
                                y=y, 
                                depth=n_hidden, 
                                eta=args.param_lr, 
                                gamma=gamma_0, 
                                T=args.n_train_iters,
                                num_steps=args.n_fixed_point_steps
                            )
                            Delta_theory = get_Delta(
                                all_H=all_H, 
                                all_G=all_G, 
                                Kx=Kx, 
                                y=y, 
                                eta=args.param_lr
                            )
                            dmft_loss = 0.5 * jnp.mean(jnp.sum(Delta_theory**2, axis=2), axis=1) 
                            # np.save(f"{args.results_dir}/all_H_{gamma_0}_gamma_0.npy", all_H)
                            # np.save(f"{args.results_dir}/all_G_{gamma_0}_gamma_0.npy", all_G)
                            # np.save(
                            #     f"{args.results_dir}/dmft_loss_{gamma_0}_gamma_0.npy", 
                            #     dmft_loss
                            # )
                            plot_dmft_kernels_and_loss(
                                all_H=all_H,
                                all_G=all_G,
                                dmft_loss=dmft_loss,
                                plots_dir=os.path.join(args.results_dir, "plots"),
                                gamma_0=gamma_0,
                                n_hidden=n_hidden,
                            )

                            # --- Calculate theory (PC) ---
                            if not args.skip_pc_theory:
                                K_inf = args.n_infer_iters
                                T_train = args.n_train_iters
                                P = args.n_samples
                                n_pc = K_inf * T_train * P
                                print(
                                    "\t\t\t\t\tCalculating PC Theory "
                                    f"(matrix size n = K*T*P = {n_pc})...\n"
                                )
                                (
                                    all_Ch,
                                    all_Cdelta,
                                    _all_Rh,
                                    _all_Rdelta,
                                    _C_delta_top,
                                    pc_dmft_loss,
                                    _mean_delta_top,
                                    pc_diagnostics,
                                ) = solve_pc_kernels(
                                    Kx=jnp.asarray(Kx, dtype=jnp.float64),
                                    y=y,
                                    depth=n_hidden,
                                    eta=args.param_lr_pc,
                                    gamma=gamma_0,
                                    beta_h=activity_lr,
                                    num_training_steps=T_train,
                                    num_inference_steps=K_inf,
                                    num_fixed_point_steps=args.n_fixed_point_steps,
                                    damping=args.pc_damping,
                                    tolerance=args.pc_tolerance,
                                )
                                print(
                                    "\t\t\t\t\tPC fixed-point residual = "
                                    f"{float(pc_diagnostics['fixed_point_residual']):.3e}, "
                                    "equation residual = "
                                    f"{float(pc_diagnostics['equation_residual']):.3e} "
                                    f"after {pc_diagnostics['iterations']} iters\n"
                                )
                                suffix = (
                                    f"{gamma_0}_gamma_0_"
                                    f"{activity_lr}_activity_lr"
                                )
                                # np.save(
                                #     f"{args.results_dir}/all_Ch_{suffix}.npy",
                                #     np.array(all_Ch, dtype=object),
                                # )
                                # np.save(
                                #     f"{args.results_dir}/all_Cdelta_{suffix}.npy",
                                #     np.array(all_Cdelta, dtype=object),
                                # )
                                # np.save(
                                #     f"{args.results_dir}/pc_dmft_loss_{suffix}.npy",
                                #     np.asarray(pc_dmft_loss),
                                # )
                                plot_pc_dmft_kernels_and_loss(
                                    all_Ch=all_Ch,
                                    all_Cdelta=all_Cdelta,
                                    pc_dmft_loss=pc_dmft_loss,
                                    plots_dir=os.path.join(
                                        args.results_dir, "plots"
                                    ),
                                    num_inference_steps=K_inf,
                                    num_training_steps=T_train,
                                    num_samples=P,
                                    gamma_0=gamma_0,
                                    n_hidden=n_hidden,
                                    activity_lr=activity_lr,
                                )

                                # --- Finite-size PC simulation (same hyperparameters) ---
                                print(
                                    "\t\t\t\t\tRunning finite-size PC simulation "
                                    f"for widths {args.widths}...\n"
                                )

                                def make_finite_pc_models(mkey, _n_hidden=n_hidden):
                                    width_keys = jax.random.split(
                                        mkey, len(args.widths)
                                    )
                                    return {
                                        width: jpc_model(
                                            width,
                                            key=wkey,
                                            input_dim=input_dim,
                                            depth=_n_hidden + 1,
                                            output_dim=output_dim,
                                            act_fn=args.act_fn,
                                            param_type=param_type,
                                            use_skips=use_skips,
                                        )
                                        for width, wkey in zip(
                                            args.widths, width_keys
                                        )
                                    }

                                finite_pc_df = get_coord_data(
                                    make_finite_pc_models,
                                    [(X_input, Y_target)],
                                    param_type=param_type,
                                    gamma=gamma_0,
                                    optimizer="sgd",
                                    lr=args.param_lr_pc,
                                    activity_lr=activity_lr,
                                    n_infer_iters=K_inf,
                                    nsteps=T_train,
                                    nseeds=1,
                                    seed=seed,
                                    fix_data=True,
                                    record="ffwd",
                                    update_mode="infer",
                                    stats=["loss"],
                                    show_progress=True,
                                )

                                plot_pc_theory_vs_finite_loss(
                                    pc_dmft_loss=pc_dmft_loss,
                                    finite_df=finite_pc_df,
                                    plots_dir=os.path.join(
                                        args.results_dir, "plots"
                                    ),
                                    gamma_0=gamma_0,
                                    n_hidden=n_hidden,
                                    activity_lr=activity_lr,
                                )

                            # # In this dataset, we treat the whole P samples as one batch
                            # X_input = X.T # Shape (P, D)
                            # Y_target = y[:, None] if y.ndim == 1 else y

                            # # Loss: toy always MSE
                            # loss_id = "mse" if args.dataset == "toy" else args.loss_id

                            # # --- Run Numerical Experiment ---
                            # for width in args.widths:
                            #     print(f"\t\t\t\t\tNumerical simulation for width N = {width}")

                            #     n_infer_iters = args.n_infer_iters if (
                            #         args.infer_mode == "closed_form"
                            #     ) else n_hidden * 100

                            #     # --- PC ---
                            #     pc_save_dir = setup_pc_experiment(
                            #         results_dir=args.results_dir,
                            #         input_dim=input_dim,
                            #         n_samples=args.n_samples,
                            #         n_hidden=n_hidden,
                            #         use_skips=use_skips,
                            #         act_fn=args.act_fn,
                            #         param_type=param_type,
                            #         param_lr=args.param_lr,
                            #         gamma_0=gamma_0,
                            #         param_optim_id=args.param_optim,
                            #         n_train_iters=args.n_train_iters,
                            #         infer_mode=args.infer_mode,
                            #         n_infer_iters=n_infer_iters,
                            #         activity_lr=activity_lr,
                            #         width=width,
                            #         loss_id=loss_id,
                            #         seed=seed
                            #     )
                            #     pc_model = jpc.make_mlp(
                            #         model_key, 
                            #         input_dim=input_dim,
                            #         width=width,
                            #         depth=n_hidden + 1,
                            #         output_dim=output_dim,
                            #         act_fn=args.act_fn,
                            #         use_bias=False,
                            #         param_type=param_type
                            #     )
                            #     pc_grads = train_pcn(
                            #         model=pc_model,
                            #         use_skips=use_skips,
                            #         X_input=X_input,
                            #         Y_target=Y_target,
                            #         width=width,
                            #         gamma_0=gamma_0,
                            #         param_type=param_type,
                            #         infer_mode=args.infer_mode,
                            #         n_infer_iters=n_infer_iters,
                            #         activity_lr=activity_lr,
                            #         param_optim_id=args.param_optim,
                            #         param_lr=args.param_lr,
                            #         n_train_iters=args.n_train_iters,
                            #         save_dir=pc_save_dir,
                            #         store_grads=args.compute_cos_sims,
                            #         loss_id=loss_id
                            #     )
                    
                            #     # --- BP ---
                            #     bp_save_dir = setup_bp_experiment(
                            #         results_dir=args.results_dir,
                            #         input_dim=input_dim,
                            #         n_samples=args.n_samples,
                            #         n_hidden=n_hidden,
                            #         use_skips=use_skips,
                            #         act_fn=args.act_fn,
                            #         param_type=param_type,
                            #         optim_id=args.param_optim,
                            #         param_lr=args.param_lr,
                            #         gamma_0=gamma_0,
                            #         n_train_iters=args.n_train_iters,
                            #         width=width,
                            #         loss_id=loss_id,
                            #         seed=seed
                            #     )
                            #     bp_model = MLP(
                            #         key=model_key,
                            #         d_in=input_dim,
                            #         N=width,
                            #         L=n_hidden + 1,
                            #         d_out=output_dim,
                            #         act_fn=args.act_fn,
                            #         param_type=param_type,
                            #         gamma=gamma_0,
                            #         use_bias=False,
                            #         use_skips=use_skips
                            #     )
                            #     # Copy weights from PC model to ensure same random initialisation
                            #     for i in range(len(pc_model)):
                            #         pc_weight = pc_model[i][1].weight
                            #         bp_model = eqx.tree_at(
                            #             lambda m: m.layers[i][1].weight,
                            #             bp_model,
                            #             pc_weight
                            #         )

                            #     # Verify all layers at once
                            #     all_match = True
                            #     for i in range(len(pc_model)):
                            #         pc_weight = pc_model[i][1].weight
                            #         bp_weight = bp_model.layers[i][1].weight
                            #         if not jnp.allclose(pc_weight, bp_weight, atol=1e-10):
                            #             all_match = False
                            #             break
                            #     if all_match:
                            #         print(f"\t\t\t\t\t✓ PC and BP models have identical random initialization\n")
                            #     else:
                            #         print(f"\n\t\t\t\t✗ WARNING: Some weights don't match!\n")

                            #     bp_grads = train_bpn(
                            #         model=bp_model,
                            #         use_skips=use_skips,
                            #         X_input=X_input,
                            #         Y_target=Y_target,
                            #         width=width,
                            #         gamma_0=gamma_0,
                            #         param_type=param_type,
                            #         optim_id=args.param_optim,
                            #         param_lr=args.param_lr,
                            #         n_train_iters=args.n_train_iters,
                            #         save_dir=bp_save_dir,
                            #         store_grads=args.compute_cos_sims,
                            #         loss_id=loss_id
                            #     )
                                
                            #     if args.compute_cos_sims:
                            #         cosine_similarities = compute_grad_cosine_similarities(pc_grads, bp_grads)
                            #         np.save(
                            #             f"{pc_save_dir}/grad_cosine_similarities.npy", 
                            #             cosine_similarities
                            #         )

### DEFAULT PARAMETERS ###
# CUDA_VISIBLE_DEVICES=1 python train.py --n_samples 5 --n_fixed_point_steps 10 --n_train_iters 20 --param_lr 0.05 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 5 --pc_damping 1.0 --gamma_0s 1

### TEST PARAMETERS ### 
# CUDA_VISIBLE_DEVICES=1 python train.py --n_samples 5 --n_fixed_point_steps 10 --n_train_iters 30 --param_lr 0.2 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 5 --pc_damping 1.0 --gamma_0s 1

# CUDA_VISIBLE_DEVICES=1 python train.py --n_samples 2 --n_fixed_point_steps 10 --n_train_iters 10 --param_lr 0.2 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 50 --n_hiddens 3 --pc_damping 1.0 --gamma_0s 1