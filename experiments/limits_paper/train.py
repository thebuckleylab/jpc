import jax
import jax.numpy as jnp
import numpy as np

import jpc
import equinox as eqx
import optax

import os
import argparse
from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import set_seed
from utils import (
    setup_pc_experiment, 
    setup_bp_experiment, 
    configure_param_optim,
    create_toy_dataset, 
    MLP, 
    flatten_grads,
    compute_grad_cosine_similarities
)
from theory_utils import solve_kernels, get_Delta


def train_pcn( 
      model,
      use_skips,
      X_input,
      Y_target,
      width,
      gamma_0,
      param_type,
      infer_mode,
      n_infer_iters,
      activity_lr,
      param_optim_id,
      param_lr,
      n_train_iters,
      loss_id,
      save_dir,
      store_grads=False
):    
    os.makedirs(save_dir, exist_ok=True)

    depth = len(model)
    skip_model = jpc.make_skip_model(depth) if use_skips else None

    # Optimisers
    activity_optim = optax.sgd(activity_lr)
    param_optim = configure_param_optim(
        param_optim_id, param_type, use_skips, param_lr, width, depth, gamma_0
    )
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )
    
    num_energies, theory_energies = [], []
    train_losses = []
    loss_rescalings = []
    pc_grads = [] if store_grads else None 
    
    # Initialize activities for the first iteration
    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=X_input,
        skip_model=skip_model,
        param_type=param_type,
        gamma=gamma_0
    )
    
    for _ in range(n_train_iters):

        if infer_mode == "closed_form":
            equilib_energy, S = jpc.linear_equilib_energy(
                params=(model, skip_model), 
                x=X_input, 
                y=Y_target,
                param_type=param_type,
                gamma=gamma_0,
                return_rescaling=True
            )
            theory_energies.append(equilib_energy)
            loss_rescaling = jnp.linalg.norm(S, ord=2) if Y_target.ndim > 1 else S
            loss_rescalings.append(loss_rescaling)
                
        # inference
        if infer_mode == "optim":
            activities = jpc.init_activities_with_ffwd(
                model=model,
                input=X_input,
                skip_model=skip_model,
                param_type=param_type,
                gamma=gamma_0
            )
            activity_opt_state = activity_optim.init(activities)
            for _ in range(n_infer_iters):
                activity_update_result = jpc.update_pc_activities(
                    params=(model, skip_model),
                    activities=activities,
                    optim=activity_optim,
                    opt_state=activity_opt_state,
                    output=Y_target,
                    input=X_input,
                    param_type=param_type,
                    gamma=gamma_0,
                    loss_id=loss_id
                )
                activities = activity_update_result["activities"]
                activity_opt_state = activity_update_result["opt_state"]
                energy = activity_update_result["energy"]
            
            num_energies.append(energy)

            param_update_result = jpc.update_pc_params(
                params=(model, skip_model),
                activities=activities,
                optim=param_optim,
                opt_state=param_opt_state,
                output=Y_target,
                input=X_input,
                param_type=param_type,
                gamma=gamma_0,
                loss_id=loss_id
            )

        else:
            # learning with closed form energy
            param_update_result = jpc.update_linear_equilib_energy_params(
                params=(model, skip_model),
                optim=param_optim,
                opt_state=param_opt_state,
                y=Y_target,
                x=X_input,
                param_type=param_type,
                gamma=gamma_0
            )
        
        model = param_update_result["model"]
        skip_model = param_update_result["skip_model"]
        param_opt_state = param_update_result["opt_state"]
        grads = param_update_result["grads"]
        
        if pc_grads is not None:
            flat_grads = flatten_grads(grads)
            # Convert JAX array to numpy immediately to free memory
            pc_grads.append(np.array(flat_grads))
            del flat_grads, grads

        activities = jpc.init_activities_with_ffwd(
            model=model,
            input=X_input,
            skip_model=skip_model,
            param_type=param_type,
            gamma=gamma_0
        )
        if loss_id == "mse":
            train_loss = jpc.mse_loss(activities[-1], Y_target)
        else:
            train_loss = jpc.cross_entropy_loss(activities[-1], Y_target)
        train_losses.append(train_loss)

    energies = (
        jnp.array(theory_energies) 
        if infer_mode == "closed_form" 
        else jnp.array(num_energies)
    )
    np.save(f"{save_dir}/energies.npy", energies)
    np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_dir}/loss_rescalings.npy", loss_rescalings)
    
    return pc_grads


def train_bpn(
      model,
      use_skips,
      X_input,
      Y_target,
      width,
      gamma_0,
      param_type,
      optim_id,
      param_lr,
      n_train_iters,
      loss_id,
      save_dir,
      store_grads=False
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimiser
    optim = configure_param_optim(
        optim_id, param_type, use_skips, param_lr, gamma_0, width, model.L
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    if loss_id == "mse":
        @eqx.filter_jit
        def loss_fn(model, x, y):
            y_pred = jax.vmap(model)(x)
            return 0.5 * jnp.mean(jnp.sum((y - y_pred) ** 2, axis=1))
    else:
        @eqx.filter_jit
        def loss_fn(model, x, y):
            y_pred = jax.vmap(model)(x)
            return jpc.cross_entropy_loss(y_pred, y)

    @eqx.filter_jit
    def make_step(model, optim, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            updates=grads, 
            state=opt_state, 
            params=eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, grads
        
    losses = []
    bp_grads = [] if store_grads else None
    
    for _ in range(n_train_iters):
        model, opt_state, loss, grads = make_step(
            model, optim, opt_state, X_input, Y_target
        )
        losses.append(float(loss))
        
        if bp_grads is not None:
            flat_grads = flatten_grads(grads)
            # Convert JAX array to numpy immediately to free memory
            bp_grads.append(np.array(flat_grads))
            del flat_grads, grads
    
    np.save(f"{save_dir}/losses.npy", losses)
    
    return bp_grads


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "Fashion-MNIST", "CIFAR10"])
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=20)
    
    # Model parameters
    parser.add_argument("--act_fn", type=str, default="linear", choices=["linear", "tanh", "relu"])
    parser.add_argument("--param_types", type=str, nargs='+', default=["mupc"], choices=["mupc", "sp"])
    parser.add_argument("--use_skips", nargs='+', default=[True, False])

    # Training parameters
    parser.add_argument("--param_optim", type=str, default="gd")
    parser.add_argument("--param_lr", type=float, default=0.025)
    parser.add_argument("--gamma_0s", type=float, nargs='+', default=[1])
    parser.add_argument("--n_train_iters", type=int, default=100)
    parser.add_argument("--loss_id", type=str, default="ce", choices=["mse", "ce"])
    
    # Inference parameters
    parser.add_argument("--infer_mode", type=str, default="closed_form", choices=["optim", "closed_form"])
    parser.add_argument("--n_infer_iters", type=int, default=20)
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[5e-1])
    
    # Loop parameters
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_hiddens", type=int, nargs='+', default=[4])
    parser.add_argument("--widths", type=int, nargs='+', 
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    
    # Other parameters
    parser.add_argument("--compute_cos_sims", action="store_true", default=True)
    args = parser.parse_args()

    if len(args.n_hiddens) > 1 and len(args.widths) > 1:
        # NOTE: need higher precision for large width & depth computation of s(theta)
        import jax
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

                            # --- Calculate theory ---
                            if args.param_optim == "gd" and param_type != "sp" and args.n_train_iters <= 100 and n_hidden <= 8 and not use_skips:
                                print("\t\t\t\t\tCalculating Theory...\n")
                                all_H, all_G, _, _ = solve_kernels(
                                    Kx=Kx, 
                                    y=y, 
                                    depth=n_hidden, 
                                    eta=args.param_lr, 
                                    gamma=gamma_0, 
                                    T=args.n_train_iters
                                )
                                Delta_theory = get_Delta(
                                    all_H=all_H, 
                                    all_G=all_G, 
                                    Kx=Kx, 
                                    y=y, 
                                    eta=args.param_lr
                                )
                                dmft_loss = 0.5 * jnp.mean(jnp.sum(Delta_theory**2, axis=2), axis=1) 
                                np.save(
                                    f"{args.results_dir}/dmft_loss_{gamma_0}_gamma_0_seed_{seed}.npy", 
                                    dmft_loss
                                )

                            # In this dataset, we treat the whole P samples as one batch
                            X_input = X.T # Shape (P, D)
                            Y_target = y[:, None] if y.ndim == 1 else y

                            # Loss: toy always MSE
                            loss_id = "mse" if args.dataset == "toy" else args.loss_id

                            # --- Run Numerical Experiment ---
                            for width in args.widths:
                                print(f"\t\t\t\t\tNumerical simulation for width N = {width}")

                                n_infer_iters = args.n_infer_iters if (
                                    args.infer_mode == "closed_form"
                                ) else n_hidden * 100

                                # --- PC ---
                                pc_save_dir = setup_pc_experiment(
                                    results_dir=args.results_dir,
                                    input_dim=input_dim,
                                    n_samples=args.n_samples,
                                    n_hidden=n_hidden,
                                    use_skips=use_skips,
                                    act_fn=args.act_fn,
                                    param_type=param_type,
                                    param_lr=args.param_lr,
                                    gamma_0=gamma_0,
                                    param_optim_id=args.param_optim,
                                    n_train_iters=args.n_train_iters,
                                    infer_mode=args.infer_mode,
                                    n_infer_iters=n_infer_iters,
                                    activity_lr=activity_lr,
                                    width=width,
                                    loss_id=loss_id,
                                    seed=seed
                                )
                                pc_model = jpc.make_mlp(
                                    model_key, 
                                    input_dim=input_dim,
                                    width=width,
                                    depth=n_hidden + 1,
                                    output_dim=output_dim,
                                    act_fn=args.act_fn,
                                    use_bias=False,
                                    param_type=param_type
                                )
                                pc_grads = train_pcn(
                                    model=pc_model,
                                    use_skips=use_skips,
                                    X_input=X_input,
                                    Y_target=Y_target,
                                    width=width,
                                    gamma_0=gamma_0,
                                    param_type=param_type,
                                    infer_mode=args.infer_mode,
                                    n_infer_iters=n_infer_iters,
                                    activity_lr=activity_lr,
                                    param_optim_id=args.param_optim,
                                    param_lr=args.param_lr,
                                    n_train_iters=args.n_train_iters,
                                    save_dir=pc_save_dir,
                                    store_grads=args.compute_cos_sims,
                                    loss_id=loss_id
                                )
                    
                                # --- BP ---
                                bp_save_dir = setup_bp_experiment(
                                    results_dir=args.results_dir,
                                    input_dim=input_dim,
                                    n_samples=args.n_samples,
                                    n_hidden=n_hidden,
                                    use_skips=use_skips,
                                    act_fn=args.act_fn,
                                    param_type=param_type,
                                    optim_id=args.param_optim,
                                    param_lr=args.param_lr,
                                    gamma_0=gamma_0,
                                    n_train_iters=args.n_train_iters,
                                    width=width,
                                    loss_id=loss_id,
                                    seed=seed
                                )
                                bp_model = MLP(
                                    key=model_key,
                                    d_in=input_dim,
                                    N=width,
                                    L=n_hidden + 1,
                                    d_out=output_dim,
                                    act_fn=args.act_fn,
                                    param_type=param_type,
                                    gamma=gamma_0,
                                    use_bias=False,
                                    use_skips=use_skips
                                )
                                # Copy weights from PC model to ensure same random initialisation
                                for i in range(len(pc_model)):
                                    pc_weight = pc_model[i][1].weight
                                    bp_model = eqx.tree_at(
                                        lambda m: m.layers[i][1].weight,
                                        bp_model,
                                        pc_weight
                                    )

                                # Verify all layers at once
                                all_match = True
                                for i in range(len(pc_model)):
                                    pc_weight = pc_model[i][1].weight
                                    bp_weight = bp_model.layers[i][1].weight
                                    if not jnp.allclose(pc_weight, bp_weight, atol=1e-10):
                                        all_match = False
                                        break
                                if all_match:
                                    print(f"\t\t\t\t\t✓ PC and BP models have identical random initialization\n")
                                else:
                                    print(f"\n\t\t\t\t✗ WARNING: Some weights don't match!\n")

                                bp_grads = train_bpn(
                                    model=bp_model,
                                    use_skips=use_skips,
                                    X_input=X_input,
                                    Y_target=Y_target,
                                    width=width,
                                    gamma_0=gamma_0,
                                    param_type=param_type,
                                    optim_id=args.param_optim,
                                    param_lr=args.param_lr,
                                    n_train_iters=args.n_train_iters,
                                    save_dir=bp_save_dir,
                                    store_grads=args.compute_cos_sims,
                                    loss_id=loss_id
                                )
                                
                                if args.compute_cos_sims:
                                    cosine_similarities = compute_grad_cosine_similarities(pc_grads, bp_grads)
                                    np.save(
                                        f"{pc_save_dir}/grad_cosine_similarities.npy", 
                                        cosine_similarities
                                    )

                                    print(f"Cosine similarities: {cosine_similarities}")
