"""Kernel-alignment analysis extracted from ``train.py``.

Runs finite-width PC simulations, then PC DMFT theory, then BP DMFT theory.
Finite-size backpropagation and PC–BP gradient cosine similarities are omitted.

Use the ``PC_dmft_env`` conda environment:
    /data/ndcn-computational-neuroscience/mert5001/envs/PC_dmft_env/bin/python analyse_alignment.py
"""

import os
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import jpc

from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import set_seed
from experiments.limits_paper.utils import setup_pc_experiment
from experiments.dmft.utils import (
    create_toy_dataset,
    cleanup_experiment_dirs,
    train_pcn,
)
from theory_utils import solve_kernels, solve_kernels_nonlin, get_Delta, solve_Delta
from theory_pc_utils import solve_pc_kernels
from theory_pc_nonlin_utils import solve_pc_kernels_nonlin
from plot_dmft_results import (
    plot_dmft_kernels_and_loss,
    plot_pc_dmft_kernels_and_loss,
    plot_pc_theory_vs_finite_loss,
)


def _train_finite_pc(
    key,
    *,
    results_dir,
    input_dim,
    output_dim,
    n_samples,
    n_hidden,
    use_skips,
    act_fn,
    param_type,
    param_lr,
    gamma_0,
    param_optim_id,
    n_train_iters,
    infer_mode,
    n_infer_iters,
    activity_lr,
    width,
    loss_id,
    seed,
    X_input,
    Y_target,
):
    """Run one finite-width PC training job and return the loss trajectory."""
    save_dir = setup_pc_experiment(
        results_dir=results_dir,
        input_dim=input_dim,
        n_samples=n_samples,
        n_hidden=n_hidden,
        use_skips=use_skips,
        act_fn=act_fn,
        param_type=param_type,
        param_lr=param_lr,
        gamma_0=gamma_0,
        param_optim_id=param_optim_id,
        n_train_iters=n_train_iters,
        infer_mode=infer_mode,
        n_infer_iters=n_infer_iters,
        activity_lr=activity_lr,
        width=width,
        loss_id=loss_id,
        seed=seed,
    )
    model = jpc.make_mlp(
        key,
        input_dim=input_dim,
        width=width,
        depth=n_hidden + 1,
        output_dim=output_dim,
        act_fn=act_fn,
        use_bias=False,
        param_type=param_type,
    )
    train_pcn(
        model=model,
        use_skips=use_skips,
        X_input=X_input,
        Y_target=Y_target,
        width=width,
        gamma_0=gamma_0,
        param_type=param_type,
        infer_mode=infer_mode,
        n_infer_iters=n_infer_iters,
        activity_lr=activity_lr,
        param_optim_id=param_optim_id,
        param_lr=param_lr,
        n_train_iters=n_train_iters,
        save_dir=save_dir,
        store_grads=False,
        loss_id=loss_id,
    )
    return np.load(f"{save_dir}/train_losses.npy")


def _loss_records(losses, **meta):
    records = []
    for t, loss in enumerate(np.asarray(losses).flatten(), start=1):
        records.append({**meta, "t": t, "loss": float(loss)})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "Fashion-MNIST", "CIFAR10"])
    parser.add_argument("--input_dim", type=int, default=20) # 40)
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
    parser.add_argument("--infer_mode", type=str, default="optim", choices=["optim", "closed_form"])
    parser.add_argument("--n_infer_iters", type=int, default=5)
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[0.05])

    # Loop parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_hiddens", type=int, nargs='+', default=[5])
    parser.add_argument("--widths", type=int, nargs='+',
        # default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        default=[128, 512] #, 2048] #, 8192]
    )

    # DMFT theory parameters (shared by BP and PC)
    parser.add_argument(
        "--nonlin_beta",
        type=float,
        default=1.0,
        help="Steepness for tanh/softplus in nonlinear DMFT theory.",
    )
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=1000,
        help="Monte-Carlo samples for nonlinear BP/PC DMFT theory.",
    )

    # BP DMFT parameters
    parser.add_argument(
        "--bp_damping",
        type=float,
        default=1.0,
        help="Kernel mixing factor for nonlinear BP DMFT fixed-point updates.",
    )
    parser.add_argument(
        "--skip_theory_bp",
        action="store_true",
        default=False,
        help="Skip BP DMFT theory (kernels and loss).",
    )

    # PC DMFT parameters
    parser.add_argument(
        "--pc_damping",
        type=float,
        default=1.0,
        help="Kernel mixing factor for PC DMFT fixed-point updates.",
    )
    parser.add_argument(
        "--pc_tolerance",
        type=float,
        default=1e-5,
        help="Early-stop tolerance for PC DMFT fixed-point residual.",
    )
    parser.add_argument(
        "--pc_backend",
        type=str,
        default="optimised",
        choices=["optimised", "reference"],
        help=(
            "PC DMFT linear solver: 'optimised' (default, reduced Delta "
            "system + jitted Jacobi sweep) or 'reference' (full 2n x 2n "
            "block system; slower, for debugging)."
        ),
    )
    parser.add_argument(
        "--num_jacobian_samples",
        type=int,
        default=None,
        help=(
            "MC samples for nonlinear PC response Jacobians "
            "(default: min(num_mc_samples, 200))."
        ),
    )
    parser.add_argument(
        "--jacobian_batch_size",
        type=int,
        default=25,
        help="Batch size for nonlinear PC Jacobian samples. Batch size fixed at 50 for BP",
    )
    parser.add_argument(
        "--skip_finite_pc",
        action="store_true",
        default=False,
        help="Skip finite-width PC simulations (iterative and closed-form).",
    )
    parser.add_argument(
        "--cleanup_npy",
        action="store_true",
        default=True,
        help=(
            "After the run, delete finite-sim result directories "
            "(*_input_dim under results_dir), keeping plot pngs."
        ),
    )
    args = parser.parse_args()

    # PC DMFT inverts (K*T*P) matrices; float64 helps stability.
    jax.config.update("jax_enable_x64", True)

    os.makedirs(args.results_dir, exist_ok=True)
    use_nonlin_theory = args.act_fn != "linear"
    if use_nonlin_theory and args.act_fn == "relu" and not args.skip_theory_bp:
        raise ValueError(
            "Nonlinear BP DMFT (solve_kernels_nonlin) supports only "
            "'tanh' (and softplus in the solver API). Use --act_fn tanh "
            "or --skip_theory_bp."
        )

    for seed in range(args.seed, args.seed + args.n_seeds):
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
        loss_id = "mse" if args.dataset == "toy" else args.loss_id

        for n_hidden in args.n_hiddens:
            print(f"\n\tn hidden H = {n_hidden}")

            for use_skips in args.use_skips:
                print(f"\n\t\tuse_skips = {use_skips}")

                for gamma_0 in args.gamma_0s:
                    print(f"\n\t\t\tgamma_0 = {gamma_0}")

                    for param_type in args.param_types:
                        print(f"\n\t\t\t\tparam_type = {param_type}")

                        width_keys = jax.random.split(
                            model_key, len(args.widths)
                        )

                        for activity_lr in args.activity_lrs:
                            print(f"\n\t\t\t\t\tactivity_lr = {activity_lr}")

                            K_inf = args.n_infer_iters
                            T_train = args.n_train_iters
                            P = args.n_samples
                            plots_dir = os.path.join(args.results_dir, "plots")

                            # --- Finite-size PC simulation (infer) ---
                            finite_pc_records = []
                            finite_pc_theory_records = []
                            if not args.skip_finite_pc:
                                print(
                                    "\t\t\t\t\tRunning finite-size PC simulation "
                                    f"for widths {args.widths}...\n"
                                )
                                for width, wkey in zip(args.widths, width_keys):
                                    print(
                                        "\t\t\t\t\tNumerical PC simulation "
                                        f"for width N = {width}"
                                    )
                                    losses = _train_finite_pc(
                                        wkey,
                                        results_dir=args.results_dir,
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        n_samples=args.n_samples,
                                        n_hidden=n_hidden,
                                        use_skips=use_skips,
                                        act_fn=args.act_fn,
                                        param_type=param_type,
                                        param_lr=args.param_lr_pc,
                                        gamma_0=gamma_0,
                                        param_optim_id=args.param_optim,
                                        n_train_iters=T_train,
                                        infer_mode="optim",
                                        n_infer_iters=K_inf,
                                        activity_lr=activity_lr,
                                        width=width,
                                        loss_id=loss_id,
                                        seed=seed,
                                        X_input=X_input,
                                        Y_target=Y_target,
                                    )
                                    finite_pc_records.extend(
                                        _loss_records(losses, width=width)
                                    )

                                # Closed-form equilib grads (linear nets).
                                if args.act_fn == "linear":
                                    print(
                                        "\t\t\t\t\tRunning finite-size PC "
                                        "simulation (theory update) for "
                                        f"widths {args.widths}...\n"
                                    )
                                    for width, wkey in zip(
                                        args.widths, width_keys
                                    ):
                                        print(
                                            "\t\t\t\t\tNumerical PC simulation "
                                            f"(theory) for width N = {width}"
                                        )
                                        theory_losses = _train_finite_pc(
                                            wkey,
                                            results_dir=args.results_dir,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            n_samples=args.n_samples,
                                            n_hidden=n_hidden,
                                            use_skips=use_skips,
                                            act_fn=args.act_fn,
                                            param_type=param_type,
                                            param_lr=args.param_lr_pc,
                                            gamma_0=gamma_0,
                                            param_optim_id=args.param_optim,
                                            n_train_iters=T_train,
                                            infer_mode="closed_form",
                                            n_infer_iters=K_inf,
                                            activity_lr=activity_lr,
                                            width=width,
                                            loss_id=loss_id,
                                            seed=seed,
                                            X_input=X_input,
                                            Y_target=Y_target,
                                        )
                                        finite_pc_theory_records.extend(
                                            _loss_records(
                                                theory_losses, width=width
                                            )
                                        )

                            # --- Calculate theory (PC) ---
                            n_pc = K_inf * T_train * P
                            if use_nonlin_theory:
                                print(
                                    "\t\t\t\t\tCalculating nonlinear PC Theory "
                                    f"(act_fn={args.act_fn}, "
                                    f"matrix size n = K*T*P = {n_pc})...\n"
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
                                ) = solve_pc_kernels_nonlin(
                                    Kx=jnp.asarray(Kx, dtype=jnp.float64),
                                    y=y,
                                    depth=n_hidden,
                                    eta=args.param_lr_pc,
                                    gamma=gamma_0,
                                    beta_h=activity_lr,
                                    hidden_energy_scaling=n_hidden + 1,
                                    num_training_steps=T_train,
                                    num_inference_steps=K_inf,
                                    num_fixed_point_steps=args.n_fixed_point_steps,
                                    num_mc_samples=args.num_mc_samples,
                                    num_jacobian_samples=args.num_jacobian_samples,
                                    jacobian_batch_size=args.jacobian_batch_size,
                                    damping=args.pc_damping,
                                    nonlinearity=args.act_fn,
                                    beta=args.nonlin_beta,
                                    tolerance=args.pc_tolerance,
                                    seed=seed,
                                )
                            else:
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
                                    hidden_energy_scaling=n_hidden + 1,
                                    num_training_steps=T_train,
                                    num_inference_steps=K_inf,
                                    num_fixed_point_steps=args.n_fixed_point_steps,
                                    damping=args.pc_damping,
                                    tolerance=args.pc_tolerance,
                                    backend=args.pc_backend,
                                )
                            print(
                                "\t\t\t\t\tPC fixed-point residual = "
                                f"{float(pc_diagnostics['fixed_point_residual']):.3e}, "
                                "equation residual = "
                                f"{float(pc_diagnostics['equation_residual']):.3e} "
                                f"after {pc_diagnostics['iterations']} iters\n"
                            )
                            plot_pc_dmft_kernels_and_loss(
                                all_Ch=all_Ch,
                                all_Cdelta=all_Cdelta,
                                pc_dmft_loss=pc_dmft_loss,
                                plots_dir=plots_dir,
                                num_inference_steps=K_inf,
                                num_training_steps=T_train,
                                num_samples=P,
                                gamma_0=gamma_0,
                                n_hidden=n_hidden,
                                activity_lr=activity_lr,
                            )

                            if finite_pc_records:
                                plot_pc_theory_vs_finite_loss(
                                    pc_dmft_loss=pc_dmft_loss,
                                    finite_df=pd.DataFrame(finite_pc_records),
                                    plots_dir=plots_dir,
                                    gamma_0=gamma_0,
                                    n_hidden=n_hidden,
                                    activity_lr=activity_lr,
                                    update_mode="infer",
                                    skip_theory=False,
                                )
                            if finite_pc_theory_records:
                                plot_pc_theory_vs_finite_loss(
                                    pc_dmft_loss=pc_dmft_loss,
                                    finite_df=pd.DataFrame(
                                        finite_pc_theory_records
                                    ),
                                    plots_dir=plots_dir,
                                    gamma_0=gamma_0,
                                    n_hidden=n_hidden,
                                    activity_lr=activity_lr,
                                    update_mode="theory",
                                    skip_theory=False,
                                )

                            # --- Calculate theory (BP) ---
                            if not args.skip_theory_bp:
                                if use_nonlin_theory:
                                    print(
                                        "\t\t\t\t\tCalculating nonlinear BP "
                                        f"Theory (act_fn={args.act_fn})...\n"
                                    )
                                    all_H, all_G, _, _ = solve_kernels_nonlin(
                                        Kx=Kx,
                                        y=y,
                                        depth=n_hidden,
                                        eta=args.param_lr,
                                        gamma=gamma_0,
                                        T=args.n_train_iters,
                                        num_iter=args.n_fixed_point_steps,
                                        samples=args.num_mc_samples,
                                        damping=args.bp_damping,
                                        nonlin=args.act_fn,
                                        beta=args.nonlin_beta,
                                    )
                                    Delta_theory = solve_Delta(
                                        Kx=Kx,
                                        y=y,
                                        all_Phi=all_H,
                                        all_G=all_G,
                                        eta=args.param_lr,
                                    )
                                    dmft_loss = 0.5 * jnp.mean(
                                        Delta_theory**2, axis=1
                                    )
                                else:
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
                                    dmft_loss = 0.5 * jnp.mean(
                                        jnp.sum(Delta_theory**2, axis=2), axis=1
                                    )

                                plot_dmft_kernels_and_loss(
                                    all_H=all_H,
                                    all_G=all_G,
                                    dmft_loss=dmft_loss,
                                    plots_dir=plots_dir,
                                    gamma_0=gamma_0,
                                    n_hidden=n_hidden,
                                )

    if args.cleanup_npy:
        removed_dirs = cleanup_experiment_dirs(args.results_dir)
        if removed_dirs:
            print(
                f"\nRemoved {len(removed_dirs)} experiment dir(s) "
                f"under {args.results_dir} (png plots kept):"
            )
            for d in removed_dirs:
                print(f"  - {d}")
        else:
            print(f"\nNo *_input_dim dirs to remove under {args.results_dir}.")


### DEFAULT PARAMETERS ###
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 5 --n_fixed_point_steps 10 --n_train_iters 20 --param_lr 0.05 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 5 --pc_damping 1.0 --gamma_0s 1

### TEST PARAMETERS ###
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 2 --n_fixed_point_steps 10 --n_train_iters 10 --param_lr 0.2 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 3 --pc_damping 1.0 --gamma_0s 1

### THEORY ONLY (skip finite PC) ###
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 5 --n_fixed_point_steps 10 --n_train_iters 20 --param_lr 0.05 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 5 --pc_damping 1.0 --gamma_0s 1 --skip_finite_pc

### PC ONLY (skip BP theory) ###
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 5 --n_fixed_point_steps 10 --n_train_iters 20 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 5 --pc_damping 1.0 --gamma_0s 1 --skip_theory_bp


############ NONLINEAR BELOW ###################
################################################

### THEORY + EMPIRICS ###
# CUDA_VISIBLE_DEVICES=1 python analyse_alignment.py --n_samples 5 --n_fixed_point_steps 10 --n_train_iters 10 --param_lr 0.5 --param_lr_pc 5.0 --activity_lrs 0.2 --n_infer_iters 10 --n_hiddens 5 --bp_damping 0.5 --pc_damping 0.5 --gamma_0s 1 --act_fn tanh --num_mc_samples 1000





