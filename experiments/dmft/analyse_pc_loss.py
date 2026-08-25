"""PC-only analysis extracted from ``train.py``.

Runs PC DMFT theory and finite-width PC simulations (iterative inference,
plus closed-form equilibrium updates for linear nets). Backpropagation
theory, finite BP simulations, and PC–BP gradient cosine similarities
are omitted.

Use the ``PC_dmft_env`` conda environment:
    /data/ndcn-computational-neuroscience/mert5001/envs/PC_dmft_env/bin/python analyse_pc_loss.py
"""

import os
import sys
from pathlib import Path

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
from theory_pc_utils import solve_pc_kernels
from theory_pc_nonlin_utils import solve_pc_kernels_nonlin
from plot_dmft_results import (
    plot_pc_theory_vs_finite_loss,
    plot_pc_param_sweep_loss,
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
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=5) # 20)

    # Model parameters
    parser.add_argument("--act_fn", type=str, default="linear", choices=["linear", "tanh", "relu"])
    parser.add_argument("--param_types", type=str, nargs='+', default=["mupc"], choices=["mupc", "sp", "my-mup"])
    parser.add_argument("--use_skips", nargs='+', default=[False])

    # Training parameters
    parser.add_argument("--param_optim", type=str, default="gd")
    parser.add_argument("--gamma_0s", type=float, nargs='+', default=[1])
    parser.add_argument("--n_train_iters", type=int, default=20) # 100)
    parser.add_argument("--loss_id", type=str, default="mse", choices=["mse", "ce"])
    parser.add_argument("--n_fixed_point_steps", type=int, default=10)

    # Inference parameters
    parser.add_argument("--param_lr_pc", type=float, default=0.5)
    parser.add_argument("--infer_mode", type=str, default="optim", choices=["optim", "closed_form"])
    parser.add_argument("--n_infer_iters", type=int, nargs='+', default=[5])
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[0.05])
    parser.add_argument(
        "--plot_closed_form",
        action="store_true",
        default=False,
        help=(
            "On n_hiddens / gamma_0s overlay plots, also draw finite-size "
            "closed-form equilibrium updates (one curve per swept value). "
            "n_infer_iters overlays always include a single closed-form "
            "curve, since it does not depend on K. Linear networks only."
        ),
    )

    # Loop parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--n_hiddens", type=int, nargs='+', default=[5])
    parser.add_argument("--widths", type=int, nargs='+',
        # default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        # default=[128, 512, 2048, 8192]
        default=[1024]
    )

    # PC DMFT theory parameters
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
        help="Monte-Carlo samples for nonlinear PC DMFT theory.",
    )
    parser.add_argument(
        "--skip_theory",
        action="store_true",
        default=False,
        help=(
            "Skip PC DMFT theory (PC matrices are K*T*P dimensional "
            "and can be costly)."
        ),
    )
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
        help="Batch size for nonlinear PC Jacobian samples.",
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
    # Also needed for large width & depth computation of s(theta).
    if (
        not args.skip_theory
        or (len(args.n_hiddens) > 1 and len(args.widths) > 1)
    ):
        jax.config.update("jax_enable_x64", True)

    os.makedirs(args.results_dir, exist_ok=True)
    use_nonlin_theory = args.act_fn != "linear"
    overlay_n_hiddens = len(args.n_hiddens) > 1
    overlay_gamma_0s = len(args.gamma_0s) > 1
    overlay_n_infer_iters = len(args.n_infer_iters) > 1
    overlay_any = (
        overlay_n_hiddens or overlay_gamma_0s or overlay_n_infer_iters
    )
    max_width = max(args.widths)
    run_widths = [max_width] if overlay_any else list(args.widths)

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
        theory_records = []
        finite_records = []

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
                        width_key_map = dict(zip(args.widths, width_keys))

                        for activity_lr in args.activity_lrs:
                            print(f"\n\t\t\t\t\tactivity_lr = {activity_lr}")

                            for K_inf in args.n_infer_iters:
                                print(f"\n\t\t\t\t\t\tn_infer_iters = {K_inf}")

                                # --- Calculate theory (PC) ---
                                pc_dmft_loss = None
                                T_train = args.n_train_iters
                                P = args.n_samples
                                if not args.skip_theory:
                                    n_pc = K_inf * T_train * P
                                    if use_nonlin_theory:
                                        print(
                                            "\t\t\t\t\tCalculating nonlinear PC Theory "
                                            f"(act_fn={args.act_fn}, "
                                            f"matrix size n = K*T*P = {n_pc})...\n"
                                        )
                                        (
                                            _all_Ch,
                                            _all_Cdelta,
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
                                            _all_Ch,
                                            _all_Cdelta,
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

                                sweep_meta = dict(
                                    n_hidden=n_hidden,
                                    gamma_0=gamma_0,
                                    activity_lr=activity_lr,
                                    n_infer_iters=K_inf,
                                    param_type=param_type,
                                    use_skips=use_skips,
                                )
                                if pc_dmft_loss is not None:
                                    theory_records.extend(
                                        _loss_records(
                                            pc_dmft_loss, **sweep_meta
                                        )
                                    )

                                # --- Finite-size PC simulation (infer) ---
                                print(
                                    "\t\t\t\t\tRunning finite-size PC simulation "
                                    f"for widths {run_widths}...\n"
                                )
                                finite_pc_records = []
                                for width in run_widths:
                                    print(
                                        "\t\t\t\t\tNumerical PC simulation "
                                        f"for width N = {width}"
                                    )
                                    losses = _train_finite_pc(
                                        width_key_map[width],
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
                                    recs = _loss_records(
                                        losses,
                                        width=width,
                                        infer_mode="infer",
                                        **sweep_meta,
                                    )
                                    finite_pc_records.extend(recs)
                                    finite_records.extend(recs)

                                if not overlay_any:
                                    finite_pc_df = pd.DataFrame(
                                        finite_pc_records
                                    )
                                    # None / zeros => finite overlays only.
                                    plot_pc_theory_vs_finite_loss(
                                        pc_dmft_loss=(
                                            pc_dmft_loss
                                            if pc_dmft_loss is not None
                                            else jnp.zeros(T_train)
                                        ),
                                        finite_df=finite_pc_df,
                                        plots_dir=os.path.join(
                                            args.results_dir, "plots"
                                        ),
                                        gamma_0=gamma_0,
                                        n_hidden=n_hidden,
                                        activity_lr=activity_lr,
                                        n_infer_iters=K_inf,
                                        update_mode="infer",
                                        skip_theory=args.skip_theory,
                                    )

                                    # Closed-form equilib grads (linear nets).
                                    if args.act_fn == "linear":
                                        print(
                                            "\t\t\t\t\tRunning finite-size PC "
                                            "simulation (theory update) for "
                                            f"widths {run_widths}...\n"
                                        )
                                        finite_pc_theory_records = []
                                        for width in run_widths:
                                            print(
                                                "\t\t\t\t\tNumerical PC "
                                                "simulation (theory) for "
                                                f"width N = {width}"
                                            )
                                            theory_losses = _train_finite_pc(
                                                width_key_map[width],
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
                                                    theory_losses,
                                                    width=width,
                                                )
                                            )
                                        plot_pc_theory_vs_finite_loss(
                                            pc_dmft_loss=(
                                                pc_dmft_loss
                                                if pc_dmft_loss is not None
                                                else jnp.zeros(T_train)
                                            ),
                                            finite_df=pd.DataFrame(
                                                finite_pc_theory_records
                                            ),
                                            plots_dir=os.path.join(
                                                args.results_dir, "plots"
                                            ),
                                            gamma_0=gamma_0,
                                            n_hidden=n_hidden,
                                            activity_lr=activity_lr,
                                            n_infer_iters=K_inf,
                                            update_mode="theory",
                                            skip_theory=args.skip_theory,
                                        )

                            if (
                                overlay_any
                                and args.act_fn == "linear"
                                and (
                                    args.plot_closed_form
                                    or overlay_n_infer_iters
                                )
                            ):
                                print(
                                    "\t\t\t\t\tRunning finite-size PC "
                                    "simulation (closed-form) for "
                                    f"widths {run_widths}...\n"
                                )
                                closed_form_k = args.n_infer_iters[0]
                                for width in run_widths:
                                    print(
                                        "\t\t\t\t\tNumerical PC simulation "
                                        f"(closed-form) for width N = {width}"
                                    )
                                    closed_losses = _train_finite_pc(
                                        width_key_map[width],
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
                                        n_train_iters=args.n_train_iters,
                                        infer_mode="closed_form",
                                        n_infer_iters=closed_form_k,
                                        activity_lr=activity_lr,
                                        width=width,
                                        loss_id=loss_id,
                                        seed=seed,
                                        X_input=X_input,
                                        Y_target=Y_target,
                                    )
                                    finite_records.extend(
                                        _loss_records(
                                            closed_losses,
                                            n_hidden=n_hidden,
                                            gamma_0=gamma_0,
                                            activity_lr=activity_lr,
                                            n_infer_iters=closed_form_k,
                                            param_type=param_type,
                                            use_skips=use_skips,
                                            width=width,
                                            infer_mode="closed_form",
                                        )
                                    )

        if overlay_any:
            theory_df = pd.DataFrame(theory_records)
            finite_df = pd.DataFrame(finite_records)
            plots_root = os.path.join(args.results_dir, "plots")
            sweep_kwargs = dict(
                theory_df=theory_df,
                finite_df=finite_df,
                plots_dir=plots_root,
                skip_theory=args.skip_theory,
            )
            if overlay_n_hiddens:
                plot_pc_param_sweep_loss(
                    swept_col="n_hidden",
                    plot_closed_form=args.plot_closed_form,
                    **sweep_kwargs,
                )
            if overlay_gamma_0s:
                plot_pc_param_sweep_loss(
                    swept_col="gamma_0",
                    plot_closed_form=args.plot_closed_form,
                    **sweep_kwargs,
                )
            if overlay_n_infer_iters:
                plot_pc_param_sweep_loss(
                    swept_col="n_infer_iters",
                    plot_closed_form=True,
                    **sweep_kwargs,
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
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_samples 5 --n_fixed_point_steps 10 --n_train_iters 20 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 5 --pc_damping 1.0 --gamma_0s 1

### TEST PARAMETERS ###
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_samples 2 --n_fixed_point_steps 10 --n_train_iters 10 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 3 --pc_damping 1.0 --gamma_0s 1

### WORKING PARAMETERS (with damping) ###
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_samples 5 --n_fixed_point_steps 60 --n_train_iters 20 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 10 --n_hiddens 5 --pc_damping 0.3 --gamma_0s 1


############ NONLINEAR BELOW ###################
################################################

### EMPIRICS ONLY ###
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_samples 2 --n_fixed_point_steps 10 --n_train_iters 10 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 3 --pc_damping 1.0 --gamma_0s 1 --act_fn tanh --skip_theory

# Optimised
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_samples 20 --n_fixed_point_steps 10 --n_train_iters 50 --param_lr_pc 20.0 --activity_lrs 0.2 --n_infer_iters 20 --n_hiddens 5 --pc_damping 1.0 --gamma_0s 1 --act_fn tanh --skip_theory

### THEORY + EMPIRICS ###
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_samples 5 --n_fixed_point_steps 10 --n_train_iters 10 --param_lr_pc 5.0 --activity_lrs 0.2 --n_infer_iters 10 --n_hiddens 5 --pc_damping 0.5 --gamma_0s 1 --act_fn tanh --num_mc_samples 1000


# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_samples 5 --n_fixed_point_steps 10 --n_train_iters 10 --param_lr_pc 0.5 --activity_lrs 0.05 --n_infer_iters 5 --n_hiddens 5 --pc_damping 1.0 --gamma_0s 1

# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 1 2 3 4 5 --widths 128 512
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --gamma_0s 0 0.5 1.0 2.0
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_infer_iters 1 2 3 4 5 --widths 1024
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_infer_iters 2 4 6 8 10--widths 1024
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 3 5 7 --plot_closed_form


# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 15 --activity_lrs 0.05 --n_infer_iters 5 --n_fixed_point_steps 100 --pc_damping 0.1 --widths 4096 --param_lr_pc 1.0

# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 4 --n_fixed_point_steps 100 --pc_damping 0.1 --widths 4096 --param_lr_pc 2.0
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 5 --n_fixed_point_steps 100 --pc_damping 0.1 --widths 4096 --param_lr_pc 2.0
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 6 --n_fixed_point_steps 100 --pc_damping 0.1 --widths 4096 --param_lr_pc 2.0

# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 4 --n_fixed_point_steps 200 --pc_damping 0.05 --widths 4096 --param_lr_pc 2.0
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 5 --n_fixed_point_steps 200 --pc_damping 0.05 --widths 4096 --param_lr_pc 2.0
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 6 --n_fixed_point_steps 200 --pc_damping 0.05 --widths 4096 --param_lr_pc 2.0

# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --act
# ivity_lrs 0.05 --n_infer_iters 4 --n_fixed_point_steps 200 --pc_damping 0.05 --widths 4096 --param_lr_pc 2.0 && CUDA_VISIBL
# E_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 5 --n_fixed_point_st
# eps 200 --pc_damping 0.05 --widths 4096 --param_lr_pc 2.0 && CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6
#  --n_samples 20 --activity_lrs 0.05 --n_infer_iters 6 --n_fixed_point_steps 200 --pc_damping 0.05 --widths 4096 --param_lr_
# pc 2.0

### Final ###
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 5 --n_fixed_point_steps 20 --pc_damping 0.2 --widths 4096 --param_lr_pc 2.0

# Across depth
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 2 3 4 5 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 5 --n_fixed_point_steps 20 --pc_damping 0.2 --widths 4096 --param_lr_pc 2.0


# Across gamma
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --gamma_0s 0.1 0.25 0.5 0.75 1.0 n_samples 20 --activity_lrs 0.05 --n_infer_iters 5 --n_fixed_point_steps 20 --pc_damping 0.2 --widths 4096 --param_lr_pc 2.0

# Across K
# CUDA_VISIBLE_DEVICES=1 python analyse_pc_loss.py --n_hiddens 6 --n_samples 20 --activity_lrs 0.05 --n_infer_iters 5 --n_fixed_point_steps 20 --pc_damping 0.2 --widths 4096 --param_lr_pc 2.0
