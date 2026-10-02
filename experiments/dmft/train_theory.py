import jax
import jax.numpy as jnp
import pandas as pd

import os
import argparse
from experiments.dmft.src.utils import (
    CIFAR_GRAY_DIM,
    bp_sample_kernel_at,
    cosine_similarity,
    create_tiny_cifar10_dataset,
    create_toy_dataset,
    final_time_pc_kernel,
)
from src.theory_utils import solve_kernels, solve_kernels_nonlin, get_Delta, solve_Delta
from src.theory_pc_utils import solve_pc_kernels
from src.theory_pc_nonlin_utils import solve_pc_kernels_nonlin
from src.plot_dmft_results import (
    feature_kernel_symbol,
    plot_dmft_kernels_and_loss,
    plot_kernel_displacement_per_timepoint,
    plot_pc_bp_alignment_vs_time,
    plot_pc_bp_loss,
    plot_pc_dmft_kernels_and_loss,
)


def _feature_kernel_cosine_frames(
    all_bp,
    all_pc,
    num_inference_steps,
    num_training_steps,
    num_samples,
):
    """PC–BP and vs-initial cosine of the P×P feature kernels along training.

    Backprop kernels are ``(T, P, T, P)``. Predictive-coding kernels are
    flattened over ``(k, t, mu)``; the compared block is the forward pass
    ``k=0`` at the same training index. Displayed time is 1-based, matching
    ``plot_dmft_loss``.
    """
    if len(all_bp) != len(all_pc):
        raise ValueError(
            "BP and PC feature-kernel counts differ: "
            f"{len(all_bp)} vs {len(all_pc)}"
        )
    slice_kw = dict(
        num_inference_steps=num_inference_steps,
        num_training_steps=num_training_steps,
        num_samples=num_samples,
    )
    align_rows = []
    disp_rows = []
    for layer, (bp_kernel, pc_kernel) in enumerate(zip(all_bp, all_pc)):
        if bp_kernel.shape[0] != num_training_steps:
            raise ValueError(
                "BP feature kernel time axis "
                f"{bp_kernel.shape[0]} != n_train_iters {num_training_steps}"
            )
        bp0 = bp_sample_kernel_at(bp_kernel, t=0)
        pc0 = final_time_pc_kernel(pc_kernel, k=0, t=0, **slice_kw)
        for t_idx in range(num_training_steps):
            bp_t = bp_sample_kernel_at(bp_kernel, t=t_idx)
            pc_t = final_time_pc_kernel(pc_kernel, k=0, t=t_idx, **slice_kw)
            t = t_idx + 1
            align_rows.append(
                {
                    "t": t,
                    "layer": layer,
                    "alignment": float(
                        cosine_similarity(pc_t, bp_t, eps=1e-30)
                    ),
                }
            )
            disp_rows.append(
                {
                    "t": t,
                    "layer": layer,
                    "method": "pc",
                    "displacement": float(
                        cosine_similarity(pc0, pc_t, eps=1e-30)
                    ),
                }
            )
            disp_rows.append(
                {
                    "t": t,
                    "layer": layer,
                    "method": "bp",
                    "displacement": float(
                        cosine_similarity(bp0, bp_t, eps=1e-30)
                    ),
                }
            )
    return pd.DataFrame(align_rows), pd.DataFrame(disp_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_temp")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "tiny-CIFAR10"])
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=5) # 20)

    # Model parameters
    parser.add_argument("--act_fn", type=str, default="linear", choices=["linear", "tanh", "relu"])

    # Training parameters. The DMFT is the gradient-descent limit, so no
    # other parameter update is accepted.
    parser.add_argument("--param_optim", type=str, default="gd", choices=["gd"])
    parser.add_argument("--param_lr", type=float, default=0.05)
    parser.add_argument("--gamma_0s", type=float, nargs='+', default=[1])
    parser.add_argument("--n_train_iters", type=int, default=20) # 100)
    parser.add_argument("--n_fixed_point_steps", type=int, default=10)

    # Inference parameters
    parser.add_argument("--param_lr_pc", type=float, default=0.5)
    parser.add_argument("--n_infer_iters", type=int, default=5)
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[0.05])

    # Loop parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_hiddens", type=int, nargs='+', default=[5])

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
        "--pc_only",
        action="store_true",
        default=False,
        help="Only run PC DMFT theory (skip BP theory).",
    )

    args = parser.parse_args()

    # PC DMFT inverts (K*T*P) matrices; float64 helps stability.
    jax.config.update("jax_enable_x64", True)

    os.makedirs(args.results_dir, exist_ok=True)
    use_nonlin_theory = args.act_fn != "linear"
    # The backprop nonlinear solver implements tanh and softplus. ReLU
    # predictive coding is compared against the softplus backprop theory.
    bp_nonlin = "softplus" if args.act_fn == "relu" else args.act_fn
    if args.act_fn == "relu":
        print(
            "act_fn=relu: PC DMFT uses ReLU; BP DMFT uses softplus."
        )
    print(f"DMFT parameter update: {args.param_optim}")

    # Three independent children of --seed: dataset, an unused model stream,
    # and PC DMFT Monte Carlo. The unused split is kept so the data and theory
    # keys match previous runs. BP Monte Carlo is folded in so it does not
    # reuse the PC key.
    data_key, _model_key, pc_theory_key = jax.random.split(
        jax.random.PRNGKey(args.seed), 3
    )
    bp_theory_key = jax.random.fold_in(pc_theory_key, 1)

    if args.dataset == "toy":
        input_dim = args.input_dim
        X, y = create_toy_dataset(
            key=data_key, D=input_dim, P=args.n_samples
        )
    else:
        input_dim = CIFAR_GRAY_DIM
        X, y = create_tiny_cifar10_dataset(
            key=data_key, D=input_dim, P=args.n_samples
        )
        print(f"Input dim: {input_dim}, Output dim: 1")

    Kx = jnp.asarray(X.T @ X / input_dim, dtype=jnp.float64)
    Y_target = jnp.asarray(y[:, None], dtype=jnp.float64)
    y_bp = jnp.squeeze(Y_target, axis=-1)

    plots_dir = os.path.join(args.results_dir, "plots")
    K_inf = args.n_infer_iters
    T_train = args.n_train_iters
    P = args.n_samples
    feat_sym = feature_kernel_symbol(args.act_fn)
    feat_tex = r"\phi" if feat_sym == "phi" else "h"

    for n_hidden in args.n_hiddens:
        print(f"\n\tn hidden H = {n_hidden}")

        for gamma_0 in args.gamma_0s:
            print(f"\n\t\tgamma_0 = {gamma_0}")

            if not args.pc_only:
                if use_nonlin_theory:
                    print(
                        "\t\tCalculating nonlinear BP "
                        f"Theory (nonlin={bp_nonlin})...\n"
                    )
                    all_H, all_G, _, _ = solve_kernels_nonlin(
                        Kx=Kx,
                        y=y_bp,
                        depth=n_hidden,
                        eta=args.param_lr,
                        gamma=gamma_0,
                        T=args.n_train_iters,
                        num_iter=args.n_fixed_point_steps,
                        samples=args.num_mc_samples,
                        damping=args.bp_damping,
                        nonlin=bp_nonlin,
                        beta=args.nonlin_beta,
                        key=bp_theory_key,
                    )
                    Delta_theory = solve_Delta(
                        Kx=Kx,
                        y=y_bp,
                        all_Phi=all_H,
                        all_G=all_G,
                        eta=args.param_lr,
                    )
                    dmft_loss = 0.5 * jnp.mean(Delta_theory**2, axis=1)
                else:
                    print("\t\tCalculating BP Theory...\n")
                    all_H, all_G, _, _ = solve_kernels(
                        Kx=Kx,
                        y=y_bp,
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
                        y=y_bp,
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

            for activity_lr in args.activity_lrs:
                print(f"\n\t\t\tactivity_lr = {activity_lr}")

                n_pc = K_inf * T_train * P
                if use_nonlin_theory:
                    print(
                        "\t\t\tCalculating nonlinear PC Theory "
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
                        Kx=Kx,
                        y=Y_target,
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
                        key=pc_theory_key,
                    )
                else:
                    print(
                        "\t\t\tCalculating PC Theory "
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
                        Kx=Kx,
                        y=Y_target,
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
                    "\t\t\tPC fixed-point residual = "
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
                    feature_symbol=feat_sym,
                )

                if args.pc_only:
                    continue

                plot_pc_bp_loss(
                    pc_losses=pc_dmft_loss,
                    bp_losses=dmft_loss,
                    plots_dir=plots_dir,
                    n_hidden=n_hidden,
                    gamma_0=gamma_0,
                    activity_lr=activity_lr,
                    n_infer_iters=K_inf,
                    dir_name="theory",
                    time_offset=1,
                )
                align_df, disp_df = _feature_kernel_cosine_frames(
                    all_bp=all_H,
                    all_pc=all_Ch,
                    num_inference_steps=K_inf,
                    num_training_steps=T_train,
                    num_samples=P,
                )
                plot_pc_bp_alignment_vs_time(
                    align_df,
                    plots_dir=plots_dir,
                    n_hidden=n_hidden,
                    gamma_0=gamma_0,
                    activity_lr=activity_lr,
                    n_infer_iters=K_inf,
                    dir_name="theory",
                    feature_symbol=feat_sym,
                    ylabel=(
                        rf"$\cos(C^{{{feat_tex},\ell}}_{{\mathrm{{PC}}}}, "
                        rf"C^{{{feat_tex},\ell}}_{{\mathrm{{BP}}}})$"
                    ),
                    filename="pc_bp_kernel_cosine_vs_time.png",
                )
                plot_kernel_displacement_per_timepoint(
                    disp_df,
                    plots_dir=plots_dir,
                    n_hidden=n_hidden,
                    gamma_0=gamma_0,
                    activity_lr=activity_lr,
                    n_infer_iters=K_inf,
                    dir_name="theory",
                    feature_symbol=feat_sym,
                    metric="cosine",
                    filename="kernel_cosine_vs_init_vs_time.png",
                )


############ LINEAR ##############
##################################

# python train_theory.py --n_samples 5 --n_hiddens 5 --gamma_0s 1.0 --param_lr 0.2 --param_lr_pc 0.2 --activity_lrs 0.01 --n_infer_iters 5 --n_train_iters 10 --n_fixed_point_steps 10 --pc_damping 0.2 --dataset toy --results_dir results_theory_linear


############ NONLINEAR #################
########################################

# python train_theory.py --n_samples 3 --n_hiddens 3 --gamma_0s 1.0 --param_lr 0.2 --param_lr_pc 0.2 --activity_lrs 0.05 --n_infer_iters 10 --n_train_iters 10 --n_fixed_point_steps 100 --num_mc_samples 3000 --pc_damping 0.1 --act_fn tanh --dataset toy --results_dir results_theory_nonlin


