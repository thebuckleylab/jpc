"""Toy-task training for energy-scaled µPC vs BP.

Mirrors ``experiments/limits_paper/train.py``: sweep width at fixed γ
and/or γ at fixed width on the Gaussian ±1 toy task and save PC/BP
metrics. Plot with ``plot_toy.py``.

The new PC parameterisation puts width, depth, and γ in the energy rather
than the PC learning rate:

* output precision ``λ = γ² N L``
* hidden precision ``κ = L``

PC GD uses plain ``param_lr`` on ``F*``. BP GD puts the µP factor
``γ² N`` in the learning rate and trains unscaled MSE (as in
``limits_paper``). Without skips, the infinite-width BP DMFT loss is the
unscaled MSE from ``limits_paper/train.py``. Both nets share the same
initial weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import jpc
from experiments.dmft.theory_utils import get_Delta, solve_kernels
from experiments.dmft.utils import (
    MLP,
    get_hidden_energy_scaling,
    get_output_energy_scaling,
    train_bpn,
    train_pcn,
)
from experiments.limits_paper.utils import (
    compute_grad_cosine_similarities,
    create_toy_dataset,
    setup_bp_experiment,
    setup_pc_experiment,
)
from experiments.mupc_paper.utils import set_seed

from plot_toy import (
    add_common_args,
    bp_dmft_loss_path,
    generate_plots,
    mup_loss_scale,
)


def compute_bp_dmft_loss(
    Kx, y, n_hidden, param_lr, gamma_0, n_train_iters, n_fixed_point_steps
):
    """Infinite-width BP DMFT loss, as in ``limits_paper/train.py``."""
    all_H, all_G, _, _ = solve_kernels(
        Kx=Kx,
        y=y,
        depth=n_hidden,
        eta=param_lr,
        gamma=gamma_0,
        T=n_train_iters,
        num_steps=n_fixed_point_steps,
    )
    Delta_theory = get_Delta(
        all_H=all_H, all_G=all_G, Kx=Kx, y=y, eta=param_lr
    )
    return 0.5 * jnp.mean(jnp.sum(Delta_theory**2, axis=2), axis=1)


def compute_bp_dmft_loss_if_applicable(args, *, Kx, y, n_hidden, use_skips, gamma_0, param_type, seed):
    if use_skips or param_type == "sp" or args.param_optim != "gd":
        return None
    path = bp_dmft_loss_path(args.results_dir, gamma_0, n_hidden, seed)
    print(
        f"\t\t\t\tCalculating BP DMFT theory "
        f"(H={n_hidden}, γ={gamma_0})..."
    )
    jax.config.update("jax_enable_x64", True)
    dmft_loss = np.asarray(
        compute_bp_dmft_loss(
            Kx=Kx,
            y=y,
            n_hidden=n_hidden,
            param_lr=args.param_lr,
            gamma_0=gamma_0,
            n_train_iters=args.n_train_iters,
            n_fixed_point_steps=args.n_fixed_point_steps,
        )
    )
    np.save(path, dmft_loss)
    print(f"\t\t\t\tSaved BP DMFT loss to {path}")
    return dmft_loss


def copy_pc_weights_to_bp(pc_model, bp_model):
    for i in range(len(pc_model)):
        bp_model = eqx.tree_at(
            lambda m, i=i: m.layers[i][1].weight,
            bp_model,
            jnp.copy(pc_model[i][1].weight),
        )
    return bp_model


def pc_bp_weights_match(pc_model, bp_model, atol=1e-10):
    return all(
        jnp.allclose(
            pc_model[i][1].weight,
            bp_model.layers[i][1].weight,
            atol=atol,
        )
        for i in range(len(pc_model))
    )


def energy_scalings(param_type, gamma_0, width, depth):
    return (
        get_output_energy_scaling(param_type, gamma_0, width, depth),
        get_hidden_energy_scaling(param_type, depth),
    )


def run_one(
    *,
    model_key,
    X_input,
    Y_target,
    input_dim,
    output_dim,
    n_samples,
    n_hidden,
    use_skips,
    act_fn,
    param_type,
    param_optim,
    param_lr,
    gamma_0,
    n_train_iters,
    infer_mode,
    n_infer_iters,
    activity_lr,
    width,
    loss_id,
    seed,
    results_dir,
    compute_cos_sims,
):
    depth = n_hidden + 1
    lam, kappa = energy_scalings(param_type, gamma_0, width, depth)
    bp_lr_scale = mup_loss_scale(param_type, gamma_0, width)
    print(
        f"\t\t\t\t\tN={width}, γ={gamma_0}, λ={lam:g}, κ={kappa:g}, "
        f"BP lr×{bp_lr_scale:g}, infer={infer_mode}"
    )

    pc_save_dir = setup_pc_experiment(
        results_dir=results_dir,
        input_dim=input_dim,
        n_samples=n_samples,
        n_hidden=n_hidden,
        use_skips=use_skips,
        act_fn=act_fn,
        param_type=param_type,
        param_lr=param_lr,
        gamma_0=gamma_0,
        param_optim_id=param_optim,
        n_train_iters=n_train_iters,
        infer_mode=infer_mode,
        n_infer_iters=n_infer_iters,
        activity_lr=activity_lr,
        width=width,
        loss_id=loss_id,
        seed=seed,
    )
    pc_model = jpc.make_mlp(
        model_key,
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=output_dim,
        act_fn=act_fn,
        use_bias=False,
        param_type=param_type,
    )
    pc_grads, _, _ = train_pcn(
        model=pc_model,
        use_skips=use_skips,
        X_input=X_input,
        Y_target=Y_target,
        width=width,
        gamma_0=gamma_0,
        param_type=param_type,
        infer_mode=infer_mode,
        n_infer_iters=n_infer_iters,
        activity_lr=activity_lr,
        param_optim_id=param_optim,
        param_lr=param_lr,
        n_train_iters=n_train_iters,
        save_dir=pc_save_dir,
        store_grads=compute_cos_sims,
        loss_id=loss_id,
    )

    bp_save_dir = setup_bp_experiment(
        results_dir=results_dir,
        input_dim=input_dim,
        n_samples=n_samples,
        n_hidden=n_hidden,
        use_skips=use_skips,
        act_fn=act_fn,
        param_type=param_type,
        optim_id=param_optim,
        param_lr=param_lr,
        gamma_0=gamma_0,
        n_train_iters=n_train_iters,
        width=width,
        loss_id=loss_id,
        seed=seed,
    )
    bp_model = MLP(
        key=model_key,
        d_in=input_dim,
        N=width,
        L=depth,
        d_out=output_dim,
        act_fn=act_fn,
        param_type=param_type,
        gamma=gamma_0,
        use_bias=False,
        use_skips=use_skips,
    )
    bp_model = copy_pc_weights_to_bp(pc_model, bp_model)
    if pc_bp_weights_match(pc_model, bp_model):
        print("\t\t\t\t\t✓ PC and BP models have identical random initialization")
    else:
        print("\t\t\t\t\t✗ WARNING: Some weights don't match!")

    bp_grads = train_bpn(
        model=bp_model,
        use_skips=use_skips,
        X_input=X_input,
        Y_target=Y_target,
        width=width,
        gamma_0=gamma_0,
        param_type=param_type,
        optim_id=param_optim,
        param_lr=param_lr,
        n_train_iters=n_train_iters,
        save_dir=bp_save_dir,
        store_grads=compute_cos_sims,
        loss_id=loss_id,
    )

    if compute_cos_sims:
        cosine_similarities = compute_grad_cosine_similarities(pc_grads, bp_grads)
        np.save(
            f"{pc_save_dir}/grad_cosine_similarities.npy",
            cosine_similarities,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Toy µPC vs BP with output energy scaling λ=γ²NL, "
            "hidden energy scaling κ=L, and BP GD LR × γ²N."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--n_fixed_point_steps",
        type=int,
        default=10,
        help="BP DMFT kernel fixed-point iterations (no-skip µPC GD only).",
    )
    parser.add_argument(
        "--compute_cos_sims",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        default=False,
        help="Train only; do not plot. Replot later with plot_toy.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.n_hiddens) > 1 and len(args.widths) > 1:
        jax.config.update("jax_enable_x64", True)

    os.makedirs(args.results_dir, exist_ok=True)
    input_dim = args.input_dim
    output_dim = 1
    loss_id = "mse"

    for seed in range(args.seed, args.seed + args.n_seeds):
        print(f"\nRunning experiment for seed: {seed}")
        set_seed(seed)
        key = jax.random.PRNGKey(seed)
        data_key, model_parent = jax.random.split(key)
        X, y = create_toy_dataset(
            key=data_key, D=input_dim, P=args.n_samples
        )
        X_input = X.T
        Y_target = y[:, None] if y.ndim == 1 else y
        Kx = X.T @ X / input_dim

        for n_hidden in args.n_hiddens:
            print(f"\n\tn hidden H = {n_hidden}")
            for use_skips in args.use_skips:
                print(f"\n\t\tuse_skips = {use_skips}")
                for gamma_0 in args.gamma_0s:
                    print(f"\n\t\t\tgamma_0 = {gamma_0}")
                    for param_type in args.param_types:
                        print(f"\n\t\t\t\tparam_type = {param_type}")
                        compute_bp_dmft_loss_if_applicable(
                            args,
                            Kx=Kx,
                            y=y,
                            n_hidden=n_hidden,
                            use_skips=use_skips,
                            gamma_0=gamma_0,
                            param_type=param_type,
                            seed=seed,
                        )
                        width_keys = jax.random.split(
                            jax.random.fold_in(model_parent, int(seed)),
                            len(args.widths),
                        )
                        for activity_lr in args.activity_lrs:
                            print(
                                f"\n\t\t\t\t\tactivity_lr = {activity_lr}"
                            )
                            for width, wkey in zip(args.widths, width_keys):
                                run_one(
                                    model_key=wkey,
                                    X_input=X_input,
                                    Y_target=Y_target,
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    n_samples=args.n_samples,
                                    n_hidden=n_hidden,
                                    use_skips=use_skips,
                                    act_fn=args.act_fn,
                                    param_type=param_type,
                                    param_optim=args.param_optim,
                                    param_lr=args.param_lr,
                                    gamma_0=gamma_0,
                                    n_train_iters=args.n_train_iters,
                                    infer_mode=args.infer_mode,
                                    n_infer_iters=args.n_infer_iters,
                                    activity_lr=activity_lr,
                                    width=width,
                                    loss_id=loss_id,
                                    seed=seed,
                                    results_dir=args.results_dir,
                                    compute_cos_sims=args.compute_cos_sims,
                                )

    if not args.skip_plot:
        generate_plots(args)


if __name__ == "__main__":
    main()


# Width sweep at fixed gamma (paper-like toy):
# python experiments/dmft/param_checks/train_toy.py \
#   --widths 8 16 32 64 128 --gamma_0s 1 --n_hiddens 3 --n_train_iters 100
#
# Replot without retraining:
# python experiments/dmft/param_checks/plot_toy.py \
#   --widths 8 16 32 64 128 --gamma_0s 1 --n_hiddens 3

# Gamma sweep at fixed width:
# python experiments/dmft/param_checks/train_toy.py \
#   --widths 128 --gamma_0s 0.1 0.5 1 2 3 4 --n_hiddens 3 --n_train_iters 100

# Closed-form inference is the default (exact F*, s(θ)). For iterative
# inference, pass --infer_mode optim --n_infer_iters 50 --activity_lrs 0.5
