"""Coordinate check for predictive coding MLPs.

Uses the activity / parameter norm recording from
``experiments.mupc_paper.test_mlp_fwd_pass``, applied to a JPC PC model
(``jpc.make_mlp`` + inference / param updates). Sweeps width only.
"""

from __future__ import annotations

import argparse
import os

import jax.numpy as jnp
import jax.random as jr
import jpc
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
from experiments.mupc_paper.utils import (
    compute_param_l2_norms,
    compute_param_spectral_norms,
)


def _layer_idxs(depth: int) -> list[int]:
    """Same layer selection as ``test_mlp_fwd_pass`` (deduped, non-negative)."""
    raw = [0, int(depth / 4) - 1, int(depth / 2) - 1, int(depth * 3 / 4) - 1, depth - 1]
    return list(dict.fromkeys(max(0, i) for i in raw))


def _record_activity_norms(activities, layer_idxs, avg_l1, avg_l2, t: int):
    i = 0
    for l, act in enumerate(activities):
        if l in layer_idxs:
            avg_l1[i, t] = float(jnp.abs(act).mean())
            avg_l2[i, t] = float(jnp.sqrt(jnp.mean(act**2)))
            i += 1


def make_dataloader(key, *, batch_size, input_dim, nsteps):
    """Synthetic regression batches: ``x, y ~ N(0, I)`` with ``y`` shaped ``(batch, 1)``."""
    batches = []
    for _ in range(nsteps):
        key, x_key, y_key = jr.split(key, 3)
        x = jr.normal(x_key, (batch_size, input_dim))
        # Must be (batch, output_dim), not (batch,): otherwise
        # y - preds broadcasts to (batch, batch) in linear_equilib_energy.
        y = jr.normal(y_key, (batch_size, 1))
        batches.append((x, y))
    return batches


def test_pc_fwd_pass(args, width: int):
    """Train a PC MLP for ``n_checks`` steps and record norms vs step."""
    key = jr.PRNGKey(args.seed)
    model_key, data_key = jr.split(key)

    model = jpc.make_mlp(
        key=model_key,
        input_dim=args.input_dim,
        width=width,
        depth=args.depth,
        output_dim=1,
        act_fn="linear",
        use_bias=False,
        param_type=args.param_type,
    )
    skip_model = jpc.make_skip_model(args.depth) if args.use_skips else None
    params = (model, skip_model)

    output_energy_scaling = (
        args.gamma**2 * width * args.depth if args.param_type == "mupc" else 1.0
    )

    if args.optimizer == "sgd":
        param_optim = optax.sgd(args.lr)
    else:
        param_optim = optax.adam(args.lr)
    param_opt_state = param_optim.init(params)

    layer_idxs = _layer_idxs(args.depth)
    n_layers = len(layer_idxs)
    avg_activity_l1 = np.zeros((n_layers, args.n_checks))
    avg_activity_l2 = np.zeros_like(avg_activity_l1)
    param_l2_norms = np.zeros_like(avg_activity_l1)
    param_spectral_norms = np.zeros_like(avg_activity_l1)

    dataloader = make_dataloader(
        data_key,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        nsteps=args.n_checks,
    )

    for t, (x, y) in enumerate(dataloader):
        ffwd_activities = jpc.init_activities_with_ffwd(
            model=model,
            input=x,
            skip_model=skip_model,
            param_type=args.param_type,
            gamma=args.gamma,
        )

        if args.record == "ffwd":
            _record_activity_norms(
                ffwd_activities,
                layer_idxs,
                avg_activity_l1,
                avg_activity_l2,
                t,
            )

        # use_bias=False => tree leaves are weights only; pass "linear" so the
        # act_fn filter in the mupc utils does not skip every other layer.
        param_l2_norms[:, t] = np.asarray(
            compute_param_l2_norms(
                model=model,
                act_fn="linear",
                layer_idxs=layer_idxs,
            )
        )
        param_spectral_norms[:, t] = np.asarray(
            compute_param_spectral_norms(
                model=model,
                act_fn="linear",
                layer_idxs=layer_idxs,
            )
        )

        if args.update_mode == "theory":
            param_result = jpc.update_linear_equilib_energy_params(
                params=params,
                optim=param_optim,
                opt_state=param_opt_state,
                x=x,
                y=y,
                param_type=args.param_type,
                gamma=args.gamma,
                output_energy_scaling=output_energy_scaling,
            )
        else:
            act_optim = optax.sgd(args.activity_lr * int(x.shape[0]))
            activities = ffwd_activities
            activity_opt_state = act_optim.init(activities)
            for _ in range(args.n_infer_iters):
                result = jpc.update_pc_activities(
                    params=params,
                    activities=activities,
                    optim=act_optim,
                    opt_state=activity_opt_state,
                    output=y,
                    input=x,
                    param_type=args.param_type,
                    gamma=args.gamma,
                    output_energy_scaling=output_energy_scaling,
                )
                activities = result["activities"]
                activity_opt_state = result["opt_state"]

            if args.record == "equilib":
                _record_activity_norms(
                    activities,
                    layer_idxs,
                    avg_activity_l1,
                    avg_activity_l2,
                    t,
                )

            param_result = jpc.update_pc_params(
                params=params,
                activities=activities,
                optim=param_optim,
                opt_state=param_opt_state,
                output=y,
                input=x,
                param_type=args.param_type,
                gamma=args.gamma,
                output_energy_scaling=output_energy_scaling,
            )

        model = param_result["model"]
        skip_model = param_result["skip_model"]
        params = (model, skip_model)
        param_opt_state = param_result["opt_state"]

    return (
        avg_activity_l1,
        avg_activity_l2,
        param_l2_norms,
        param_spectral_norms,
        layer_idxs,
    )


def plot_norms_vs_width(
    metric: np.ndarray,
    widths: list[int],
    layer_idxs: list[int],
    *,
    ylabel: str,
    title: str,
    save_path: str,
    loglog: bool = True,
):
    """Plot norm vs width with one panel per training step and a curve per layer.

    ``metric`` has shape ``(n_layers, n_checks, n_widths)``.
    """
    n_layers, n_checks, _ = metric.shape
    sns.set()
    fig, axes = plt.subplots(
        1,
        n_checks,
        figsize=(4.5 * n_checks, 4.0),
        sharey=True,
        squeeze=False,
    )
    colors = sns.color_palette("husl", n_layers)

    for t in range(n_checks):
        ax = axes[0, t]
        for i, layer in enumerate(layer_idxs):
            ax.plot(
                widths,
                metric[i, t, :],
                marker="o",
                color=colors[i],
                label=rf"$\ell={layer}$",
            )
        ax.set_title(f"t = {t}")
        ax.set_xlabel("width")
        if t == 0:
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        if loglog:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths], rotation=45)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"plot saved to {save_path}")


def run_coord_check(args) -> None:
    """Sweep widths and save activity / param-norm arrays."""
    if args.update_mode == "theory" and args.record == "equilib":
        raise ValueError("record='equilib' requires update_mode='infer'")

    save_dir = os.path.join(
        args.results_dir,
        "linear",
        args.optimizer,
        args.param_type,
        f"gamma{args.gamma}",
        "skips" if args.use_skips else "no_skips",
        f"{args.record}_{args.update_mode}",
        str(args.seed),
    )
    os.makedirs(save_dir, exist_ok=True)

    layer_idxs = _layer_idxs(args.depth)
    n_layers = len(layer_idxs)
    shape = (n_layers, args.n_checks, len(args.widths))
    avg_activity_l1_per_N = np.zeros(shape)
    avg_activity_l2_per_N = np.zeros(shape)
    param_l2_norms_per_N = np.zeros(shape)
    param_spectral_norms_per_N = np.zeros(shape)

    for w, width in enumerate(args.widths):
        print(f"N = {width}")
        (
            avg_activity_l1,
            avg_activity_l2,
            param_l2_norms,
            param_spectral_norms,
            _,
        ) = test_pc_fwd_pass(args, width)
        avg_activity_l1_per_N[:, :, w] = avg_activity_l1
        avg_activity_l2_per_N[:, :, w] = avg_activity_l2
        param_l2_norms_per_N[:, :, w] = param_l2_norms
        param_spectral_norms_per_N[:, :, w] = param_spectral_norms

    # np.save(f"{save_dir}/avg_activity_l1_per_N.npy", avg_activity_l1_per_N)
    # np.save(f"{save_dir}/avg_activity_l2_per_N.npy", avg_activity_l2_per_N)
    # np.save(f"{save_dir}/param_l2_norms_per_N.npy", param_l2_norms_per_N)
    # np.save(
    #     f"{save_dir}/param_spectral_norms_per_N.npy",
    #     param_spectral_norms_per_N,
    # )
    # np.save(f"{save_dir}/layer_idxs.npy", np.asarray(layer_idxs))
    # np.save(f"{save_dir}/widths.npy", np.asarray(args.widths))
    print(f"coord check results saved to {save_dir}")

    if args.plot:
        prm = "μPC" if args.param_type == "mupc" else args.param_type.upper()
        base_title = (
            f"{prm} PC MLP {args.optimizer} lr={args.lr} "
            f"γ={args.gamma} record={args.record} update={args.update_mode}"
        )
        plot_specs = [
            (avg_activity_l1_per_N, r"$\|\mathbf{z}_\ell\|_1$", "activity_l1"),
            (avg_activity_l2_per_N, r"$\|\mathbf{z}_\ell\|_2$", "activity_l2"),
            (param_l2_norms_per_N, r"$\|W_\ell\|_F$", "param_l2"),
            (param_spectral_norms_per_N, r"$\|W_\ell\|_2$", "param_spectral"),
        ]
        for metric, ylabel, name in plot_specs:
            if name not in args.plot_metrics:
                continue
            plot_norms_vs_width(
                metric,
                args.widths,
                layer_idxs,
                ylabel=ylabel,
                title=f"{base_title} ({name})",
                save_path=os.path.join(save_dir, f"{name}_vs_width.png"),
                loglog=not args.no_loglog,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "PC MLP coordinate check: record activity/param norms from "
            "test_mlp_fwd_pass on a jpc.make_mlp predictive coding model."
        )
    )
    parser.add_argument("--results_dir", type=str, default="coord_checks_original")
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--param_types",
        type=str,
        nargs="+",
        default=["mupc"],
        choices=["mupc", "sp", "ntp"],
    )
    parser.add_argument(
        "--use_skips",
        type=int,
        default=0,
        help="1 to use jpc.make_skip_model; 0 for params=(model, None).",
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--activity_lrs", type=float, nargs="+", default=[5e-1])
    parser.add_argument("--n_infer_iters", type=int, default=50)
    parser.add_argument("--n_checks", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--update_mode",
        type=str,
        default="infer",
        choices=["infer", "theory"],
    )
    parser.add_argument(
        "--record",
        type=str,
        default="ffwd",
        choices=["ffwd", "equilib"],
        help="Record feedforward or equilibrated activities.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save norm-vs-width figures (one panel per training step).",
    )
    parser.add_argument(
        "--plot_metrics",
        type=str,
        nargs="+",
        default=["activity_l1", "activity_l2", "param_l2", "param_spectral"],
        choices=["activity_l1", "activity_l2", "param_l2", "param_spectral"],
        help="Which norms to plot when --plot is set.",
    )
    parser.add_argument(
        "--no_loglog",
        action="store_true",
        help="Use linear axes instead of log-log for width/norm plots.",
    )

    args = parser.parse_args()

    for gamma in args.gammas:
        for param_type in args.param_types:
            for activity_lr in args.activity_lrs:
                run_args = argparse.Namespace(**vars(args))
                run_args.gamma = gamma
                run_args.param_type = param_type
                run_args.activity_lr = activity_lr
                print(
                    f"\nparam_type={param_type}, gamma={gamma}, "
                    f"activity_lr={activity_lr}"
                )
                run_coord_check(run_args)

# Example:
# python test_coord_check_original.py --widths 128 512 2048 --depth 5 \
#   --n_checks 5 --n_infer_iters 100 --lr 0.05 --activity_lrs 0.05 \
#   --update_mode theory --param_types mupc --record ffwd --plot
