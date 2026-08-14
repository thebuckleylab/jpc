"""Coordinate check for μPC / SP MLPs (μP-style width scaling of activations).

Mirrors the mup.coord_check workflow:

    models = {width: jpc_model(width), ...}
    df = get_coord_data(models, dataloader)
    plot_coord_data(df, save_to=filename)

but uses JPC (JAX) models and discrete PC training steps instead of PyTorch.
Plotting reuses ``plot_coord_data`` from ``coord_check.py``.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import pandas as pd

import jpc
from experiments.dmft.coord_check import plot_coord_data


#: JAX equivalents of mup.coord_check.FDICT (activation coordinate statistics).
FDICT = {
    "l1": lambda x: float(jnp.mean(jnp.abs(x))),
    "l2": lambda x: float(jnp.sqrt(jnp.mean(x**2))),
    "mean": lambda x: float(jnp.mean(x)),
    "std": lambda x: float(jnp.std(x)),
}

#: Stats that are activation coordinates (per-layer) vs network-level.
ACTIVITY_STATS = set(FDICT.keys())
RESCALING_STAT = "rescaling"
LOSS_STAT = "loss"
ALL_STATS = sorted(ACTIVITY_STATS | {RESCALING_STAT, LOSS_STAT})


def convert_fdict(d: Optional[dict]) -> dict:
    """Convert string values in an fdict to callables via ``FDICT``."""
    if d is None:
        return {"l1": FDICT["l1"]}
    return {k: (FDICT[v] if isinstance(v, str) else v) for k, v in d.items()}


def make_dataloader(
    key: jr.PRNGKey,
    *,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    nsteps: int,
):
    """Synthetic binary classification batches for a short coord-check training run.

    Draws ``x ~ N(0, I)`` in ``input_dim`` dimensions (default 40) and labels
    ``y ∈ {-1, +1}`` from a fixed random linear teacher:

        y = sign(x · w),   w ~ N(0, I)

    ``output_dim`` must be 1 (scalar ±1 targets reshaped to ``(batch, 1)``).
    """
    if output_dim != 1:
        raise ValueError(
            f"Binary classification requires output_dim=1, got {output_dim}"
        )

    key, teacher_key = jr.split(key)
    # Fixed decision boundary shared across batches.
    w = jr.normal(teacher_key, (input_dim,))

    batches = []
    for _ in range(nsteps):
        key, x_key = jr.split(key)
        x = jr.normal(x_key, (batch_size, input_dim))
        logits = x @ w
        y = jnp.where(logits >= 0, 1.0, -1.0).reshape(batch_size, 1)
        batches.append((x, y))
    return batches


def jpc_model(
    width: int,
    *,
    key: jr.PRNGKey,
    input_dim: int,
    depth: int,
    output_dim: int,
    act_fn: str,
    param_type: str,
    use_skips: bool,
) -> Callable:
    """Return a zero-arg factory that builds a ``(model, skip_model)`` pair."""

    def factory():
        model = jpc.make_mlp(
            key=key,
            input_dim=input_dim,
            width=width,
            depth=depth,
            output_dim=output_dim,
            act_fn=act_fn,
            use_bias=False,
            param_type=param_type,
        )
        skip_model = jpc.make_skip_model(depth) if use_skips else None
        return model, skip_model

    return factory


def _record_activities(
    records: list,
    width: int,
    activities,
    t: int,
    output_fdict: dict,
):
    """Append per-layer coordinate statistics for ``activities``."""
    for layer_idx, act in enumerate(activities):
        row = {
            "width": width,
            "module": str(layer_idx),
            "t": t,
        }
        for fname, fn in output_fdict.items():
            row[fname] = fn(act)
        records.append(row)


def _record_rescaling(
    records: list,
    width: int,
    params,
    x,
    t: int,
    *,
    param_type: str,
    gamma: float,
    output_energy_scaling: float,
):
    """Append the scalar equilibrated-energy rescaling ``S[0, 0]``."""
    S = jpc.compute_linear_equilib_rescaling(
        params,
        x,
        param_type=param_type,
        gamma=gamma,
        output_energy_scaling=output_energy_scaling,
    )
    records.append(
        {
            "width": width,
            "module": "S",
            "t": t,
            RESCALING_STAT: float(S[0, 0]),
        }
    )


def _record_loss(
    records: list,
    width: int,
    preds,
    y,
    t: int,
):
    """Append the supervised MSE training loss for the current batch."""
    records.append(
        {
            "width": width,
            "module": "loss",
            "t": t,
            LOSS_STAT: float(jpc.mse_loss(preds, y)),
        }
    )


def get_coord_data(
    models: Union[Dict[int, Callable], Callable],
    dataloader,
    *,
    param_type: str = "mupc",
    gamma: float = 1.0,
    optimizer: str = "sgd",
    lr: float = 0.1,
    activity_lr: float = 0.5,
    n_infer_iters: int = 50,
    nsteps: int = 3,
    nseeds: int = 1,
    seed: int = 0,
    fix_data: bool = True,
    output_fdict: Optional[dict] = None,
    record: str = "ffwd",
    update_mode: str = "infer",
    stats: Optional[list] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Train JPC models for a few steps and record activation / rescaling stats.

    Args:
        models: Dict mapping width -> zero-arg factory returning ``(model, skip)``,
            or a callable ``make_models(key) -> dict`` for independent seeds.
        dataloader: Iterable of ``(x, y)`` batches.
        param_type: ``"mupc"``, ``"sp"``, or ``"ntp"``.
        gamma: Output scaling used by μPC energy / grads.
        optimizer: Parameter optimiser, ``"sgd"`` or ``"adam"``.
        lr: Parameter learning rate.
        activity_lr: Activity (inference) learning rate.
        n_infer_iters: Discrete inference steps per parameter update
            (ignored when ``update_mode="theory"``).
        nsteps: Number of parameter updates to record.
        nseeds: Number of independent runs.
        seed: Base seed when ``models`` is a callable factory builder.
        fix_data: If True, reuse the first batch for all steps (mup default).
        output_fdict: Activity stats to record (default: from ``stats``, or
            ``{"l1": "l1"}``). Ignored when only ``rescaling`` / ``loss``
            are requested.
        record: ``"ffwd"`` records feedforward activities (μP-style);
            ``"equilib"`` records activities after inference (requires
            ``update_mode="infer"``).
        update_mode: ``"infer"`` runs discrete PC inference then
            ``update_pc_params``; ``"theory"`` skips inference and updates
            via ``update_linear_equilib_energy_params`` (closed-form
            equilibrated energy gradients).
        stats: Which quantities to record. Activity stats (``l1``, ``l2``,
            ``mean``, ``std``), ``rescaling`` (``S[0,0]`` from
            ``compute_linear_equilib_rescaling``), and/or ``loss``
            (supervised MSE of feedforward predictions). Default: ``["l1"]``.
        show_progress: Print a simple progress line.

    Returns:
        DataFrame with columns ``width``, ``module``, ``t``, plus recorded stats.
    """
    if update_mode not in ("infer", "theory"):
        raise ValueError("update_mode must be 'infer' or 'theory'")
    if update_mode == "theory" and record == "equilib":
        raise ValueError(
            "record='equilib' requires update_mode='infer' "
            "(theory mode has no inferred activities to record)."
        )

    if stats is None:
        stats = ["l1"]
    unknown = set(stats) - set(ALL_STATS)
    if unknown:
        raise ValueError(f"Unknown stats {unknown}; choose from {ALL_STATS}")
    record_activities = bool(set(stats) & ACTIVITY_STATS)
    record_rescaling = RESCALING_STAT in stats
    record_loss = LOSS_STAT in stats

    if output_fdict is None:
        activity_keys = [s for s in stats if s in ACTIVITY_STATS]
        output_fdict = (
            {k: k for k in activity_keys} if activity_keys else {"l1": "l1"}
        )
    output_fdict = convert_fdict(output_fdict)

    if fix_data:
        batch = next(iter(dataloader))
        dataloader = [batch] * nsteps

    if optimizer == "sgd":
        param_optim_fn = lambda: optax.sgd(lr)
    elif optimizer == "adam":
        param_optim_fn = lambda: optax.adam(lr)
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'")

    make_models = models if callable(models) else (lambda _key: models)

    records = []
    done = 0
    probe_models = make_models(jr.PRNGKey(seed))
    total = nseeds * len(probe_models)

    for seed_i in range(nseeds):
        models_i = make_models(jr.PRNGKey(seed + seed_i))
        for width, model_fn in models_i.items():
            model, skip_model = model_fn()
            params = (model, skip_model)
            param_optim = param_optim_fn()
            param_opt_state = param_optim.init(params)

            output_energy_scaling = (
                # gamma**2 * width * len(model) if param_type == "mupc" else 1.0
                gamma**2 * width if param_type == "mupc" else 1.0
            )

            for t, (x, y) in enumerate(dataloader, start=1):
                need_ffwd = (
                    record_activities
                    or record_loss
                    or update_mode == "infer"
                )
                if need_ffwd:
                    ffwd_activities = jpc.init_activities_with_ffwd(
                        model=model,
                        input=x,
                        skip_model=skip_model,
                        param_type=param_type,
                        gamma=gamma,
                    )
                else:
                    ffwd_activities = None

                if record_activities and record == "ffwd":
                    _record_activities(
                        records, width, ffwd_activities, t, output_fdict
                    )

                if record_rescaling:
                    _record_rescaling(
                        records,
                        width,
                        params,
                        x,
                        t,
                        param_type=param_type,
                        gamma=gamma,
                        output_energy_scaling=output_energy_scaling,
                    )

                if record_loss:
                    _record_loss(
                        records, width, ffwd_activities[-1], y, t
                    )

                if update_mode == "theory":
                    param_result = jpc.update_linear_equilib_energy_params(
                        params=params,
                        optim=param_optim,
                        opt_state=param_opt_state,
                        x=x,
                        y=y,
                        param_type=param_type,
                        gamma=gamma,
                        output_energy_scaling=output_energy_scaling,
                    )
                else:
                    batch_size = int(x.shape[0])
                    act_optim = optax.sgd(activity_lr * batch_size)
                    activities = ffwd_activities
                    activity_opt_state = act_optim.init(activities)
                    for _ in range(n_infer_iters):
                        result = jpc.update_pc_activities(
                            params=params,
                            activities=activities,
                            optim=act_optim,
                            opt_state=activity_opt_state,
                            output=y,
                            input=x,
                            param_type=param_type,
                            gamma=gamma,
                            output_energy_scaling=output_energy_scaling,
                        )
                        activities = result["activities"]
                        activity_opt_state = result["opt_state"]

                    if record_activities and record == "equilib":
                        _record_activities(
                            records, width, activities, t, output_fdict
                        )

                    param_result = jpc.update_pc_params(
                        params=params,
                        activities=activities,
                        optim=param_optim,
                        opt_state=param_opt_state,
                        output=y,
                        input=x,
                        param_type=param_type,
                        gamma=gamma,
                        output_energy_scaling=output_energy_scaling,
                    )

                model = param_result["model"]
                skip_model = param_result["skip_model"]
                params = (model, skip_model)
                param_opt_state = param_result["opt_state"]

                if t == nsteps:
                    break

            done += 1
            if show_progress:
                print(f"coord check: {done}/{total} (width={width}, seed={seed_i})")

    df = pd.DataFrame(records)
    df["optimizer"] = optimizer
    df["lr"] = lr
    df["param_type"] = param_type
    df["gamma"] = gamma
    df["update_mode"] = update_mode
    return df


def run_coord_check(args) -> pd.DataFrame:
    """Run a coord check with the same model setup as the original energy test:

    ``jpc.make_mlp(..., output_dim=1, act_fn="linear", use_bias=False)``
    and ``params = (model, None)`` unless ``--use_skips 1``.
    """
    key = jr.PRNGKey(args.seed)
    _, data_key = jr.split(key)
    use_skips = bool(args.use_skips)

    def make_models(model_key):
        width_keys = jr.split(model_key, len(args.widths))
        return {
            width: jpc_model(
                width,
                key=wkey,
                input_dim=args.input_dim,
                depth=args.depth,
                output_dim=args.output_dim,
                act_fn=args.act_fn,
                param_type=args.param_type,
                use_skips=use_skips,
            )
            for width, wkey in zip(args.widths, width_keys)
        }

    dataloader = make_dataloader(
        data_key,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        nsteps=args.nsteps,
    )

    df = get_coord_data(
        make_models,
        dataloader,
        param_type=args.param_type,
        gamma=args.gamma,
        optimizer=args.optimizer,
        lr=args.lr,
        activity_lr=args.activity_lr,
        n_infer_iters=args.n_infer_iters,
        nsteps=args.nsteps,
        nseeds=args.nseeds,
        seed=args.seed,
        record=args.record,
        update_mode=args.update_mode,
        stats=args.stats,
        show_progress=True,
    )

    os.makedirs(args.plotdir, exist_ok=True)
    prm = "μPC" if args.param_type == "mupc" else args.param_type.upper()
    base = (
        f"{args.param_type}_{args.optimizer}_lr{args.lr}"
        f"_actlr{args.activity_lr}_gamma{args.gamma}"
        f"_nseeds{args.nseeds}_{args.record}_{args.update_mode}"
    )
    if args.save_csv:
        csv_path = os.path.join(args.plotdir, f"{base}_coord.csv")
        df.to_csv(csv_path, index=False)
        print(f"coord check data saved to {csv_path}")

    for stat in args.stats:
        plot_df = df[df[stat].notna()].copy()
        filename = os.path.join(args.plotdir, f"{base}_{stat}_coord.png")
        plot_kwargs = dict(
            y=stat,
            save_to=filename,
            suptitle=(
                f"{prm} MLP {args.optimizer} lr={args.lr} "
                f"activity_lr={args.activity_lr} gamma={args.gamma} "
                f"record={args.record} update={args.update_mode} "
                f"stat={stat} nseeds={args.nseeds}"
            ),
            face_color=None if args.param_type == "mupc" else "xkcd:light grey",
            legend=args.legend,
        )
        # Loss is a single network-level curve; also plot loss vs step by width.
        if stat == LOSS_STAT:
            plot_kwargs["legend"] = False
            # plot_coord_data(plot_df, **plot_kwargs)
            _plot_loss_by_width(
                plot_df,
                save_to=os.path.join(args.plotdir, f"{base}_loss_vs_t.png"),
                suptitle=plot_kwargs["suptitle"],
                face_color=plot_kwargs["face_color"],
            )
        else:
            plot_coord_data(plot_df, **plot_kwargs)
    return df


def _plot_loss_by_width(
    df: pd.DataFrame,
    *,
    save_to: str,
    suptitle: Optional[str] = None,
    face_color: Optional[str] = None,
):
    """Plot training MSE vs step with one curve per width."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()
    fig = plt.figure(figsize=(7, 4.5))
    if face_color is not None:
        fig.patch.set_facecolor(face_color)
    sns.lineplot(
        data=df,
        x="t",
        y=LOSS_STAT,
        hue="width",
        marker="o",
        legend="full",
    )
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_to)
    print(f"loss-vs-step plot saved to {save_to}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="μP-style coordinate check for JPC μPC / SP MLPs."
    )

    # Model parameters (binary ±1 classification with 40-D Gaussian inputs)
    parser.add_argument("--input_dim", type=int, default=40)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--act_fn", type=str, default="linear")
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
        help="1 to use jpc.make_skip_model; 0 matches original params=(model, None).",
    )
    # Multiple widths needed for a coord-check plot (original used a single 1024).
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192],
    )

    # Data parameters
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    # Inference / training parameters
    parser.add_argument("--gammas", type=float, nargs="+", default=[1])
    parser.add_argument("--activity_lrs", type=float, nargs="+", default=[5e-1])
    parser.add_argument("--n_infer_iters", type=int, default=50)
    parser.add_argument("--nsteps", type=int, default=3)
    parser.add_argument("--nseeds", type=int, default=3)
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=["sgd", "adam"]
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--update_mode",
        type=str,
        default="infer",
        choices=["infer", "theory"],
        help=(
            "'infer': discrete PC inference then update_pc_params; "
            "'theory': skip inference, update via closed-form "
            "linear equilibrium energy gradients."
        ),
    )

    # Recording / plotting
    parser.add_argument(
        "--record",
        type=str,
        default="ffwd",
        choices=["ffwd", "equilib"],
        help=(
            "Record feedforward (μP-style) or equilibrated activities. "
            "'equilib' requires --update_mode infer."
        ),
    )
    parser.add_argument(
        "--stats",
        type=str,
        nargs="+",
        default=["l1"],
        choices=ALL_STATS,
        help=(
            "Quantities to record/plot. Activity coords (l1/l2/mean/std), "
            "'rescaling' (S[0,0] from compute_linear_equilib_rescaling), "
            "and/or 'loss' (supervised MSE of feedforward preds vs targets). "
            "One plot is written per requested stat; 'loss' also writes a "
            "loss-vs-step curve colored by width."
        ),
    )
    parser.add_argument("--plotdir", type=str, default="coord_checks")
    parser.add_argument("--legend", type=str, default="full")
    parser.add_argument("--save_csv", action="store_true")

    args = parser.parse_args()

    for gamma in args.gammas:
        for param_type in args.param_types:
            for activity_lr in args.activity_lrs:
                run_args = argparse.Namespace(**vars(args))
                run_args.gamma = gamma
                run_args.param_type = param_type
                run_args.activity_lr = activity_lr
                run_coord_check(run_args)

# Parameters used for simulation
# python test_coord_check.py --batch_size 20 --gammas 1.0 --depth 5 --nsteps 5 --activity_lrs 0.05 --n_infer_iters 1000 --lr 0.05 --record ffwd --save_csv
# Theory (closed-form equilib grads, no inference):
# python test_coord_check.py --batch_size 20 --gammas 1.0 --depth 5 --nsteps 5 --lr 0.05 --record ffwd --update_mode theory --save_csv
# Plot L1 and equilibrated-energy rescaling S[0,0]:
# python test_coord_check.py --batch_size 20 --gammas 1.0 --depth 5 --nsteps 5 --lr 0.05 --record ffwd --update_mode theory --stats l1 rescaling --save_csv
# Plot training MSE loss across widths:
# python test_coord_check.py --batch_size 20 --gammas 1.0 --depth 5 --nsteps 5 --lr 0.05 --record ffwd --update_mode theory --stats loss --save_csv

# Testing
# python test_coord_check.py --batch_size 10 --gammas 1.0 --depth 5 --nsteps 5 --lr 0.5 --record ffwd --update_mode theory --stats l1 --save_csv --widths 128 512 2048 8192 --nseeds 1 --seed 10

# python test_coord_check.py --batch_size 10 --gammas 1.0 --depth 5 --nsteps 5 --activity_lrs 0.05 --n_infer_iters 1000 --lr 0.5 --record ffwd --update_mode infer --stats l1 --save_csv --widths 128 512 2048 8192 --nseeds 1 --seed 10

# python test_coord_check.py --batch_size 10 --gammas 1.0 --depth 5 --nsteps 50 --activity_lrs 0.05 --n_infer_iters 1000 --lr 0.5 --record ffwd --update_mode infer --stats loss --save_csv --widths 128 512 2048 8192 --nseeds 1 --seed 10