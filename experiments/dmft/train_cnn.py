"""Train a ResNet CNN with PC (and BP) on ImageNet.

This trains on a fixed ImageNet (or CIFAR / TinyImageNet) batch with the same
μPC ResNet and CNN optimiser. It uses the output-energy scaling
parameterisation from ``train.py`` / ``test_coord_check.py``

    λ = γ² N    (μPC)     or     λ = 1    (SP)

in the PC energy / activity / parameter updates. BP gradients are multiplied by
the same λ so the two methods share a learning-rate convention.
"""

import argparse
import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jpc
import numpy as np
import optax


_CNN_DIR = Path(__file__).resolve().parents[1] / "limits_paper" / "cnn"
if str(_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_DIR))

from model import ResNet  # noqa: E402
from optim import configure_cnn_param_optim  # noqa: E402
from utils import (  # noqa: E402
    load_cifar10_batch,
    load_imagenet_batch,
    load_tinyimagenet_batch,
)


def get_output_energy_scaling(param_type, gamma, width):
    """Match ``train.py`` / ``test_coord_check.get_coord_data``."""
    return (gamma**2) * width if param_type == "mupc" else 1.0


def load_batch(args):
    if args.dataset == "cifar":
        return load_cifar10_batch(args.batch_size, seed=args.seed)
    if args.dataset == "tinyimagenet":
        return load_tinyimagenet_batch(args.batch_size, seed=args.seed)
    if args.dataset == "imagenet":
        return load_imagenet_batch(args.batch_size, seed=args.seed)
    raise ValueError(
        f"Unknown dataset '{args.dataset}'. "
        "Expected 'cifar', 'tinyimagenet', or 'imagenet'."
    )


def make_resnet(key, args):
    return ResNet(
        key=key,
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


def setup_pc_save_dir(args):
    return os.path.join(
        args.results_dir,
        args.dataset,
        args.loss_id,
        "pc",
        f"{args.width}_width",
        f"{args.n_res_blocks}_n_res_blocks",
        f"{args.act_fn}_act_fn",
        f"{args.param_type}_param_type",
        f"{args.gamma}_gamma",
        f"{args.param_optim}_param_optim",
        f"{args.param_lr}_param_lr",
        f"{args.batch_size}_batch_size",
        f"{args.n_infer_iters}_n_infer_iters",
        f"{args.activity_lr}_activity_lr",
        f"{args.n_steps}_n_steps",
        str(args.seed),
    )


def setup_bp_save_dir(args):
    return os.path.join(
        args.results_dir,
        args.dataset,
        args.loss_id,
        "bp",
        f"{args.width}_width",
        f"{args.n_res_blocks}_n_res_blocks",
        f"{args.act_fn}_act_fn",
        f"{args.param_type}_param_type",
        f"{args.gamma}_gamma",
        f"{args.param_optim}_param_optim",
        f"{args.param_lr}_param_lr",
        f"{args.batch_size}_batch_size",
        f"{args.n_steps}_n_steps",
        str(args.seed),
    )


def supervised_loss(preds, y, loss_id):
    if loss_id == "mse":
        return jpc.mse_loss(preds, y)
    return jpc.cross_entropy_loss(preds, y)


def accuracy(preds, y):
    return jnp.mean(jnp.argmax(preds, axis=-1) == jnp.argmax(y, axis=-1))


def ffwd_metrics(model, x, y, loss_id):
    preds = jax.vmap(model)(x)
    return float(supervised_loss(preds, y, loss_id)), float(accuracy(preds, y))


def train_pc(args, x, y, model, output_energy_scaling, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    activity_optim = optax.sgd(args.activity_lr * args.batch_size)
    depth = args.n_res_blocks + args.additive_depth_factor
    param_optim = configure_cnn_param_optim(
        model,
        optim_id=args.param_optim,
        param_type=args.param_type,
        param_lr=args.param_lr,
        width=args.width,
        depth=depth,
        gamma_0=args.gamma,
        params_for_pc=True,
    )
    param_opt_state = param_optim.init((eqx.filter(model, eqx.is_array), None))

    energies, losses, accs = [], [], []
    init_loss, init_acc = ffwd_metrics(model, x, y, args.loss_id)
    losses.append(init_loss)
    accs.append(init_acc)

    for step in range(args.n_steps):
        params = (model, None)
        activities = jpc.init_activities_with_ffwd(model=model, input=x)
        activity_opt_state = activity_optim.init(activities)

        for _ in range(args.n_infer_iters):
            result = jpc.update_pc_activities(
                params=params,
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=y,
                input=x,
                loss_id=args.loss_id,
                output_energy_scaling=output_energy_scaling,
            )
            activities = result["activities"]
            activity_opt_state = result["opt_state"]
            energy = result["energy"]

        energy = float(energy)
        if not np.isfinite(energy):
            print(
                f"Warning: PC energy is non-finite at step {step}. Skipping this step."
            )
            continue
        energies.append(energy)

        param_result = jpc.update_pc_params(
            params=params,
            activities=activities,
            optim=param_optim,
            opt_state=param_opt_state,
            output=y,
            input=x,
            loss_id=args.loss_id,
            output_energy_scaling=output_energy_scaling,
        )
        model = param_result["model"]
        param_opt_state = param_result["opt_state"]

        loss, acc = ffwd_metrics(model, x, y, args.loss_id)
        losses.append(loss)
        accs.append(acc)
        print(f"  PC step {step}: energy={energy:.6f}  loss={loss:.6f}  acc={acc:.4f}")

    np.save(os.path.join(save_dir, "energies.npy"), np.asarray(energies))
    np.save(os.path.join(save_dir, "train_losses.npy"), np.asarray(losses))
    np.save(os.path.join(save_dir, "train_accs.npy"), np.asarray(accs))
    return model


def train_bp(args, x, y, model, output_energy_scaling, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    depth = args.n_res_blocks + args.additive_depth_factor
    optim = configure_cnn_param_optim(
        model,
        optim_id=args.param_optim,
        param_type=args.param_type,
        param_lr=args.param_lr,
        width=args.width,
        depth=depth,
        gamma_0=args.gamma,
        params_for_pc=False,
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(model, x_batch, y_batch):
        preds = jax.vmap(model)(x_batch)
        return supervised_loss(preds, y_batch, args.loss_id)

    @eqx.filter_jit
    def step(model, opt_state, x_batch, y_batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_batch, y_batch)
        scaled_grads = jtu.tree_map(lambda g: g * output_energy_scaling, grads)
        updates, opt_state = optim.update(
            updates=scaled_grads,
            state=opt_state,
            params=eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    losses, accs = [], []
    init_loss, init_acc = ffwd_metrics(model, x, y, args.loss_id)
    losses.append(init_loss)
    accs.append(init_acc)

    for t in range(args.n_steps):
        model, opt_state, _ = step(model, opt_state, x, y)
        loss, acc = ffwd_metrics(model, x, y, args.loss_id)
        losses.append(loss)
        accs.append(acc)
        print(f"  BP step {t}: loss={loss:.6f}  acc={acc:.4f}")

    np.save(os.path.join(save_dir, "losses.npy"), np.asarray(losses))
    np.save(os.path.join(save_dir, "train_accs.npy"), np.asarray(accs))
    return model


def main(args):
    key = jr.PRNGKey(args.seed)
    model_key, _ = jr.split(key, 2)

    x, y = load_batch(args)
    args.in_channels = x.shape[1]
    args.input_size, args.out_features = x.shape[-1], y.shape[-1]

    output_energy_scaling = get_output_energy_scaling(
        args.param_type, args.gamma, args.width
    )
    pc_model = make_resnet(model_key, args)
    bp_model = make_resnet(model_key, args)

    pc_save_dir = setup_pc_save_dir(args)
    print(
        f"PC training (width={args.width}, n_res_blocks={args.n_res_blocks}, "
        f"gamma={args.gamma}, output_energy_scaling={output_energy_scaling}, "
        f"activity_lr={args.activity_lr}, n_infer_iters={args.n_infer_iters})..."
    )
    train_pc(args, x, y, pc_model, output_energy_scaling, pc_save_dir)

    bp_save_dir = setup_bp_save_dir(args)
    print(f"BP training (same init, same data, grads × {output_energy_scaling})...")
    train_bp(args, x, y, bp_model, output_energy_scaling, bp_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir", type=str, default="pc_results")
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["cifar", "tinyimagenet", "imagenet"],
    )

    # Model parameters
    parser.add_argument("--widths", type=int, nargs="+", default=[64])
    parser.add_argument("--n_res_blocks", type=int, default=3)
    parser.add_argument(
        "--param_type", type=str, default="mupc", choices=["sp", "mupc"]
    )
    parser.add_argument("--act_fn", type=str, default="tanh")
    parser.add_argument("--scale_non_res_layers", action="store_true", default=False)
    parser.add_argument("--additive_depth_factor", type=int, default=4)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument(
        "--param_optim", type=str, default="adam", choices=["gd", "adam"]
    )
    parser.add_argument("--param_lr", type=float, default=1e-3)
    parser.add_argument("--loss_id", type=str, default="ce", choices=["mse", "ce"])
    parser.add_argument("--seed", type=int, default=0)

    # Inference / output-energy scaling
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--activity_lrs", type=float, nargs="+", default=[3e-1])
    parser.add_argument("--n_infer_iters", type=int, nargs="+", default=[10])

    args = parser.parse_args()

    for n_infer_iters in args.n_infer_iters:
        for width in args.widths:
            for activity_lr in args.activity_lrs:
                for gamma in args.gammas:
                    run_args = argparse.Namespace(**vars(args))
                    run_args.n_infer_iters = n_infer_iters
                    run_args.width = width
                    run_args.activity_lr = activity_lr
                    run_args.gamma = gamma
                    main(run_args)
