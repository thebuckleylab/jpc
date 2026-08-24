"""Train a linear MLP on a linear regression task with BP and PC.

Uses the same μPC / SP parameterisation and ``output_energy_scaling`` as
``test_equilib_energy.py``.
"""

import argparse
import os

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jpc
import numpy as np
import optax
from experiments.limits_paper.utils import flatten_grads, MLP
from jax import vmap
from jax.tree_util import tree_map


def create_linear_dataset(key, n_samples, input_dim, noise_std=0.0):
    """Teacher-student linear regression: y = x w / sqrt(d) + noise."""
    x_key, w_key, noise_key = jr.split(key, 3)
    x = jr.normal(x_key, (n_samples, input_dim))
    w = jr.normal(w_key, (input_dim,))
    y = (x @ w) / jnp.sqrt(input_dim)
    if noise_std > 0:
        y = y + noise_std * jr.normal(noise_key, (n_samples,))
    return x, y


def get_output_energy_scaling(param_type, gamma, width, depth):
    if param_type == "mupc":
        return gamma**2 * width * depth
    return 1.0


def make_models(key, input_dim, width, depth, param_type, gamma):
    pc_model = jpc.make_mlp(
        key=key,
        input_dim=input_dim,
        width=width,
        depth=depth,
        output_dim=1,
        act_fn="linear",
        use_bias=False,
        param_type=param_type,
    )
    bp_model = MLP(
        key=key,
        d_in=input_dim,
        N=width,
        L=depth,
        d_out=1,
        act_fn="linear",
        param_type=param_type,
        gamma=gamma,
        use_bias=False,
    )
    for i in range(len(pc_model)):
        bp_model = eqx.tree_at(
            lambda m, i=i: m.layers[i][1].weight,
            bp_model,
            pc_model[i][1].weight,
        )
    return pc_model, bp_model


def compute_bp_loss(model, x, y):
    y_pred = vmap(model)(x)
    return 0.5 * jnp.mean(jnp.sum((y - y_pred) ** 2, axis=1))


def train_bp(
    model,
    x,
    y,
    *,
    param_lr,
    n_train_iters,
    output_energy_scaling,
    store_grads=False,
):
    optim = optax.sgd(param_lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        mse, grads = eqx.filter_value_and_grad(compute_bp_loss)(model, x, y)
        # Scale grads by λ so updates match PC under the same param_lr
        # (see test_equilib_energy.py: λ ∇L_BP ≈ ∇F*_PC)
        scaled_grads = tree_map(lambda g: g * output_energy_scaling, grads)
        updates, opt_state = optim.update(
            updates=scaled_grads,
            state=opt_state,
            params=eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, mse, scaled_grads

    losses, grads_list = [], [] if store_grads else None
    losses.append(float(compute_bp_loss(model, x, y)))
    for _ in range(n_train_iters):
        model, opt_state, _, grads = step(model, opt_state, x, y)
        losses.append(float(compute_bp_loss(model, x, y)))
        if grads_list is not None:
            grads_list.append(np.array(flatten_grads(grads)))
    return model, np.array(losses), grads_list


def train_pc(
    model,
    x,
    y,
    *,
    param_type,
    gamma,
    output_energy_scaling,
    infer_mode,
    n_infer_iters,
    activity_lr,
    param_lr,
    n_train_iters,
    store_grads=False,
):
    # Analytical energy / PC updates expect y shaped (B, d_out)
    y = y[:, None] if y.ndim == 1 else y
    batch_size = x.shape[0]
    activity_optim = optax.sgd(activity_lr * batch_size)
    param_optim = optax.sgd(param_lr)
    param_opt_state = param_optim.init((eqx.filter(model, eqx.is_array), None))

    energies, losses = [], []
    grads_list = [] if store_grads else None

    # Loss at initialisation (before any updates)
    init_activities = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        param_type=param_type,
        gamma=gamma,
    )
    losses.append(float(jpc.mse_loss(init_activities[-1], y)))

    for _ in range(n_train_iters):
        if infer_mode == "closed_form":
            energy = jpc.linear_equilib_energy(
                (model, None),
                x,
                y,
                param_type=param_type,
                gamma=gamma,
                output_energy_scaling=output_energy_scaling,
            )
            energies.append(float(energy))
            param_update = jpc.update_linear_equilib_energy_params(
                params=(model, None),
                optim=param_optim,
                opt_state=param_opt_state,
                y=y,
                x=x,
                param_type=param_type,
                gamma=gamma,
                output_energy_scaling=output_energy_scaling,
            )
        else:
            activities = jpc.init_activities_with_ffwd(
                model=model,
                input=x,
                param_type=param_type,
                gamma=gamma,
            )
            activity_opt_state = activity_optim.init(activities)
            for _ in range(n_infer_iters):
                activity_update = jpc.update_pc_activities(
                    params=(model, None),
                    activities=activities,
                    optim=activity_optim,
                    opt_state=activity_opt_state,
                    output=y,
                    input=x,
                    param_type=param_type,
                    gamma=gamma,
                    output_energy_scaling=output_energy_scaling,
                )
                activities = activity_update["activities"]
                activity_opt_state = activity_update["opt_state"]
                energy = activity_update["energy"]
            energies.append(float(energy))
            param_update = jpc.update_pc_params(
                params=(model, None),
                activities=activities,
                optim=param_optim,
                opt_state=param_opt_state,
                output=y,
                input=x,
                param_type=param_type,
                gamma=gamma,
                output_energy_scaling=output_energy_scaling,
            )

        model = param_update["model"]
        param_opt_state = param_update["opt_state"]
        if grads_list is not None:
            grads_list.append(np.array(flatten_grads(param_update["grads"][0])))

        # Feedforward MSE after the parameter update
        activities = jpc.init_activities_with_ffwd(
            model=model,
            input=x,
            param_type=param_type,
            gamma=gamma,
        )
        losses.append(float(jpc.mse_loss(activities[-1], y)))

    return model, np.array(losses), np.array(energies), grads_list


def compute_cosine_similarities(pc_grads, bp_grads):
    n = min(len(pc_grads), len(bp_grads))
    sims = np.zeros(n)
    for i in range(n):
        a, b = pc_grads[i].ravel(), bp_grads[i].ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        sims[i] = (np.dot(a, b) / denom) if denom > 1e-10 else 0.0
    return sims


def log_every_n(n_train_iters, log_every=None):
    """Print every step for short runs; otherwise every ~20 updates."""
    if log_every is not None:
        return max(1, log_every)
    if n_train_iters <= 50:
        return 1
    return max(1, n_train_iters // 20)


def print_training_losses(pc_losses, bp_losses, pc_energies, log_every):
    # losses include init at index 0; energies are length n_train_iters
    n = len(pc_losses)
    print(f"  {'step':>6}  {'pc_loss':>12}  {'bp_loss':>12}  {'pc_energy':>12}")
    for t in range(n):
        if t == 0 or t == n - 1 or t % log_every == 0:
            energy_str = f"{pc_energies[t - 1]:12.6f}" if t > 0 else f"{'—':>12}"
            print(f"  {t:6d}  {pc_losses[t]:12.6f}  {bp_losses[t]:12.6f}  {energy_str}")


def run(args):
    key = jr.PRNGKey(args.seed)
    model_key, data_key = jr.split(key)

    x, y = create_linear_dataset(
        data_key, args.n_samples, args.input_dim, args.noise_std
    )
    y_target = y[:, None] if y.ndim == 1 else y

    output_energy_scaling = get_output_energy_scaling(
        args.param_type, args.gamma, args.width, args.depth
    )
    pc_model, bp_model = make_models(
        model_key,
        args.input_dim,
        args.width,
        args.depth,
        args.param_type,
        args.gamma,
    )

    save_dir = os.path.join(
        args.results_dir,
        f"{args.input_dim}_input_dim",
        f"{args.n_samples}_n_samples",
        f"{args.depth}_depth",
        f"{args.param_type}_param_type",
        f"{args.param_lr}_param_lr",
        f"{args.gamma}_gamma",
        f"{args.n_train_iters}_n_train_iters",
        f"{args.infer_mode}_infer_mode",
        f"{args.n_infer_iters}_n_infer_iters",
        f"{args.activity_lr}_activity_lr",
        f"{args.width}_width",
        str(args.seed),
    )
    os.makedirs(save_dir, exist_ok=True)

    print(
        f"width={args.width}, gamma={args.gamma}, param_type={args.param_type}, "
        f"infer_mode={args.infer_mode}, activity_lr={args.activity_lr}, "
        f"output_energy_scaling={output_energy_scaling}"
    )

    _, pc_losses, pc_energies, pc_grads = train_pc(
        pc_model,
        x,
        y_target,
        param_type=args.param_type,
        gamma=args.gamma,
        output_energy_scaling=output_energy_scaling,
        infer_mode=args.infer_mode,
        n_infer_iters=args.n_infer_iters,
        activity_lr=args.activity_lr,
        param_lr=args.param_lr,
        n_train_iters=args.n_train_iters,
        store_grads=args.compute_cos_sims,
    )
    _, bp_losses, bp_grads = train_bp(
        bp_model,
        x,
        y_target,
        param_lr=args.param_lr,
        n_train_iters=args.n_train_iters,
        output_energy_scaling=output_energy_scaling,
        store_grads=args.compute_cos_sims,
    )

    np.save(os.path.join(save_dir, "pc_losses.npy"), pc_losses)
    np.save(os.path.join(save_dir, "pc_energies.npy"), pc_energies)
    np.save(os.path.join(save_dir, "bp_losses.npy"), bp_losses)

    print_training_losses(
        pc_losses,
        bp_losses,
        pc_energies,
        log_every_n(args.n_train_iters, args.log_every),
    )
    print(f"  PC final loss: {pc_losses[-1]:.6f}  (init {pc_losses[0]:.6f})")
    print(f"  BP final loss: {bp_losses[-1]:.6f}  (init {bp_losses[0]:.6f})")
    print(f"  PC final energy: {pc_energies[-1]:.6f}")

    if args.compute_cos_sims:
        cos_sims = compute_cosine_similarities(pc_grads, bp_grads)
        np.save(os.path.join(save_dir, "grad_cosine_similarities.npy"), cos_sims)
        print(f"  grad cos sim: mean={cos_sims.mean():.4f}, final={cos_sims[-1]:.4f}")

    print(f"  saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")

    # Model parameters
    parser.add_argument("--input_dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--param_types",
        type=str,
        nargs="+",
        default=["mupc"],
        choices=["mupc", "sp"],
    )

    # Data parameters
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    # Training parameters
    parser.add_argument("--param_lr", type=float, default=1e-1)
    parser.add_argument("--n_train_iters", type=int, default=500)
    parser.add_argument(
        "--log_every",
        type=int,
        default=None,
        help="Print losses every N steps (default: every step if ≤50 iters, else ~20 prints).",
    )

    # Inference parameters
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="closed_form",
        choices=["closed_form", "optim"],
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--activity_lrs", type=float, nargs="+", default=[5e-1])
    parser.add_argument("--n_infer_iters", type=int, default=10)

    # Loop parameters
    parser.add_argument("--widths", type=int, nargs="+", default=[1024])
    parser.add_argument("--compute_cos_sims", action="store_true", default=False)

    args = parser.parse_args()

    for width in args.widths:
        for gamma in args.gammas:
            for param_type in args.param_types:
                for activity_lr in args.activity_lrs:
                    run_args = argparse.Namespace(**vars(args))
                    run_args.width = width
                    run_args.gamma = gamma
                    run_args.param_type = param_type
                    run_args.activity_lr = activity_lr
                    run(run_args)
