"""Compare closed-form equilibrated energy with numerical PC inference."""

import argparse

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_map

import jpc
import optax
from experiments.limits_paper.utils import MLP, flatten_grads

  
def compute_cosine_similarity(a, b):
    a = jnp.asarray(a).reshape(-1)
    b = jnp.asarray(b).reshape(-1)
    dot = jnp.dot(a, b)
    norms = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    return float(dot / norms) if norms > 1e-10 else 0.0


def main(args):
    key = jr.PRNGKey(args.seed)
    model_key, data_key = jr.split(key)

    model = jpc.make_mlp(
        key=model_key,
        input_dim=args.input_dim,
        width=args.width,
        depth=args.depth,
        output_dim=1,
        act_fn="linear",
        use_bias=False,
        param_type=args.param_type,
    )
    bp_model = MLP(
        key=model_key,
        d_in=args.input_dim,
        N=args.width,
        L=args.depth,
        d_out=1,
        act_fn="linear",
        param_type=args.param_type,
        gamma=args.gamma,
        use_bias=False,
    )
    for i in range(len(model)):
        bp_model = eqx.tree_at(
            lambda m, i=i: m.layers[i][1].weight,
            bp_model,
            model[i][1].weight,
        )

    data_key, x_key, y_key = jr.split(data_key, 3)
    x = jr.normal(x_key, (args.batch_size, args.input_dim))
    y = jr.normal(y_key, (args.batch_size,))
    activity_optim = optax.sgd(args.activity_lr * args.batch_size)
    params = (model, None)

    y_target = y[:, None] if y.ndim == 1 else y
    bp_preds = vmap(bp_model)(x)
    bp_loss = jpc.mse_loss(bp_preds, y_target)

    output_energy_scaling = (
        args.gamma ** 2 * args.width * args.depth 
        if args.param_type == "mupc" else 1.0
    )

    theory_energy = jpc.linear_equilib_energy(
        params, 
        x, 
        y, 
        param_type=args.param_type,
        gamma=args.gamma, 
        output_energy_scaling=output_energy_scaling
    )
    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        param_type=args.param_type,
        gamma=args.gamma,
    )
    activity_opt_state = activity_optim.init(activities)
    for _ in range(args.n_infer_iters):
        activity_update_result = jpc.update_pc_activities(
            params=(model, None),
            activities=activities,
            optim=activity_optim,
            opt_state=activity_opt_state,
            output=y,
            input=x,
            param_type=args.param_type,
            gamma=args.gamma,
            output_energy_scaling=output_energy_scaling,
        )
        activities = activity_update_result["activities"]
        activity_opt_state = activity_update_result["opt_state"]
        numerical_energy = activity_update_result["energy"]

    theory_energy_from_loss = bp_loss / float(
        jpc.compute_linear_equilib_rescaling(
            params,
            x,
            param_type=args.param_type,
            gamma=args.gamma,
            output_energy_scaling=output_energy_scaling,
        )[0, 0]
    )
    rel_err = jnp.abs(theory_energy - numerical_energy) / (
        jnp.abs(theory_energy) + 1e-8
    )

    def bp_loss_fn(model, x_batch, y_batch):
        y_pred = vmap(model)(x_batch)
        return 0.5 * jnp.mean(jnp.sum((y_batch - y_pred) ** 2, axis=1))

    pc_grads_theory = jpc.compute_linear_equilib_energy_grads(
        params, 
        x, 
        y, 
        param_type=args.param_type,
        gamma=args.gamma, 
        output_energy_scaling=output_energy_scaling
    )
    pc_grads_numerical = jpc.compute_pc_param_grads(
        params=(model, None),
        activities=activities,
        y=y,
        x=x,
        param_type=args.param_type,
        gamma=args.gamma,
        output_energy_scaling=output_energy_scaling,
    )
    _, bp_grads = eqx.filter_value_and_grad(bp_loss_fn)(bp_model, x, y_target)
    bp_grads = tree_map(lambda g: g * output_energy_scaling, bp_grads)
    pc_theory_flat = flatten_grads(pc_grads_theory[0])
    pc_numerical_flat = flatten_grads(pc_grads_numerical[0])
    bp_flat = flatten_grads(bp_grads)
    cos_sim_theory = compute_cosine_similarity(pc_theory_flat, bp_flat)
    cos_sim_numerical = compute_cosine_similarity(pc_numerical_flat, bp_flat)

    print(f"width={args.width}, gamma={args.gamma}, param_type={args.param_type}, "
          f"activity_lr={args.activity_lr}, output_energy_scaling={output_energy_scaling}")
    print(f"theory energy:    {float(theory_energy):.8f}")
    print(f"numerical energy: {float(numerical_energy):.8f}")
    print(f"relative error:   {float(rel_err):.2e}")
    print(f"theory energy from loss: {float(theory_energy_from_loss):.8f}")
    print(f"bp loss:          {float(bp_loss):.8f}")
    print(f"cos sim (theory PC grad, BP grad):    {cos_sim_theory:.6f}")
    print(f"cos sim (numerical PC grad, BP grad): {cos_sim_numerical:.6f}")

    if not jnp.allclose(
        theory_energy, numerical_energy, rtol=args.rtol, atol=args.atol
    ):
        raise SystemExit("Theory and numerical energies do not match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--input_dim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--param_types", type=str, nargs="+", default=["mupc"], choices=["mupc", "sp"])

    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    # Inference parameters
    parser.add_argument("--gammas", type=float, nargs="+", default=[1])
    parser.add_argument("--activity_lrs", type=float, nargs="+", default=[5e-1])
    parser.add_argument("--n_infer_iters", type=int, default=50)

    # Loop parameters
    parser.add_argument("--widths", type=int, nargs="+", default=[1024])

    # Tolerance parameters
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)

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
                    main(run_args)
