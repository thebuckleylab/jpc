"""Compare Blake closed-form equilibrated energy with numerical PC inference."""

import argparse

import jax.numpy as jnp
import jax.random as jr

import jpc
import optax


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

    data_key, x_key, y_key = jr.split(data_key, 3)
    x = jr.normal(x_key, (args.batch_size, args.input_dim))
    y = jr.normal(y_key, (args.batch_size,))
    activity_optim = optax.sgd(args.activity_lr * args.batch_size)
    params = (model, None)

    theory_energy = jpc.new_linear_equilib_energy(params, x, y, gamma=args.gamma)
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
        )
        activities = activity_update_result["activities"]
        activity_opt_state = activity_update_result["opt_state"]
        numerical_energy = activity_update_result["energy"]

    rel_err = jnp.abs(theory_energy - numerical_energy) / (
        jnp.abs(theory_energy) + 1e-8
    )
    print(f"width={args.width}, gamma={args.gamma}, param_type={args.param_type}, "
          f"activity_lr={args.activity_lr}")
    print(f"theory energy:    {float(theory_energy):.8f}")
    print(f"numerical energy: {float(numerical_energy):.8f}")
    print(f"relative error:   {float(rel_err):.2e}")

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
    parser.add_argument("--n_infer_iters", type=int, default=20)

    # Loop parameters
    parser.add_argument("--widths", type=int, nargs="+", default=[2048])

    # Tolerance parameters
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-4)

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
