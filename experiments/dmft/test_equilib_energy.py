"""Compare Blake closed-form equilibrated energy with numerical PC inference."""

import jax.numpy as jnp
import jax.random as jr

import jpc
import optax


def main():
    seed = 0
    key = jr.PRNGKey(seed)
    model_key, data_key = jr.split(key)

    D, N = 16, 2048
    batch_size = 1
    gamma = 1
    param_type = "mupc"

    model = jpc.make_mlp(
        key=model_key,
        input_dim=D,
        width=N,
        depth=2,
        output_dim=1,
        act_fn="linear",
        use_bias=False,
        param_type=param_type,
    )

    data_key, x_key, y_key = jr.split(data_key, 3)
    x = jr.normal(x_key, (batch_size, D))
    y = jr.normal(y_key, (batch_size,))
    activity_optim = optax.sgd(5e-1 * batch_size)
    params = (model, None)

    theory_energy = jpc.blake_linear_equilib_energy(params, x, y, gamma=gamma)
    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=x,
        param_type=param_type,
        gamma=gamma,
    )
    activity_opt_state = activity_optim.init(activities)
    for _ in range(20):
        activity_update_result = jpc.update_pc_activities(
            params=(model, None),
            activities=activities,
            optim=activity_optim,
            opt_state=activity_opt_state,
            output=y,
            input=x,
            param_type=param_type,
            gamma=gamma
        )
        activities = activity_update_result["activities"]
        activity_opt_state = activity_update_result["opt_state"]
        numerical_energy = activity_update_result["energy"]
    
    rel_err = jnp.abs(theory_energy - numerical_energy) / (
        jnp.abs(theory_energy) + 1e-8
    )
    print(f"theory energy:    {float(theory_energy):.8f}")
    print(f"numerical energy: {float(numerical_energy):.8f}")
    print(f"relative error:   {float(rel_err):.2e}")

    if not jnp.allclose(theory_energy, numerical_energy, rtol=1e-2, atol=1e-4):
        raise SystemExit("Theory and numerical energies do not match.")


if __name__ == "__main__":
    main()
