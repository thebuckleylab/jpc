import os
import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import jpc

from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import set_seed, unwrap_hessian_pytree
from experiments.mupc_paper.plotting import plot_activity_hessian
from utils import setup_hessian_analysis


def compute_hessian_metrics(generator, amortiser, skip_model, y, x):
    activities = jpc.init_activities_with_ffwd(generator, x)
    hessian_pytree = jax.hessian(jpc.bpc_energy_fn, argnums=2)(
        generator,
        amortiser,
        activities,
        y,
        x,
        skip_model=skip_model
    )
    H = unwrap_hessian_pytree(hessian_pytree, activities)

    eigenvals, _ = jnp.linalg.eigh(H)
    cond_num = jnp.linalg.cond(H)

    return H, eigenvals, cond_num


def run_analysis(
        seed,
        in_out_dims,
        act_fn,
        width,
        n_hidden,
        use_skips,
        save_dir
):
    set_seed(seed)
    key = jr.PRNGKey(seed)
    gen_key, amort_key = jr.split(key, 2)

    input_dim = width if in_out_dims == "width" else in_out_dims[0]
    output_dim = width if in_out_dims == "width" else in_out_dims[1]

    # models
    generator = jpc.make_mlp(
        key=gen_key,
        input_dim=input_dim,
        width=width,
        depth=n_hidden + 1,
        output_dim=output_dim,
        act_fn=act_fn
    )
    amortiser = jpc.make_mlp(
        key=amort_key,
        input_dim=output_dim,
        width=width,
        depth=n_hidden + 1,
        output_dim=input_dim,
        act_fn=act_fn
    )[::-1]
    skip_model = jpc.make_skip_model(n_hidden + 1) if use_skips else None

    # data
    if input_dim == width:
        x = y = np.array([[1]])

        # NOTE: set all the weights to 1 for simplicity for testing purposes
        import equinox as eqx   
        where1, where2 = lambda l: l[0][1].weight, lambda l: l[1][1].weight
        generator = eqx.tree_at(where1, generator, jnp.array([[1]]))
        generator = eqx.tree_at(where2, generator, jnp.array([[1]]))
        amortiser = eqx.tree_at(where1, amortiser, jnp.array([[1]]))
        amortiser = eqx.tree_at(where2, amortiser, jnp.array([[1]]))
    else:
        train_loader, _ = get_dataloaders("MNIST", batch_size=1)
        img_batch, label_batch = next(iter(train_loader))
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        x, y = (img_batch, label_batch) if input_dim == 784 else (label_batch, img_batch)

    H, eigenvals, cond_num = compute_hessian_metrics(
        generator=generator,
        amortiser=amortiser,
        skip_model=skip_model,
        y=y,
        x=x
    )
    print(f"Hessian matrix: {H}")
    plot_activity_hessian(H, f"{save_dir}/hessian_matrix.pdf")
    
    np.save(f"{save_dir}/hessian_eigenvals", eigenvals)
    np.save(f"{save_dir}/cond_num", cond_num)


if __name__ == "__main__":
    RESULTS_DIR = "activity_hessian_results"
    IN_OUT_DIMS = [784, 10]
    ACT_FNS = ["linear","tanh", "relu"]
    WIDTHS = [2 ** i for i in range(11)]
    N_HIDDENS = [2 ** i for i in range(4)]
    USE_SKIPS = [True, False]
    N_SEEDS = 3

    for act_fn in ACT_FNS:
        for width in WIDTHS:
            for n_hidden in N_HIDDENS:
                for use_skips in USE_SKIPS:
                    for seed in range(N_SEEDS):
                        save_dir = setup_hessian_analysis(
                            results_dir=RESULTS_DIR,
                            in_out_dims=IN_OUT_DIMS,
                            act_fn=act_fn,
                            width=width,
                            n_hidden=n_hidden,
                            use_skips=use_skips,
                            seed=seed
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        run_analysis(
                            seed=seed,
                            in_out_dims=IN_OUT_DIMS,
                            act_fn=act_fn,
                            width=width,
                            n_hidden=n_hidden,
                            use_skips=use_skips,
                            save_dir=save_dir
                        )
