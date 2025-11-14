import os
import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np

from experiments.datasets import make_gaussian_dataset, get_dataloaders
from experiments.mupc_paper.utils import (
    setup_hessian_analysis,
    set_seed,
    init_weights,
    get_network_weights,
    unwrap_hessian_pytree
)
import jpc
from experiments.mupc_paper.plotting import plot_activity_hessian


def compute_hessian_metrics(
        network,
        act_fn,
        skip_model,
        y,
        x,
        use_skips,
        param_type,
        activity_decay,
        mode,
        layer_sizes,
        key
):
    if act_fn == "linear":
        # theoretical activity Hessian
        weights = get_network_weights(network)
        theory_H = jpc.compute_linear_activity_hessian(
            weights,
            param_type=param_type,
            use_skips=use_skips,
            activity_decay=activity_decay
        )
        D = jpc.compute_linear_activity_hessian(
            weights,
            param_type=param_type,
            off_diag=False,
            use_skips=use_skips,
            activity_decay=activity_decay
        )
        O = jpc.compute_linear_activity_hessian(
            weights,
            param_type=param_type,
            diag=False,
            use_skips=use_skips,
            activity_decay=activity_decay
        )

    # numerical activity Hessian
    if mode == "supervised":
        activities = jpc.init_activities_with_ffwd(
            network,
            x,
            skip_model=skip_model,
            param_type=param_type
        )
    elif mode == "unsupervised":
        activities = jpc.init_activities_from_normal(
            key=key,
            layer_sizes=layer_sizes,
            mode=mode,
            batch_size=1,
            sigma=1
        )

    hessian_pytree = jax.hessian(jpc.pc_energy_fn, argnums=1)(
        (network, skip_model),
        activities,
        y,
        x=x,
        param_type=param_type,
        activity_decay=activity_decay
    )
    num_H = unwrap_hessian_pytree(
        hessian_pytree,
        activities,
    )

    # compute eigenthings
    num_H_eigenvals, _ = jnp.linalg.eigh(num_H)
    cond_num = jnp.linalg.cond(num_H)
    if act_fn == "linear":
        theory_H_eigenvals, _ = jnp.linalg.eigh(theory_H)
        D_eigenvals, _ = jnp.linalg.eigh(D)
        O_eigenvals, _ = jnp.linalg.eigh(O)

    return {
        "hessian": num_H,
        "num": num_H_eigenvals,
        "cond_num": cond_num,
        "theory": theory_H_eigenvals if act_fn == "linear" else None,
        "D": D_eigenvals if act_fn == "linear" else None,
        "O": O_eigenvals if act_fn == "linear" else None
    }


def run_analysis(
        seed,
        in_out_dims,
        act_fn,
        use_biases,
        mode,
        use_skips,
        weight_init,
        param_type,
        activity_decay,
        width,
        n_hidden,
        save_dir
):
    set_seed(seed)
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 4)

    d_in = width if in_out_dims == "width" else in_out_dims[0]
    d_out = width if in_out_dims == "width" else in_out_dims[1]

    # create and initialise model
    L = n_hidden+1
    network = jpc.make_mlp(
        key=keys[0],
        input_dim=d_in,
        width=width,
        depth=L,
        output_dim=d_out,
        act_fn=act_fn,
        use_bias=use_biases,
        param_type=param_type
    )
    if weight_init != "standard":
        network = init_weights(
            key=keys[1],
            model=network,
            init_fn_id=weight_init
        )
    skip_model = jpc.make_skip_model(L) if use_skips else None

    # data
    if in_out_dims != "width":
        train_loader, _ = get_dataloaders("MNIST", batch_size=1)
        img_batch, label_batch = next(iter(train_loader))
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        x, y = (img_batch, label_batch) if d_in == 784 else (label_batch, img_batch)
    else:
        x, y = make_gaussian_dataset(keys[2], 1, 0.1, (1, width))
    if mode == "unsupervised":
        x = None

    layer_sizes = [d_in] + [width]*n_hidden + [d_out]
    metrics = compute_hessian_metrics(
        network=network,
        act_fn=act_fn,
        skip_model=skip_model,
        y=y,
        x=x,
        use_skips=use_skips,
        param_type=param_type,
        activity_decay=activity_decay,
        mode=mode,
        layer_sizes=layer_sizes,
        key=keys[3]
    )
    plot_activity_hessian(
        metrics["hessian"],
        f"{save_dir}/hessian_matrix.pdf"
    )
    np.save(
        f"{save_dir}/num_hessian_eigenvals",
        metrics["num"]
    )
    np.save(
        f"{save_dir}/cond_num",
        metrics["cond_num"]
    )
    if act_fn == "linear":
        np.save(
            f"{save_dir}/theory_hessian_eigenvals",
            metrics["theory"]
        )
        np.save(
            f"{save_dir}/theory_D_eigenvals",
            metrics["D"]
        )
        np.save(
            f"{save_dir}/theory_O_eigenvals",
            metrics["O"]
        )


if __name__ == "__main__":
    RESULTS_DIR = "activity_hessian_results"
    IN_OUT_DIMS = [[784, 10]]  #, [784, 10], [10, 784]]
    ACT_FNS = ["linear"]#, "tanh", "relu"]
    USE_BIASES = [False]
    MODES = ["supervised"]  #,"unsupervised"]
    USE_SKIPS = [False, True]
    WEIGHT_INITS = ["standard"]#["one_over_N", "standard", "orthogonal"] 
    PARAM_TYPES = ["sp"]#, "mupc", "ntp"]
    ACTIVITY_DECAY = [False]#, True]
    WIDTHS = [2 ** i for i in range(11)]
    N_HIDDENS = [2 ** i for i in range(4)]
    N_SEEDS = 3

    for in_out_dims in IN_OUT_DIMS:
        for act_fn in ACT_FNS:
            for use_biases in USE_BIASES:
                for mode in MODES:
                    for use_skips in USE_SKIPS:
                        for weight_init in WEIGHT_INITS:
                            for param_type in PARAM_TYPES:
                                for activity_decay in ACTIVITY_DECAY:
                                    for width in WIDTHS:
                                        for n_hidden in N_HIDDENS:
                                            for seed in range(N_SEEDS):
                                                save_dir = setup_hessian_analysis(
                                                    results_dir=RESULTS_DIR,
                                                    in_out_dims=in_out_dims,
                                                    act_fn=act_fn,
                                                    use_biases=use_biases,
                                                    mode=mode,
                                                    use_skips=use_skips,
                                                    weight_init=weight_init,
                                                    param_type=param_type,
                                                    activity_decay=activity_decay,
                                                    width=width,
                                                    n_hidden=n_hidden,
                                                    seed=seed
                                                )
                                                os.makedirs(save_dir, exist_ok=True)
                                                run_analysis(
                                                    seed=seed,
                                                    in_out_dims=in_out_dims,
                                                    act_fn=act_fn,
                                                    use_biases=use_biases,
                                                    mode=mode,
                                                    use_skips=use_skips,
                                                    weight_init=weight_init,
                                                    param_type=param_type,
                                                    activity_decay=activity_decay,
                                                    width=width,
                                                    n_hidden=n_hidden,
                                                    save_dir=save_dir
                                                )
