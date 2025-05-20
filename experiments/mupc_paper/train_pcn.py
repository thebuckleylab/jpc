import os
import pickle
import argparse
import numpy as np

import jax
import jax.random as jr
import jax.numpy as jnp
from jax.tree_util import tree_map

import equinox as eqx
import jpc
import optax
from optimistix import rms_norm

from utils import (
    setup_experiment,
    set_seed,
    init_weights,
    compute_param_l2_norms,
    compute_param_spectral_norms,
    compute_hessian_eigens,
    compute_cond_num
)
from experiments.datasets import get_dataloaders
from plotting import (
    plot_loss,
    plot_loss_and_accuracy,
    plot_n_infer_iters,
    plot_norms,
    plot_energies,
    plot_hessian_eigenvalues_during_training,
    plot_max_min_eigenvals,
    plot_max_min_eigenvals_2_axes,
    plot_metric_stats
)


def evaluate(params, test_loader, n_skip, param_type):
    model, skip_model = params
    avg_test_loss, avg_test_acc = 0, 0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        test_loss, test_acc = jpc.test_discriminative_pc(
            model=model,
            output=label_batch,
            input=img_batch,
            skip_model=skip_model,
            n_skip=n_skip,
            param_type=param_type
        )
        avg_test_loss += test_loss
        avg_test_acc += test_acc

    return avg_test_loss / len(test_loader), avg_test_acc / len(test_loader)


def train_mlp(
        seed,
        dataset,
        width,
        n_hidden,
        act_fn,
        n_skip,
        weight_init,
        param_type,
        param_optim_id,
        param_lr,
        batch_size,
        max_infer_iters,
        activity_optim_id,
        activity_lr,
        activity_decay,
        weight_decay,
        spectral_penalty,
        max_epochs,
        test_every,
        compute_activity_vec,
        compute_hessian,
        save_dir
):
    set_seed(seed)
    key = jax.random.PRNGKey(seed)
    keys = jr.split(key, 4)
    os.makedirs(save_dir, exist_ok=True)

    # create and initialise model
    d_in, d_out = 784, 10
    L = n_hidden + 1
    model = jpc.make_mlp_preactiv(
        key=keys[0],
        d_in=d_in,
        N=width,
        L=L,
        d_out=d_out,
        act_fn=jpc.get_act_fn(act_fn),
        use_bias=False
    )
    if weight_init != "standard":
        gain = 1.05 if (weight_init == "orthogonal" and act_fn == "tanh") else 1
        model = init_weights(
            key=keys[1],
            model=model,
            init_fn_id=weight_init,
            gain=gain
        )
    skip_model = jpc.make_skip_model(model) if n_skip == 1 else None

    # optimisers
    if param_optim_id == "SGD":
        param_optim = optax.sgd(param_lr)
    elif param_optim_id == "Adam":
        param_optim = optax.adam(param_lr)
    else:
        raise ValueError("Invalid param optim id. Options are 'SGD' and 'Adam'.")

    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )
    activity_optim = optax.sgd(activity_lr) if (
            activity_optim_id == "GD"
    ) else optax.adam(activity_lr)

    # data
    train_loader, test_loader = get_dataloaders(dataset, batch_size)

    # metrics
    train_losses = []
    test_losses, test_accs = [], []

    n_train_iters = len(train_loader.dataset) // batch_size * max_epochs
    n_test_iters = n_train_iters // test_every * max_epochs
    layer_idxs = [0, int(L / 4) - 1, int(L / 2) - 1, int(L * 3 / 4) - 1, L - 1]

    mean_abs_activities = np.zeros(
        (n_train_iters, max_infer_iters + 1, len(layer_idxs))
    )
    activity_l2_norms = np.zeros_like(mean_abs_activities)
    n_infer_iters = np.ones(n_train_iters) * max_infer_iters
    train_theory_activities = {}
    if compute_activity_vec:
        mean_activity_vec = np.zeros(
            (n_train_iters, max_infer_iters, width * (L - 1) + d_out)
        )

    param_l2_norms = np.zeros((n_train_iters + 1, len(layer_idxs)))
    param_spectral_norms = np.zeros_like(param_l2_norms)

    if compute_hessian:
        hessian_eigenvals = np.zeros((n_test_iters + 1, width * n_hidden))
        max_min_hess_eigenvals = np.zeros((2, n_test_iters + 1))

    train_theory_energies = np.zeros((n_test_iters + 1, len(layer_idxs)))
    train_num_energies = np.zeros_like(train_theory_energies)

    has_diverged, no_learning = False, False
    global_batch_id = 0
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}\n-------------------------------")

        for train_iter, (img_batch, label_batch) in enumerate(train_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

            # initialise activities
            activities = jpc.init_activities_with_ffwd(
                model=model,
                input=img_batch,
                skip_model=skip_model,
                n_skip=n_skip,
                param_type=param_type
            )
            activity_opt_state = activity_optim.init(activities)
            train_loss = jpc.mse_loss(activities[-1], label_batch)

            # record metrics at init
            i = 0
            for l, act in enumerate(activities):
                if l in layer_idxs:
                    mean_abs_activities[global_batch_id, 0, i] = np.array(
                        jnp.mean(jnp.abs(act))
                    )
                    activity_l2_norms[global_batch_id, 0, i] = np.array(
                        jnp.linalg.norm(act, axis=1, ord=2).mean()
                    )
                    i += 1

            if global_batch_id == 0:
                if act_fn == "linear":
                    theory_activities = jpc.linear_activities_solution(
                        network=model,
                        x=img_batch,
                        y=label_batch,
                        n_skip=n_skip,
                        param_type=param_type,
                        activity_decay=activity_decay
                    )
                    theory_energies = jpc.pc_energy_fn(
                        params=(model, skip_model),
                        activities=theory_activities,
                        y=label_batch,
                        x=img_batch,
                        n_skip=n_skip,
                        param_type=param_type,
                        activity_decay=activity_decay,
                        weight_decay=weight_decay,
                        spectral_penalty=spectral_penalty,
                        record_layers=True
                    )
                    train_theory_activities[0] = [
                        a for l, a in enumerate(theory_activities) if l in layer_idxs
                    ]
                    train_theory_energies[0] = np.array([
                        e for l, e in enumerate(reversed(theory_energies)) if l in layer_idxs
                    ])

                param_l2_norms[0] = compute_param_l2_norms(
                    model=model,
                    act_fn=act_fn,
                    layer_idxs=layer_idxs
                )
                param_spectral_norms[0] = compute_param_spectral_norms(
                    model=model,
                    act_fn=act_fn,
                    layer_idxs=layer_idxs
                )
                if compute_hessian:
                    eigenvals, _ = compute_hessian_eigens(
                        params=(model, skip_model),
                        activities=tree_map(lambda a: a[[0], :], activities),
                        y=label_batch[[0], :],
                        x=img_batch[[0], :],
                        n_skip=n_skip,
                        param_type=param_type,
                        activity_decay=activity_decay,
                        weight_decay=weight_decay,
                        spectral_penalty=spectral_penalty
                    )
                    hessian_eigenvals[0] = eigenvals
                    max_min_hess_eigenvals[:, 0] = np.array(
                        [max(eigenvals), min(eigenvals)]
                    )

            # inference
            for t in range(max_infer_iters):
                activity_update_result = jpc.update_activities(
                    params=(model, skip_model),
                    activities=activities,
                    optim=activity_optim,
                    opt_state=activity_opt_state,
                    output=label_batch,
                    input=img_batch,
                    n_skip=n_skip,
                    param_type=param_type,
                    activity_decay=activity_decay,
                    weight_decay=weight_decay,
                    spectral_penalty=spectral_penalty
                )
                activities = activity_update_result["activities"]
                activity_opt_state = activity_update_result["opt_state"]
                activity_grads = activity_update_result["grads"]
                if rms_norm(activity_grads) < 1e-3 + 1e-3 * rms_norm(activity_grads):
                    n_infer_iters[global_batch_id] = t            
                    #break
                
                if global_batch_id == 0 or global_batch_id % test_every == 0:
                    num_energies = jpc.pc_energy_fn(
                        params=(model, skip_model),
                        activities=activities,
                        y=label_batch,
                        x=img_batch,
                        n_skip=n_skip,
                        param_type=param_type,
                        activity_decay=activity_decay,
                        weight_decay=weight_decay,
                        spectral_penalty=spectral_penalty,
                        record_layers=True
                    )
                    test_iter = 0 if (
                            global_batch_id == 0
                    ) else int(global_batch_id / test_every)
                    train_num_energies[test_iter] = np.array([
                        e for l, e in enumerate(reversed(num_energies)) if l in layer_idxs
                    ])

                i = 0
                for l, act in enumerate(activities):
                    if l in layer_idxs:
                        mean_abs_activities[global_batch_id, t + 1, i] = np.array(
                            jnp.mean(jnp.abs(act))
                        )
                        activity_l2_norms[global_batch_id, t + 1, i] = np.array(
                            jnp.linalg.norm(act, axis=1, ord=2).mean()
                        )
                        i += 1

                if compute_activity_vec:
                    mean_activity_vec[global_batch_id, t] = jnp.concatenate(
                        [activities[i].mean(axis=0) for i in range(len(activities))],
                        axis=0
                    )

            # update parameters
            param_update_result = jpc.update_params(
                params=(model, skip_model),
                activities=activities,
                optim=param_optim,
                opt_state=param_opt_state,
                output=label_batch,
                input=img_batch,
                n_skip=n_skip,
                param_type=param_type,
                activity_decay=activity_decay,
                weight_decay=weight_decay,
                spectral_penalty=spectral_penalty
            )
            model = param_update_result["model"]
            skip_model = param_update_result["skip_model"]
            param_opt_state = param_update_result["opt_state"]

            param_l2_norms[global_batch_id + 1] = compute_param_l2_norms(
                model=model,
                act_fn=act_fn,
                layer_idxs=layer_idxs
            )
            param_spectral_norms[global_batch_id + 1] = compute_param_spectral_norms(
                model=model,
                act_fn=act_fn,
                layer_idxs=layer_idxs
            )
            train_losses.append(train_loss)
            global_batch_id += 1

            if global_batch_id % test_every == 0:
                print(
                    f"Train loss: {train_loss:.7f} [{train_iter * len(img_batch)}/{len(train_loader.dataset)}]"
                )
                avg_test_loss, avg_test_acc = evaluate(
                    params=(model, skip_model),
                    test_loader=test_loader,
                    n_skip=n_skip,
                    param_type=param_type
                )
                test_losses.append(avg_test_loss)
                test_accs.append(avg_test_acc)
                print(f"Avg test accuracy: {avg_test_acc:.4f}\n")

                test_iter = int(global_batch_id / test_every)
                if compute_hessian:
                    eigenvals, _ = compute_hessian_eigens(
                        params=(model, skip_model),
                        activities=tree_map(lambda a: a[[0], :], activities),
                        y=label_batch[[0], :],
                        x=img_batch[[0], :],
                        n_skip=n_skip,
                        param_type=param_type,
                        activity_decay=activity_decay,
                        weight_decay=weight_decay,
                        spectral_penalty=spectral_penalty
                    )
                    hessian_eigenvals[test_iter] = eigenvals
                    max_min_hess_eigenvals[:, test_iter] = np.array(
                        [max(eigenvals), min(eigenvals)]
                    )
                if act_fn == "linear":
                    theory_activities = jpc.linear_activities_solution(
                        network=model,
                        x=img_batch,
                        y=label_batch,
                        n_skip=n_skip,
                        param_type=param_type,
                        activity_decay=activity_decay
                    )
                    theory_energies = jpc.pc_energy_fn(
                        params=(model, skip_model),
                        activities=theory_activities,
                        y=label_batch,
                        x=img_batch,
                        n_skip=n_skip,
                        param_type=param_type,
                        activity_decay=activity_decay,
                        weight_decay=weight_decay,
                        spectral_penalty=spectral_penalty,
                        record_layers=True
                    )
                    train_theory_activities[global_batch_id] = [
                        a for l, a in enumerate(theory_activities) if l in layer_idxs
                    ]
                    train_theory_energies[test_iter] = np.array([
                        e for l, e in enumerate(reversed(theory_energies)) if l in layer_idxs
                    ])

            if np.isinf(train_loss) or np.isnan(train_loss):
                has_diverged = True
                break
            
            if global_batch_id >= test_every and avg_test_acc < 15:
                no_learning = True
                break
        
        if has_diverged:
            print(
                f"Stopping training because of diverging loss: {train_loss}"
            )
            break

        if no_learning:
            print(
                f"Stopping training because of chance accuracy (no learning): {avg_test_acc}"
            )
            break
    
    cond_nums = [compute_cond_num(eig) for eig in hessian_eigenvals] if (
        compute_hessian
    ) else None

    plot_loss(
        loss=train_losses,
        yaxis_title="Train loss",
        xaxis_title="Iteration",
        save_path=f"{save_dir}/train_losses.pdf"
    )
    plot_loss_and_accuracy(
        loss=test_losses,
        accuracy=test_accs,
        mode="test",
        xaxis_title="Training iteration",
        save_path=f"{save_dir}/test_losses_and_accs.pdf",
        test_every=test_every
    )
    plot_n_infer_iters(
        n_infer_iters=n_infer_iters,
        save_path=f"{save_dir}/n_infer_iters.pdf"
    )
    plot_norms(
        norms=param_l2_norms,
        norm_type="param_l2",
        save_path=f"{save_dir}/param_l2_norms.pdf"
    )
    plot_norms(
        norms=param_spectral_norms,
        norm_type="param_spectral",
        save_path=f"{save_dir}/param_spectral_norms.pdf"
    )
    plot_energies(
        energies=train_num_energies.T,
        test_every=test_every,
        save_path=f"{save_dir}/energies.pdf",
        theory_energies=train_theory_energies.T if act_fn == "linear" else None,
        log=False
    )
    plot_energies(
        energies=train_num_energies.T,
        test_every=test_every,
        save_path=f"{save_dir}/log_energies.pdf",
        theory_energies=train_theory_energies.T if act_fn == "linear" else None,
        log=True
    )
    if compute_hessian:
        plot_hessian_eigenvalues_during_training(
            eigenvals=[e for i, e in enumerate(hessian_eigenvals) if i % 2 == 0],
            test_every=200,
            save_path=f"{save_dir}/hessian_eigenvals.pdf"
        )
        plot_max_min_eigenvals(
            max_min_eigenvals=max_min_hess_eigenvals,
            test_every=test_every,
            save_path=f"{save_dir}/max_min_eigenvals.pdf"
        )
        plot_max_min_eigenvals_2_axes(
            max_min_eigenvals=max_min_hess_eigenvals,
            test_every=test_every,
            save_path=f"{save_dir}/max_min_eigenvals_2_axes.pdf"
        )
    if act_fn != "linear":
        #last_T = np.argmin(activity_l2_norms[-1, :, 0]) - 1
        #first_T = np.argmin(activity_l2_norms[0, :, 0]) - 1
        #if first_T > 0:
        plot_norms(
            norms=activity_l2_norms[0],  #:first_T]
            norm_type="activity",
            save_path=f"{save_dir}/activity_l2_norms_at_init.pdf"
        )
        plot_norms(
            norms=activity_l2_norms[0],  #:first_T]
            norm_type="activity",
            save_path=f"{save_dir}/log_activity_l2_norms_at_init.pdf",
            log=True
        )
        #if last_T > 0:
        plot_norms(
            norms=activity_l2_norms[-1],  #:last_T],
            norm_type="activity",
            save_path=f"{save_dir}/activity_l2_norms_last_train_iter.pdf"
        )
        plot_norms(
            norms=activity_l2_norms[-1],  #:last_T],
            norm_type="activity",
            save_path=f"{save_dir}/log_activity_l2_norms_last_train_iter.pdf",
            log=True
        )

    np.save(f"{save_dir}/batch_train_losses.npy", train_losses)
    np.save(f"{save_dir}/test_losses.npy", test_losses)
    np.save(f"{save_dir}/test_accs.npy", test_accs)
    np.save(f"{save_dir}/num_energies.npy", train_num_energies)

    np.save(f"{save_dir}/mean_abs_activities.npy", mean_abs_activities)
    np.save(f"{save_dir}/activity_l2_norms.npy", activity_l2_norms)
    np.save(f"{save_dir}/n_infer_iters.npy", n_infer_iters)
    if compute_activity_vec:
        np.save(f"{save_dir}/mean_activity_vec.npy", mean_activity_vec)

    np.save(f"{save_dir}/param_l2_norms.npy", param_l2_norms)
    np.save(f"{save_dir}/param_spectral_norms.npy", param_spectral_norms)
    if compute_hessian:
        np.save(f"{save_dir}/hessian_eigenvals.npy", hessian_eigenvals)
        np.save(f"{save_dir}/cond_nums.npy", cond_nums)

    if act_fn == "linear":
        np.save(f"{save_dir}/theory_energies.npy", train_theory_energies)
        with open(f"{save_dir}/theory_activities.pkl", "wb") as f:
            pickle.dump(train_theory_activities, f)

        theory_activity_l2_norms = np.zeros(
            (len(train_theory_activities.keys()), len(train_theory_activities[0]))
        )
        for i, t in enumerate(train_theory_activities.keys()):
            for l, act in enumerate(train_theory_activities[t]):
                theory_activity_l2_norms[i, l] = np.array(
                    np.linalg.norm(act, axis=1, ord=2).mean()
                )

        plot_norms(
            norms=activity_l2_norms[0],
            norm_type="activity",
            save_path=f"{save_dir}/theory_activity_l2_norms_at_init.pdf",
            theory_norms=theory_activity_l2_norms[0],
            log=False
        )
        plot_norms(
            norms=activity_l2_norms[0],
            norm_type="activity",
            save_path=f"{save_dir}/log_theory_activity_l2_norms_at_init.pdf",
            theory_norms=theory_activity_l2_norms[0],
            log=True
        )
        plot_norms(
            norms=activity_l2_norms[-1],
            norm_type="activity",
            save_path=f"{save_dir}/theory_activity_l2_norms_last_train_iter.pdf",
            theory_norms=theory_activity_l2_norms[-1],
            log=False
        )
        plot_norms(
            norms=activity_l2_norms[-1],
            norm_type="activity",
            save_path=f"{save_dir}/log_theory_activity_l2_norms_last_train_iter.pdf",
            theory_norms=theory_activity_l2_norms[-1],
            log=True
        )

    return test_accs, cond_nums


if __name__ == "__main__":
    device = jax.devices()[0]
    print(f"device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="pcn_results")
    parser.add_argument("--datasets", type=str, nargs='+', default=["MNIST"])  # , "Fashion-MNIST"]
    parser.add_argument("--widths", type=int, nargs='+', default=[128])
    parser.add_argument("--n_hiddens", type=int, nargs='+', default=[32])
    parser.add_argument("--act_fns", type=str, nargs='+', default=["linear", "tanh", "relu"])  # , "tanh", "relu"]
    parser.add_argument("--n_skips", type=int, nargs='+', default=[1])  # , 1]
    parser.add_argument("--weight_inits", type=str, nargs='+', default=["standard_gauss"])  # "one_over_N", "standard_gauss", "orthogonal"]
    parser.add_argument("--param_types", type=str, nargs='+', default=["μP"])  # , "NTP", "μP"]
    parser.add_argument("--param_lrs", type=float, nargs='+', default=[1e-1])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_infer_iters", type=int, default=32)
    parser.add_argument("--param_optim_ids", type=str, nargs='+', default=["Adam"])  # , "SGD"]
    parser.add_argument("--activity_optim_ids", type=str, nargs='+', default=["GD"])  # , "Adam"]
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[5e-1])  # 5e-1, 1e-1,
    parser.add_argument("--activity_decays", type=float, nargs='+', default=[0])
    parser.add_argument("--weight_decays", type=float, nargs='+', default=[0])
    parser.add_argument("--spectral_penalties", type=float, nargs='+', default=[0])
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=300)
    parser.add_argument("--compute_activity_vec", type=bool, default=False)
    parser.add_argument("--compute_hessian", type=bool, default=True)
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    for dataset in args.datasets:
        for width in args.widths:
            for n_hidden in args.n_hiddens:
                for act_fn in args.act_fns:
                    for n_skip in args.n_skips:
                        for weight_init in args.weight_inits:
                            for param_type in args.param_types:
                                for param_optim_id in args.param_optim_ids:
                                    for param_lr in args.param_lrs:
                                        for activity_optim_id in args.activity_optim_ids:
                                            for activity_lr in args.activity_lrs:
                                                for activity_decay in args.activity_decays:
                                                    for weight_decay in args.weight_decays:
                                                        for spectral_penalty in args.spectral_penalties:

                                                            test_accs_seeds = [[] for _ in range(args.n_seeds)]
                                                            cond_nums_seeds = [[] for _ in range(args.n_seeds)]
                                                            for seed in range(args.n_seeds):
                                                                save_dir = setup_experiment(
                                                                    results_dir=args.results_dir,
                                                                    dataset=dataset,
                                                                    width=width,
                                                                    n_hidden=n_hidden,
                                                                    act_fn=act_fn,
                                                                    n_skip=n_skip,
                                                                    weight_init=weight_init,
                                                                    param_type=param_type,
                                                                    param_optim_id=param_optim_id,
                                                                    param_lr=param_lr,
                                                                    batch_size=args.batch_size,
                                                                    max_infer_iters=args.max_infer_iters,
                                                                    activity_optim_id=activity_optim_id,
                                                                    activity_lr=activity_lr,
                                                                    activity_decay=activity_decay,
                                                                    weight_decay=weight_decay,
                                                                    spectral_penalty=spectral_penalty,
                                                                    max_epochs=args.max_epochs,
                                                                    seed=seed
                                                                )
                                                                test_accs, cond_nums = train_mlp(
                                                                    seed=seed,
                                                                    dataset=dataset,
                                                                    width=width,
                                                                    n_hidden=n_hidden,
                                                                    act_fn=act_fn,
                                                                    n_skip=n_skip,
                                                                    weight_init=weight_init,
                                                                    param_type=param_type,
                                                                    param_optim_id=param_optim_id,
                                                                    param_lr=param_lr,
                                                                    batch_size=args.batch_size,
                                                                    max_infer_iters=args.max_infer_iters,
                                                                    activity_optim_id=activity_optim_id,
                                                                    activity_lr=activity_lr,
                                                                    activity_decay=activity_decay,
                                                                    weight_decay=weight_decay,
                                                                    spectral_penalty=spectral_penalty,
                                                                    max_epochs=args.max_epochs,
                                                                    test_every=args.test_every,
                                                                    compute_activity_vec=args.compute_activity_vec,
                                                                    compute_hessian=args.compute_hessian,
                                                                    save_dir=save_dir
                                                                )
                                                                test_accs_seeds[seed] = test_accs
                                                                cond_nums_seeds[seed] = cond_nums

                                                            plot_metric_stats(
                                                                metric=test_accs_seeds,
                                                                metric_id="test_acc",
                                                                test_every=args.test_every,
                                                                save_path=f"{save_dir[:-1]}/test_accs.pdf"
                                                            )
                                                            if args.compute_hessian:
                                                                plot_metric_stats(
                                                                    metric=cond_nums_seeds,
                                                                    metric_id="cond_num",
                                                                    test_every=args.test_every,
                                                                    save_path=f"{save_dir[:-1]}/cond_nums.pdf"
                                                                )
