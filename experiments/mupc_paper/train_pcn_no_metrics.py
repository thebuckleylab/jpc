import os
import argparse
import numpy as np

import jax
import jax.random as jr

import equinox as eqx
import jpc
import optax
from optimistix import rms_norm

from utils import (
    setup_experiment,
    set_seed,
    init_weights
)
from experiments.datasets import get_dataloaders


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
        save_dir
):
    set_seed(seed)
    key = jax.random.PRNGKey(seed)
    keys = jr.split(key, 4)
    os.makedirs(save_dir, exist_ok=True)

    # create and initialise model
    d_in, d_out = 784, 10
    L = n_hidden + 1
    model = jpc.make_mlp(
        key=keys[0],
        input_dim=d_in,
        width=width,
        depth=L,
        output_dim=d_out,
        act_fn=act_fn,
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
    if param_optim_id == "sgd":
        param_optim = optax.sgd(param_lr)
    elif param_optim_id == "adam":
        param_optim = optax.adam(param_lr)
    else:
        raise ValueError("Invalid param optim id. Options are 'sgd' and 'adam'.")

    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )
    activity_optim = optax.sgd(activity_lr) if (
            activity_optim_id == "gd"
    ) else optax.adam(activity_lr)

    # data
    train_loader, test_loader = get_dataloaders(dataset, batch_size)

    # key metrics
    train_losses = []
    test_losses, test_accs = [], []
    
    n_train_iters = len(train_loader.dataset) // batch_size * max_epochs
    n_infer_iters = np.ones(n_train_iters) * max_infer_iters

    diverged = no_learning = False
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

            if np.isinf(train_loss) or np.isnan(train_loss):
                diverged = True
                break
            
            if global_batch_id >= test_every and avg_test_acc < 15:
                no_learning = True
                break
        
        if diverged:
            print(
                f"Stopping training because of diverging loss: {train_loss}"
            )
            break
        
        if no_learning:
            print(
                f"Stopping training because of chance accuracy (no learning): {avg_test_acc}"
            )
            break

    np.save(f"{save_dir}/batch_train_losses.npy", train_losses)
    np.save(f"{save_dir}/test_losses.npy", test_losses)
    np.save(f"{save_dir}/test_accs.npy", test_accs)
    np.save(f"{save_dir}/n_infer_iters.npy", n_infer_iters)


if __name__ == "__main__":
    device = jax.devices()[0]
    print(f"device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="pcn_results")
    parser.add_argument("--datasets", type=str, nargs='+', default=["MNIST"])
    parser.add_argument("--widths", type=int, nargs='+', default=[512])
    parser.add_argument("--n_hiddens", type=int, nargs='+', default=[8])
    parser.add_argument("--act_fns", type=str, nargs='+', default=["relu"])
    parser.add_argument("--n_skips", type=int, nargs='+', default=[1])
    parser.add_argument("--weight_inits", type=str, nargs='+', default=["standard_gauss"]) 
    parser.add_argument("--param_types", type=str, nargs='+', default=["mupc"]) 
    parser.add_argument("--param_lrs", type=float, nargs='+', default=[1e-1])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_infer_iters", type=int, default=8)
    parser.add_argument("--param_optim_ids", type=str, nargs='+', default=["adam"])
    parser.add_argument("--activity_optim_ids", type=str, nargs='+', default=["gd"])
    parser.add_argument("--activity_lrs", type=float, nargs='+', default=[5e-1])
    parser.add_argument("--activity_decays", type=float, nargs='+', default=[0])
    parser.add_argument("--weight_decays", type=float, nargs='+', default=[0])
    parser.add_argument("--spectral_penalties", type=float, nargs='+', default=[0])
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=300)
    parser.add_argument("--n_seeds", type=int, default=1)
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
                                                                train_mlp(
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
                                                                    save_dir=save_dir
                                                                )
