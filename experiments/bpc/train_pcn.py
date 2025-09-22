import os
import argparse
import numpy as np

import jax
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx
import jpc
import optax

from experiments.datasets import get_dataloaders
from experiments.mupc_paper.utils import set_seed
from experiments.bpc.utils import setup_experiment
from experiments.bpc.plotting import plot_imgs, fig_to_pil


def evaluate(generator, amortiser, test_loader, activity_optim):
    amort_accs, bpc_accs = 0., 0.
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        activities = jpc.init_activities_with_ffwd(
            model=amortiser[::-1],
            input=img_batch
        )
        amort_accs += jpc.compute_accuracy(label_batch, activities[-1])
        activity_opt_state = activity_optim.init(activities)

        for t in range(len(amortiser)-1):
            activity_update_result = jpc.update_bpc_activities(
                top_down_model=generator,
                bottom_up_model=amortiser,
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=img_batch,
                input=label_batch
            )
            activities = activity_update_result["activities"]
            activity_opt_state = activity_update_result["opt_state"]

        bpc_accs += jpc.compute_accuracy(label_batch, activities[-1])

    img_preds = jpc.init_activities_with_ffwd(
        model=generator,
        input=label_batch
    )[-1]

    return (
        amort_accs / len(test_loader),
        bpc_accs / len(test_loader),
        label_batch,
        img_preds
    )


def poly_basis(x):
    return jnp.concatenate([x, x**2], axis=-1)


def train(
      seed,
      dataset,
      width,
      n_hidden,
      layer_type,   
      init_type,
      activity_lr,
      param_lr,
      batch_size,
      n_train_iters,
      test_every,
      save_dir
):  
    set_seed(seed)
    key = jax.random.PRNGKey(seed)
    gen_key, amort_key = jax.random.split(key, 2)
    os.makedirs(save_dir, exist_ok=True)

    # models (NOTE: input and output are inverted for the amortiser)
    input_dim, output_dim = 10, 784
    depth = n_hidden + 1
    if layer_type == "mlp":
        generator = jpc.make_mlp(
            gen_key, 
            input_dim=input_dim,
            width=width,
            depth=depth,
            output_dim=output_dim,
            act_fn="relu"
        )
        amortiser = jpc.make_mlp(
            amort_key,
            input_dim=output_dim,
            width=width,
            depth=depth,
            output_dim=input_dim,
            act_fn="relu"
        )[::-1]

    else:
        generator = jpc.make_basis_mlp(
            gen_key, 
            input_dim=input_dim,
            width=width,
            depth=depth,
            output_dim=output_dim,
            basis_fn=poly_basis
        )
        amortiser = jpc.make_basis_mlp(
            amort_key,
            input_dim=output_dim,
            width=width,
            depth=depth,
            output_dim=input_dim,
            basis_fn=poly_basis
        )[::-1]

    # optimisers
    activity_optim = optax.sgd(activity_lr)

    gen_optim = optax.adam(param_lr)
    amort_optim = optax.adam(param_lr)
    optims = [gen_optim, amort_optim]
    
    gen_opt_state = gen_optim.init(eqx.filter(generator, eqx.is_array))
    amort_opt_state = amort_optim.init(eqx.filter(amortiser, eqx.is_array))
    opt_states = [gen_opt_state, amort_opt_state]

    # data
    train_loader, test_loader = get_dataloaders(dataset, batch_size)

    # metrics
    train_gen_losses, train_amort_losses = [], []
    test_amort_accs, test_bpc_accs = [], []
    img_preds_gif = []

    for step, (img_batch, label_batch) in enumerate(train_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        
        # initialise activities
        gen_activities = jpc.init_activities_with_ffwd(
            model=generator,
            input=label_batch
        )
        gen_loss = jpc.mse_loss(gen_activities[-1], img_batch)

        # discriminative loss
        amort_activities = jpc.init_activities_with_ffwd(
            model=amortiser[::-1],
            input=img_batch
        )
        amort_loss = jpc.mse_loss(amort_activities[-1], label_batch)

        activities = gen_activities if init_type == "gen" else amort_activities
        activity_opt_state = activity_optim.init(activities)

        # inference
        for t in range(n_hidden):
            activity_update_result = jpc.update_bpc_activities(
                top_down_model=generator,
                bottom_up_model=amortiser,
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=img_batch,
                input=label_batch
            )
            activities = activity_update_result["activities"]
            activity_opt_state = activity_update_result["opt_state"]

        # learning
        param_update_result = jpc.update_bpc_params(
            top_down_model=generator,
            bottom_up_model=amortiser,
            activities=activities,
            top_down_optim=gen_optim,
            bottom_up_optim=amort_optim,
            top_down_opt_state=gen_opt_state,
            bottom_up_opt_state=amort_opt_state,
            output=img_batch,
            input=label_batch
        )
        generator, amortiser = param_update_result["models"]
        gen_opt_state, amort_opt_state  = param_update_result["opt_states"]

        train_gen_losses.append(gen_loss)
        train_amort_losses.append(amort_loss)

        if step % test_every == 0:
            amort_acc, bpc_acc, label_batch, img_preds = evaluate(
                generator,
                amortiser,
                test_loader,
                activity_optim
            )
            test_amort_accs.append(amort_acc)
            test_bpc_accs.append(bpc_acc)

            fig = plot_imgs(img_preds, dataset, label_batch, step)
            frame = fig_to_pil(fig)
            img_preds_gif.append(frame)

            print(
                f"step {step}, gen loss={gen_loss:4f}, "
                f"amort loss={amort_loss:4f}, "
                f"avg amort test accuracy={amort_acc:4f}, "
                f"avg bpc test accuracy={bpc_acc:4f}"
            )
            if (step+1) >= n_train_iters:
                break

    np.save(f"{save_dir}/train_gen_losses.npy", train_gen_losses)
    np.save(f"{save_dir}/train_amort_losses.npy", train_amort_losses)

    np.save(f"{save_dir}/test_amort_accs.npy", test_amort_accs)
    np.save(f"{save_dir}/test_bpc_accs.npy", test_bpc_accs)

    img_preds_gif[0].save(
        f"{save_dir}/train_img_preds.gif",
        save_all=True,
        append_images=img_preds_gif[1:],
        duration=900,
        loop=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="pcn_results")
    parser.add_argument("--datasets", type=str, nargs='+', default=["MNIST", "Fashion-MNIST"])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--layer_types", type=str, nargs='+', default=["mlp", "basis_fn"])
    parser.add_argument("--init_type", type=str, default="gen")
    parser.add_argument("--activity_lr", type=float, default=5e-1)
    parser.add_argument("--param_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--n_train_iters", type=int, default=300)
    parser.add_argument("--test_every", type=int, default=50)
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    for dataset in args.datasets:
        for layer_type in args.layer_types:
            for seed in range(args.n_seeds):
                save_dir = setup_experiment(
                    results_dir=args.results_dir,
                    dataset=dataset,
                    width=args.width,
                    n_hidden=args.n_hidden,
                    layer_type=layer_type,
                    init_type=args.init_type,
                    activity_lr=args.activity_lr,
                    param_lr=args.param_lr,
                    batch_size=args.batch_size,
                    n_train_iters=args.n_train_iters,
                    test_every=args.test_every,
                    seed=seed
                )
                train(
                    seed=seed,
                    dataset=dataset,
                    width=args.width,
                    n_hidden=args.n_hidden,
                    layer_type=layer_type,
                    init_type=args.init_type,
                    activity_lr=args.activity_lr,
                    param_lr=args.param_lr,
                    batch_size=args.batch_size,
                    n_train_iters=args.n_train_iters,
                    test_every=args.test_every,
                    save_dir=save_dir
                )
