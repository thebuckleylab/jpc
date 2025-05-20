import os
import time
import numpy as np

import jax
import equinox as eqx
import jpc
import optax
from diffrax import PIDController, ConstantStepSize

from utils import (
    setup_mlp_experiment,
    get_ode_solver,
    set_seed
)
from plotting import (
    plot_loss,
    plot_loss_and_accuracy,
    plot_runtimes,
    plot_norms
)
from experiments.datasets import get_dataloaders


def evaluate(model, test_loader):
    avg_test_loss, avg_test_acc = 0, 0
    for batch_id, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        test_loss, test_acc = jpc.test_discriminative_pc(
            model=model,
            output=label_batch,
            input=img_batch
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
        max_t1,
        activity_lr,
        param_lr,
        batch_size,
        activity_optim_id,
        max_epochs,
        test_every,
        save_dir
):
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    layer_sizes = [784] + [width] * n_hidden + [10]
    model = jpc.make_mlp(key, layer_sizes, act_fn)

    param_optim = optax.adam(param_lr)
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), None)
    )
    train_loader, test_loader = get_dataloaders(dataset, batch_size)

    train_losses = []
    test_losses, test_accs = [], []
    activity_norms, param_norms, param_grad_norms = [], [], []
    inference_runtimes = []

    if activity_optim_id != "SGD":
        stepsize_controller = ConstantStepSize() if (
                activity_optim_id == "Euler"
        ) else PIDController(rtol=1e-3, atol=1e-3)
        ode_solver = get_ode_solver(activity_optim_id)

    elif activity_optim_id == "SGD":
        activity_optim = optax.sgd(activity_lr)

    global_batch_id = 0
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}\n-------------------------------")

        for batch_id, (img_batch, label_batch) in enumerate(train_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

            if activity_optim_id != "SGD":
                result = jpc.make_pc_step(
                    model,
                    param_optim,
                    param_opt_state,
                    output=label_batch,
                    input=img_batch,
                    ode_solver=ode_solver,
                    max_t1=max_t1,
                    dt=activity_lr,
                    stepsize_controller=stepsize_controller,
                    activity_norms=True,
                    param_norms=True,
                    grad_norms=True
                )
                model, param_optim, param_opt_state = result["model"], result["optim"], result["opt_state"]
                train_loss, t_max = result["loss"], result["t_max"]
                activity_norms.append(result["activity_norms"][:-1])
                param_norms.append(
                    [p for p in result["model_param_norms"] if p == 0 or p % 2 == 0]
                )
                param_grad_norms.append(
                    [p for p in result["model_grad_norms"] if p == 0 or p % 2 == 0]
                )

            elif activity_optim_id == "SGD":
                activities = jpc.init_activities_with_ffwd(
                    model=model,
                    input=img_batch
                )
                activity_opt_state = activity_optim.init(activities)
                train_loss = jpc.mse_loss(activities[-1], label_batch)

                for t in range(max_t1):
                    activity_update_result = jpc.update_activities(
                        params=(model, None),
                        activities=activities,
                        optim=activity_optim,
                        opt_state=activity_opt_state,
                        output=label_batch,
                        input=img_batch
                    )
                    activities = activity_update_result["activities"]
                    activity_optim = activity_update_result["activity_optim"]
                    activity_opt_state = activity_update_result["activity_opt_state"]

                param_update_result = jpc.update_params(
                    params=(model, None),
                    activities=activities,
                    optim=param_optim,
                    opt_state=param_opt_state,
                    output=label_batch,
                    input=img_batch
                )
                model = param_update_result["model"]
                param_grads = param_update_result["param_grads"]
                param_optim = param_update_result["param_optim"]
                param_opt_state = param_update_result["param_opt_state"]

                activity_norms.append(jpc.compute_activity_norms(activities[:-1]))
                param_norms.append(jpc.compute_param_norms((model, None))[0])
                param_grad_norms.append(jpc.compute_param_norms(param_grads)[0])

            if activity_optim_id != "SGD":
                activities0 = jpc.init_activities_with_ffwd(model, img_batch)
                start_time = time.time()
                jax.block_until_ready(
                    jpc.solve_inference(
                        (model, None),
                        activities0,
                        output=label_batch,
                        input=img_batch,
                        solver=ode_solver,
                        max_t1=max_t1,
                        dt=activity_lr,
                        stepsize_controller=stepsize_controller
                    )
                )
                end_time = time.time()

            elif activity_optim_id == "SGD":
                activities = jpc.init_activities_with_ffwd(
                    model=model,
                    input=img_batch
                )
                activity_opt_state = activity_optim.init(activities)
                start_time = time.time()
                for t in range(max_t1):
                    jax.block_until_ready(
                        jpc.update_activities(
                            (model, None),
                            activities,
                            activity_optim,
                            activity_opt_state,
                            label_batch,
                            img_batch
                        )
                    )
                end_time = time.time()

            train_losses.append(train_loss)
            inference_runtimes.append((end_time - start_time) * 1000)
            global_batch_id += 1

            if global_batch_id % test_every == 0:
                print(f"Train loss: {train_loss:.7f} [{batch_id * len(img_batch)}/{len(train_loader.dataset)}]")

                avg_test_loss, avg_test_acc = evaluate(model, test_loader)
                test_losses.append(avg_test_loss)
                test_accs.append(avg_test_acc)
                print(f"Avg test accuracy: {avg_test_acc:.4f}\n")

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
        save_path=f"{save_dir}/test_losses_and_accs.pdf"
    )
    plot_norms(
        norms=param_norms,
        norm_type="param",
        save_path=f"{save_dir}/param_norms.pdf"
    )
    plot_norms(
        norms=param_grad_norms,
        norm_type="param_grad",
        save_path=f"{save_dir}/param_grad_norms.pdf"
    )
    plot_norms(
        norms=activity_norms,
        norm_type="activity",
        save_path=f"{save_dir}/activity_norms.pdf"
    )
    plot_runtimes(
        runtimes=inference_runtimes,
        save_path=f"{save_dir}/inference_runtimes.pdf"
    )

    np.save(f"{save_dir}/batch_train_losses.npy", train_losses)
    np.save(f"{save_dir}/test_losses.npy", test_losses)
    np.save(f"{save_dir}/test_accs.npy", test_accs)

    np.save(f"{save_dir}/activity_norms.npy", activity_norms)
    np.save(f"{save_dir}/param_norms.npy", param_norms)
    np.save(f"{save_dir}/param_grad_norms.npy", param_grad_norms)

    np.save(f"{save_dir}/inference_runtimes.npy", inference_runtimes)


if __name__ == "__main__":
    RESULTS_DIR = "mlp_results"
    DATASETS = ["MNIST", "Fashion-MNIST"]
    N_SEEDS = 3

    WIDTH = 300
    N_HIDDENS = [3, 5, 10]
    ACT_FN = "tanh"

    ACTIVITY_OPTIMS_ID = ["Euler", "Heun"]
    MAX_T1S = [5, 10, 20, 50, 100, 200, 500]
    ACTIVITY_LRS = [5e-1, 1e-1, 5e-2]

    PARAM_LR = 1e-3
    BATCH_SIZE = 64
    MAX_EPOCHS = 1
    TEST_EVERY = 100

    for dataset in DATASETS:
        for n_hidden in N_HIDDENS:
            for activity_optim_id in ACTIVITY_OPTIMS_ID:
                for max_t1 in MAX_T1S:
                    for activity_lr in ACTIVITY_LRS:
                        for seed in range(N_SEEDS):
                            save_dir = setup_mlp_experiment(
                                results_dir=RESULTS_DIR,
                                dataset=dataset,
                                width=WIDTH,
                                n_hidden=n_hidden,
                                act_fn=ACT_FN,
                                max_t1=max_t1,
                                activity_lr=activity_lr,
                                param_lr=PARAM_LR,
                                activity_optim_id=activity_optim_id,
                                seed=seed
                            )
                            train_mlp(
                                seed=seed,
                                dataset=dataset,
                                width=WIDTH,
                                n_hidden=n_hidden,
                                act_fn=ACT_FN,
                                max_t1=max_t1,
                                activity_lr=activity_lr,
                                param_lr=PARAM_LR,
                                batch_size=BATCH_SIZE,
                                activity_optim_id=activity_optim_id,
                                max_epochs=MAX_EPOCHS,
                                test_every=TEST_EVERY,
                                save_dir=save_dir
                            )
