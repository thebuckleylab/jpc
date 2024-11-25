import os
import numpy as np

import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import jpc

from utils import set_seed
from datasets import get_dataloaders

import plotly.graph_objs as go
import plotly.colors as pc


def plot_accuracies(accuracies, save_path):
    n_train_iters = len(accuracies[10])
    train_iters = [t+1 for t in range(n_train_iters)]

    colorscale = "Blues"
    colors = pc.sample_colorscale(colorscale, len(accuracies)+2)[2:][::-1]
    fig = go.Figure()
    for i, (max_t1, accuracy) in enumerate(accuracies.items()):
        fig.add_trace(
            go.Scatter(
                x=train_iters,
                y=accuracy,
                mode="lines",
                line=dict(width=2, color=colors[i]),
                name=f"$t = {max_t1}$"
            )
        )
    fig.update_layout(
        height=350,
        width=500,
        xaxis=dict(
            title="Training iteration",
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2)*10, train_iters[-1]*10]
        ),
        yaxis=dict(title="Test accuracy (%)"),
        font=dict(size=16),
        margin=dict(r=120)
    )
    fig.write_image(save_path)


def plot_energies_across_ts(theory_energies, num_energies, save_path):
    n_train_iters = len(theory_energies)
    train_iters = [t+1 for t in range(n_train_iters)]

    colorscale = "Greens"
    colors = pc.sample_colorscale(colorscale, len(num_energies)+3)[2:][::-1]
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=train_iters,
            y=theory_energies,
            name="theory",
            mode="lines",
            line=dict(
                width=3,
                dash="dash",
                color=colors[0]
            ),
        )
    )
    for i, (max_t1, num_energy) in enumerate(num_energies.items()):
        fig.add_trace(
            go.Scatter(
                x=train_iters,
                y=num_energy,
                mode="lines",
                line=dict(width=2, color=colors[i+1]),
                name=f"$t = {max_t1}$"
            )
        )
    fig.update_layout(
        height=350,
        width=500,
        xaxis=dict(
            title="Training iteration",
            tickvals=[1, int(train_iters[-1]/2), train_iters[-1]],
            ticktext=[1, int(train_iters[-1]/2), train_iters[-1]]
        ),
        yaxis=dict(title="Energy"),
        font=dict(size=16),
        margin=dict(r=120)
    )
    fig.write_image(save_path)


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


def train(
      dataset,
      width,
      n_hidden,
      lr,
      batch_size,
      max_t1,
      test_every,
      n_train_iters,
      save_dir
):
    key = jr.PRNGKey(0)
    input_dim = 3072 if dataset == "CIFAR10" else 784
    model = jpc.make_mlp(
        key,
        [input_dim] + [width]*n_hidden + [10],
        act_fn="linear",
        use_bias=False
    )
    optim = optax.adam(lr)
    opt_state = optim.init(
        (eqx.filter(model, eqx.is_array), None)
    )
    train_loader, test_loader = get_dataloaders(dataset, batch_size)

    test_accs = []
    theory_energies, num_energies = [], []
    for batch_id, (img_batch, label_batch) in enumerate(train_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        theory_energies.append(
            jpc.linear_equilib_energy(
                network=model,
                x=img_batch,
                y=label_batch
            )
        )
        result = jpc.make_pc_step(
            model,
            optim,
            opt_state,
            output=label_batch,
            input=img_batch,
            max_t1=max_t1,
            record_energies=True
        )
        model, optim, opt_state = result["model"], result["optim"], result["opt_state"]
        train_loss, t_max = result["loss"], result["t_max"]
        num_energies.append(result["energies"][:, t_max-1].sum())

        if ((batch_id+1) % test_every) == 0:
            avg_test_loss, avg_test_acc = evaluate(model, test_loader)
            test_accs.append(avg_test_acc)
            print(
                f"Train iter {batch_id+1}, train loss={train_loss:4f}, "
                f"avg test accuracy={avg_test_acc:4f}"
            )
            if (batch_id+1) >= n_train_iters:
                break

    np.save(f"{save_dir}/test_accs.npy", test_accs)
    np.save(f"{save_dir}/theory_energies.npy", theory_energies)
    np.save(f"{save_dir}/num_energies.npy", num_energies)

    return test_accs, jnp.array(theory_energies), jnp.array(num_energies)


if __name__ == "__main__":
    RESULTS_DIR = "theory_energies_results"
    DATASETS = ["MNIST", "Fashion-MNIST"]
    SEED = 916
    WIDTH = 300
    N_HIDDEN = 10
    LR = 1e-3
    BATCH_SIZE = 64
    MAX_T1S = [200, 100, 50, 20, 10]
    TEST_EVERY = 10
    N_TRAIN_ITERS = 100

    for dataset in DATASETS:
        set_seed(SEED)
        all_test_accs, all_theory_energies, all_num_energies = {}, {}, {}
        for max_t1 in MAX_T1S:
            print(f"\nmax_t1: {max_t1}")
            save_dir = os.path.join(RESULTS_DIR, dataset, f"max_t1_{max_t1}")
            os.makedirs(save_dir, exist_ok=True)
            test_accs, theory_energies, num_energies = train(
                dataset=dataset,
                width=WIDTH,
                n_hidden=N_HIDDEN,
                lr=LR,
                batch_size=BATCH_SIZE,
                max_t1=max_t1,
                test_every=TEST_EVERY,
                n_train_iters=N_TRAIN_ITERS,
                save_dir=save_dir
            )
            all_test_accs[max_t1] = test_accs
            all_theory_energies[max_t1] = theory_energies
            all_num_energies[max_t1] = num_energies

        plot_accuracies(
            all_test_accs,
            f"{RESULTS_DIR}/{dataset}/test_accs.pdf"
        )
        plot_energies_across_ts(
            all_theory_energies[MAX_T1S[0]],
            all_num_energies,
            f"{RESULTS_DIR}/{dataset}/energies.pdf"
        )
