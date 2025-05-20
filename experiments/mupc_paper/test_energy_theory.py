import os
import numpy as np
import jax.random as jr

import jpc
import optax
import equinox as eqx

from utils import set_seed, init_weights
from experiments.datasets import get_dataloaders


def run_test(
        seed,
        dataset,
        width,
        n_hidden,
        act_fn,
        n_skip,
        param_type,
        param_optim_id,
        param_lr,
        batch_size,
        save_dir
):
    set_seed(seed)
    key = jr.PRNGKey(seed)
    model_key, init_key = jr.split(key, 2)
    os.makedirs(save_dir, exist_ok=True)

    # create and initialise model
    d_in, d_out = 784, 10
    L = n_hidden + 1
    model = jpc.make_mlp_preactiv(
        key=model_key,
        d_in=d_in,
        N=width,
        L=L,
        d_out=d_out,
        act_fn=jpc.get_act_fn(act_fn),
        use_bias=False
    )
    if param_type != "SP":
        model = init_weights(
            key=init_key,
            model=model,
            init_fn_id="standard_gauss"
        )
    skip_model = jpc.make_skip_model(model) if n_skip == 1 else None

    # optimisers
    if param_optim_id == "SGD":
        param_optim = optax.sgd(param_lr)
    elif param_optim_id == "Adam":
        param_optim = optax.adam(param_lr)
    
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )

    # data & metrics
    train_loader, _ = get_dataloaders(dataset, batch_size)
    train_losses, train_energies = [], []
    loss_energy_ratios = []
    
    for t, (img_batch, label_batch) in enumerate(train_loader):
        x, y = img_batch.numpy(), label_batch.numpy()
    
        # compute loss
        activities = jpc.init_activities_with_ffwd(
            model=model,
            input=x,
            skip_model=skip_model,
            n_skip=n_skip,
            param_type=param_type
        )
        loss = 0.5 * np.sum((y - activities[-1])**2) / batch_size

        # compute theoretical activities & energy
        activities = jpc.linear_activities_solution(
            network=model,
            x=x,
            y=y,
            n_skip=n_skip,
            param_type=param_type
        )       
        energy = jpc.pc_energy_fn(
            params=(model, skip_model),
            activities=activities,
            y=y,
            x=x,
            n_skip=n_skip,
            param_type=param_type
        )
                                                          
        # update parameters
        param_update_result = jpc.update_params(
            params=(model, skip_model),
            activities=activities,
            optim=param_optim,
            opt_state=param_opt_state,
            output=y,
            input=x,
            n_skip=n_skip,
            param_type=param_type
        )
        model = param_update_result["model"]
        skip_model = param_update_result["skip_model"]
        param_opt_state = param_update_result["opt_state"]

        train_losses.append(loss)
        train_energies.append(energy)
        loss_energy_ratios.append(loss/energy)
    
        if t % 200 == 0:
            print(
                f"\t{t * len(img_batch)}/{len(train_loader.dataset)}, "
                f"loss: {loss:.4f}, energy: {energy:.4f}, ratio: {loss/energy:.4f} "
            )
    
    np.save(f"{save_dir}/train_losses.npy", train_losses)
    np.save(f"{save_dir}/train_energies.npy", train_energies)
    np.save(f"{save_dir}/loss_energy_ratios.npy", loss_energy_ratios)


if __name__ == "__main__":

    RESULTS_DIR = "energy_theory_results"
    DATASET = "MNIST"
    N_SKIP = 1
    PARAM_OPTIM_ID = "Adam"
    PARAM_LR = 1e-2
    BATCH_SIZE = 64

    ACT_FNS = ["linear"]
    PARAM_TYPES = ["μP"]  #"SP", 
    WIDTHS = [2**i for i in range(7)]           # 7 or 10 max
    N_HIDDENS = [2**i for i in range(7)]        # 4 or 7 max
    SEED = 4320

    for act_fn in ACT_FNS:
        for param_type in PARAM_TYPES:
            for width in WIDTHS:
                for n_hidden in N_HIDDENS:
                    print(
                        f"\nAct fn: {act_fn}\n"
                        f"Param type: {param_type}\n"
                        f"N skip: {N_SKIP}\n"
                        f"Param optim: {PARAM_OPTIM_ID}\n"
                        f"Param lr: {PARAM_LR}\n"
                        f"Width: {width}\n"
                        f"N hidden: {n_hidden}\n"
                        f"Seed: {SEED}\n"
                    )
                    save_dir = os.path.join(
                        RESULTS_DIR,
                        act_fn,
                        param_type,
                        f"{N_SKIP}_skip",
                        PARAM_OPTIM_ID,
                        f"param_lr_{PARAM_LR}",
                        f"width_{width}",
                        f"{n_hidden}_n_hidden",
                        str(SEED)
                    )
                    run_test(
                        seed=SEED,
                        dataset=DATASET,
                        width=width,
                        n_hidden=n_hidden,
                        act_fn=act_fn,
                        n_skip=N_SKIP,
                        param_type=param_type,
                        param_optim_id=PARAM_OPTIM_ID,
                        param_lr=PARAM_LR,
                        batch_size=BATCH_SIZE,
                        save_dir=save_dir
                    )
