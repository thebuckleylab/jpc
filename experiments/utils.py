import os
import random
import numpy as np
from torch import manual_seed
from diffrax import Euler, Heun, Midpoint, Ralston, Bosh3, Tsit5, Dopri5, Dopri8


def setup_mlp_experiment(
        results_dir,
        dataset,
        width,
        n_hidden,
        act_fn,
        max_t1,
        activity_lr,
        param_lr,
        activity_optim_id,
        seed
):
    print(
        f"""
Starting experiment with configuration:

  Dataset: {dataset}
  Width: {width}
  N hidden: {n_hidden}
  Act fn: {act_fn}
  Max t1: {max_t1}
  Activity step size: {activity_lr}
  Param learning rate: {param_lr}
  Activity optim: {activity_optim_id}
  Seed: {seed}
"""
    )
    return os.path.join(
        results_dir,
        dataset,
        f"width_{width}",
        f"{n_hidden}_n_hidden",
        act_fn,
        f"max_t1_{max_t1}",
        f"activity_lr_{activity_lr}",
        f"param_lr_{param_lr}",
        activity_optim_id,
        str(seed)
    )


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    manual_seed(seed)


def get_ode_solver(name):
    if name == "Euler":
        return Euler()
    elif name == "Heun":
        return Heun()
    elif name == "Midpoint":
        return Midpoint()
    elif name == "Ralston":
        return Ralston()
    elif name == "Bosh3":
        return Bosh3()
    elif name == "Tsit5":
        return Tsit5()
    elif name == "Dopri5":
        return Dopri5()
    elif name == "Dopri8":
        return Dopri8()
