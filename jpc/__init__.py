import importlib.metadata

from ._core import (
    init_activities_with_ffwd as init_activities_with_ffwd,
    init_activities_from_normal as init_activities_from_normal,
    init_activities_with_amort as init_activities_with_amort,
    pc_energy_fn as pc_energy_fn,
    hpc_energy_fn as hpc_energy_fn,
    neg_activity_grad as neg_activity_grad,
    solve_inference as solve_inference,
    compute_activity_grad as compute_activity_grad,
    compute_pc_param_grads as compute_pc_param_grads,
    compute_hpc_param_grads as compute_hpc_param_grads,
    update_activities as update_activities,
    update_params as update_params,
    linear_equilib_energy as linear_equilib_energy,
    linear_activities_hessian as linear_activities_hessian,
    linear_activities_solution as linear_activities_solution
)
from ._utils import (
    make_mlp as make_mlp,
    make_mlp_preactiv as make_mlp_preactiv,
    make_skip_model as make_skip_model,
    get_act_fn as get_act_fn,
    mse_loss as mse_loss,
    cross_entropy_loss as cross_entropy_loss,
    compute_accuracy as compute_accuracy,
    get_t_max as get_t_max,
    compute_activity_norms as compute_activity_norms,
    compute_infer_energies as compute_infer_energies,
    compute_param_norms as compute_param_norms
)
from ._train import (
    make_pc_step as make_pc_step,
    make_hpc_step as make_hpc_step
)
from ._test import (
    test_discriminative_pc as test_discriminative_pc,
    test_generative_pc as test_generative_pc,
    test_hpc as test_hpc
)


__version__ = importlib.metadata.version("jpc")
