import importlib.metadata

from ._core import (
    _check_param_type as _check_param_type,
    _get_param_scalings as _get_param_scalings,
    bpc_energy_fn as bpc_energy_fn,
    compute_bpc_activity_grad as compute_bpc_activity_grad,
    compute_bpc_param_grads as compute_bpc_param_grads,
    compute_epc_error_grad as compute_epc_error_grad,
    compute_epc_param_grads as compute_epc_param_grads,
    compute_hpc_param_grads as compute_hpc_param_grads,
    compute_linear_activity_hessian as compute_linear_activity_hessian,
    compute_linear_activity_solution as compute_linear_activity_solution,
    compute_linear_equilib_energy_grads as compute_linear_equilib_energy_grads,
    compute_linear_equilib_rescaling as compute_linear_equilib_rescaling,
    compute_pc_activity_grad as compute_pc_activity_grad,
    compute_pc_param_grads as compute_pc_param_grads,
    compute_pdm_activity_grad as compute_pdm_activity_grad,
    compute_pdm_param_grads as compute_pdm_param_grads,
    epc_energy_fn as epc_energy_fn,
    hpc_energy_fn as hpc_energy_fn,
    init_activities_from_normal as init_activities_from_normal,
    init_activities_with_amort as init_activities_with_amort,
    init_activities_with_ffwd as init_activities_with_ffwd,
    init_epc_errors as init_epc_errors,
    linear_equilib_energy as linear_equilib_energy,
    neg_pc_activity_grad as neg_pc_activity_grad,
    pc_energy_fn as pc_energy_fn,
    pdm_energy_fn as pdm_energy_fn,
    solve_inference as solve_inference,
    update_bpc_activities as update_bpc_activities,
    update_bpc_params as update_bpc_params,
    update_epc_errors as update_epc_errors,
    update_epc_params as update_epc_params,
    update_linear_equilib_energy_params as update_linear_equilib_energy_params,
    update_pc_activities as update_pc_activities,
    update_pc_params as update_pc_params,
    update_pdm_activities as update_pdm_activities,
    update_pdm_params as update_pdm_params,
)
from ._test import (
    test_discriminative_pc as test_discriminative_pc,
    test_generative_pc as test_generative_pc,
    test_hpc as test_hpc,
)
from ._train import make_hpc_step as make_hpc_step, make_pc_step as make_pc_step
from ._utils import (
    compute_accuracy as compute_accuracy,
    compute_activity_norms as compute_activity_norms,
    compute_infer_energies as compute_infer_energies,
    compute_param_norms as compute_param_norms,
    cross_entropy_loss as cross_entropy_loss,
    get_act_fn as get_act_fn,
    get_t_max as get_t_max,
    make_mlp as make_mlp,
    make_skip_model as make_skip_model,
    mse_loss as mse_loss,
)


__version__ = importlib.metadata.version("jpc")
