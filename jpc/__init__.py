import importlib.metadata

from ._core import (
    init_activities_with_ffwd as init_activities_with_ffwd,
    init_activities_from_normal as init_activities_from_normal,
    init_activities_with_amort as init_activities_with_amort,
    pc_energy_fn as pc_energy_fn,
    neg_activity_grad as neg_activity_grad,
    solve_pc_activities as solve_pc_activities,
    compute_pc_param_grads as compute_pc_param_grads,
    linear_equilib_energy as linear_equilib_energy
)
from ._utils import (
    make_mlp as make_mlp,
    compute_accuracy as compute_accuracy,
    get_t_max as get_t_max,
    compute_infer_energies as compute_infer_energies
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
