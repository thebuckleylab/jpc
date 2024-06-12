import importlib.metadata

from ._utils import get_fc_network as get_fc_network
from ._init import (
    init_activities_with_ffwd as init_activities_with_ffwd,
    init_activities_from_gaussian as init_activities_from_gaussian
)
from ._energies import (
    pc_energy_fn as pc_energy_fn,
    hpc_energy_fn as hpc_energy_fn
)
from ._infer import solve_pc_activities as solve_pc_activities
from ._grads import (
    compute_pc_param_grads as compute_pc_param_grads,
    compute_gen_param_grads as compute_gen_param_grads,
    compute_amort_param_grads as compute_amort_param_grads
)


__version__ = importlib.metadata.version("jpc_local")
