import importlib.metadata

from .core import (
    init_activities_with_ffwd as init_activities_with_ffwd,
    init_activities_from_gaussian as init_activities_from_gaussian,
    init_activities_with_amort as init_activities_with_amort
)
from .core import (
    pc_energy_fn as pc_energy_fn,
    hpc_energy_fn as hpc_energy_fn
)
from .core import (
    compute_pc_param_grads as compute_pc_param_grads,
    compute_gen_param_grads as compute_gen_param_grads,
    compute_amort_param_grads as compute_amort_param_grads
)
from .core import solve_pc_activities as solve_pc_activities

from ._utils import (
    get_fc_network as get_fc_network,
    compute_accuracy as compute_accuracy
)
from ._train import (
    make_pc_step as make_pc_step,
    make_hpc_step as make_hpc_step
)
from ._test import (
    test_generative_pc as test_generative_pc,
    test_hpc as test_hpc
)


__version__ = importlib.metadata.version("jpc")
