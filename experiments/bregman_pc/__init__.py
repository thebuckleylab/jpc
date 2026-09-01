"""Activation-matched Bregman predictive coding experiment.

Energy, inference, and weight updates are implemented in jpc core.
"""

from .model import BregmanMLP
from .steps import bregman_pc_step, standard_pc_step
