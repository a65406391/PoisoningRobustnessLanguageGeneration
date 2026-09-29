from dataclasses import dataclass
from enum import IntEnum

import gurobipy as gp
import numpy as np


class Threats(IntEnum):
    L0 = 0
    L2 = 1


@dataclass
class Constraints:
    """This class holds f-divergences constraints for a threat model"""

    epsilon_f: float
    divergence: callable = lambda x: x * gp.nlfunc.log(x)  # KL-divergence by default

    def __post_init__(self):
        assert isinstance(self.epsilon_f, float), """The f-divergence constraint `epsilon_f` must be a float"""
        assert self.epsilon_f >= 0, """The f-divergence constraint `epsilon_f` must be non-negative"""
        assert callable(self.divergence), "The f-divergence must be a callable"

    @property
    def num_constraint_sets(self) -> int:
        # 1 constraint set, represented by the (KL-, or potentially other f-) divergence
        return 1
