"""LP container used by every SAMADHAN component.

Canonical form (all solvers work on this):
    minimize    c'x + obj_const
    subject to  K[:n_eq] x  = q[:n_eq]
                K[n_eq:] x >= q[n_eq:]
                l <= x <= u          (entries may be +-inf)
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class LP:
    c: np.ndarray
    K: sp.csr_matrix
    q: np.ndarray
    n_eq: int
    l: np.ndarray
    u: np.ndarray
    obj_const: float = 0.0
    name: str = "lp"
    col_names: list = field(default_factory=list)
    integer: np.ndarray | None = None  # bool mask, used by the MILP layer
    Q: sp.csr_matrix | None = None      # quadratic objective 0.5 x'Qx (symmetric PSD), used by the QP layer

    @property
    def shape(self):
        return self.K.shape

    def summary(self):
        m, n = self.K.shape
        return f"{self.name}: {n:,} variables, {m:,} constraints ({self.n_eq:,} equalities), {self.K.nnz:,} nonzeros"
