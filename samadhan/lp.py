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

    @property
    def shape(self):
        return self.K.shape

    def summary(self):
        m, n = self.K.shape
        return f"{self.name}: {n:,} variables, {m:,} constraints ({self.n_eq:,} equalities), {self.K.nnz:,} nonzeros"


def from_rows(c, rows_eq, rhs_eq, rows_ge, rhs_ge, l, u, n, name="lp"):
    """Build an LP from COO triplets (row, col, val) lists for = rows and >= rows."""
    def coo(rows, m):
        if m == 0:
            return sp.csr_matrix((0, n))
        r, cidx, v = rows
        return sp.csr_matrix((v, (r, cidx)), shape=(m, n))

    Keq = coo(rows_eq, len(rhs_eq))
    Kge = coo(rows_ge, len(rhs_ge))
    K = sp.vstack([Keq, Kge]).tocsr()
    K.sum_duplicates()
    q = np.concatenate([np.asarray(rhs_eq, float), np.asarray(rhs_ge, float)])
    return LP(np.asarray(c, float), K, q, len(rhs_eq), np.asarray(l, float), np.asarray(u, float), name=name)
