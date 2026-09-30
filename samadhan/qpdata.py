"""Loader for the Maros-Meszaros convex QP test set in the qpbenchmark .mat format:
    minimise 0.5 x'Px + q'x + r   subject to   l <= A x <= u      (|value| >= 1e20 means infinite)
Single-entry rows with coefficient +-1 become variable bounds; other rows become = or >= rows.
"""
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from .lp import LP

BIG = 1e20


def read_maros(path, name=None):
    d = sio.loadmat(path)
    P = sp.csr_matrix(d["P"], dtype=float)
    q = np.asarray(d["q"], float).ravel()
    r = float(np.asarray(d["r"]).ravel()[0]) if "r" in d else 0.0
    A = sp.csr_matrix(d["A"], dtype=float)
    lo = np.asarray(d["l"], float).ravel()
    hi = np.asarray(d["u"], float).ravel()
    lo[lo <= -BIG] = -np.inf
    hi[hi >= BIG] = np.inf
    m, n = A.shape
    lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
    eq_rows, eq_rhs, ge_rows, ge_rhs = [], [], [], []
    A.sort_indices()
    for i in range(m):
        s, e = A.indptr[i], A.indptr[i + 1]
        if e - s == 1 and A.data[s] != 0:          # variable bound row
            j, a = A.indices[s], A.data[s]
            blo, bhi = (lo[i] / a, hi[i] / a) if a > 0 else (hi[i] / a, lo[i] / a)
            lb[j], ub[j] = max(lb[j], blo), min(ub[j], bhi)
            continue
        if e == s:
            continue
        if np.isfinite(lo[i]) and lo[i] == hi[i]:
            eq_rows.append(A[i]); eq_rhs.append(lo[i]); continue
        if np.isfinite(lo[i]):
            ge_rows.append(A[i]); ge_rhs.append(lo[i])
        if np.isfinite(hi[i]):
            ge_rows.append(-A[i]); ge_rhs.append(-hi[i])
    rows = eq_rows + ge_rows
    K = sp.vstack(rows).tocsr() if rows else sp.csr_matrix((0, n))
    Q = ((P + P.T) * 0.5).tocsr() if (P != P.T).nnz else P.tocsr()
    return LP(q, K, np.array(eq_rhs + ge_rhs, float), len(eq_rows), lb, ub, obj_const=r,
              name=name or str(path), Q=Q)
