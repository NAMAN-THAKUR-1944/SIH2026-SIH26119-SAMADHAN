"""From-scratch dense two-phase primal simplex (tableau form, Bland's rule).

Exact vertex solutions for small LPs; used for branch-and-bound node relaxations. Works on the canonical
SAMADHAN LP form (= rows, >= rows, bounds l <= x <= u with finite l after shifting; free variables split).
"""
import numpy as np

from .lp import LP

EPS = 1e-9


def _pivot(T, r, c):
    T[r] /= T[r, c]
    col = T[:, c].copy()
    col[r] = 0.0
    T -= np.outer(col, T[r])


def _simplex(T, basis, n_cols, max_iter=50_000):
    """Minimise the objective in the last row of tableau T over the first n_cols columns (Bland's rule)."""
    m = T.shape[0] - 1
    for _ in range(max_iter):
        obj = T[-1, :n_cols]
        enter = next((j for j in range(n_cols) if obj[j] < -EPS), None)
        if enter is None:
            return "optimal"
        col = T[:m, enter]
        rhs = T[:m, -1]
        rows = np.where(col > EPS)[0]
        if rows.size == 0:
            return "unbounded"
        ratios = rhs[rows] / col[rows]
        best = ratios.min()
        cand = rows[np.abs(ratios - best) <= 1e-12 * max(1.0, abs(best))]
        leave = min(cand, key=lambda i: basis[i])  # Bland: smallest basic index
        _pivot(T, leave, enter)
        basis[leave] = enter
    return "iteration_limit"


def solve_lp_dense(lp: LP, l=None, u=None):
    """Return (status, x, obj). l/u override the LP bounds (used by branch-and-bound)."""
    l = lp.l if l is None else l
    u = lp.u if u is None else u
    K = lp.K.toarray()
    m, n = K.shape
    if np.any(l > u + EPS):
        return "infeasible", None, np.inf

    # variable substitution: finite lower bound -> x = l + x'; free -> x = x+ - x-
    cols, shift, sign, src = [], np.zeros(n), [], []
    for j in range(n):
        if np.isfinite(l[j]):
            shift[j] = l[j]; cols.append(K[:, j]); sign.append(1.0); src.append(j)
        elif np.isfinite(u[j]):  # x = u - x'
            shift[j] = u[j]; cols.append(-K[:, j]); sign.append(-1.0); src.append(j)
        else:
            cols.append(K[:, j]); sign.append(1.0); src.append(j)
            cols.append(-K[:, j]); sign.append(-1.0); src.append(j)
    A = np.column_stack(cols) if cols else np.zeros((m, 0))
    sign, src = np.array(sign), np.array(src)
    nv = A.shape[1]
    c = lp.c[src] * sign
    b = lp.q - K @ shift

    # upper bounds on shifted variables become rows  x' <= u - l
    ub_rows = []
    for k in range(nv):
        j = src[k]
        if np.isfinite(l[j]) and np.isfinite(u[j]) and sign[k] > 0:
            ub_rows.append((k, u[j] - l[j]))
    # build rows: equalities, >= rows (surplus), <= upper-bound rows (slack)
    n_eq, n_ge, n_ub = lp.n_eq, m - lp.n_eq, len(ub_rows)
    n_slack = n_ge + n_ub
    R = m + n_ub
    M = np.zeros((R, nv + n_slack))
    rhs = np.zeros(R)
    M[:m, :nv] = A
    rhs[:m] = b
    for i in range(n_ge):
        M[n_eq + i, nv + i] = -1.0
    for t, (k, ub) in enumerate(ub_rows):
        M[m + t, k] = 1.0
        M[m + t, nv + n_ge + t] = 1.0
        rhs[m + t] = ub
    neg = rhs < 0
    M[neg] *= -1; rhs[neg] *= -1

    # phase 1 with one artificial per row
    ncol = nv + n_slack
    T = np.zeros((R + 1, ncol + R + 1))
    T[:R, :ncol] = M
    T[:R, ncol:ncol + R] = np.eye(R)
    T[:R, -1] = rhs
    T[-1, :ncol] = -M.sum(axis=0)
    T[-1, -1] = -rhs.sum()
    basis = list(range(ncol, ncol + R))
    st = _simplex(T, basis, ncol + R)
    if st != "optimal" or -T[-1, -1] > 1e-7 * max(1.0, np.abs(rhs).max(initial=0)):
        return "infeasible", None, np.inf
    # drive remaining artificials out of the basis where possible
    for i, bv in enumerate(basis):
        if bv >= ncol:
            nz = np.where(np.abs(T[i, :ncol]) > EPS)[0]
            if nz.size:
                _pivot(T, i, nz[0]); basis[i] = nz[0]
    # phase 2: drop artificial columns, set real objective
    T2 = np.zeros((R + 1, ncol + 1))
    T2[:R, :ncol] = T[:R, :ncol]
    T2[:R, -1] = T[:R, -1]
    cfull = np.zeros(ncol); cfull[:nv] = c
    T2[-1, :ncol] = cfull
    for i, bv in enumerate(basis):
        if bv < ncol and abs(cfull[bv]) > 0:
            T2[-1] -= cfull[bv] * T2[i]
    st = _simplex(T2, basis, ncol)
    if st != "optimal":
        return st, None, -np.inf if st == "unbounded" else np.inf
    z = np.zeros(ncol)
    for i, bv in enumerate(basis):
        if bv < ncol:
            z[bv] = T2[i, -1]
    x = shift.copy()
    np.add.at(x, src, sign * z[:nv])
    return "optimal", x, float(lp.c @ x + lp.obj_const)
