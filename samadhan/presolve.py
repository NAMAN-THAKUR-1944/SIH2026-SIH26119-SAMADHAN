"""Presolve: shrink an LP or MILP before it is solved, and map the solution back (postsolve).

Reductions, repeated until a pass changes nothing:
  * fixed columns (l = u) are substituted into the rows and the objective;
  * empty rows are dropped (or prove infeasibility);
  * singleton rows  a x_j (= or >=) q  become bounds on x_j (rounded for integer columns);
  * rows whose activity bounds already satisfy them are dropped as redundant;
  * forcing rows (activity bound equal to the right-hand side) fix all their columns at a bound;
  * integer columns get tighter bounds from the rows they appear in (domain propagation);
  * empty columns are fixed at the bound their cost prefers (or prove unboundedness).
Only the primal solution is mapped back. Models with a quadratic objective are passed through unchanged.
"""
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp

from .lp import LP

FEAS_TOL = 1e-9    # feasibility tolerance when rows and bounds are compared
INT_TOL = 1e-6     # integer rounding tolerance for tightened bounds (relative to the bound)


def _floor(v):
    return np.floor(v + INT_TOL * np.maximum(1.0, np.abs(np.where(np.isfinite(v), v, 0.0))))


def _ceil(v):
    return np.ceil(v - INT_TOL * np.maximum(1.0, np.abs(np.where(np.isfinite(v), v, 0.0))))


@dataclass
class Presolved:
    status: str                       # "reduced", "infeasible" or "unbounded"
    lp: LP | None                     # the reduced model
    cols: np.ndarray                  # original indices of the kept columns
    x_fixed: np.ndarray               # values of the removed columns (kept ones are overwritten by postsolve)
    stats: dict = field(default_factory=dict)
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # original indices of the kept rows

    def postsolve(self, x_reduced):
        x = self.x_fixed.copy()
        if len(self.cols):
            x[self.cols] = x_reduced
        return x


def _activity(A, l, u):
    """Row activity bounds of A x over l <= x <= u, as (finite part, number of infinite terms) for min and max."""
    Ap, An = A.maximum(0), A.minimum(0)
    lf, uf = np.where(np.isfinite(l), l, 0.0), np.where(np.isfinite(u), u, 0.0)
    li, ui = (~np.isfinite(l)).astype(float), (~np.isfinite(u)).astype(float)
    mn, mn_inf = Ap @ lf + An @ uf, (Ap != 0) @ li + (An != 0) @ ui
    mx, mx_inf = Ap @ uf + An @ lf, (Ap != 0) @ ui + (An != 0) @ li
    return mn, mn_inf, mx, mx_inf


def presolve(lp: LP, max_passes=50) -> Presolved:
    m, n = lp.K.shape
    if lp.Q is not None and lp.Q.nnz:
        return Presolved("reduced", lp, np.arange(n), np.zeros(n), dict(passes=0), np.arange(m))
    A = sp.csr_matrix(lp.K, dtype=float, copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    lo = lp.q.astype(float).copy()
    hi = np.full(m, np.inf)
    hi[:lp.n_eq] = lp.q[:lp.n_eq]
    l, u, c = lp.l.astype(float).copy(), lp.u.astype(float).copy(), lp.c.astype(float).copy()
    isint = np.zeros(n, bool) if lp.integer is None else np.asarray(lp.integer, bool).copy()
    l[isint] = _ceil(l[isint])
    u[isint] = _floor(u[isint])
    row_on, col_on = np.ones(m, bool), np.ones(n, bool)
    x_fixed = np.zeros(n)
    const = lp.obj_const
    st = dict(passes=0, fixed_cols=0, empty_rows=0, singleton_rows=0, redundant_rows=0, forcing_rows=0,
              tightened=0, empty_cols=0)
    fail = lambda s: Presolved(s, None, np.arange(0), x_fixed, st)

    def fix(cols, vals):
        nonlocal const
        x_fixed[cols] = vals
        col_on[cols] = False
        const += float(c[cols] @ vals)
        shift = A[:, cols] @ vals
        lo[:] -= shift
        hi[:] -= shift

    for _ in range(max_passes):
        st["passes"] += 1
        changed = False
        if np.any(l > u + FEAS_TOL):
            return fail("infeasible")
        # fixed columns
        fx = np.flatnonzero(col_on & (u - l <= FEAS_TOL))
        if len(fx):
            fix(fx, l[fx]); st["fixed_cols"] += len(fx); changed = True
        S = A[row_on][:, col_on]                       # active submatrix
        rows, cols = np.flatnonzero(row_on), np.flatnonzero(col_on)
        cnt = np.diff(S.indptr)
        # empty rows
        e = cnt == 0
        if e.any():
            ri = rows[e]
            if np.any(lo[ri] > FEAS_TOL) or np.any(hi[ri] < -FEAS_TOL):
                return fail("infeasible")
            row_on[ri] = False; st["empty_rows"] += len(ri); changed = True
        # singleton rows -> column bounds
        one = cnt == 1
        if one.any():
            S1 = S[np.flatnonzero(one)].tocoo()
            ri, j, a = rows[np.flatnonzero(one)][S1.row], cols[S1.col], S1.data
            b1, b2 = lo[ri] / a, hi[ri] / a
            nl, nu = np.where(a > 0, b1, b2), np.where(a > 0, b2, b1)
            nl = np.where(isint[j], _ceil(nl), nl)
            nu = np.where(isint[j], _floor(nu), nu)
            np.maximum.at(l, j, nl)
            np.minimum.at(u, j, nu)
            row_on[ri] = False; st["singleton_rows"] += len(ri); changed = True
            continue                                   # bounds changed: rebuild the active submatrix
        # activity bounds: redundant, infeasible and forcing rows
        Sl, Su = l[cols], u[cols]
        mn, mn_inf, mx, mx_inf = _activity(S, Sl, Su)
        rmn = np.where(mn_inf > 0, -np.inf, mn)
        rmx = np.where(mx_inf > 0, np.inf, mx)
        rlo, rhi = lo[rows], hi[rows]
        scale = 1.0 + np.abs(np.where(np.isfinite(rlo), rlo, 0.0)) + np.abs(np.where(np.isfinite(rhi), rhi, 0.0))
        tol = FEAS_TOL * scale
        if np.any(rmn > rhi + 1e3 * tol) or np.any(rmx < rlo - 1e3 * tol):
            return fail("infeasible")
        red = (rmn >= rlo - tol) & (rmx <= rhi + tol)
        if red.any():
            row_on[rows[red]] = False; st["redundant_rows"] += int(red.sum()); changed = True
        # forcing: the row can only be met with every column at the bound that maximises (or minimises) it.
        # The row stays active: once its columns are fixed it is checked again (redundant, or infeasible if two
        # forcing rows disagree on a column).
        f_up = ~red & np.isfinite(rmx) & (np.abs(rmx - rlo) <= tol)
        f_dn = ~red & ~f_up & np.isfinite(rmn) & (np.abs(rmn - rhi) <= tol)
        if f_up.any() or f_dn.any():
            for mask, at_max in ((f_up, True), (f_dn, False)):
                if not mask.any():
                    continue
                F = S[np.flatnonzero(mask)].tocoo()
                j, a = cols[F.col], F.data
                val = np.where((a > 0) == at_max, u[j], l[j])
                jj, first = np.unique(j, return_index=True)
                l[jj] = u[jj] = val[first]             # becomes a fixed column in the next pass
            st["forcing_rows"] += int(f_up.sum() + f_dn.sum()); changed = True
            continue
        # domain propagation on integer columns
        if isint[cols].any():
            T = S.tocoo()
            keep = isint[cols[T.col]]
            r_, j_, a = T.row[keep], cols[T.col[keep]], T.data[keep]
            lj, uj = l[j_], u[j_]
            # contribution of the entry itself to the min / max activity of its row
            cmin = np.where(a > 0, a * lj, a * uj)
            cmax = np.where(a > 0, a * uj, a * lj)
            inf_min, inf_max = ~np.isfinite(cmin), ~np.isfinite(cmax)
            rest_min = np.where(mn_inf[r_] - inf_min == 0, mn[r_] - np.where(inf_min, 0.0, cmin), -np.inf)
            rest_max = np.where(mx_inf[r_] - inf_max == 0, mx[r_] - np.where(inf_max, 0.0, cmax), np.inf)
            with np.errstate(invalid="ignore", divide="ignore"):
                from_hi = (rhi[r_] - rest_min) / a       # a x_j <= hi - rest_min
                from_lo = (rlo[r_] - rest_max) / a       # a x_j >= lo - rest_max
            new_u = np.where(a > 0, from_hi, from_lo)
            new_l = np.where(a > 0, from_lo, from_hi)
            new_u = np.where(np.isnan(new_u), np.inf, _floor(new_u))
            new_l = np.where(np.isnan(new_l), -np.inf, _ceil(new_l))
            nu, nl = u.copy(), l.copy()
            np.minimum.at(nu, j_, new_u)
            np.maximum.at(nl, j_, new_l)
            k = int(np.sum(nu < u) + np.sum(nl > l))
            if k:
                l, u = nl, nu
                st["tightened"] += k; changed = True
                continue
        # empty columns: fix at the bound the cost prefers
        ccnt = np.diff(S.tocsc().indptr)
        ec = cols[ccnt == 0]
        if len(ec):
            cj = c[ec]
            val = np.where(cj > 0, l[ec], np.where(cj < 0, u[ec], np.clip(0.0, l[ec], u[ec])))
            if not np.all(np.isfinite(val)):
                return fail("unbounded")
            fix(ec, val); st["empty_cols"] += len(ec); changed = True
        if not changed:
            break

    rows, cols = np.flatnonzero(row_on), np.flatnonzero(col_on)
    S = A[rows][:, cols].tocsr()
    eq = rows < lp.n_eq
    order = np.concatenate([np.flatnonzero(eq), np.flatnonzero(~eq)])
    S, q = S[order], lo[rows][order]
    names = [lp.col_names[j] for j in cols] if lp.col_names else []
    red = replace(lp, c=c[cols], K=S, q=q, n_eq=int(eq.sum()), l=l[cols], u=u[cols], obj_const=const,
                  col_names=names, integer=isint[cols] if lp.integer is not None else None,
                  name=f"{lp.name} (presolved)")
    st.update(rows=(m, len(rows)), cols=(n, len(cols)))
    return Presolved("reduced", red, cols, x_fixed, st, rows[order])
