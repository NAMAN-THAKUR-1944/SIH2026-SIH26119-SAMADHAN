"""Presolve: shrink an LP or MILP before it is solved, and map the solution back (postsolve).

Reductions, repeated until a pass changes nothing:
  * fixed columns (l = u) are substituted into the rows and the objective;
  * empty rows are dropped (or prove infeasibility);
  * singleton rows  a x_j (= or >=) q  become bounds on x_j (rounded for integer columns);
  * rows whose activity bounds already satisfy them are dropped as redundant;
  * forcing rows (activity bound equal to the right-hand side) fix all their columns at a bound;
  * parallel rows (one a multiple of the other) are merged into one, where the result is an equality or a >= row;
  * integer columns get tighter bounds from the rows they appear in (domain propagation);
  * free column singletons: a continuous column in a single equality row that already implies its bounds is
    substituted out with the row (its value is recovered from the row in postsolve);
  * doubleton equations  a_j x_j + a_k x_k = b  with x_k continuous: x_k = (b - a_j x_j) / a_k is substituted into
    the other rows and the objective, and the bounds of x_k become bounds of x_j;
  * dual fixing: a column that neither its cost nor any row pushes upwards is fixed at its lower bound (and the
    mirror case at its upper bound);
  * empty columns are fixed at the bound their cost prefers (or prove unboundedness);
  * probing (MILP, when the reductions above find nothing more): every binary column is tentatively fixed at 0 and
    at 1 and the fixing propagated through the rows (C++ core); an impossible value fixes the column at the other,
    bounds implied by both values are kept, and a binary that follows another in both cases is substituted by it
    (x_k = x_j or x_k = 1 - x_j, recovered in postsolve).
Only the primal solution is mapped back. Models with a quadratic objective are passed through unchanged.
"""
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp

from .lp import LP

FEAS_TOL = 1e-9    # feasibility tolerance when rows and bounds are compared
PROBE_WORK = 6e7   # probing budget: row-entry operations of the propagation (well under a second)
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
    # substituted columns, in order: (k, b, cols, coefs, a)  means  x_k = (b - coefs . x[cols]) / a
    stack: list = field(default_factory=list)

    def postsolve(self, x_reduced):
        x = self.x_fixed.copy()
        if len(self.cols):
            x[self.cols] = x_reduced
        for k, b, cols, coefs, a in reversed(self.stack):
            x[k] = (b - coefs @ x[cols]) / a
        return x


def _activity(A, l, u):
    """Row activity bounds of A x over l <= x <= u, as (finite part, number of infinite terms) for min and max."""
    Ap, An = A.maximum(0), A.minimum(0)
    lf, uf = np.where(np.isfinite(l), l, 0.0), np.where(np.isfinite(u), u, 0.0)
    li, ui = (~np.isfinite(l)).astype(float), (~np.isfinite(u)).astype(float)
    mn, mn_inf = Ap @ lf + An @ uf, (Ap != 0) @ li + (An != 0) @ ui
    mx, mx_inf = Ap @ uf + An @ lf, (Ap != 0) @ ui + (An != 0) @ li
    return mn, mn_inf, mx, mx_inf


def _parallel_rows(S):
    """Pairs (i, k, lam) of rows of the CSR matrix S (two or more entries) with row k = lam * row i. Rows are grouped by
    a hash of their pattern and of their values divided by the first one; every pair is then checked exactly."""
    S = sp.csr_matrix(S)
    S.sort_indices()
    m, n = S.shape
    cnt = np.diff(S.indptr)
    cand = np.flatnonzero(cnt > 1)
    if len(cand) < 2:
        return []
    rowid = np.repeat(np.arange(m), cnt)
    first = np.zeros(m)
    first[cnt > 0] = S.data[S.indptr[:-1][cnt > 0]]
    scaled = S.data / first[rowid]
    rng = np.random.default_rng(12345)
    w1, w2 = rng.random(n), rng.random(n)
    hp = np.round(np.bincount(rowid, weights=w1[S.indices], minlength=m), 9)
    hv = np.round(np.bincount(rowid, weights=scaled * w2[S.indices], minlength=m), 7)
    order = cand[np.lexsort((hv[cand], hp[cand], cnt[cand]))]
    pairs = []
    start = 0
    for t in range(1, len(order) + 1):
        if t < len(order) and cnt[order[t]] == cnt[order[start]] and hp[order[t]] == hp[order[start]]                 and hv[order[t]] == hv[order[start]]:
            continue
        if t - start > 1:
            head = order[start]
            hi_, hv_ = S.indices[S.indptr[head]:S.indptr[head + 1]], scaled[S.indptr[head]:S.indptr[head + 1]]
            for k in order[start + 1:t]:
                ki, kv = S.indices[S.indptr[k]:S.indptr[k + 1]], scaled[S.indptr[k]:S.indptr[k + 1]]
                if np.array_equal(ki, hi_) and np.allclose(kv, hv_, rtol=1e-12, atol=0.0):
                    pairs.append((int(head), int(k), float(first[k] / first[head])))
        start = t
    return pairs


def presolve(lp: LP, max_passes=50, probe_work=PROBE_WORK, probe_seconds=60.0) -> Presolved:
    """probe_work: probing budget per probing call, in row entries visited by the propagation (0: no probing);
    probe_seconds is only a safety net (the work budget keeps the result independent of the machine)."""
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
              tightened=0, free_singletons=0, dual_fixed=0, empty_cols=0, probe_fixed=0, probe_tightened=0,
              probe_equiv=0, parallel_rows=0, doubletons=0)
    probes_left = parallel_left = 2                 # expensive steps: only when a pass finds nothing else
    stack = []
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
        # free column singletons: a continuous column that appears only in one equality row, whose other terms
        # already imply its bounds, is substituted out together with the row:  x_k = (b - sum_j a_j x_j) / a_k
        Scsc = S.tocsc()
        ccnt1 = np.diff(Scsc.indptr)
        cand = np.flatnonzero((ccnt1 == 1) & ~isint[cols])
        done_rows = set()
        subs = 0
        for kl in cand:
            p0 = Scsc.indptr[kl]
            il, a = int(Scsc.indices[p0]), float(Scsc.data[p0])
            i = int(rows[il])
            if not np.isfinite(hi[i]) or not row_on[i] or il in done_rows or abs(a) < 1e-9:
                continue
            k = int(cols[kl])
            lk, uk = l[k], u[k]
            cmin = a * lk if a > 0 else a * uk
            cmax = a * uk if a > 0 else a * lk
            rmin_ok = mn_inf[il] - (0 if np.isfinite(cmin) else 1) == 0
            rmax_ok = mx_inf[il] - (0 if np.isfinite(cmax) else 1) == 0
            rest_min = mn[il] - (cmin if np.isfinite(cmin) else 0.0) if rmin_ok else -np.inf
            rest_max = mx[il] - (cmax if np.isfinite(cmax) else 0.0) if rmax_ok else np.inf
            b = lo[i]
            lo_imp, hi_imp = (b - rest_max) / a, (b - rest_min) / a
            if a < 0:
                lo_imp, hi_imp = hi_imp, lo_imp
            tol = 1e-9 * (1.0 + abs(b))
            if not (lo_imp >= lk - tol and hi_imp <= uk + tol):
                continue                                         # the column's own bounds still matter
            r0, r1 = S.indptr[il], S.indptr[il + 1]
            others = [(int(cols[jl]), float(v)) for jl, v in zip(S.indices[r0:r1], S.data[r0:r1]) if jl != kl]
            ocols = np.array([j for j, _ in others], int)
            ocoef = np.array([v for _, v in others], float)
            if c[k] != 0.0:
                c[ocols] -= c[k] * ocoef / a
                const += c[k] * b / a
            stack.append((k, b, ocols, ocoef, a))
            col_on[k] = False
            row_on[i] = False
            done_rows.add(il)
            subs += 1
        if subs:
            st["free_singletons"] += subs; changed = True
            continue
        # doubleton equations: a_j x_j + a_k x_k = b with x_k continuous -> x_k = (b - a_j x_j) / a_k everywhere
        dbl = np.flatnonzero((cnt == 2) & np.isfinite(hi[rows]))
        if len(dbl):
            used = np.zeros(n, bool)
            ks, js, bs, ajs, aks = [], [], [], [], []
            for il in dbl:
                i = rows[il]
                if not row_on[i]:
                    continue
                p0 = S.indptr[il]
                (j1, j2), (v1, v2) = cols[S.indices[p0:p0 + 2]], S.data[p0:p0 + 2]
                if used[j1] or used[j2] or v1 == 0.0 or v2 == 0.0:
                    continue
                # x_k: a continuous column, the one with the larger coefficient (smaller multiplier)
                opts = [(k, ak, j, aj) for k, ak, j, aj in ((j2, v2, j1, v1), (j1, v1, j2, v2)) if not isint[k]]
                if not opts:
                    continue
                k, ak, j, aj = max(opts, key=lambda o: abs(o[1]))
                if abs(aj / ak) > 1e3:
                    continue
                b = lo[i]
                # x_k in [l_k, u_k]  <=>  a_j x_j in [b - a_k u_k, b - a_k l_k]  (ends swapped if a_k < 0)
                with np.errstate(over="ignore", invalid="ignore"):
                    e1, e2 = b - ak * u[k], b - ak * l[k]
                    ylo, yhi = (e1, e2) if ak > 0 else (e2, e1)
                    nl, nu = (ylo / aj, yhi / aj) if aj > 0 else (yhi / aj, ylo / aj)
                nl = float(np.nan_to_num(nl, nan=-np.inf, posinf=np.inf, neginf=-np.inf))
                nu = float(np.nan_to_num(nu, nan=np.inf, posinf=np.inf, neginf=-np.inf))
                if isint[j]:
                    nl, nu = _ceil(np.array([nl]))[0], _floor(np.array([nu]))[0]
                nl, nu = max(l[j], nl), min(u[j], nu)
                if np.isfinite(nl) and np.isfinite(nu) and nl - nu > FEAS_TOL * (1.0 + abs(nl)):
                    return fail("infeasible")
                l[j], u[j] = nl, max(nl, nu)
                used[j] = used[k] = True
                ks.append(k); js.append(j); bs.append(b); ajs.append(aj); aks.append(ak)
                row_on[i] = False
            if ks:
                ks, js = np.array(ks), np.array(js)
                bs, ajs, aks = np.array(bs), np.array(ajs), np.array(aks)
                keep = np.ones(n, bool)
                keep[ks] = False
                diag = np.flatnonzero(keep)
                T = sp.csc_matrix((np.concatenate([np.ones(len(diag)), -ajs / aks]),
                                   (np.concatenate([diag, ks]), np.concatenate([diag, js]))), shape=(n, n))
                b0 = np.zeros(n)
                b0[ks] = bs / aks
                shift = A @ b0
                lo -= shift
                hi -= shift
                const += float(c @ b0)
                c = T.T @ c
                A = sp.csr_matrix(A @ T)
                A.eliminate_zeros()
                for k, j, b, aj, ak in zip(ks, js, bs, ajs, aks):
                    col_on[k] = False
                    stack.append((int(k), float(b), np.array([j]), np.array([aj]), float(ak)))
                st["doubletons"] += len(ks); changed = True
                continue
        # dual fixing: if the cost does not reward increasing a column and no row needs it larger (no down-lock),
        # some optimal solution has it at its lower bound (and symmetrically at the upper bound)
        T = S.tocoo()
        eqr = np.isfinite(hi[rows[T.row]])                 # equality rows (the only ones with a finite upper side)
        down = np.bincount(T.col, weights=(eqr | (T.data > 0)).astype(float), minlength=len(cols))
        up = np.bincount(T.col, weights=(eqr | (T.data < 0)).astype(float), minlength=len(cols))
        cc, lc, uc = c[cols], l[cols], u[cols]
        at_l = (cc >= 0) & (down == 0) & np.isfinite(lc)
        at_u = ~at_l & (cc <= 0) & (up == 0) & np.isfinite(uc)
        if at_l.any() or at_u.any():
            jl, ju = cols[at_l], cols[at_u]
            fix(np.concatenate([jl, ju]), np.concatenate([l[jl], u[ju]]))
            st["dual_fixed"] += len(jl) + len(ju); changed = True
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
        # parallel rows: a_k = lam a_i. Row k becomes a bound on a_i x and is merged into row i when the result is
        # an equality or a >= row (a true range lo < a_i x < hi cannot be written in this LP form: both stay)
        merged = 0
        for il, kl, lam in (_parallel_rows(S) if not changed and parallel_left else []):
            i, k = rows[il], rows[kl]
            if not (row_on[i] and row_on[k]):
                continue
            k_lo, k_hi = (lo[k] / lam, hi[k] / lam) if lam > 0 else (hi[k] / lam, lo[k] / lam)
            new_lo, new_hi = max(lo[i], k_lo), min(hi[i], k_hi)
            tol = FEAS_TOL * (1.0 + abs(new_lo) if np.isfinite(new_lo) else 1.0)
            if new_lo > new_hi + 1e3 * tol:
                return fail("infeasible")
            if np.isfinite(new_hi):
                if new_hi - new_lo > tol:
                    continue
                v = lo[i] if np.isfinite(hi[i]) else (lo[k] / lam if np.isfinite(hi[k]) else new_lo)
                lo[i] = hi[i] = v
            else:
                lo[i] = new_lo
            row_on[k] = False
            merged += 1
        if not changed and parallel_left:
            parallel_left -= 1
        if merged:
            st["parallel_rows"] += merged; changed = True
            continue
        if not changed and probes_left and probe_work > 0 and isint[cols].any():
            probes_left -= 1
            from .core import probe
            infeasible, nl, nu, fx, tg, eqv, _ = probe(S, lo[rows], hi[rows], isint[cols], l[cols], u[cols],
                                                       probe_work, probe_seconds)
            if infeasible:
                return fail("infeasible")
            if fx or tg:
                l[cols], u[cols] = nl, nu
                st["probe_fixed"] += fx; st["probe_tightened"] += tg; changed = True
                parallel_left = max(parallel_left, 1)     # fixed columns can leave rows parallel
            if len(eqv):
                # x_k = x_r (sign 1) or 1 - x_r (sign -1), r found by following the chain k -> j -> ...
                link = {int(cols[k]): (int(cols[j]), int(sg)) for k, j, sg in eqv}
                ks, rs, signs = [], [], []
                for k in link:
                    r, sg = k, 1
                    while r in link:
                        r, t = link[r]
                        sg *= t
                    ks.append(k); rs.append(r); signs.append(sg)
                ks, rs, signs = np.array(ks), np.array(rs), np.array(signs, float)
                keep = np.ones(n, bool)
                keep[ks] = False
                diag = np.flatnonzero(keep)
                T = sp.csc_matrix((np.concatenate([np.ones(len(diag)), signs]),
                                   (np.concatenate([diag, ks]), np.concatenate([diag, rs]))), shape=(n, n))
                b0 = np.zeros(n)
                b0[ks] = (signs < 0).astype(float)
                shift = A @ b0
                lo -= shift
                hi -= shift
                const += float(c @ b0)
                c = T.T @ c
                A = sp.csr_matrix(A @ T)
                A.eliminate_zeros()
                for k, r, sg in zip(ks, rs, signs):         # the bounds of x_k now restrict x_r
                    if sg > 0:
                        l[r], u[r] = max(l[r], l[k]), min(u[r], u[k])
                    else:
                        l[r], u[r] = max(l[r], 1.0 - u[k]), min(u[r], 1.0 - l[k])
                    col_on[k] = False
                    stack.append((int(k), float(sg < 0), np.array([r]), np.array([1.0 if sg < 0 else -1.0]), 1.0))
                st["probe_equiv"] += len(ks); changed = True
        if not changed:
            break

    rows, cols = np.flatnonzero(row_on), np.flatnonzero(col_on)
    S = A[rows][:, cols].tocsr()
    eq = np.isfinite(hi[rows])
    order = np.concatenate([np.flatnonzero(eq), np.flatnonzero(~eq)])
    S, q = S[order], lo[rows][order]
    names = [lp.col_names[j] for j in cols] if lp.col_names else []
    red = replace(lp, c=c[cols], K=S, q=q, n_eq=int(eq.sum()), l=l[cols], u=u[cols], obj_const=const,
                  col_names=names, integer=isint[cols] if lp.integer is not None else None,
                  name=f"{lp.name} (presolved)")
    st.update(rows=(m, len(rows)), cols=(n, len(cols)))
    return Presolved("reduced", red, cols, x_fixed, st, rows[order], stack)
