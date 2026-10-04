"""Interior-point method for LP: Mehrotra predictor-corrector, written from scratch.

The LP is presolved and brought to the standard form   min c'x  s.t.  A x = b,  0 <= x <= u  (u may be +inf),
scaled by Ruiz equilibration, and solved by a primal-dual path-following method. Each iteration factorises the
normal equations  A Theta A' dy = r  once with the C++ sparse LU (samadhan.core.SparseLU) and solves twice
(predictor and corrector). With crossover=True the interior solution is turned into an exact optimal vertex by
the C++ simplex.
"""
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .lp import LP


@dataclass
class IPMResult:
    status: str                 # "optimal", "iteration_limit", "time_limit", "infeasible", "numerical_error"
    x: np.ndarray | None
    y: np.ndarray | None        # duals of the rows K x (=, >=) q of the original LP
    primal_obj: float
    dual_obj: float
    rel_gap: float
    rel_primal_res: float
    rel_dual_res: float
    iterations: int
    time: float
    vertex: bool = False        # x is an exact optimal vertex (after crossover)
    crossover_time: float = 0.0


def _standard_form(lp):
    """min c'x + const  s.t.  A x = b,  x_j >= 0 (except free columns),  x <= u.  Returns
    (A, b, c, u, free, const, recover) with recover(x_std) giving the original columns. Columns with a finite lower
    bound are shifted (x = l + t), columns with only an upper bound are mirrored (x = u - t), free columns are kept
    as they are (flagged in `free`); >= rows get surplus columns."""
    K = sp.csc_matrix(lp.K, dtype=float)
    m, n = K.shape
    l, u = lp.l.astype(float), lp.u.astype(float)
    fl, fu = np.isfinite(l), np.isfinite(u)
    upper_only, free = ~fl & fu, ~fl & ~fu
    sign = np.where(upper_only, -1.0, 1.0)
    shift = np.where(fl, l, np.where(upper_only, u, 0.0))
    nge = m - lp.n_eq
    surplus = sp.vstack([sp.csc_matrix((lp.n_eq, nge)), -sp.identity(nge, format="csc")])
    A = sp.hstack([K @ sp.diags(sign), surplus]).tocsr()
    c = np.concatenate([lp.c * sign, np.zeros(nge)])
    ub = np.concatenate([np.where(fl, u - l, np.inf), np.full(nge, np.inf)])
    fr = np.concatenate([free, np.zeros(nge, bool)])
    b = lp.q - K @ shift
    const = lp.obj_const + float(lp.c @ shift)

    def recover(xs):
        return shift + sign * xs[:n]
    return A, b, c, ub, fr, const, recover


def _ruiz(A, passes=10):
    """Row and column factors r, s with diag(r) A diag(s) having all row and column max-norms close to 1."""
    m, n = A.shape
    r, s = np.ones(m), np.ones(n)
    B = A.tocsr().copy()
    for _ in range(passes):
        rn = np.sqrt(abs(B).max(axis=1).toarray().ravel())
        cn = np.sqrt(abs(B).max(axis=0).toarray().ravel())
        rn[rn == 0] = 1.0
        cn[cn == 0] = 1.0
        B = sp.diags(1 / rn) @ B @ sp.diags(1 / cn)
        r, s = r / rn, s / cn
    return r, s


def _max_step(v, dv):
    neg = dv < 0
    return min(1.0, float(np.min(-v[neg] / dv[neg]))) if neg.any() else 1.0


class _Normal:
    """Factorised normal equations M = A Theta A'. The factorisation includes a small diagonal regularisation
    (so that it exists for rank-deficient A); iterative refinement against the unregularised M then recovers the
    accurate solution."""

    def __init__(self, A, AT, theta):
        from .core import SparseLU
        m = A.shape[0]
        self.M = (A @ sp.diags(theta) @ AT).tocsc()
        d = self.M.diagonal()
        self.s = 1.0 / np.sqrt(np.where(d > 0, d, 1.0))         # symmetric scaling to a unit diagonal
        Ms = (sp.diags(self.s) @ self.M @ sp.diags(self.s)).tocsc()
        reg = 1e-12
        self.lu = None
        for _ in range(6):
            lu = SparseLU(Ms + reg * sp.identity(m, format="csc"), abs_tol=1e-30, symmetric=True)
            if lu.ok:
                self.lu = lu
                break
            reg *= 100.0

    def _apply(self, r):
        return self.s * self.lu.solve(self.s * r)

    def solve(self, r, steps=3):
        x = self._apply(r)
        nr = np.linalg.norm(r)
        for _ in range(steps):
            e = r - self.M @ x
            if np.linalg.norm(e) <= 1e-14 * (1.0 + nr):
                break
            x += self._apply(e)
        return x


def _direction(state, rxz, rwv):
    """Newton direction for the complementarity right-hand sides rxz (x z) and rwv (w v), from the factorised
    normal equations:  A Theta A' dy = rb + A Theta rhat,  dx = Theta (A' dy - rhat). Free columns have no z."""
    A, AT, ne, N, B, x, z, w, v, theta, rb, rc, ru = state
    rhat = rc.copy()
    rhat[N] -= rxz[N] / x[N]
    rhat[B] += (rwv - v * ru) / w
    dy = ne.solve(rb + A @ (theta * rhat))
    dx = theta * (AT @ dy - rhat)
    dz = np.zeros_like(z)
    dz[N] = (rxz[N] - z[N] * dx[N]) / x[N]
    dw = ru - dx[B]
    dv = (rwv - v * dw) / w
    return dx, dy, dz, dw, dv


def _mehrotra(A, b, c, u, free, tol, max_iter, time_limit, verbose):
    t0 = time.perf_counter()
    m, n = A.shape
    AT = A.T.tocsr()
    N, B = ~free, np.isfinite(u)
    ub = u[B]
    npair = int(N.sum() + B.sum())                 # complementarity pairs
    nb_, nc_ = np.linalg.norm(b), np.linalg.norm(c)
    rho = 1e-8                                     # primal regularisation of free columns

    # starting point (Mehrotra): least-squares x and y, shifted into the interior
    ne = _Normal(A, AT, np.ones(n))
    if ne.lu is None:
        return "numerical_error", None, None, 0, {}
    x = AT @ ne.solve(b)
    y = ne.solve(A @ c)
    zt = c - AT @ y
    xN = x[N] + max(-1.5 * x[N].min(initial=0.0), 0.0)
    zN = np.maximum(zt[N], 0.0) + max(-1.5 * zt[N].min(initial=0.0), 0.0)
    xz = float(xN @ zN)
    x[N] = np.maximum(xN + 0.5 * xz / max(zN.sum(), 1e-12), 1e-2)
    z = np.zeros(n)
    z[N] = np.maximum(zN + 0.5 * xz / max(xN.sum(), 1e-12), 1e-2)
    x[B] = np.clip(x[B], 0.1 * ub, 0.9 * ub)
    w = ub - x[B]
    v = np.maximum(-zt[B], 0.0) + 1e-2
    z[B] = np.maximum(z[B], 1e-2)

    status, it = "iteration_limit", 0
    best, best_it, best_pt = np.inf, 0, None
    for it in range(1, max_iter + 1):
        rb = b - A @ x
        rc = c - AT @ y - z
        rc[B] += v
        ru = ub - x[B] - w
        mu = (x[N] @ z[N] + w @ v) / max(npair, 1)
        pobj, dobj = c @ x, b @ y - ub @ v
        rel_p = np.linalg.norm(rb) / (1 + nb_)
        rel_d = np.linalg.norm(rc) / (1 + nc_)
        gap = abs(pobj - dobj) / (1 + abs(pobj))
        merit = max(rel_p, rel_d, gap)
        if verbose:
            print(f"  ipm {it:3d}  pobj {pobj:+.10e}  dobj {dobj:+.10e}  pres {rel_p:.1e}  dres {rel_d:.1e}  "
                  f"gap {gap:.1e}  mu {mu:.1e}  {time.perf_counter() - t0:.1f}s")
        if merit < best:
            best, best_it = merit, it
            best_pt = (x.copy(), y.copy(), dict(pobj=float(pobj), dobj=float(dobj), rel_p=float(rel_p),
                                                rel_d=float(rel_d), gap=float(gap)))
        if merit < tol:
            status = "optimal"
            break
        if time.perf_counter() - t0 > time_limit:
            status = "time_limit"
            break
        if not np.isfinite(mu) or np.abs(x).max() > 1e30:
            status = "infeasible" if best > 1e-3 else "stalled"   # iterates diverge
            break
        if it - best_it >= 15:
            status = "stalled"                                     # no progress: the best point goes on
            break
        theta_inv = np.full(n, rho)
        theta_inv[N] = z[N] / x[N]
        theta_inv[B] += v / w
        theta = 1.0 / theta_inv
        ne = _Normal(A, AT, theta)
        if ne.lu is None:
            status = "numerical_error"
            break
        st = (A, AT, ne, N, B, x, z, w, v, theta, rb, rc, ru)
        # predictor (affine scaling) step
        dx, dy, dz, dw, dv = _direction(st, -x * z, -w * v)
        ap = min(_max_step(x[N], dx[N]), _max_step(w, dw))
        ad = min(_max_step(z[N], dz[N]), _max_step(v, dv))
        mu_aff = ((x[N] + ap * dx[N]) @ (z[N] + ad * dz[N]) + (w + ap * dw) @ (v + ad * dv)) / max(npair, 1)
        sigma = (mu_aff / mu) ** 3
        # corrector step with centring
        rxz = sigma * mu - x * z - dx * dz
        rxz[free] = 0.0
        dx, dy, dz, dw, dv = _direction(st, rxz, sigma * mu - w * v - dw * dv)
        ap = min(1.0, 0.995 * min(_max_step(x[N], dx[N]), _max_step(w, dw)))
        ad = min(1.0, 0.995 * min(_max_step(z[N], dz[N]), _max_step(v, dv)))
        x += ap * dx
        w += ap * dw
        y += ad * dy
        z += ad * dz
        v += ad * dv
    if status != "optimal" and best_pt is not None:
        x, y, info = best_pt
    else:
        info = dict(pobj=float(c @ x), dobj=float(b @ y - ub @ v), rel_p=float(rel_p), rel_d=float(rel_d),
                    gap=float(gap))
    info["merit"] = best if status != "optimal" else merit
    return status, x, y, it, info


def solve_ipm(lp: LP, tol=1e-8, max_iter=200, time_limit=600.0, crossover=False, verbose=False) -> IPMResult:
    """Interior-point method for an LP (integer markers are ignored). With crossover=True the result is an exact
    optimal vertex (C++ simplex started from the interior solution)."""
    from .presolve import presolve
    if lp.Q is not None and lp.Q.nnz:
        raise ValueError("solve_ipm is for LPs")
    t0 = time.perf_counter()
    m0 = lp.K.shape[0]
    P = presolve(lp)
    nan = float("nan")
    if P.status != "reduced":
        return IPMResult(P.status, None, None, nan, nan, nan, nan, nan, 0, time.perf_counter() - t0)
    red = P.lp
    if red.K.shape[1] == 0:
        x = P.postsolve(np.zeros(0))
        obj = red.obj_const
        return IPMResult("optimal", x, np.zeros(m0), obj, obj, 0.0, 0.0, 0.0, 0, time.perf_counter() - t0, True)
    A, b, c, u, free, const, recover = _standard_form(red)
    r, s = _ruiz(A)
    As = (sp.diags(r) @ A @ sp.diags(s)).tocsr()
    status, xs, ys, it, info = _mehrotra(As, r * b, s * c, u / s, free, tol, max_iter,
                                         time_limit - (time.perf_counter() - t0), verbose)
    if xs is None:
        return IPMResult(status, None, None, nan, nan, nan, nan, nan, it, time.perf_counter() - t0)
    x = P.postsolve(recover(s * xs))
    y = np.zeros(m0)
    y[P.rows] = r * ys
    res = IPMResult(status, x, y, info["pobj"] + const, info["dobj"] + const, info["gap"], info["rel_p"],
                    info["rel_d"], it, time.perf_counter() - t0)
    if crossover and (status == "optimal" or info.get("merit", 1.0) < 1e-2):   # crossover finishes the job
        from .core import crossover as run_crossover
        tc = time.perf_counter()
        cr = run_crossover(lp, x, time_limit=max(time_limit - res.time, 1.0), verbose=verbose)
        if cr.status == "optimal":
            res = IPMResult("optimal", cr.x, y, cr.obj, cr.obj, 0.0, 0.0, res.rel_dual_res, it,
                            time.perf_counter() - t0, True, time.perf_counter() - tc)
    return res
