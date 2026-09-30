"""From-scratch MILP: best-first branch-and-bound on exact simplex relaxations.

Prototype of the SAMADHAN MILP layer: node LPs by samadhan.simplex, most-fractional branching,
a rounding heuristic for early incumbents, and bound-based pruning with a relative gap stop.
Cutting planes, presolve and GPU relaxations for large nodes are the next steps.
"""
import heapq
import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from .lp import LP
from .simplex import solve_lp_dense

INT_TOL = 1e-6


@dataclass
class MILPResult:
    status: str
    x: np.ndarray | None
    obj: float
    bound: float
    gap: float
    nodes: int
    time: float


def _feasible(lp: LP, x, tol=1e-6):
    r = lp.K @ x - lp.q
    scale = 1 + np.abs(lp.q)
    return (np.all(np.abs(r[:lp.n_eq]) <= tol * scale[:lp.n_eq]) and
            np.all(r[lp.n_eq:] >= -tol * scale[lp.n_eq:]) and
            np.all(x >= lp.l - tol) and np.all(x <= lp.u + tol))


def _cutoff(best_obj, gap_tol):
    return best_obj - gap_tol * max(1.0, abs(best_obj)) if math.isfinite(best_obj) else math.inf


def solve_milp(lp: LP, integer=None, gap_tol=1e-6, node_limit=100_000, time_limit=300.0, verbose=False):
    ints = np.where(lp.integer if integer is None else integer)[0]
    t0 = time.perf_counter()
    best_x, best_obj = None, math.inf
    tie = itertools.count()
    st, x, obj = solve_lp_dense(lp)
    if st != "optimal":
        return MILPResult(st, None, math.inf, math.inf, math.inf, 1, time.perf_counter() - t0)
    heap = [(obj, next(tie), lp.l.copy(), lp.u.copy(), x)]
    nodes = 1

    while heap:
        bound, _, l, u, x = heapq.heappop(heap)
        if bound >= _cutoff(best_obj, gap_tol):
            continue  # pruned by bound
        frac = np.abs(x[ints] - np.round(x[ints]))
        if frac.max(initial=0.0) <= INT_TOL:  # integral: new incumbent
            if bound < best_obj:
                best_obj, best_x = bound, x.copy()
            continue
        # rounding heuristic
        xr = x.copy(); xr[ints] = np.round(xr[ints])
        if _feasible(lp, xr):
            o = float(lp.c @ xr + lp.obj_const)
            if o < best_obj:
                best_obj, best_x = o, xr
        # branch on the most fractional variable
        j = ints[int(np.argmax(frac))]
        for side in (0, 1):
            l2, u2 = l.copy(), u.copy()
            if side == 0:
                u2[j] = math.floor(x[j])
            else:
                l2[j] = math.ceil(x[j])
            st, x2, o2 = solve_lp_dense(lp, l2, u2)
            nodes += 1
            if st == "optimal" and o2 < _cutoff(best_obj, gap_tol):
                heapq.heappush(heap, (o2, next(tie), l2, u2, x2))
        glob = min([h[0] for h in heap] + [best_obj])
        gap = (best_obj - glob) / max(1.0, abs(best_obj)) if math.isfinite(best_obj) else math.inf
        if verbose and nodes % 200 < 2:
            print(f"nodes {nodes:6d}  open {len(heap):5d}  incumbent {best_obj:.6g}  bound {glob:.6g}  gap {gap:.2e}")
        if gap <= gap_tol:
            break
        if nodes >= node_limit or time.perf_counter() - t0 > time_limit:
            return MILPResult("limit", best_x, best_obj, glob, gap, nodes, time.perf_counter() - t0)

    glob = min([h[0] for h in heap] + [best_obj])
    status = "optimal" if best_x is not None else "infeasible"
    gap = 0.0 if not heap else (best_obj - glob) / max(1.0, abs(best_obj))
    return MILPResult(status, best_x, best_obj, glob, max(gap, 0.0), nodes, time.perf_counter() - t0)
