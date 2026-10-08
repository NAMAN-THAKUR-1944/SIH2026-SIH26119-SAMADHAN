"""PuLP plugin: solve PuLP models with SAMADHAN.

    import pulp
    from samadhan.pulp_solver import SAMADHAN

    prob = pulp.LpProblem("plan", pulp.LpMinimize)
    ...
    prob.solve(SAMADHAN(msg=False, timeLimit=60))

LPs and MILPs go to the C++ core (dual simplex to an exact vertex, branch-and-cut for integer columns).
engine="gpu" sends an LP to the GPU engine and finishes it with crossover to an exact vertex; engine="ipm" uses the
interior-point method with crossover; engine="auto" picks like the command line (C++ core up to 50k variables).
"""
import math

import numpy as np
import scipy.sparse as sp
from pulp import (
    LpBinary,
    LpConstraintEQ,
    LpConstraintGE,
    LpConstraintLE,
    LpInteger,
    LpMaximize,
    LpSolver,
    LpSolveStatus,
)
from pulp.apis.core import PulpSolverError, clocks

from .lp import LP

STATUS = {"optimal": LpSolveStatus.Optimal, "infeasible": LpSolveStatus.Infeasible,
          "unbounded": LpSolveStatus.Unbounded, "time_limit": LpSolveStatus.TimeLimit,
          "no_solution_found": LpSolveStatus.TimeLimit, "node_limit": LpSolveStatus.NodeLimit,
          "iteration_limit": LpSolveStatus.IterationLimit, "numerical_error": LpSolveStatus.NumericalError}


def to_lp(prob, mip=True):
    """PuLP problem -> (samadhan LP, its variables in column order, objective sign).
    Rows become K x = q (equalities first) and K x >= q; a maximisation is solved as min -c'x."""
    cols = prob.exported_variables()
    index = {v.id: j for j, v in enumerate(cols)}
    n = len(cols)
    sign = -1.0 if prob.sense == LpMaximize else 1.0
    c = np.array([sign * prob.objective.get(v, 0.0) for v in cols], float)
    lo = np.array([-math.inf if v.lowBound is None else float(v.lowBound) for v in cols])
    up = np.array([math.inf if v.upBound is None else float(v.upBound) for v in cols])
    integer = np.zeros(n, bool)
    for j, v in enumerate(cols):
        if mip and v.cat in (LpInteger, LpBinary):
            integer[j] = True
            if v.cat == LpBinary:
                lo[j], up[j] = max(lo[j], 0.0), min(up[j], 1.0)
    eq, ge = ([], [], []), ([], [], [])                # (row index, column, value) for each block
    q_eq, q_ge = [], []
    for con in prob.constraints():
        if con.sense == LpConstraintEQ:
            block, rhs, s = eq, q_eq, 1.0
        elif con.sense == LpConstraintGE:
            block, rhs, s = ge, q_ge, 1.0
        elif con.sense == LpConstraintLE:
            block, rhs, s = ge, q_ge, -1.0              # a x <= b  ->  -a x >= -b
        else:
            raise PulpSolverError(f"SAMADHAN: unsupported constraint sense in {con.name}")
        i = len(rhs)
        for v, a in con.items():
            block[0].append(i); block[1].append(index[v.id]); block[2].append(s * a)
        rhs.append(-s * con.constant)
    K = sp.vstack([sp.csr_matrix((eq[2], (eq[0], eq[1])), shape=(len(q_eq), n)),
                   sp.csr_matrix((ge[2], (ge[0], ge[1])), shape=(len(q_ge), n))]).tocsr()
    lp = LP(c, K, np.array(q_eq + q_ge, float), len(q_eq), lo, up,
            obj_const=sign * float(prob.objective.constant), name=prob.name or "pulp",
            col_names=[v.name for v in cols], integer=integer if integer.any() else None, sense=int(sign))
    return lp, cols, sign


class SAMADHAN(LpSolver):
    """SAMADHAN through PuLP. timeLimit in seconds; gapRel the relative MIP gap; engine "core", "gpu", "ipm" or
    "auto" for LPs (MILPs always use the C++ core); device for the GPU engine ("cuda" or "cpu")."""

    name = "SAMADHAN"

    def __init__(self, mip=True, msg=True, timeLimit=None, gapRel=None, engine="core", device="cuda", **kwargs):
        LpSolver.__init__(self, mip=mip, msg=msg, timeLimit=timeLimit, gapRel=gapRel, **kwargs)
        self.engine, self.device = engine, device

    def available(self):
        return True

    def copy(self):
        other = super().copy()
        other.timeLimit, other.engine, other.device = self.timeLimit, self.engine, self.device
        other.optionsDict = dict(self.optionsDict)
        return other

    def actualSolve(self, lp, **kwargs):
        start = clocks()
        model, cols, sign = to_lp(lp, self.mip)
        time_limit = float(self.timeLimit) if self.timeLimit else 600.0
        x, status, bound = self._solve(model, time_limit)
        has_solution = x is not None and status in (LpSolveStatus.Optimal, LpSolveStatus.TimeLimit,
                                                     LpSolveStatus.NodeLimit, LpSolveStatus.IterationLimit)
        if has_solution:
            for j, v in enumerate(cols):
                val = float(x[j])
                if model.integer is not None and model.integer[j]:
                    val = float(round(val))
                v.varValue = val
        best = None if bound is None or not math.isfinite(bound) else sign * bound
        return self.buildStats(lp, status, has_solution, start=start, best_bound=best)

    def _solve(self, model, time_limit):
        engine = self.engine
        if model.integer is not None or engine == "core" or (engine == "auto" and model.K.shape[1] <= 50_000):
            from .core import solve_core
            gap = self.optionsDict.get("gapRel")
            r = solve_core(model, time_limit=time_limit, verbose=self.msg,
                           **({} if gap is None else dict(gap=float(gap))))
            x = r.x if r.x is not None and math.isfinite(r.obj) else None
            return x, STATUS.get(r.status, LpSolveStatus.Undefined), r.bound
        if engine == "ipm":
            from .ipm import solve_ipm
            r = solve_ipm(model, crossover=True, time_limit=time_limit, verbose=self.msg)
            return (r.x if r.status == "optimal" else None), STATUS.get(r.status, LpSolveStatus.Undefined), None
        from .pdlp import solve
        r = solve(model, device=self.device, crossover=True, tol=1e-6, time_limit=time_limit)
        return (r.x if r.status == "optimal" else None), STATUS.get(r.status, LpSolveStatus.Undefined), None
