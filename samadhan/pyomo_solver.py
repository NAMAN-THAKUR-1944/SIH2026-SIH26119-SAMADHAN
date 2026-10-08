"""Pyomo plugin: SolverFactory("samadhan").

    import pyomo.environ as pyo
    import samadhan.pyomo_solver            # registers the solver

    results = pyo.SolverFactory("samadhan").solve(model, timelimit=60)
    print(results.solver.termination_condition, pyo.value(model.obj))

The model is written with Pyomo's own MPS writer (symbolic labels), read by SAMADHAN's MPS reader and solved: LPs and
MILPs by the C++ core (exact vertices, branch-and-cut); options engine="gpu" or "ipm" send an LP to the GPU engine or
the interior-point method, both finished by crossover to an exact vertex. The values are loaded back into the model's
variables by name. Options: timelimit (s), mip_gap (relative), engine, device.
"""
import math
import os
import tempfile
import time

from pyomo.opt import SolverFactory, SolverResults, SolverStatus, TerminationCondition

from .mps import read_mps

TERMINATION = {"optimal": TerminationCondition.optimal, "infeasible": TerminationCondition.infeasible,
               "unbounded": TerminationCondition.unbounded, "time_limit": TerminationCondition.maxTimeLimit,
               "no_solution_found": TerminationCondition.maxTimeLimit,
               "node_limit": TerminationCondition.maxEvaluations,
               "iteration_limit": TerminationCondition.maxIterations, "numerical_error": TerminationCondition.error}


@SolverFactory.register("samadhan", doc="SAMADHAN LP / MILP solver (C++ core; GPU engine and interior point for LPs)")
class SamadhanSolver:
    def __init__(self, **kwds):
        self.options = dict(kwds.get("options") or {})

    def available(self, exception_flag=True):
        return True

    def license_is_valid(self):
        return True

    def version(self):
        return (1, 0, 0)

    def solve(self, model, tee=False, timelimit=None, load_solutions=True, options=None, **kwds):
        opts = {**self.options, **(options or {})}
        time_limit = float(timelimit or opts.get("timelimit", 600.0))
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            path, smap_id = model.write(os.path.join(d, "model.mps"), io_options={"symbolic_solver_labels": True})
            lp = read_mps(path)
        symbols = model.solutions.symbol_map[smap_id].bySymbol
        status, obj, bound, x = self._solve(lp, time_limit, opts, tee)
        res = SolverResults()
        res.solver.name = "SAMADHAN"
        res.solver.termination_condition = TERMINATION.get(status, TerminationCondition.unknown)
        res.solver.status = SolverStatus.ok if status in ("optimal", "time_limit", "node_limit") else \
            SolverStatus.warning if status == "no_solution_found" else SolverStatus.error
        res.solver.wallclock_time = time.perf_counter() - start
        if x is not None and math.isfinite(obj):
            # incumbent and proven bound in the model's sense; sorted they are the lower and upper bound
            lo, hi = sorted((lp.sense * obj, lp.sense * (bound if bound is not None and math.isfinite(bound) else obj)))
            res.problem.lower_bound, res.problem.upper_bound = lo, hi
            if load_solutions:
                for name, val in zip(lp.col_names, x):
                    var = symbols.get(name)
                    if var is None or name == "ONE_VAR_CONSTANT":
                        continue
                    if var.is_integer() or var.is_binary():
                        val = float(round(val))
                    var.set_value(float(val), skip_validation=True)
        return res

    def _solve(self, lp, time_limit, opts, tee):
        engine = opts.get("engine", "core")
        if lp.integer is not None or engine == "core" or (engine == "auto" and lp.K.shape[1] <= 50_000):
            from .core import solve_core
            gap = opts.get("mip_gap")
            r = solve_core(lp, time_limit=time_limit, verbose=tee, **({} if gap is None else dict(gap=float(gap))))
            return r.status, r.obj, r.bound, (r.x if r.x is not None and math.isfinite(r.obj) else None)
        if engine == "ipm":
            from .ipm import solve_ipm
            r = solve_ipm(lp, crossover=True, time_limit=time_limit, verbose=tee)
            return r.status, r.primal_obj, r.dual_obj, (r.x if r.status == "optimal" else None)
        from .pdlp import solve
        r = solve(lp, device=opts.get("device", "cuda"), crossover=True, tol=1e-6, time_limit=time_limit)
        return r.status, r.primal_obj, r.dual_obj, (r.x if r.status == "optimal" else None)
