"""Runs inside the NVIDIA cuOpt container (see benchmarks/mittelmann.py --cuopt): solves each MPS file in /data with
the given cuOpt LP method (0 concurrent, 1 PDLP), tolerance and time limit, printing one 'RESULT {json}' per model.

    python /bench/cuopt_runner.py TIME_LIMIT TOL METHOD file.mps [file.mps ...]
"""
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version

from cuopt.linear_programming.problem import Problem
from cuopt.linear_programming.solver.solver_wrapper import LPTerminationStatus
from cuopt.linear_programming.solver_settings import SolverSettings

STATUS = {s.value: s.name for s in LPTerminationStatus}

for pkg in ("cuopt-cu13", "cuopt-cu12", "cuopt"):
    try:
        print("cuOpt", pkg, version(pkg), flush=True)
        break
    except PackageNotFoundError:
        pass

time_limit, tol, method = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
for f in sys.argv[4:]:
    name = f.rsplit(".", 1)[0]
    t0 = time.perf_counter()
    problem = Problem.read(f"/data/{f}") if hasattr(Problem, "read") else Problem.readMPS(f"/data/{f}")
    t_read = time.perf_counter() - t0
    settings = SolverSettings()
    settings.set_optimality_tolerance(tol)
    settings.set_parameter("time_limit", time_limit)
    settings.set_parameter("method", method)
    t0 = time.perf_counter()
    problem.solve(settings)
    wall = time.perf_counter() - t0
    status = STATUS.get(int(problem.Status), str(problem.Status))
    obj = problem.ObjValue
    res = dict(name=name, status=status, obj=float(obj) if obj is not None else None, time=wall,
               solver_time=float(problem.SolveTime) if problem.SolveTime is not None else None, read=t_read)
    print("RESULT " + json.dumps(res), flush=True)
