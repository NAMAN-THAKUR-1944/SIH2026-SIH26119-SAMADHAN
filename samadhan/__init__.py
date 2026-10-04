"""SAMADHAN: sovereign optimization solver for LP, QP and MILP (SIH 2026, PS SIH26119).

    from samadhan import read_mps, solve, solve_core, solve_ipm
    lp = read_mps("model.mps")
    solve(lp, device="cuda", tol=1e-6)     # GPU engine: LP, or QP when lp.Q is set (crossover=True: exact vertex)
    solve_ipm(lp, crossover=True)          # interior-point method + crossover (LP)
    solve_core(lp, time_limit=60)          # C++ core: LP or MILP (presolve, simplex, branch-and-cut)
"""
from .core import crossover, solve_core
from .ipm import solve_ipm
from .lp import LP
from .mps import read_mps
from .pdlp import PDLP, Result, solve
from .qpdata import read_maros

__all__ = ["LP", "PDLP", "Result", "crossover", "read_maros", "read_mps", "solve", "solve_core", "solve_ipm"]
