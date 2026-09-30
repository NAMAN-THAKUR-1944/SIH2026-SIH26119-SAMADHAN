"""SAMADHAN: sovereign optimization solver for LP, QP and MILP (SIH 2026, PS SIH26119).

    from samadhan import read_mps, solve, solve_core
    lp = read_mps("model.mps")
    solve(lp, device="cuda", tol=1e-6)     # GPU engine: LP, or QP when lp.Q is set
    solve_core(lp, time_limit=60)          # C++ core: LP or MILP (dual simplex, branch-and-cut)
"""
from .core import solve_core
from .lp import LP
from .mps import read_mps
from .pdlp import PDLP, Result, solve
from .qpdata import read_maros

__all__ = ["LP", "PDLP", "Result", "read_maros", "read_mps", "solve", "solve_core"]
