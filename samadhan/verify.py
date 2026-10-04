"""Evaluator self-check:  python -m samadhan verify

Solves small LP, QP and MILP models with every engine and checks every answer two ways:
  1. the objective is compared with an independent reference: an optimum worked out by hand, or HiGHS
     (skipped if highspy is not installed);
  2. feasibility is recomputed here from the returned solution vector (constraints, bounds, integrality),
     so the solver's own report is not trusted.
Takes about a minute on a laptop CPU.
"""
import time

import numpy as np
import scipy.sparse as sp

from .generate import refinery_lp, refinery_milp
from .lp import LP

GPU_TOL, CORE_TOL, FEAS_TOL = 1e-5, 1e-8, 1e-5   # relative objective error allowed per engine; feasibility


def _wyndor():
    """max 3x + 5y  s.t.  x <= 4,  2y <= 12,  3x + 2y <= 18,  x, y >= 0   ->   x = 2, y = 6, value 36."""
    K = sp.csr_matrix([[-1.0, 0.0], [0.0, -2.0], [-3.0, -2.0]])
    return LP(np.array([-3.0, -5.0]), K, np.array([-4.0, -12.0, -18.0]), 0, np.zeros(2), np.full(2, np.inf))


def _hs21():
    """HS21 (Maros-Meszaros): min 0.01 x1^2 + x2^2 - 100,  10 x1 - x2 >= 10,  2 <= x1 <= 50,  -50 <= x2 <= 50
    ->  x = (2, 0), value -99.96."""
    return LP(np.zeros(2), sp.csr_matrix([[10.0, -1.0]]), np.array([10.0]), 0, np.array([2.0, -50.0]),
              np.array([50.0, 50.0]), obj_const=-100.0, Q=sp.csr_matrix(np.diag([0.02, 2.0])))


def _knapsack():
    """max 5x + 4y  s.t.  6x + 4y <= 24,  x + 2y <= 6,  0 <= x, y <= 10 integer
    ->  LP relaxation 21 at (3, 1.5), integer optimum 20 at (4, 0)."""
    K = sp.csr_matrix([[-6.0, -4.0], [-1.0, -2.0]])
    return LP(np.array([-5.0, -4.0]), K, np.array([-24.0, -6.0]), 0, np.zeros(2), np.full(2, 10.0),
              integer=np.array([True, True]))


SMALL = dict(R=3, C=5, P=4, D=20, T=4, seed=1)             # 1,708-variable refinery planning model
CHECKS = [   # (problem class, description, model, engine, reference: number worked out by hand, or "highs")
    ("LP", "textbook (2 vars)", _wyndor, "gpu", -36.0),
    ("LP", "textbook (2 vars)", _wyndor, "core", -36.0),
    ("LP", "refinery planning", lambda: refinery_lp(**SMALL), "gpu", "highs"),
    ("LP", "refinery planning", lambda: refinery_lp(**SMALL), "core", "highs"),
    ("LP", "textbook (2 vars)", _wyndor, "ipm", -36.0),
    ("LP", "refinery planning", lambda: refinery_lp(**SMALL), "ipm", "highs"),
    ("QP", "HS21, Maros-Meszaros", _hs21, "gpu", -99.96),
    ("QP", "refinery, convex costs", lambda: refinery_lp(**SMALL, quad=0.3), "gpu", "highs"),
    ("MILP", "textbook knapsack", _knapsack, "core", -20.0),
] + [("MILP", f"refinery contracts #{s}", lambda s=s: refinery_milp(R=4, C=8, P=3, D=6, seed=s), "core", "highs")
     for s in range(3)]


def violation(lp: LP, x):
    """Largest violation of rows (relative to the right-hand side), bounds and integrality by the vector x."""
    r = lp.q - lp.K @ x
    r[lp.n_eq:] = np.maximum(r[lp.n_eq:], 0.0)
    v = np.abs(r).max(initial=0.0) / (1.0 + np.abs(lp.q).max(initial=0.0))
    v = max(v, np.maximum(lp.l - x, x - lp.u).max(initial=0.0))
    if lp.integer is not None and lp.integer.any():
        xi = x[lp.integer]
        v = max(v, np.abs(xi - np.round(xi)).max())
    return v


def _solve(lp, engine, device):
    if engine == "gpu":
        from .pdlp import PDLP
        r = PDLP(lp, device=device).solve(tol=1e-6, max_iter=2_000_000, time_limit=120)
        return r.status, r.primal_obj, r.x
    if engine == "ipm":
        from .ipm import solve_ipm
        r = solve_ipm(lp, time_limit=120, crossover=True)
        return r.status, r.primal_obj, r.x
    from .core import solve_core
    r = solve_core(lp, time_limit=120)
    return r.status, r.obj, r.x


def run(device="cuda"):
    import torch
    try:
        from importlib.metadata import version

        from .baseline import solve_highs
        referee = f"HiGHS {version('highspy')}"
    except ImportError:
        solve_highs, referee = None, "HiGHS not installed (pip install highspy): those checks are skipped"
    where = "CPU"
    if device == "cuda":
        if torch.cuda.is_available():
            where = torch.cuda.get_device_name(0)
        else:
            device, where = "cpu", "CPU (no CUDA GPU found)"
    print(f"SAMADHAN self-check   GPU engine on {where}   referee: {referee}\n")

    from . import core
    t = time.perf_counter()
    core._load()                                   # compiles the C++ core with zig on the first run
    if time.perf_counter() - t > 1:
        print(f"C++ core compiled in {time.perf_counter() - t:.1f} s (first run only)\n")

    head = f"{'':5}{'model':25}{'vars':>6}  {'engine':12}{'SAMADHAN':>15}{'reference':>15}  {'':9}" \
           f"{'obj. err':>9}{'infeas.':>9}{'time':>8}"
    print(head + "\n" + "-" * len(head))
    passed = run_count = 0
    t_all = time.perf_counter()
    for cls, desc, make, engine, ref in CHECKS:
        lp = make()
        n = lp.K.shape[1]
        if ref == "highs":
            if solve_highs is None:
                continue
            ref_obj, ref_src = solve_highs(lp)["obj"], "HiGHS"
        else:
            ref_obj, ref_src = ref, "by hand"
        t = time.perf_counter()
        status, obj, x = _solve(lp, engine, device)
        dt = time.perf_counter() - t
        err = abs(obj - ref_obj) / max(1.0, abs(ref_obj))
        feas = violation(lp, x) if x is not None else np.inf
        ok = status == "optimal" and err <= (GPU_TOL if engine == "gpu" else CORE_TOL) and feas <= FEAS_TOL
        passed += ok
        run_count += 1
        name = {"gpu": "GPU engine", "core": "C++ core", "ipm": "IPM+xover"}[engine]
        print(f"{cls:5}{desc:25}{n:>6,}  {name:12}{obj:>15.8g}{ref_obj:>15.8g}  {ref_src:9}{err:>9.1e}{feas:>9.1e}"
              f"{dt:>7.2f}s  {'PASS' if ok else 'FAIL (' + status + ')'}")
    print(f"\n{passed}/{run_count} checks passed in {time.perf_counter() - t_all:.1f} s.  "
          f"obj. err = relative distance to the reference;\ninfeas. = worst constraint, bound or integrality "
          f"violation, recomputed here from the solution vector.")
    return 0 if passed == run_count else 1
