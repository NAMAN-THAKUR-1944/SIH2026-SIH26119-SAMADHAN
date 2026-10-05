"""Python binding for the SAMADHAN C++ core (cpp/samadhan_core.cpp): dual simplex + branch-and-cut.

solve_core() presolves the model first (samadhan/presolve.py) and maps the solution back.

The shared library is compiled on first use with the zig C++ toolchain (`pip install ziglang`), so no system
compiler is needed. Rebuilt automatically when the C++ source is newer than the library.
"""
import ctypes
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .lp import LP

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "cpp" / "samadhan_core.cpp"
LIBDIR = ROOT / "_lib"
LIBNAME = "samadhan_core.dll" if os.name == "nt" else "libsamadhan_core.so"
# branch-and-cut features (bit mask): 1 feasibility pump, 2 diving, 4 cover cuts, 8 reliability branching,
# 16 node domain propagation, 32 c-MIR cuts
FEATURES = 59         # all but cover cuts, which cost more than they gave on MIPLIB 3 (ablation)
STATUS = {0: "optimal", 1: "infeasible", 2: "unbounded", 3: "time_limit", 4: "node_limit", 5: "numerical_error",
          6: "no_solution_found"}
_lib = None


def build(verbose=False):
    """Compile the C++ core into samadhan/_lib/ with zig (clang-based, bundled via pip)."""
    LIBDIR.mkdir(exist_ok=True)
    out = LIBDIR / LIBNAME
    target = "x86_64-windows-gnu" if os.name == "nt" else "x86_64-linux-gnu"
    cmd = [sys.executable, "-m", "ziglang", "c++", "-O3", "-std=c++17", "-shared", "-w"]
    if os.name != "nt":
        cmd.append("-fPIC")
    cmd += ["-target", target, "-o", str(out), str(SRC)]
    res = subprocess.run(cmd, capture_output=not verbose, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"C++ core build failed:\n{res.stderr}")
    return out


def _load():
    global _lib
    if _lib is not None:
        return _lib
    out = LIBDIR / LIBNAME
    if not out.exists() or (SRC.exists() and SRC.stat().st_mtime > out.stat().st_mtime):
        build()
    lib = ctypes.CDLL(str(out))
    P = ctypes.POINTER
    lib.sm_solve.restype = ctypes.c_int
    lib.sm_solve.argtypes = [ctypes.c_int, ctypes.c_int, P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_double),
                             P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_double),
                             P(ctypes.c_double), P(ctypes.c_char), P(ctypes.c_double), P(ctypes.c_double),
                             P(ctypes.c_double)]
    lib.sm_crossover.restype = ctypes.c_int
    lib.sm_crossover.argtypes = [ctypes.c_int, ctypes.c_int, P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_double),
                                 P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_double),
                                 P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_double),
                                 P(ctypes.c_double)]
    lib.sm_lu_create.restype = ctypes.c_void_p
    lib.sm_lu_create.argtypes = [ctypes.c_int, P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_double), ctypes.c_double,
                                 ctypes.c_int, P(ctypes.c_int)]
    lib.sm_lu_solve.restype = None
    lib.sm_lu_solve.argtypes = [ctypes.c_void_p, P(ctypes.c_double)]
    lib.sm_lu_nnz.restype = ctypes.c_long
    lib.sm_lu_nnz.argtypes = [ctypes.c_void_p]
    lib.sm_lu_free.restype = None
    lib.sm_lu_free.argtypes = [ctypes.c_void_p]
    lib.sm_lu_check.restype = ctypes.c_int
    lib.sm_lu_check.argtypes = [ctypes.c_int, P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_double), ctypes.c_int,
                                P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_double)]
    _lib = lib
    return lib


class SparseLU:
    """Sparse LU of a square matrix by the C++ core (Markowitz order, threshold pivoting), kept for repeated
    solves. symmetric=True is for symmetric positive definite matrices: diagonal pivots only (minimum-degree
    order), i.e. an LDL' factorisation. rank < m means the matrix is singular to the pivot tolerance abs_tol and
    nothing is stored."""

    def __init__(self, B, abs_tol=1e-14, symmetric=False):
        self._lib = _load()
        A = sp.csc_matrix(B, dtype=np.float64)
        A.sum_duplicates()
        A.sort_indices()
        self.m = A.shape[0]
        cp, ri = np.ascontiguousarray(A.indptr, np.int32), np.ascontiguousarray(A.indices, np.int32)
        v = np.ascontiguousarray(A.data, np.float64)
        rank = ctypes.c_int(0)
        self._h = self._lib.sm_lu_create(self.m, _ip(cp), _ip(ri), _dp(v), abs_tol, int(symmetric),
                                         ctypes.byref(rank))
        self.rank = rank.value

    @property
    def ok(self):
        return bool(self._h)

    def nnz(self):
        return int(self._lib.sm_lu_nnz(self._h))

    def solve(self, b):
        x = np.array(b, np.float64)
        self._lib.sm_lu_solve(self._h, _dp(x))
        return x

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.sm_lu_free(self._h)
            self._h = None


def lu_check(B, b, d, replace=None):
    """Test hook for the C++ sparse LU: returns (rank, x, y) with B x = b and B' y = d. With replace=(r, col)
    column r of B is first replaced through a product-form (eta) update."""
    lib = _load()
    A = sp.csc_matrix(B, dtype=np.float64)
    A.sum_duplicates(); A.sort_indices()
    m = A.shape[0]
    r, col = replace if replace is not None else (-1, np.zeros(m))
    cp, ri = np.ascontiguousarray(A.indptr, np.int32), np.ascontiguousarray(A.indices, np.int32)
    v, col = np.ascontiguousarray(A.data, np.float64), np.ascontiguousarray(col, np.float64)
    x, y = np.array(b, np.float64), np.array(d, np.float64)
    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ip = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    rank = lib.sm_lu_check(m, ip(cp), ip(ri), dp(v), r, dp(col), dp(x), dp(y))
    return rank, x, y


@dataclass
class CoreResult:
    status: str
    x: np.ndarray | None
    obj: float
    bound: float
    gap: float
    nodes: int
    lp_iters: int
    time: float
    cuts: int
    root_bound: float
    root_bound_cuts: float


def solve_core(lp: LP, integer=None, time_limit=300.0, node_limit=50_000_000, cut_rounds=8, gap=1e-6,
               verbose=False, presolve=True, features=None) -> CoreResult:
    """Solve an LP (dual simplex) or MILP (branch-and-cut) with the C++ core. Objective includes obj_const."""
    if integer is not None:
        lp = replace(lp, integer=np.asarray(integer, bool))
    if not presolve:
        return _solve_core(lp, time_limit, node_limit, cut_rounds, gap, verbose, FEATURES if features is None else features)
    from .presolve import presolve as run_presolve
    t0 = time.perf_counter()
    P = run_presolve(lp)
    tp = time.perf_counter() - t0
    if verbose:
        print(f"presolve: {P.status}, rows {P.stats.get('rows')}, columns {P.stats.get('cols')}, {tp:.2f} s")
    if P.status != "reduced":
        return CoreResult(P.status, None, np.inf, -np.inf, np.inf, 0, 0, tp, 0, -np.inf, -np.inf)
    if P.lp.K.shape[1] == 0:                       # presolve fixed every column
        obj = P.lp.obj_const
        return CoreResult("optimal", P.postsolve(np.zeros(0)), obj, obj, 0.0, 0, 0, tp, 0, obj, obj)
    r = _solve_core(P.lp, max(time_limit - tp, 0.0), node_limit, cut_rounds, gap, verbose,
                    FEATURES if features is None else features)
    return replace(r, x=P.postsolve(r.x) if r.x is not None else None, time=r.time + tp)


def _model_arrays(lp):
    """Column-wise model arrays in the layout of the C ABI (rows as rlo <= a x <= rhi)."""
    A = lp.K.tocsc()
    A.sum_duplicates()
    A.sort_indices()
    m = A.shape[0]
    rhi = np.full(m, np.inf)
    rhi[:lp.n_eq] = lp.q[:lp.n_eq]
    f = lambda a: np.ascontiguousarray(a, np.float64)
    return dict(cp=np.ascontiguousarray(A.indptr, np.int32), ri=np.ascontiguousarray(A.indices, np.int32),
                v=f(A.data), c=f(lp.c), lb=f(lp.l), ub=f(lp.u), rlo=f(lp.q), rhi=rhi)


_dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
_ip = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _model_args(a):
    return (_ip(a["cp"]), _ip(a["ri"]), _dp(a["v"]), _dp(a["c"]), _dp(a["lb"]), _dp(a["ub"]), _dp(a["rlo"]),
            _dp(a["rhi"]))


def _result(st, lp, x, info):
    k = lp.obj_const
    has_x = st in (0, 3, 4)
    return CoreResult(STATUS.get(st, str(st)), x.copy() if has_x else None, info[0] + k, info[1] + k,
                      info[2], int(info[3]), int(info[4]), info[5], int(info[6]), info[7] + k, info[8] + k)


def _solve_core(lp, time_limit, node_limit, cut_rounds, gap, verbose, features=FEATURES):
    lib = _load()
    m, n = lp.K.shape
    a = _model_arrays(lp)
    isint = np.ascontiguousarray(np.zeros(n) if lp.integer is None else lp.integer, np.int8)
    opts = np.array([time_limit, node_limit, cut_rounds, 1 if verbose else 0, gap, features], np.float64)
    x, info = np.zeros(n), np.zeros(9)
    st = lib.sm_solve(n, m, *_model_args(a), isint.ctypes.data_as(ctypes.POINTER(ctypes.c_char)), _dp(opts),
                      _dp(x), _dp(info))
    sys.stdout.flush()
    return _result(st, lp, x, info)


def crossover(lp: LP, x0, time_limit=300.0, verbose=False, presolve=True, crash_tol=0.0) -> CoreResult:
    """Exact optimal vertex of an LP from an approximate solution x0 (for example from the GPU engine): crash basis
    from x0, dual simplex on shifted costs, primal simplex clean-up. Falls back to a cold-start dual simplex if
    the crossover does not finish."""
    if lp.integer is not None and lp.integer.any():
        raise ValueError("crossover is for LPs; use solve_core for models with integer variables")
    x0 = np.asarray(x0, np.float64)
    t0 = time.perf_counter()
    P = None
    if presolve:
        from .presolve import presolve as run_presolve
        P = run_presolve(lp)
        if P.status != "reduced" or P.lp.K.shape[1] == 0:
            return solve_core(lp, time_limit=time_limit, verbose=verbose)
        lp, x0 = P.lp, x0[P.cols]
    lib = _load()
    m, n = lp.K.shape
    a = _model_arrays(lp)
    tp = time.perf_counter() - t0
    opts = np.array([max(time_limit - tp, 0.0), 1 if verbose else 0, crash_tol], np.float64)
    x, info = np.zeros(n), np.zeros(9)
    st = lib.sm_crossover(n, m, *_model_args(a), _dp(np.ascontiguousarray(x0)), _dp(opts), _dp(x), _dp(info))
    sys.stdout.flush()
    r = _result(st, lp, x, info)
    if r.status != "optimal":
        if verbose:
            print(f"crossover ended with {r.status}: cold start")
        r = _solve_core(lp, max(time_limit - (time.perf_counter() - t0), 0.0), 1, 0, 1e-6, verbose)
    r = replace(r, time=time.perf_counter() - t0)
    if P is not None:
        r = replace(r, x=P.postsolve(r.x) if r.x is not None else None)
    return r
