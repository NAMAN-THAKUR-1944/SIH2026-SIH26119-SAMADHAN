"""Python binding for the SAMADHAN C++ core (cpp/samadhan_core.cpp): dual simplex + branch-and-cut.

The shared library is compiled on first use with the zig C++ toolchain (`pip install ziglang`), so no system
compiler is needed. Rebuilt automatically when the C++ source is newer than the library.
"""
import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lp import LP

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "cpp" / "samadhan_core.cpp"
LIBDIR = ROOT / "_lib"
LIBNAME = "samadhan_core.dll" if os.name == "nt" else "libsamadhan_core.so"
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
    _lib = lib
    return lib


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
               verbose=False) -> CoreResult:
    """Solve an LP (dual simplex) or MILP (branch-and-cut) with the C++ core. Objective includes obj_const."""
    lib = _load()
    A = lp.K.tocsc()
    A.sort_indices()
    m, n = A.shape
    rlo = lp.q.astype(np.float64).copy()
    rhi = np.full(m, np.inf)
    rhi[:lp.n_eq] = lp.q[:lp.n_eq]
    ints = lp.integer if integer is None else integer
    isint = np.zeros(n, np.int8) if ints is None else np.asarray(ints, np.int8)
    arrs = dict(cp=np.ascontiguousarray(A.indptr, np.int32), ri=np.ascontiguousarray(A.indices, np.int32),
                v=np.ascontiguousarray(A.data, np.float64), c=np.ascontiguousarray(lp.c, np.float64),
                lb=np.ascontiguousarray(lp.l, np.float64), ub=np.ascontiguousarray(lp.u, np.float64),
                rlo=rlo, rhi=rhi, isint=np.ascontiguousarray(isint),
                opts=np.array([time_limit, node_limit, cut_rounds, 1 if verbose else 0, gap], np.float64),
                x=np.zeros(n), info=np.zeros(9))
    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ip = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    st = lib.sm_solve(n, m, ip(arrs["cp"]), ip(arrs["ri"]), dp(arrs["v"]), dp(arrs["c"]), dp(arrs["lb"]),
                      dp(arrs["ub"]), dp(arrs["rlo"]), dp(arrs["rhi"]),
                      arrs["isint"].ctypes.data_as(ctypes.POINTER(ctypes.c_char)), dp(arrs["opts"]),
                      dp(arrs["x"]), dp(arrs["info"]))
    sys.stdout.flush()
    info = arrs["info"]
    has_x = st in (0, 3, 4)
    k = lp.obj_const
    return CoreResult(STATUS.get(st, str(st)), arrs["x"].copy() if has_x else None, info[0] + k, info[1] + k,
                      info[2], int(info[3]), int(info[4]), info[5], int(info[6]), info[7] + k, info[8] + k)
