"""Crossover benchmark: three ways to an exact optimal vertex, each checked against a cold-start solve.

  cold      C++ dual simplex from the slack basis
  gpu+x     GPU engine to 1e-4 relative KKT, then crossover in the C++ core
  ipm+x     interior-point method to 1e-8, then crossover

    git clone https://github.com/coin-or-tools/Data-Netlib data/netlib
    python -m benchmarks.crossover [--with-L]                 -> results/lp_crossover.json
"""
import argparse
import json
import time
from pathlib import Path

from samadhan.core import crossover, solve_core
from samadhan.generate import BENCH_SEED, REFINERY_SIZES, refinery_lp
from samadhan.ipm import solve_ipm
from samadhan.mps import read_mps
from samadhan.pdlp import PDLP
from samadhan.verify import violation

OUT = Path(__file__).resolve().parent.parent / "results" / "lp_crossover.json"
NETLIB = ["25fv47", "80bau3b", "d2q06c", "degen3", "fit2p", "maros-r7", "pilot", "greenbea"]


def check(lp, ref, status, obj, x):
    if x is None:
        return dict(status=status, err=None, viol=None)
    return dict(status=status, err=abs(obj - ref) / max(1.0, abs(ref)), viol=violation(lp, x))


def model(name):
    if name.startswith("refinery-"):
        return refinery_lp(seed=BENCH_SEED, **REFINERY_SIZES[name[-1]])
    return read_mps(f"data/netlib/{name}.mps.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-L", action="store_true", help="also the 406k-variable refinery LP (slow)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    names = ["refinery-S", "refinery-M"] + (["refinery-L"] if a.with_L else []) + NETLIB
    res = []
    for name in names:
        lp = model(name)
        m, n = lp.K.shape
        cold = solve_core(lp, time_limit=1800)
        ref = cold.obj
        p = PDLP(lp, device=a.device).solve(tol=1e-4, max_iter=5_000_000, time_limit=300)
        t = time.perf_counter()
        xg = crossover(lp, p.x, time_limit=1800)
        tg = time.perf_counter() - t
        if n <= 50_000:
            xi = solve_ipm(lp, crossover=True, time_limit=300)
            ipm = dict(time=xi.time, iterations=xi.iterations, crossover_time=xi.crossover_time,
                       **check(lp, ref, xi.status, xi.primal_obj, xi.x))
        else:
            ipm = None
        out = dict(name=name, m=m, n=n, cold=dict(time=cold.time, iterations=cold.lp_iters, status=cold.status),
                   gpu_x=dict(gpu_time=p.solve_time, crossover_time=tg, iterations=xg.lp_iters,
                              **check(lp, ref, xg.status, xg.obj, xg.x)), ipm_x=ipm)
        res.append(out)
        print(f"{name:11} n={n:8,}  cold {cold.time:7.2f}s | gpu {p.solve_time:5.1f}s + crossover {tg:7.2f}s "
              f"err {out['gpu_x']['err']:.0e} | ipm+x "
              + (f"{ipm['time']:6.2f}s err {ipm['err']:.0e}" if ipm and ipm["err"] is not None else "-"), flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
