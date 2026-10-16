"""Crossover benchmark: three ways to an exact optimal vertex, each checked against a cold-start solve.

  cold      C++ dual simplex from the slack basis
  gpu+x     GPU engine to 1e-4 relative KKT, then crossover in the C++ core
  ipm+x     interior-point method to 1e-8, then crossover

    git clone https://github.com/coin-or-tools/Data-Netlib data/netlib
    python -m benchmarks.crossover [--with-L] [--with-XL] [--reuse-cold]       -> results/lp_crossover.json

--with-XL adds the 1.16 M-variable refinery LP from a 1e-6 GPU point; a cold simplex start is not attempted there
(HiGHS's dual simplex does not finish it in 900 s), its objective is checked against HiGHS's interior point
(results/lp_refinery.json). --reuse-cold keeps cold-start results over a minute from the existing results file.
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
        return refinery_lp(seed=BENCH_SEED, **REFINERY_SIZES[name.split("-")[1]])
    return read_mps(f"data/netlib/{name}.mps.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-L", action="store_true", help="also the 406k-variable refinery LP (slow)")
    ap.add_argument("--with-XL", action="store_true", help="also the 1.16 M-variable refinery LP, from a 1e-6 GPU point")
    ap.add_argument("--reuse-cold", action="store_true", help="keep cold starts over 60 s from the existing results")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    names = ["refinery-S", "refinery-M"] + (["refinery-L"] if a.with_L else []) + NETLIB +         (["refinery-XL"] if a.with_XL else [])
    prev = {r["name"]: r for r in json.loads(OUT.read_text())} if a.reuse_cold and OUT.exists() else {}
    res = []
    for name in names:
        lp = model(name)
        m, n = lp.K.shape
        tol = 1e-6 if name == "refinery-XL" else 1e-4
        highs = {r["size"]: r["highs"]["best"] for r in json.loads((OUT.parent / "lp_refinery.json").read_text())}
        if name == "refinery-XL":
            ref = highs["XL"]["obj"]
            cold_rec = dict(time=None, iterations=None, status="not attempted", highs_time=highs["XL"]["time"])
        elif name in prev and prev[name]["cold"]["time"] > 60 and name.startswith("refinery-"):
            cold_rec = dict(prev[name]["cold"], reused=True)
            ref = highs[name.split("-")[1]]["obj"]
        else:
            cold = solve_core(lp, time_limit=1800)
            ref = cold.obj
            cold_rec = dict(time=cold.time, iterations=cold.lp_iters, status=cold.status)
        p = PDLP(lp, device=a.device).solve(tol=tol, max_iter=5_000_000, time_limit=300)
        t = time.perf_counter()
        xg = crossover(lp, p.x, p.y, time_limit=1800)
        tg = time.perf_counter() - t
        if n <= 50_000:
            xi = solve_ipm(lp, crossover=True, time_limit=300)
            ipm = dict(time=xi.time, iterations=xi.iterations, crossover_time=xi.crossover_time,
                       **check(lp, ref, xi.status, xi.primal_obj, xi.x))
        else:
            ipm = None
        out = dict(name=name, m=m, n=n, cold=cold_rec,
                   gpu_x=dict(gpu_time=p.solve_time, setup_time=p.setup_time, tol=tol, crossover_time=tg,
                              iterations=xg.lp_iters, obj=xg.obj, **check(lp, ref, xg.status, xg.obj, xg.x)),
                   ipm_x=ipm)
        res.append(out)
        ct = cold_rec["time"]
        print(f"{name:11} n={n:8,}  cold {ct if ct is not None else float('nan'):7.2f}s | gpu {p.solve_time:5.1f}s + crossover {tg:7.2f}s "
              f"err {out['gpu_x']['err']:.0e} | ipm+x "
              + (f"{ipm['time']:6.2f}s err {ipm['err']:.0e}" if ipm and ipm["err"] is not None else "-"), flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
