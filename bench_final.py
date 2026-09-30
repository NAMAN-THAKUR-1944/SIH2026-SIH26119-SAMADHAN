"""Final benchmark used in the SIH26119 deck.

For every refinery planning LP size and two accuracy targets (1e-4 and 1e-6 relative KKT), SAMADHAN runs
both of its GPU methods (adaptive-step PDLP, constant-step PDLP replayed as CUDA graphs). HiGHS runs both of
its methods (dual simplex default, interior point). Each solver is credited with its faster method, and all
raw timings are kept in results/final.json.
"""
import json
import sys

from samadhan.baseline import solve_highs
from samadhan.generate import refinery_lp
from samadhan.pdlp import PDLP
from bench import SIZES

which = sys.argv[1:] or list(SIZES)
prev = {r["size"]: r for r in json.load(open("results/bench_all.json"))}
rows = []
for key in which:
    lp = refinery_lp(seed=7, **SIZES[key])
    m, n = lp.K.shape
    print(f"\n== {key}: {lp.summary()}", flush=True)
    if key in prev:  # HiGHS timings already measured (simplex 900 s limit + IPM); reuse
        p = prev[key]
        highs = dict(simplex=p["highs_simplex"], ipm=None, best=p["highs"])
    else:
        s_, i_ = solve_highs(lp, time_limit=900), solve_highs(lp, solver="ipm", time_limit=900)
        best = min([c for c in (s_, i_) if c["status"] == "Optimal"] or [s_], key=lambda c: c["time"])
        highs = dict(simplex=s_, ipm=i_, best=best)
    ref = highs["best"]["obj"]
    solver = PDLP(lp, "cuda")
    row = dict(size=key, n=n, m=m, nnz=int(lp.K.nnz), highs=highs, samadhan={})
    for tol in (1e-4, 1e-6):
        runs = {}
        for mode, adaptive in (("adaptive", True), ("graph", False)):
            r = solver.solve(tol=tol, adaptive=adaptive, time_limit=900, max_iter=10**7)
            err = abs(r.primal_obj - ref) / abs(ref)
            runs[mode] = dict(status=r.status, time=r.solve_time, iters=r.iterations, obj=r.primal_obj,
                              obj_err=err, rel_gap=r.rel_gap, rel_p=r.rel_primal_res, rel_d=r.rel_dual_res)
            print(f"  tol {tol:g} {mode:8} {r.status:>8} {r.solve_time:7.2f}s  iters {r.iterations:7d}  "
                  f"cost err {err:.1e}", flush=True)
        ok = [v | dict(mode=k) for k, v in runs.items() if v["status"] == "optimal"]
        best = min(ok, key=lambda v: v["time"]) if ok else None
        row["samadhan"][f"{tol:g}"] = dict(runs=runs, best=best)
        if best:
            print(f"  -> best {best['mode']} {best['time']:.2f}s vs HiGHS {highs['best']['time']:.1f}s "
                  f"({highs['best']['time'] / best['time']:.1f}x)", flush=True)
    rows.append(row)
    json.dump(rows, open("results/final.json", "w"), indent=1)
