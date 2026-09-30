"""Benchmark behind the README table and the SIH26119 deck -> results/lp_refinery.json.

For every refinery planning LP size and two accuracy targets (1e-4 and 1e-6 relative KKT), SAMADHAN runs both
of its GPU methods (adaptive-step PDLP, and constant-step PDLP replayed as CUDA graphs). HiGHS runs both of its
methods (dual simplex, the default, and interior point). Each solver is credited with its faster method; every
raw timing is kept.

    python -m benchmarks.lp                  # all sizes (HiGHS simplex on XL alone takes 15 min)
    python -m benchmarks.lp M L --reuse-highs  # rerun SAMADHAN only, keep HiGHS timings from results/lp_refinery.json
"""
import argparse
import json
from pathlib import Path

from samadhan.baseline import solve_highs
from samadhan.generate import BENCH_SEED, REFINERY_SIZES, refinery_lp
from samadhan.pdlp import PDLP

OUT = Path(__file__).resolve().parent.parent / "results" / "lp_refinery.json"


def run_highs(lp, time_limit):
    runs = {"simplex": solve_highs(lp, time_limit=time_limit), "ipm": solve_highs(lp, solver="ipm", time_limit=time_limit)}
    runs["simplex"]["method"], runs["ipm"]["method"] = "dual simplex", "interior point"
    ok = [r for r in runs.values() if r["status"] == "Optimal"] or [runs["simplex"]]
    return dict(simplex=runs["simplex"], ipm=runs["ipm"], best=min(ok, key=lambda r: r["time"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", default=list(REFINERY_SIZES), choices=list(REFINERY_SIZES))
    ap.add_argument("--reuse-highs", action="store_true", help="take HiGHS timings from the existing results file")
    ap.add_argument("--time-limit", type=float, default=900.0)
    a = ap.parse_args()

    rows = {r["size"]: r for r in json.loads(OUT.read_text())} if OUT.exists() else {}
    for key in a.sizes:
        lp = refinery_lp(seed=BENCH_SEED, **REFINERY_SIZES[key])
        m, n = lp.K.shape
        print(f"\n== {key}: {lp.summary()}", flush=True)
        highs = rows[key]["highs"] if a.reuse_highs and key in rows else run_highs(lp, a.time_limit)
        print(f"  HiGHS best: {highs['best']['method']} {highs['best']['time']:.1f}s", flush=True)
        ref = highs["best"]["obj"]
        solver = PDLP(lp, "cuda")
        row = dict(size=key, n=n, m=m, nnz=int(lp.K.nnz), highs=highs, samadhan={})
        for tol in (1e-4, 1e-6):
            runs = {}
            for mode, adaptive in (("adaptive", True), ("graph", False)):
                r = solver.solve(tol=tol, adaptive=adaptive, time_limit=a.time_limit, max_iter=10**7)
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
        rows[key] = row
        OUT.write_text(json.dumps([rows[k] for k in REFINERY_SIZES if k in rows], indent=1))


if __name__ == "__main__":
    main()
