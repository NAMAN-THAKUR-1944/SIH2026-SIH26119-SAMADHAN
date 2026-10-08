"""MRPL-shaped MILPs: crude oil scheduling at the marine terminal and multi-period refinery planning with crude
cargoes and unit modes (samadhan/crude.py, samadhan/generate.py). SAMADHAN's C++ core against HiGHS, one thread
each, time limit and gap as for MIPLIB; every SAMADHAN solution is checked against the model's rows.

    python -m benchmarks.mrpl [--time-limit 60] [--workers 6]        -> results/milp_mrpl.json
"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "milp_mrpl.json"
SEEDS = (7, 8, 9)


def instances():
    from samadhan.crude import CRUDE_SIZES
    from samadhan.generate import PLAN_SIZES
    out = []
    for size in CRUDE_SIZES:
        out += [("crude", size, s) for s in SEEDS]
    for size in PLAN_SIZES:
        out += [("plan", size, s) for s in SEEDS]
    return out


def build(kind, size, seed):
    from samadhan.crude import CRUDE_SIZES, crude_schedule
    from samadhan.generate import PLAN_SIZES, refinery_plan
    return crude_schedule(**CRUDE_SIZES[size], seed=seed) if kind == "crude" else \
        refinery_plan(**PLAN_SIZES[size], seed=seed)


def run_one(args):
    kind, size, seed, time_limit, gap = args
    from samadhan.baseline import solve_highs
    from samadhan.core import solve_core
    from samadhan.verify import violation
    lp = build(kind, size, seed)
    m, n = lp.K.shape
    h = solve_highs(lp, time_limit=time_limit, threads=1, gap=gap)
    r = solve_core(lp, time_limit=time_limit, gap=gap)
    out = dict(model=kind, size=size, seed=seed, name=lp.name, n=n, m=m, ints=int(lp.integer.sum()),
               highs=h, samadhan=dict(status=r.status, obj=r.obj, bound=r.bound, gap=r.gap, nodes=r.nodes,
                                      time=r.time,
                                      viol=violation(lp, r.x) if r.x is not None and math.isfinite(r.obj) else None))
    if h["status"] == "Optimal" and r.x is not None:
        out["obj_rel_diff"] = abs(r.obj - h["obj"]) / max(1.0, abs(h["obj"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from samadhan.core import _load
    _load()   # build the C++ core once before starting the workers
    jobs = [(k, s, sd, a.time_limit, a.gap) for k, s, sd in instances()]
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_one, jobs):
            res.append(r)
            h, s = r["highs"], r["samadhan"]
            fmt = lambda v: f"{v:14.6g}" if v is not None and math.isfinite(v) else f"{'-':>14}"
            print(f"{r['name']:34} n={r['n']:6d} ints={r['ints']:5d}  SAMADHAN {s['status']:17} {fmt(s['obj'])} "
                  f"{s['time']:5.1f}s | HiGHS {h['status']:18} {fmt(h['obj'])} {h['time']:5.1f}s", flush=True)
            OUT.write_text(json.dumps(res, indent=1, default=float))
    opt = sum(r["samadhan"]["status"] == "optimal" for r in res)
    hopt = sum(r["highs"]["status"] == "Optimal" for r in res)
    print(f"\nran {len(res)}: SAMADHAN proved optimal {opt}, HiGHS {hopt}")


if __name__ == "__main__":
    main()
