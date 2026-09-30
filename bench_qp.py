"""QP benchmarks: SAMADHAN QP (restarted PDHG with a quadratic term) vs the HiGHS QP solver.

    python bench_qp.py maros   [--time-limit 60] [--workers 8]   -> results/qp_maros.json   (CPU, Maros-Meszaros)
    python bench_qp.py refinery                                  -> results/qp_refinery.json (GPU, refinery QP)

Maros-Meszaros .mat files: https://github.com/qpsolvers/maros_meszaros_qpbenchmark (data/, files < 300 KB).
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def maros_one(path, time_limit, tol):
    import torch
    torch.set_num_threads(1)
    from samadhan.baseline import solve_highs
    from samadhan.pdlp import PDLP
    from samadhan.qpdata import read_maros

    name = os.path.basename(path).split(".")[0]
    out = dict(name=name)
    try:
        lp = read_maros(path, name)
        m, n = lp.K.shape
        out.update(n=n, m=m, nnz=int(lp.K.nnz), qnnz=int(lp.Q.nnz))
        h = solve_highs(lp, time_limit=time_limit, threads=1)
        out["highs"] = h
        r = PDLP(lp, "cpu").solve(tol=tol, time_limit=time_limit, max_iter=10**8)
        out["samadhan"] = dict(status=r.status, obj=r.primal_obj, time=r.solve_time, iters=r.iterations,
                               rel_gap=r.rel_gap, rel_p=r.rel_primal_res, rel_d=r.rel_dual_res)
        if h["status"] == "Optimal":
            out["obj_rel_diff"] = abs(r.primal_obj - h["obj"]) / max(1.0, abs(h["obj"]))
    except Exception as e:
        out["error"] = repr(e)[:300]
    return out


def maros_isolated(path, time_limit, tol):
    """Run one problem in its own process, so a crash in any solver cannot take the benchmark down."""
    import subprocess, sys
    name = os.path.basename(path).split(".")[0]
    try:
        p = subprocess.run([sys.executable, "-W", "ignore", __file__, "_one", path, str(time_limit), str(tol)],
                           capture_output=True, text=True, timeout=4 * time_limit + 120)
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if p.returncode == 0 and lines:
            return json.loads(lines[-1])
        return dict(name=name, error=f"process exited with code {p.returncode}")
    except subprocess.TimeoutExpired:
        return dict(name=name, error="process timeout")


def maros(a):
    from concurrent.futures import ThreadPoolExecutor
    files = sorted(glob.glob("data/maros/*.mat"))
    if a.only:
        files = [f for f in files if os.path.basename(f).split(".")[0] in a.only]
    prev = {}
    if a.only and os.path.exists("results/qp_maros.json"):
        prev = {r["name"]: r for r in json.load(open("results/qp_maros.json"))}
    res = []
    with ThreadPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(maros_isolated, f, a.time_limit, a.tol) for f in files]):
            r = fu.result()
            res.append(r)
            s, h = r.get("samadhan", {}), r.get("highs", {})
            print(f"{r['name']:10} n={r.get('n', 0):6} m={r.get('m', 0):6}  SAMADHAN {s.get('status', r.get('error', '?'))[:16]:16}"
                  f" {s.get('time', 0):6.1f}s  HiGHS {h.get('status', '-')[:14]:14} {h.get('time', 0):6.1f}s"
                  f"  diff {r.get('obj_rel_diff', float('nan')):.1e}", flush=True)
            prev[r["name"]] = r
            json.dump(sorted(prev.values(), key=lambda x: x["name"]), open("results/qp_maros.json", "w"), indent=1,
                      default=float)
    res = list(prev.values())
    ok = [r for r in res if r.get("samadhan", {}).get("status") == "optimal"]
    hok = [r for r in res if r.get("highs", {}).get("status") == "Optimal"]
    both = [r for r in ok if "obj_rel_diff" in r]
    print(f"\n{len(res)} QPs: SAMADHAN solved {len(ok)} to {a.tol:g} rel. KKT, HiGHS {len(hok)}; "
          f"{sum(r['obj_rel_diff'] <= 1e-4 for r in both)}/{len(both)} of the common ones agree to 1e-4")


def refinery(a):
    from samadhan.baseline import solve_highs
    from samadhan.generate import BENCH_SEED, REFINERY_SIZES, refinery_lp
    from samadhan.pdlp import PDLP
    rows = []
    for key in a.sizes:
        lp = refinery_lp(seed=BENCH_SEED, quad=0.3, **REFINERY_SIZES[key])
        print(f"\n== {key}: {lp.summary()}", flush=True)
        h = solve_highs(lp, time_limit=a.time_limit)
        print(f"  HiGHS QP {h['status']} {h['time']:.1f}s", flush=True)
        solver = PDLP(lp, "cuda")
        r = solver.solve(tol=a.tol, time_limit=a.time_limit, max_iter=10**8)
        err = abs(r.primal_obj - h["obj"]) / abs(h["obj"]) if h["status"] == "Optimal" else None
        print(f"  SAMADHAN {r.status} {r.solve_time:.1f}s iters {r.iterations} err {err}", flush=True)
        rows.append(dict(size=key, n=lp.K.shape[1], m=lp.K.shape[0], nnz=int(lp.K.nnz), highs=h,
                         samadhan=dict(status=r.status, obj=r.primal_obj, time=r.solve_time, iters=r.iterations,
                                       obj_err=err)))
        json.dump(rows, open("results/qp_refinery.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "_one":        # worker mode: one problem, JSON on stdout
        print(json.dumps(maros_one(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])), default=float))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["maros", "refinery"])
    ap.add_argument("sizes", nargs="*", default=["S", "M", "L", "XL"])
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=[], help="rerun only these problems (merged into the results)")
    a = ap.parse_args()
    maros(a) if a.which == "maros" else refinery(a)
