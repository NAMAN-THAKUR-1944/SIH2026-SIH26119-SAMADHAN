"""Parallel tree search: SAMADHAN's branch-and-cut with 1, 4 and 8 threads, same time limit and gap. One model at a
time, so that the threads have the machine to themselves; every run in its own process.

    python -m benchmarks.threads [--threads 1,4,8] [--time-limit 60]             -> results/milp_threads.json
    python -m benchmarks.threads --set mrpl                                      -> results/milp_mrpl_threads.json

miplib3: the MIPLIB 3 models; proven optima are checked against HiGHS's (results/milp_miplib3.json, from
benchmarks/miplib.py). mrpl: the MRPL-shaped models of benchmarks/mrpl.py, every solution checked against the
model's rows, HiGHS's results from results/milp_mrpl.json.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = {"miplib3": (ROOT / "results" / "milp_threads.json", ROOT / "results" / "milp_miplib3.json"),
        "mrpl": (ROOT / "results" / "milp_mrpl_threads.json", ROOT / "results" / "milp_mrpl.json")}


def load(spec):
    """a MIPLIB 3 file, or kind:size:seed of an MRPL-shaped model"""
    if spec.count(":") == 2 and not os.path.exists(spec):
        from benchmarks.mrpl import build
        kind, size, seed = spec.split(":")
        return build(kind, size, int(seed))
    from samadhan.mps import read_mps
    return read_mps(spec)


def run_one(spec, threads, time_limit, gap):
    from samadhan.core import solve_core
    from samadhan.verify import violation
    lp = load(spec)
    r = solve_core(lp, time_limit=time_limit, gap=gap, threads=threads)
    has_x = r.x is not None and math.isfinite(r.obj)
    return dict(status=r.status, obj=r.obj, bound=r.bound, gap=r.gap, nodes=r.nodes, lp_iters=r.lp_iters, time=r.time,
                viol=violation(lp, r.x) if has_x else None)


def sgm(times, limit, shift=10.0):
    """shifted geometric mean of the times (unsolved runs count as the time limit)"""
    return math.exp(sum(math.log(min(t, limit) + shift) for t in times) / len(times)) - shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="miplib3", choices=list(SETS))
    ap.add_argument("--threads", default="1,4,8")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--models", default=None, help="comma-separated subset (MIPLIB names or MRPL model names)")
    ap.add_argument("--one", nargs=2, metavar=("MODEL", "THREADS"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print("RESULT " + json.dumps(run_one(a.one[0], int(a.one[1]), a.time_limit, a.gap), default=float))
        return
    threads = [int(t) for t in a.threads.split(",")]
    out_file, ref_file = SETS[a.set]
    if a.set == "miplib3":
        files = sorted(f for f in glob.glob(str(ROOT / "data" / "miplib3" / "*.gz")) if not f.endswith(".tar.gz"))
        jobs = [(os.path.basename(f).split(".")[0], f) for f in files]
        ref = {r["name"]: r["highs"] for r in json.loads(ref_file.read_text())} if ref_file.exists() else {}
    else:
        from benchmarks.mrpl import build, instances
        jobs = [(build(k, s, sd).name, f"{k}:{s}:{sd}") for k, s, sd in instances()]
        ref = {r["name"]: r["highs"] for r in json.loads(ref_file.read_text())} if ref_file.exists() else {}
    if a.models:
        jobs = [j for j in jobs if j[0] in a.models.split(",")]
    from samadhan.core import _load
    _load()                                         # build the C++ core once, before the runs
    res = []
    for name, spec in jobs:
        row = dict(name=name)
        for k in threads:
            p = subprocess.run([sys.executable, "-m", "benchmarks.threads", "--one", spec, str(k),
                                "--time-limit", str(a.time_limit), "--gap", str(a.gap)], cwd=ROOT,
                               capture_output=True, text=True, timeout=a.time_limit + 600)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            r = json.loads(lines[-1][7:]) if lines else dict(status="crash", obj=None, time=a.time_limit,
                                                              err=p.stderr[-300:])
            h = ref.get(name, {})
            if r["status"] == "optimal" and h.get("status") == "Optimal":
                r["wrong"] = abs(r["obj"] - h["obj"]) / max(1.0, abs(h["obj"])) > a.gap
            row[f"t{k}"] = r
        if ref.get(name):
            row["highs"] = {key: ref[name].get(key) for key in ("status", "obj", "time")}
        res.append(row)
        show = lambda r: f"{r['status'][:8]:8} {r['obj'] if r.get('obj') is not None else math.nan:>13.6g} {r['time']:6.1f}s"
        print(f"{name:12}" + "".join(f" | T{k}: {show(row[f't{k}'])}" for k in threads)
              + (f" | HiGHS {show(dict(row['highs'], status=str(row['highs']['status'])))}" if "highs" in row else ""),
              flush=True)
        out_file.write_text(json.dumps(res, indent=1, default=float))
    print()
    if not res:
        print("no models run")
        return
    for k in threads:
        rs = [r[f"t{k}"] for r in res]
        print(f"{k} threads: proved optimal {sum(r['status'] == 'optimal' for r in rs)}, wrong "
              f"{sum(r.get('wrong', False) for r in rs)}, infeasible points "
              f"{sum((r.get('viol') or 0) > 1e-6 for r in rs)}, shifted geometric mean time "
              f"{sgm([r['time'] for r in rs], a.time_limit):.2f} s")
    # speed-up on the models that every thread count solves and that take the first one at least a second
    k0 = threads[0]
    both = [r for r in res if all(r[f"t{k}"]["status"] == "optimal" for k in threads) and r[f"t{k0}"]["time"] >= 1.0]
    for k in threads[1:]:
        if both:
            g = math.exp(sum(math.log(r[f"t{k0}"]["time"] / r[f"t{k}"]["time"]) for r in both) / len(both))
            print(f"{k} threads vs {k0}: {g:.2f}x geometric mean speed-up on {len(both)} models solved by all, "
                  f">= 1 s with {k0}")


if __name__ == "__main__":
    main()
