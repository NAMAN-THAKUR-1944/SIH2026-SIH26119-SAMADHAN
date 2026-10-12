"""Parallel tree search: SAMADHAN's branch-and-cut on MIPLIB 3 with 1, 4 and 8 threads, same time limit and gap.
One model at a time, so that the threads have the machine to themselves; every run in its own process.

    python -m benchmarks.threads [--threads 1,4,8] [--time-limit 60]        -> results/milp_threads.json

Proven optima are checked against HiGHS's (results/milp_miplib3.json, from benchmarks/miplib.py).
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
OUT = ROOT / "results" / "milp_threads.json"
REF = ROOT / "results" / "milp_miplib3.json"


def run_one(path, threads, time_limit, gap):
    from samadhan.core import solve_core
    from samadhan.mps import read_mps
    r = solve_core(read_mps(path), time_limit=time_limit, gap=gap, threads=threads)
    return dict(status=r.status, obj=r.obj, bound=r.bound, gap=r.gap, nodes=r.nodes, lp_iters=r.lp_iters, time=r.time)


def sgm(times, limit, shift=10.0):
    """shifted geometric mean of the times (unsolved runs count as the time limit)"""
    return math.exp(sum(math.log(min(t, limit) + shift) for t in times) / len(times)) - shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,4,8")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--models", default=None, help="comma-separated subset of the MIPLIB 3 names")
    ap.add_argument("--one", nargs=2, metavar=("FILE", "THREADS"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print("RESULT " + json.dumps(run_one(a.one[0], int(a.one[1]), a.time_limit, a.gap), default=float))
        return
    threads = [int(t) for t in a.threads.split(",")]
    ref = {r["name"]: r["highs"] for r in json.loads(REF.read_text())} if REF.exists() else {}
    files = sorted(f for f in glob.glob(str(ROOT / "data" / "miplib3" / "*.gz")) if not f.endswith(".tar.gz"))
    if a.models:
        files = [f for f in files if os.path.basename(f).split(".")[0] in a.models.split(",")]
    from samadhan.core import _load
    _load()                                         # build the C++ core once, before the runs
    res = []
    for f in files:
        name = os.path.basename(f).split(".")[0]
        row = dict(name=name)
        for k in threads:
            p = subprocess.run([sys.executable, "-m", "benchmarks.threads", "--one", f, str(k),
                                "--time-limit", str(a.time_limit), "--gap", str(a.gap)], cwd=ROOT,
                               capture_output=True, text=True, timeout=a.time_limit + 600)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            r = json.loads(lines[-1][7:]) if lines else dict(status="crash", obj=None, time=a.time_limit,
                                                              err=p.stderr[-300:])
            h = ref.get(name, {})
            if r["status"] == "optimal" and h.get("status") == "Optimal":
                r["wrong"] = abs(r["obj"] - h["obj"]) / max(1.0, abs(h["obj"])) > a.gap
            row[f"t{k}"] = r
        res.append(row)
        show = lambda r: f"{r['status'][:8]:8} {r['obj'] if r.get('obj') is not None else math.nan:>13.6g} {r['time']:6.1f}s"
        print(f"{name:10}" + "".join(f" | T{k}: {show(row[f't{k}'])}" for k in threads), flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))
    print()
    for k in threads:
        rs = [r[f"t{k}"] for r in res]
        print(f"{k} threads: proved optimal {sum(r['status'] == 'optimal' for r in rs)}, wrong "
              f"{sum(r.get('wrong', False) for r in rs)}, shifted geometric mean time "
              f"{sgm([r['time'] for r in rs], a.time_limit):.2f} s")
    # speed-up on the models that every thread count solves and that take the serial search at least a second
    k0 = threads[0]
    both = [r for r in res if all(r[f"t{k}"]["status"] == "optimal" for k in threads) and r[f"t{k0}"]["time"] >= 1.0]
    for k in threads[1:]:
        if both:
            g = math.exp(sum(math.log(r[f"t{k0}"]["time"] / r[f"t{k}"]["time"]) for r in both) / len(both))
            print(f"{k} threads vs {k0}: {g:.2f}x geometric mean speed-up on {len(both)} models solved by all, "
                  f">= 1 s with {k0}")


if __name__ == "__main__":
    main()
