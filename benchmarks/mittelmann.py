"""LP instances from Hans Mittelmann's LPfeas benchmark (plato.asu.edu/ftp/lpfeas.html): SAMADHAN's GPU engine
against HiGHS and NVIDIA cuOpt on the same laptop, each to a relative tolerance of 1e-4 (the LPfeas criterion: a
primal-dual feasible point) within a time limit.

    mkdir data/mittelmann; cd data/mittelmann
    curl -O https://plato.asu.edu/ftp/lptestset/<file> for each file in FILES; bunzip2 *.bz2
    (plain .mps files, named as in FILES without .bz2; datt256_lp.mps is renamed datt256.mps)
    python -m benchmarks.mittelmann [--time-limit 300]           SAMADHAN and HiGHS -> results/lp_mittelmann.json
    python -m benchmarks.mittelmann --cuopt [--time-limit 300]   cuOpt in Docker (nvidia/cuopt:latest-cu13, GPU)

SAMADHAN runs its GPU engine as samadhan.pdlp.solve does by default (presolve, then adaptive-step PDLP on its native
CUDA kernels; the time includes presolve and setup); HiGHS its default LP solver
with default threads; cuOpt its PDLP method (cuOpt 26.08) with absolute and relative tolerances 1e-4. Every
objective is compared with HiGHS's where HiGHS proves optimality.

The cuOpt image is built for CUDA 13.3 and the laptop's driver (592.00) supports CUDA 13.1, so the container starts
with NVIDIA_DISABLE_REQUIRE=1; cuOpt's PDLP runs that way, its default concurrent mode (PDLP, barrier and dual
simplex at once) crashes on start.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mittelmann"
OUT = ROOT / "results" / "lp_mittelmann.json"
CUOPT_IMAGE = "nvidia/cuopt:latest-cu13"
FILES = ("datt256_lp.mps.bz2", "ex10.mps.bz2", "graph40-40.mps.bz2", "irish-electricity.mps.bz2",
         "neos-5052403-cygnet.mps.bz2", "neos-5251015.mps.bz2", "physiciansched3-3.mps.bz2", "qap15.mps.bz2",
         "rmine15.mps.bz2", "s100.mps.bz2", "s250r10.mps.bz2", "savsched1.mps.bz2", "supportcase10.mps.bz2",
         "woodlands09.mps.bz2")


def load():
    return {r["name"]: r for r in json.loads(OUT.read_text())} if OUT.exists() else {}


def save(rows):
    OUT.write_text(json.dumps([rows[k] for k in sorted(rows)], indent=1, default=float))


def rel_err(obj, ref):
    if obj is None or ref is None or not (math.isfinite(obj) and math.isfinite(ref)):
        return None
    return abs(obj - ref) / max(1.0, abs(ref))


def run_native(time_limit, tol, names, reuse_highs=False):
    from samadhan.baseline import solve_highs
    from samadhan.mps import read_mps
    from samadhan.pdlp import solve
    rows = load()
    from samadhan.generate import refinery_lp
    solve(refinery_lp(R=3, C=5, P=4, D=20, T=4, seed=1), "cuda", tol=tol)   # untimed: CUDA context and kernels
    for path in sorted(DATA.glob("*.mps")):
        name = path.stem
        if names and name not in names:
            continue
        lp = read_mps(str(path))
        m, n = lp.K.shape
        print(f"\n== {name}: {m:,} rows, {n:,} columns, {lp.K.nnz:,} nonzeros", flush=True)
        h = rows[name]["highs"] if reuse_highs and "highs" in rows.get(name, {}) else solve_highs(lp, time_limit=time_limit)
        print(f"  HiGHS     {h['status']:>20} {h['obj']:+.10e} {h['time']:8.2f}s", flush=True)
        r = solve(lp, "cuda", tol=tol, time_limit=time_limit, max_iter=10**8)
        ref = h["obj"] if h["status"] == "Optimal" else None
        sam = dict(status=r.status, time=r.solve_time, setup=r.setup_time, iters=r.iterations, obj=r.primal_obj,
                   rel_p=r.rel_primal_res, rel_d=r.rel_dual_res, rel_gap=r.rel_gap, device=r.device,
                   obj_err=rel_err(r.primal_obj, ref))
        print(f"  SAMADHAN  {r.status:>20} {r.primal_obj:+.10e} {r.solve_time:8.2f}s  iters {r.iterations}  "
              f"err {sam['obj_err'] if sam['obj_err'] is not None else float('nan'):.1e}  {r.device}", flush=True)
        row = rows.get(name, {})
        row.update(name=name, m=m, n=n, nnz=int(lp.K.nnz), highs=h, samadhan=sam, time_limit=time_limit, tol=tol)
        if "cuopt" in row and row["cuopt"].get("obj") is not None:
            row["cuopt"]["obj_err"] = rel_err(row["cuopt"]["obj"], ref)
        rows[name] = row
        save(rows)


def run_cuopt(time_limit, tol, names):
    """cuOpt inside its own container on the same GPU; benchmarks/cuopt_runner.py does the solving."""
    rows = load()
    files = [p.name for p in sorted(DATA.glob("*.mps")) if not names or p.stem in names]
    cmd = ["docker", "run", "--rm", "--gpus", "all", "-e", "NVIDIA_DISABLE_REQUIRE=1", "-v", f"{DATA}:/data:ro",
           "-v", f"{ROOT / 'benchmarks'}:/bench:ro", CUOPT_IMAGE, "python", "/bench/cuopt_runner.py", str(time_limit),
           str(tol), "1", *files]
    print(" ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        line = line.rstrip()
        if not line.startswith("RESULT "):
            print(line, flush=True)
            continue
        res = json.loads(line[len("RESULT "):])
        name = res.pop("name")
        row = rows.setdefault(name, dict(name=name))
        ref = row.get("highs", {}).get("obj") if row.get("highs", {}).get("status") == "Optimal" else None
        res["obj_err"] = rel_err(res.get("obj"), ref)
        row["cuopt"] = res
        print(f"  cuOpt {name:22s} {res['status']:>20} {res.get('obj') or float('nan'):+.10e} {res['time']:8.2f}s", flush=True)
        save(rows)
    return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="instances (default: every data/mittelmann/*.mps)")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--cuopt", action="store_true", help="run cuOpt in Docker instead of SAMADHAN and HiGHS")
    ap.add_argument("--reuse-highs", action="store_true", help="keep HiGHS results from the existing results file")
    a = ap.parse_args()
    if a.cuopt:
        raise SystemExit(run_cuopt(a.time_limit, a.tol, set(a.names)))
    run_native(a.time_limit, a.tol, set(a.names), a.reuse_highs)


if __name__ == "__main__":
    main()
