"""MIPLIB 3 benchmark: SAMADHAN C++ branch-and-cut vs HiGHS, same time limit and gap, one thread each.

    git clone https://github.com/coin-or-tools/Data-miplib3 data/miplib3
    python -m benchmarks.miplib [--time-limit 60] [--workers 6]      -> results/milp_miplib3.json
"""
import argparse
import glob
import gzip
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "milp_miplib3.json"

MAX_ROWS = 1500   # the prototype keeps a dense basis inverse


def run_one(path, time_limit, gap):
    import highspy
    import numpy as np

    from samadhan.core import solve_core
    from samadhan.mps import read_mps

    name = os.path.basename(path).split(".")[0]
    out = dict(name=name)
    try:
        lp = read_mps(path)
        m, n = lp.K.shape
        out.update(n=n, m=m, nnz=int(lp.K.nnz), ints=int(lp.integer.sum()) if lp.integer is not None else 0)
        if m > MAX_ROWS:
            out["skipped"] = f"{m} rows > {MAX_ROWS}"
            return out
        tmp = os.path.join(tempfile.gettempdir(), f"samadhan_mip_{name}.mps")
        with gzip.open(path, "rb") as s, open(tmp, "wb") as d:
            shutil.copyfileobj(s, d)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", gap)
        h.readModel(tmp)
        # parser check: bounds and integrality exactly as HiGHS reads them
        hl = h.getLp()
        hlo, hup = np.array(hl.col_lower_), np.array(hl.col_upper_)
        hint = np.array([int(v) == 1 for v in hl.integrality_]) if len(hl.integrality_) else np.zeros(n, bool)
        ours_int = lp.integer if lp.integer is not None else np.zeros(n, bool)
        same = lambda a, b: np.allclose(np.where(np.isinf(a), 1e30 * np.sign(a), a),
                                        np.where(np.abs(b) >= 1e30, 1e30 * np.sign(b), b))
        out["parser_match"] = bool(same(lp.l, hlo) and same(lp.u, hup) and np.array_equal(ours_int, hint))
        t0 = time.perf_counter(); h.run(); ht = time.perf_counter() - t0
        info = h.getInfo()
        out["highs"] = dict(status=h.modelStatusToString(h.getModelStatus()), obj=info.objective_function_value,
                            bound=info.mip_dual_bound, gap=info.mip_gap, nodes=int(info.mip_node_count), time=ht)
        r = solve_core(lp, time_limit=time_limit, gap=gap)
        out["samadhan"] = dict(status=r.status, obj=r.obj, bound=r.bound, gap=r.gap, nodes=r.nodes,
                               lp_iters=r.lp_iters, time=r.time, cuts=r.cuts)
        if out["highs"]["status"] == "Optimal" and r.x is not None:
            ref = out["highs"]["obj"]
            out["obj_rel_diff"] = abs(r.obj - ref) / max(1.0, abs(ref))
    except Exception as e:
        out["error"] = repr(e)[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob("data/miplib3/*.gz") if not f.endswith(".tar.gz"))
    from samadhan.core import _load
    _load()   # build the C++ core once before forking workers
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, f, a.time_limit, a.gap) for f in files]
        for fu in as_completed(futs):
            r = fu.result()
            res.append(r)
            s, h = r.get("samadhan", {}), r.get("highs", {})
            print(f"{r['name']:10} m={r.get('m', 0):5} n={r.get('n', 0):6}  "
                  f"SAMADHAN {s.get('status', r.get('skipped', r.get('error', '?')))[:22]:22} "
                  f"{s.get('obj', float('nan')):>14.6g} {s.get('time', 0):6.1f}s  | "
                  f"HiGHS {h.get('status', '-')[:18]:18} {h.get('obj', float('nan')):>14.6g} {h.get('time', 0):6.1f}s"
                  f"  parser={r.get('parser_match')}", flush=True)
            OUT.write_text(json.dumps(sorted(res, key=lambda x: x["name"]), indent=1, default=float))
    ran = [r for r in res if "samadhan" in r]
    opt = [r for r in ran if r["samadhan"]["status"] == "optimal"]
    ok = [r for r in opt if r.get("obj_rel_diff", 1) <= 1e-4]
    hopt = [r for r in ran if r["highs"]["status"] == "Optimal"]
    print(f"\nran {len(ran)}: SAMADHAN proved optimal {len(opt)} ({len(ok)} matching HiGHS), HiGHS {len(hopt)}")


if __name__ == "__main__":
    main()
