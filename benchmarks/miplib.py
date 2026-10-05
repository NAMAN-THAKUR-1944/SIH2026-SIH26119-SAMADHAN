"""MIPLIB benchmark: SAMADHAN C++ branch-and-cut vs HiGHS, same time limit and gap, one thread each.

    git clone https://github.com/coin-or-tools/Data-miplib3 data/miplib3
    python -m benchmarks.miplib [--time-limit 60] [--workers 6] [--reuse-highs]   -> results/milp_miplib3.json

    # MIPLIB 2017: the 50 instances listed in MIPLIB2017 below, from https://miplib.zib.de/WebData/instances/
    python -m benchmarks.miplib --set miplib2017                                    -> results/milp_miplib2017.json

--reuse-highs keeps the HiGHS results already in the output file and reruns only SAMADHAN.
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

RESULTS = Path(__file__).resolve().parent.parent / "results"
SETS = {"miplib3": ("data/miplib3/*.gz", RESULTS / "milp_miplib3.json"),
        "miplib2017": ("data/miplib2017/*.mps.gz", RESULTS / "milp_miplib2017.json")}
# MIPLIB 2017 benchmark-set instances used here (each <name>.mps.gz from miplib.zib.de/WebData/instances/)
MIPLIB2017 = [
    "30n20b8", "50v-10", "air05", "beasleyC3", "binkar10_1", "bppc4-08", "dano3_3", "dano3_5", "eil33-2",
    "enlight_hard", "ex9", "gen-ip054", "glass4", "graph20-20-1rand", "h80x6320d", "markshare_4_0", "mas74", "mas76",
    "mik-250-20-75-4", "n5-3", "neos-662469", "neos-787933", "neos-860300", "neos-911970", "neos-933966",
    "neos-957323", "neos17", "neos5", "neos8", "net12", "ns1830653", "nu25-pr12", "nw04", "p200x1188c", "pg",
    "pg5_34", "pk1", "qap10", "rail507", "ran14x18-disj-8", "rmatr100-p10", "rococoC10-001000", "roi2alpha3n4",
    "sct2", "seymour1", "sp150x300d", "swath1", "swath3", "timtab1", "tr12-30",
]


def run_one(path, time_limit, gap, highs_prev=None):
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
        if highs_prev:
            out["highs"] = highs_prev
        else:
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
    ap.add_argument("--reuse-highs", action="store_true", help="keep HiGHS results from the existing output file")
    ap.add_argument("--set", default="miplib3", choices=list(SETS))
    a = ap.parse_args()
    pattern, OUT = SETS[a.set]
    files = sorted(f for f in glob.glob(pattern) if not f.endswith(".tar.gz"))
    prev = {}
    if a.reuse_highs and OUT.exists():
        prev = {r["name"]: r["highs"] for r in json.loads(OUT.read_text()) if "highs" in r}
    from samadhan.core import _load
    _load()   # build the C++ core once before forking workers
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, f, a.time_limit, a.gap, prev.get(os.path.basename(f).split(".")[0]))
                for f in files]
        for fu in as_completed(futs):
            r = fu.result()
            res.append(r)
            s, h = r.get("samadhan", {}), r.get("highs", {})
            print(f"{r['name']:10} m={r.get('m', 0):5} n={r.get('n', 0):6}  "
                  f"SAMADHAN {s.get('status', r.get('error', '?'))[:22]:22} "
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
