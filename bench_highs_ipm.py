"""Time HiGHS interior point on the same refinery LPs, then merge into results/bench_all.json,
keeping the FASTER of HiGHS default (dual simplex) and HiGHS IPM as the comparator."""
import json
import sys

from samadhan.baseline import solve_highs
from samadhan.generate import refinery_lp
from bench import SIZES

which = sys.argv[1:] or list(SIZES)
ipm = {}
try:
    ipm = json.load(open("results/highs_ipm.json"))
except FileNotFoundError:
    pass
for key in which:
    lp = refinery_lp(seed=7, **SIZES[key])
    r = solve_highs(lp, solver="ipm", time_limit=900)
    print(key, lp.summary(), r, flush=True)
    ipm[key] = r
    json.dump(ipm, open("results/highs_ipm.json", "w"), indent=1)

rows = []
for f in ("results/bench_S_M.json", "results/bench_L_XL.json"):
    try:
        rows += json.load(open(f))
    except FileNotFoundError:
        pass
for row in rows:
    row["highs_simplex"] = row["highs"]
    cands = [dict(row["highs"], method="default (dual simplex)")]
    if row["size"] in ipm and ipm[row["size"]]["status"] == "Optimal":
        cands.append(dict(ipm[row["size"]], method="interior point"))
    opt = [c for c in cands if c["status"] == "Optimal"] or cands
    row["highs"] = min(opt, key=lambda c: c["time"])
    ref = row["highs"]["obj"]
    for dev in ("cuda", "cpu"):
        if dev in row:
            row[dev]["rel_diff"] = (row[dev]["obj"] - ref) / max(1.0, abs(ref))
json.dump(rows, open("results/bench_all.json", "w"), indent=1)
for r in rows:
    print(r["size"], r["n"], "HiGHS best", r["highs"]["method"], f"{r['highs']['time']:.1f}s",
          "GPU", f"{r['cuda']['time']:.1f}s", f"speedup {r['highs']['time'] / r['cuda']['time']:.1f}x")
