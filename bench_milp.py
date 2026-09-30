"""MILP check: SAMADHAN branch-and-bound (own simplex) vs HiGHS MILP on refinery contract/refinery-selection MILPs."""
import json

from samadhan.baseline import solve_highs
from samadhan.generate import refinery_milp
from samadhan.milp import solve_milp

rows = []
for seed in range(5):
    for R, C, P, D in [(3, 6, 3, 5), (4, 8, 3, 6)]:
        lp = refinery_milp(R=R, C=C, P=P, D=D, seed=seed)
        r = solve_milp(lp, time_limit=300)
        h = solve_highs(lp)
        diff = abs(r.obj - h["obj"]) / abs(h["obj"])
        rows.append(dict(name=lp.name, n=lp.K.shape[1], m=lp.K.shape[0], ints=int(lp.integer.sum()),
                         status=r.status, obj=r.obj, nodes=r.nodes, time=r.time,
                         highs_obj=h["obj"], highs_time=h["time"], rel_diff=diff))
        print(f"{lp.name:32} ours {r.status} {r.obj:.4f} ({r.nodes} nodes, {r.time:.2f}s)  "
              f"HiGHS {h['obj']:.4f}  diff {diff:.1e}", flush=True)
json.dump(rows, open("results/milp.json", "w"), indent=1)
ok = sum(1 for r in rows if r["status"] == "optimal" and r["rel_diff"] < 1e-9)
print(f"\n{ok}/{len(rows)} proven optimal and equal to HiGHS")
