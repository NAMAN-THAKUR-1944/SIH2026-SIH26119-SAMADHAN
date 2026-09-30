"""Refinery MILP check: both SAMADHAN MILP engines vs HiGHS.

The C++ core (dual simplex + branch-and-cut) is the production engine; the pure-Python branch-and-bound on a
dense simplex (samadhan/milp.py) is a reference implementation used as an independent cross-check.

    python -m benchmarks.milp        -> results/milp_refinery.json
"""
import json
from pathlib import Path

from samadhan.baseline import solve_highs
from samadhan.core import solve_core
from samadhan.generate import refinery_milp
from samadhan.milp import solve_milp

OUT = Path(__file__).resolve().parent.parent / "results" / "milp_refinery.json"


def rel_diff(obj, ref):
    return abs(obj - ref) / abs(ref)


def main():
    rows = []
    for seed in range(5):
        for R, C, P, D in [(3, 6, 3, 5), (4, 8, 3, 6)]:
            lp = refinery_milp(R=R, C=C, P=P, D=D, seed=seed)
            h = solve_highs(lp)
            core = solve_core(lp, time_limit=300)
            ref = solve_milp(lp, time_limit=300)
            d_core, d_ref = rel_diff(core.obj, h["obj"]), rel_diff(ref.obj, h["obj"])
            rows.append(dict(name=lp.name, n=lp.K.shape[1], m=lp.K.shape[0], ints=int(lp.integer.sum()),
                             status=core.status, obj=core.obj, nodes=core.nodes, time=core.time, cuts=core.cuts,
                             rel_diff=d_core, highs_obj=h["obj"], highs_time=h["time"],
                             reference=dict(status=ref.status, obj=ref.obj, nodes=ref.nodes, time=ref.time,
                                            rel_diff=d_ref)))
            print(f"{lp.name:32} C++ core {core.status} {core.obj:.4f} ({core.time:.3f}s)  "
                  f"reference {ref.status} ({ref.time:.2f}s)  HiGHS {h['obj']:.4f}  "
                  f"diff {d_core:.1e} / {d_ref:.1e}", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    ok = sum(r["status"] == "optimal" and r["rel_diff"] < 1e-9 for r in rows)
    ok_ref = sum(r["reference"]["status"] == "optimal" and r["reference"]["rel_diff"] < 1e-9 for r in rows)
    print(f"\nC++ core: {ok}/{len(rows)} proven optimal and equal to HiGHS; reference: {ok_ref}/{len(rows)}")


if __name__ == "__main__":
    main()
