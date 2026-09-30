"""SAMADHAN benchmark: GPU PDLP vs CPU PDLP vs HiGHS on refinery planning LPs of growing size."""
import json, sys, time
import numpy as np
from samadhan.generate import refinery_lp
from samadhan.pdlp import solve
from samadhan.baseline import solve_highs

SIZES = {
    "S":  dict(R=4,  C=8,  P=6, D=60,  T=6),
    "M":  dict(R=8,  C=10, P=6, D=150, T=12),
    "L":  dict(R=12, C=12, P=8, D=300, T=12),
    "XL": dict(R=16, C=12, P=8, D=500, T=16),
}
if __name__ == "__main__":
    which = sys.argv[1:] or list(SIZES)
    tol = 1e-4
    out = []
    for key in which:
        lp = refinery_lp(seed=7, **SIZES[key])
        m, n = lp.K.shape
        print(f"\n== {key}: {lp.summary()}", flush=True)
        hb = solve_highs(lp, time_limit=900)
        print(f"HiGHS  {hb['status']:>10}  obj {hb['obj']:.6e}  {hb['time']:.2f}s", flush=True)
        row = dict(size=key, n=n, m=m, nnz=int(lp.K.nnz), highs=hb)
        for dev in (["cuda", "cpu"] if n < 400_000 else ["cuda"]):
            r = solve(lp, device=dev, tol=tol, time_limit=900, max_iter=500_000)
            rel = (r.primal_obj - hb["obj"]) / max(1.0, abs(hb["obj"]))
            print(f"{dev:5}  {r.status:>10}  obj {r.primal_obj:.6e}  rel.diff {rel:+.1e}  iters {r.iterations}  "
                  f"{r.solve_time:.2f}s (+{r.setup_time:.2f}s setup)", flush=True)
            row[dev] = dict(status=r.status, obj=r.primal_obj, rel_diff=rel, iters=r.iterations,
                            time=r.solve_time, setup=r.setup_time, rel_gap=r.rel_gap,
                            rel_p=r.rel_primal_res, rel_d=r.rel_dual_res)
        out.append(row)
        json.dump(out, open(f"results/bench_{'_'.join(which)}.json", "w"), indent=1)
