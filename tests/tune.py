import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from samadhan.generate import refinery_lp
from samadhan.pdlp import PDLP
from bench import SIZES
ref = {r["size"]: r["highs"]["obj"] for r in json.load(open("results/bench_all.json"))}
for key in sys.argv[1:]:
    lp = refinery_lp(seed=7, **SIZES[key])
    solver = PDLP(lp, "cuda")
    for adaptive, tol in [(True, 1e-4), (False, 1e-4), (True, 1e-6), (False, 1e-6)]:
        r = solver.solve(tol=tol, adaptive=adaptive, time_limit=300, max_iter=1_000_000)
        print(f"{key} {'adaptive' if adaptive else 'constant'} tol={tol:g} {r.status} {r.solve_time:7.2f}s "
              f"iters={r.iterations} obj_err={abs(r.primal_obj - ref[key]) / abs(ref[key]):.1e}", flush=True)
