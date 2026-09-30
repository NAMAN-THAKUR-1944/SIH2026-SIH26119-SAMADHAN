import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, time
from samadhan.generate import refinery_lp
from samadhan.pdlp import solve
from samadhan.baseline import solve_highs
lp = refinery_lp(R=3, C=5, P=4, D=20, T=4, seed=1)
print(lp.summary())
hb = solve_highs(lp); print("HiGHS", hb)
for dev in ["cuda", "cpu"]:
    r = solve(lp, device=dev, tol=1e-6, verbose=(dev=="cuda"), eval_every=64, max_iter=50000)
    print(dev, r.status, f"obj {r.primal_obj:.6f} gap_vs_highs {(r.primal_obj-hb['obj'])/abs(hb['obj']):.2e} iters {r.iterations} restarts {r.restarts} time {r.solve_time:.2f}s")
