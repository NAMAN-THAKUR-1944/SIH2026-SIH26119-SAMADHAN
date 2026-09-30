"""Command line:
    python -m samadhan solve model.mps [--device cuda|cpu] [--tol 1e-6] [--method adaptive|graph]
    python -m samadhan demo [--size S|M|L|XL]      (generated refinery planning LP)
"""
import argparse
import sys

from .pdlp import PDLP


def main(argv=None):
    ap = argparse.ArgumentParser(prog="samadhan", description="SAMADHAN sovereign GPU optimization engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="solve an LP from an MPS file")
    s.add_argument("file")
    d = sub.add_parser("demo", help="solve a generated refinery planning LP")
    d.add_argument("--size", default="S", choices=["S", "M", "L", "XL"])
    for p in (s, d):
        p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
        p.add_argument("--tol", type=float, default=1e-4, help="relative KKT tolerance")
        p.add_argument("--method", default="adaptive", choices=["adaptive", "graph"],
                       help="adaptive-step PDLP, or constant-step PDLP replayed as CUDA graphs")
        p.add_argument("--time-limit", type=float, default=600.0)
        p.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "solve":
        from .mps import read_mps
        lp = read_mps(a.file)
    else:
        from .generate import BENCH_SEED, REFINERY_SIZES, refinery_lp
        lp = refinery_lp(seed=BENCH_SEED, **REFINERY_SIZES[a.size])
    print(lp.summary())
    r = PDLP(lp, device=a.device).solve(tol=a.tol, time_limit=a.time_limit, verbose=not a.quiet,
                                         adaptive=a.method == "adaptive")
    print(f"\nstatus      {r.status}\nobjective   {r.primal_obj:.10g}\ndual bound  {r.dual_obj:.10g}\n"
          f"rel. gap    {r.rel_gap:.2e}   primal res {r.rel_primal_res:.2e}   dual res {r.rel_dual_res:.2e}\n"
          f"iterations  {r.iterations}   restarts {r.restarts}\n"
          f"time        {r.solve_time:.2f} s solve + {r.setup_time:.2f} s setup on {r.device}")
    return 0 if r.status == "optimal" else 1


if __name__ == "__main__":
    sys.exit(main())
