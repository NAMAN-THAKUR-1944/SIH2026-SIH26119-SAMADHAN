"""Command line:
    python -m samadhan solve model.mps [--engine auto|gpu|core] [--device cuda|cpu] [--tol 1e-6]
    python -m samadhan demo [--size S|M|L|XL]      (generated refinery planning LP on the GPU engine)

--engine auto (default) sends models with integer variables to the C++ branch-and-cut core and all other
models to the GPU engine.
"""
import argparse
import sys


def _gpu(lp, a):
    from .pdlp import PDLP
    r = PDLP(lp, device=a.device).solve(tol=a.tol, time_limit=a.time_limit, verbose=not a.quiet,
                                         adaptive=a.method == "adaptive")
    print(f"\nstatus      {r.status}\nobjective   {r.primal_obj:.10g}\ndual bound  {r.dual_obj:.10g}\n"
          f"rel. gap    {r.rel_gap:.2e}   primal res {r.rel_primal_res:.2e}   dual res {r.rel_dual_res:.2e}\n"
          f"iterations  {r.iterations}   restarts {r.restarts}\n"
          f"time        {r.solve_time:.2f} s solve + {r.setup_time:.2f} s setup on {r.device}")
    return r.status == "optimal"


def _core(lp, a):
    from .core import solve_core
    r = solve_core(lp, time_limit=a.time_limit, verbose=not a.quiet)
    print(f"\nstatus      {r.status}\nobjective   {r.obj:.10g}\nbound       {r.bound:.10g}\n"
          f"gap         {r.gap:.2e}   nodes {r.nodes}   simplex iterations {r.lp_iters}   cuts {r.cuts}\n"
          f"time        {r.time:.2f} s (C++ core)")
    return r.status == "optimal"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="samadhan", description="SAMADHAN: sovereign LP / QP / MILP solver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="solve an LP or MILP from an MPS file")
    s.add_argument("file")
    s.add_argument("--engine", default="auto", choices=["auto", "gpu", "core"],
                   help="auto: C++ core for models with integer variables, GPU engine otherwise")
    d = sub.add_parser("demo", help="solve a generated refinery planning LP")
    d.add_argument("--size", default="S", choices=["S", "M", "L", "XL"])
    for p in (s, d):
        p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="GPU engine device")
        p.add_argument("--tol", type=float, default=1e-4, help="GPU engine: relative KKT tolerance")
        p.add_argument("--method", default="adaptive", choices=["adaptive", "graph"],
                       help="GPU engine: adaptive-step PDLP, or constant-step PDLP replayed as CUDA graphs")
        p.add_argument("--time-limit", type=float, default=600.0)
        p.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "solve":
        from .mps import read_mps
        lp = read_mps(a.file)
        engine = a.engine
        if engine == "auto":
            engine = "core" if lp.integer is not None and lp.integer.any() else "gpu"
    else:
        from .generate import BENCH_SEED, REFINERY_SIZES, refinery_lp
        lp = refinery_lp(seed=BENCH_SEED, **REFINERY_SIZES[a.size])
        engine = "gpu"
    print(lp.summary(), f"-> {'C++ core' if engine == 'core' else 'GPU engine'}")
    ok = _core(lp, a) if engine == "core" else _gpu(lp, a)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
