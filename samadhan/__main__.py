"""Command line:
    python -m samadhan solve model.mps [--engine auto|gpu|core|ipm] [--device cuda|cpu] [--tol 1e-6]
    python -m samadhan demo [--size S|M|L|XL]      (generated refinery planning LP on the GPU engine)
    python -m samadhan demo --model plan|crude [--size S|M|L]   (MRPL-shaped MILPs on the C++ core, 60 s)
    python -m samadhan verify                      (self-check: every engine against HiGHS or a hand-worked optimum)

--engine auto (default) picks the engine from the model: integer variables -> C++ branch-and-cut; an LP with at
most GPU_FROM variables -> C++ dual simplex (exact vertex solution); larger LPs and all QPs -> GPU engine, which
pays off from roughly 100k variables. --engine ipm runs the interior-point method and crosses over to an exact
vertex.
"""
import argparse
import sys

GPU_FROM = 50_000


def pick_engine(lp):
    if lp.integer is not None and lp.integer.any():
        return "core"
    if lp.Q is None and lp.K.shape[1] <= GPU_FROM:
        return "core"
    return "gpu"


def _gpu(lp, a):
    from .pdlp import solve
    r = solve(lp, device=a.device, native=False if a.no_native else None, presolve=not a.no_presolve,
              tol=a.tol, time_limit=a.time_limit, verbose=not a.quiet, adaptive=a.method == "adaptive")
    print(f"\nstatus      {r.status}\nobjective   {lp.sense * r.primal_obj:.10g}\n"
          f"dual bound  {lp.sense * r.dual_obj:.10g}\n"
          f"rel. gap    {r.rel_gap:.2e}   primal res {r.rel_primal_res:.2e}   dual res {r.rel_dual_res:.2e}\n"
          f"iterations  {r.iterations}   restarts {r.restarts}\n"
          f"time        {r.solve_time:.2f} s solve + {r.setup_time:.2f} s setup on {r.device}")
    if a.crossover and r.status == "optimal" and lp.Q is None:
        from .core import crossover
        c = crossover(lp, r.x, r.y, time_limit=max(a.time_limit - r.solve_time, 1.0), verbose=not a.quiet)
        print(f"crossover   {c.status}: exact vertex, objective {lp.sense * c.obj:.10g}, {c.lp_iters} simplex "
              f"iterations, "
              f"{c.time:.2f} s (C++ core)")
        return c.status == "optimal"
    return r.status == "optimal"


def _ipm(lp, a):
    from .ipm import solve_ipm
    r = solve_ipm(lp, time_limit=a.time_limit, crossover=True, verbose=not a.quiet)
    print(f"\nstatus      {r.status}{'  (exact vertex after crossover)' if r.vertex else ''}\n"
          f"objective   {lp.sense * r.primal_obj:.10g}\n"
          f"iterations  {r.iterations} interior-point\n"
          f"time        {r.time:.2f} s (crossover {r.crossover_time:.2f} s)")
    return r.status == "optimal"


def _core(lp, a):
    from .core import solve_core
    r = solve_core(lp, time_limit=a.time_limit, verbose=not a.quiet, **({} if a.gap is None else dict(gap=a.gap)))
    print(f"\nstatus      {r.status}\nobjective   {lp.sense * r.obj:.10g}\nbound       {lp.sense * r.bound:.10g}\n"
          f"gap         {r.gap:.2e}   nodes {r.nodes}   simplex iterations {r.lp_iters}   cuts {r.cuts}\n"
          f"time        {r.time:.2f} s (C++ core)")
    return r.status == "optimal"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="samadhan", description="SAMADHAN: sovereign LP / QP / MILP solver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="solve an LP or MILP from an MPS file")
    s.add_argument("file")
    s.add_argument("--engine", default="auto", choices=["auto", "gpu", "core", "ipm"],
                   help=f"auto: C++ core for MILPs and for LPs up to {GPU_FROM:,} variables, GPU engine otherwise")
    d = sub.add_parser("demo", help="solve a generated refinery model (MRPL-shaped)")
    d.add_argument("--model", default="lp", choices=["lp", "plan", "crude"],
                   help="lp: refinery planning LP (GPU engine); plan: refinery planning MILP with crude cargoes and "
                        "unit modes; crude: crude oil scheduling at the marine terminal (both MILPs: C++ core)")
    d.add_argument("--size", default="S", choices=["S", "M", "L", "XL"], help="XL: LP only")
    d.add_argument("--seed", type=int, default=None, help="instance seed (default: the one of the benchmarks)")
    v = sub.add_parser("verify", help="self-check: solve small LP, QP and MILP models with both engines and "
                                      "compare every answer with HiGHS or an optimum worked out by hand")
    v.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="GPU engine device")
    for p in (s, d):
        p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="GPU engine device")
        p.add_argument("--tol", type=float, default=1e-4, help="GPU engine: relative KKT tolerance")
        p.add_argument("--method", default="adaptive", choices=["adaptive", "graph"],
                       help="GPU engine: adaptive-step PDLP, or constant-step PDLP replayed as CUDA graphs")
        p.add_argument("--no-presolve", action="store_true",
                       help="GPU engine: solve the model as given, without reducing it first")
        p.add_argument("--no-native", action="store_true",
                       help="GPU engine: PyTorch operations instead of the fused native CUDA kernels")
        p.add_argument("--time-limit", type=float, default=None, help="seconds (default 600; MILP demos 60)")
        p.add_argument("--gap", type=float, default=None,
                       help="C++ core, MILP: relative gap (default 1e-6; MILP demos and the benchmarks use 1e-4)")
        p.add_argument("--quiet", action="store_true")
        p.add_argument("--crossover", action="store_true",
                       help="GPU engine, LP: turn the first-order solution into an exact optimal vertex (C++ simplex)")
    a = ap.parse_args(argv)

    if a.cmd == "verify":
        from .verify import run
        return run(device=a.device)
    if a.cmd == "solve":
        from .mps import read_mps
        lp = read_mps(a.file)
        engine = a.engine
        if engine == "auto":
            engine = pick_engine(lp)
    else:
        from .generate import BENCH_SEED, PLAN_SIZES, REFINERY_SIZES, refinery_lp, refinery_plan
        seed = BENCH_SEED if a.seed is None else a.seed
        if a.model == "lp":
            lp, engine = refinery_lp(seed=seed, **REFINERY_SIZES[a.size]), "gpu"
        else:
            if a.size == "XL":
                ap.error("--size XL exists for --model lp only")
            from .crude import CRUDE_SIZES, crude_schedule
            if a.model == "plan":
                lp = refinery_plan(seed=seed, **PLAN_SIZES[a.size])
            else:
                lp = crude_schedule(seed=seed, **CRUDE_SIZES[a.size])
            engine = "core"
            a.time_limit = 60.0 if a.time_limit is None else a.time_limit
            a.gap = 1e-4 if a.gap is None else a.gap
    if a.time_limit is None:
        a.time_limit = 600.0
    name = {"core": "C++ core", "ipm": "interior point + crossover", "gpu": "GPU engine"}[engine]
    print(lp.summary(), f"-> {name}")
    ok = {"core": _core, "ipm": _ipm, "gpu": _gpu}[engine](lp, a)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
