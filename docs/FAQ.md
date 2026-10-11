# Questions a reviewer will ask

Short, honest answers. Every number quoted in the [README](../README.md) has its raw data in
[`results/`](../results) and the command that produced it under
[Reproduce every number](../README.md#reproduce-every-number).

### Is SAMADHAN really written from scratch? What does it depend on?

The solver code is ours: the MPS/QPS readers, presolve (bound propagation, probing, parallel rows, doubleton
substitution), scaling, the GPU engine (restarted PDHG with its own CUDA C++ kernels), the C++17 core (sparse LU
with Forrest–Tomlin updates, primal and dual simplex, crossover, branch-and-cut with Gomory and c-MIR cuts,
reliability branching, node propagation, feasibility pump, diving, fix-and-propagate and RINS), the
interior-point method and the PuLP / Pyomo plugins. No code from CPLEX, Gurobi, Xpress, HiGHS, CBC or SCIP is
inside.

Libraries we build on: **PyTorch** for GPU tensors, sparse products in the convergence checks and CUDA graphs
(its NVRTC compiles our kernels at run time), **NumPy / SciPy** for data handling, and **ziglang**, a pip-packaged
C++ compiler, so that the core builds without a system toolchain. **HiGHS** appears only as an outside referee in
the tests and benchmarks; it is never called by the solver.

### How does it compare with CPLEX, Gurobi or Xpress?

We have no licences for them, so we compare with HiGHS, the strongest open-source solver. On hard MILPs the
commercial solvers are generally faster than HiGHS, so expect the gap to them to be larger than the gap to HiGHS
shown in the README. We do not claim parity with them on MILP.

### Where is SAMADHAN faster than HiGHS, and where slower?

Faster: large LPs on the GPU (the 1.16 M-variable refinery planning LP), large convex QPs (from 110k variables HiGHS
does not finish in 600 s), the small refinery-planning MILP, and 19 MIPLIB 3 models, plus four (nw04, pk1, qiu,
mas76) that HiGHS does not finish in 60 s.

Slower: small LPs (HiGHS's simplex median is lower), MILP in general (MIPLIB 3: 44 proved against HiGHS's 50;
MIPLIB 2017: 5 against 18), and the crude-scheduling MILP, where HiGHS still finds cheaper schedules on most
instances (on one medium instance 113× cheaper). The README lists all of it under
[Honest limitations](../README.md#honest-limitations).

### A first-order method on the GPU: how accurate are its answers?

The GPU engine stops at a relative KKT error of 1e-4 or 1e-6 (primal residual, dual residual and duality gap all
below it) and reports the three numbers. When an exact answer is needed, **crossover** turns the GPU solution
into an exact optimal vertex with the C++ simplex (`--crossover`). The C++ core returns exact vertices that
match HiGHS to 1e-9 on all 91 Netlib models (the interior-point method with crossover on 89 of them).

### What needs a GPU?

Nothing. The GPU engine runs on the CPU too (slower), and the C++ core (LP and MILP) and the interior-point
method are CPU programs. A GPU speeds up large LPs and QPs. MILPs are solved on the CPU.

### Isn't CUDA itself a dependency on a foreign vendor?

Yes, for the GPU engine: it needs an NVIDIA GPU and driver. Everything else runs without it, and the GPU engine
also runs on the CPU. The native kernels are plain CUDA C++ of a few hundred lines; porting them to another GPU
platform would be work, but bounded work. The C++ core is self-contained C++17 with no outside libraries.

### How would MRPL use it?

* Models exported by existing planning tools in **MPS** format: `samadhan solve model.mps` picks the engine
  from the model.
* Models written in **PuLP or Pyomo**: one changed line (`prob.solve(SAMADHAN())`,
  `SolverFactory("samadhan")`).
* **Docker** images for CPU and GPU, so nothing else has to be installed.
* The benchmarks include MRPL-shaped models: refinery planning (LP, QP with convex cost curves, and MILP with
  crude cargoes and unit modes) and crude-oil scheduling at a marine terminal (MILP).

### How do you know the answers are right?

* `python -m samadhan verify` solves LP, QP and MILP models with every engine, compares each answer with HiGHS
  or a hand-worked optimum, and recomputes feasibility from the returned solution vector instead of trusting the
  solver's report.
* The test suite (71 tests) runs in CI on Ubuntu and Windows on every push.
* Every benchmark answer is compared with HiGHS; every "proved optimal" MILP result matches the known optimum,
  and every MILP solution is checked against the model's original rows.

### Are the benchmarks fair?

Both solvers run on the same laptop, with the same time limit and, for MILP, the same relative gap (1e-4) and one
CPU thread each. On LP, HiGHS is credited with the faster of its two methods (dual simplex and interior point).
GPU-against-CPU comparisons are exactly that, and the README says so. Benchmarks run alone and on mains power;
since 2026-10-09 also not while the battery charges, which we found slows everything, HiGHS included, by
1.2–1.5×.

### Are runs repeatable?

LP solves are: the simplex refactorises after a counted amount of work rather than after a time, and the GPU
engine's native kernels add up in a fixed order, so two runs give the same iterations and the same answer. In
branch-and-cut, some budgets are measured in seconds (the root cut rounds, the spacing of RINS calls), so MILP
runs that reach those points, and every result that stops at a time limit, depend on the machine's speed.

### What is next?

The [roadmap](../README.md#roadmap-sih-build-phase): crossover at a million variables, local branching for
scheduling models, clique and enumeration presolve, lifted cover and flow-cover cuts, parallel tree search, GPU
LP relaxations inside branch-and-bound, and a REST service for plant planning systems.
