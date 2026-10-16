# Questions a reviewer will ask

Short, honest answers. Every number quoted in the [README](../README.md) has its raw data in
[`results/`](../results) and the command that produced it under
[Reproduce every number](../README.md#reproduce-every-number).

### Is SAMADHAN really written from scratch? What does it depend on?

The solver code is ours: the MPS/QPS readers, presolve (bound propagation, probing, parallel rows, doubleton
substitution), scaling, the GPU engine (restarted PDHG with its own CUDA C++ kernels), the C++17 core (sparse LU
with Forrest–Tomlin updates, primal and dual simplex, crossover, branch-and-cut with Gomory, c-MIR, flow cover and
network cut-set cuts, reliability branching, node propagation, feasibility pump, diving, fix-and-propagate, RINS,
local branching and a deterministic parallel tree search), the
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

### And with NVIDIA cuOpt, the other GPU solver?

Measured on the same laptop GPU on 14 of Hans Mittelmann's LP benchmark instances (README, "LP — Mittelmann's
instances"): SAMADHAN is faster on 7 of the 14, cuOpt on the other 7. SAMADHAN wins the small and medium ones
(datt256 0.30 s against 1.08 s, savsched1 0.78 s against 1.76 s), since its presolve and scaling moved to C++ and its
setup takes at most 0.7 s; cuOpt is 2.2–2.7× faster on the long runs and solves two instances, s100 and
irish-electricity, on which SAMADHAN does not converge within 300 s. SAMADHAN's objectives agree with cuOpt's to
7e-4 or better; that took a presolve before the GPU engine, without which three answers passed the 1e-4 test with a
wrong objective. Both are far ahead of HiGHS on these models. A stopping test that bounds the objective error and
a better restart scheme are next.

### Where is SAMADHAN faster than HiGHS, and where slower?

Faster: large LPs on the GPU (the 1.16 M-variable refinery planning LP: 33× at 1e-4, and an exact optimal vertex in
91 s against 323 s), large convex QPs (from 110k variables HiGHS does not finish in 600 s), the small
refinery-planning MILP, 20 MIPLIB 3 models (nw04: 2.4 s against 50 s) plus two (pk1, arki001) that HiGHS does not
finish in 60 s, and network-design MIPLIB 2017 models such as beasleyC3 (1.6 s against 22 s).

Slower: small LPs (HiGHS's simplex median is lower), MILP in general (MIPLIB 3: 45 proved against HiGHS's 51;
MIPLIB 2017: 8 against 20), and the crude-scheduling MILP, where HiGHS still finds cheaper schedules on six of
nine instances with one thread each (on one medium instance 31× cheaper; 3.3× against SAMADHAN on 8 threads). The
README lists all of it under [Honest limitations](../README.md#honest-limitations).

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
* The test suite (85 tests) runs in CI on Ubuntu and Windows on every push.
* Every benchmark answer is compared with HiGHS, and every "proved optimal" MILP result matches the known optimum.
  Since 2026-10-13 the MIPLIB, MRPL and thread benchmarks also record how far each MILP solution is from satisfying
  the model's original rows, bounds and integrality. That check found a real bug: the solver tested new incumbents
  on the scaled rows, and on MIPLIB 3's enigma it reported as optimal a point that misses one equality row by 1.0.
  Incumbents are now tested in each row's own units; no solution in those tables is off by more than 1e-6.

### Are the benchmarks fair?

Both solvers run on the same laptop, with the same time limit and, for MILP, the same relative gap (1e-4) and one
CPU thread each. On LP, HiGHS is credited with the faster of its two methods (dual simplex and interior point).
GPU-against-CPU comparisons are exactly that, and the README says so. The parallel tree search is compared with
SAMADHAN itself on one thread, each model alone on the machine; every comparison with HiGHS stays at one thread
each. Benchmarks run alone and on mains power;
since 2026-10-09 also not while the battery charges, which we found slows everything, HiGHS included, by
1.2–1.5×. Since 2026-10-15 every solver process, HiGHS's included, opts out of Windows 11's background throttling,
which runs windowless processes on the efficiency cores while the desktop is in use (up to 3.3× slower); a fixed
speed check runs before and after each benchmark, the load of other programs is logged every 20 s, and runs that
overlap heavy outside load are repeated.

### Are runs repeatable?

LP solves are: the simplex refactorises after a counted amount of work rather than after a time, and the GPU
engine's native kernels add up in a fixed order, so two runs give the same iterations and the same answer. In
branch-and-cut, some budgets are measured in seconds (the root cut rounds, the spacing of RINS calls), so MILP
runs that reach those points, and every result that stops at a time limit, depend on the machine's speed. The
parallel tree search keeps this property: its threads work in rounds that end after a fixed number of simplex
iterations, and their results are merged in a fixed order, so the same model with the same thread count gives the
same tree on every run.

### Does it use more than one CPU core?

For MILP, yes: `--threads N` (or `threads=` in the Python API and both plugins) runs the branch-and-bound tree on N
threads. On MIPLIB 3, alone on the laptop, 4 threads prove 47 models against 46 with one, and 8 threads are 2.0×
faster on the models all prove (but prove 45); on the medium crude scheduling instances they find schedules up to
9.5× cheaper within the minute. Presolve, the root LP and the cuts still run on one thread, so the gain is about 2×,
not 8×.
The GPU engine uses the GPU's thousands of threads; the simplex for LP is sequential.

### What is next?

The [roadmap](../README.md#roadmap-sih-build-phase). Done in the latest round: crossover at a million variables,
local branching, network cut-set cuts, presolve and scaling in C++. Next: a GPU stopping test that bounds the
objective error and better restarts, root heuristics that hold up on large scheduling models, clique and
enumeration presolve, lifted cover cuts and multi-row flow covers, an asynchronous parallel mode, GPU LP
relaxations inside branch-and-bound, and a REST service for plant planning systems.
