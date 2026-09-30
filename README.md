# SAMADHAN — a sovereign optimization solver (LP · QP · MILP), GPU-first

[![tests](https://github.com/NAMAN-THAKUR-1944/SIH2026-SIH26119-SAMADHAN/actions/workflows/tests.yml/badge.svg)](https://github.com/NAMAN-THAKUR-1944/SIH2026-SIH26119-SAMADHAN/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.12%20|%203.14-informational)
![c++](https://img.shields.io/badge/C%2B%2B-17-informational)

**Smart India Hackathon 2026 · PS SIH26119 (MRPL): *Indigenous GPU-Accelerated Optimization Solver
(Sovereign Alternative to Xpress / CPLEX)* · Team VIGHNAX (127364)**

SAMADHAN (समाधान, "solution") is an optimization solver written from scratch: no CPLEX, Gurobi, Xpress,
HiGHS, CBC or SCIP code inside. It has two engines:

* a **GPU engine for LP and convex QP**: restarted primal-dual hybrid gradient (PDLP / PDQP family), running
  entirely on the GPU with sparse mat-vecs and CUDA graphs;
* a **C++17 core for MILP**: bounded dual simplex and branch-and-cut (warm-started nodes, pseudocost
  branching, Gomory mixed-integer cuts).

HiGHS, the leading open-source solver, is used only as an outside referee for answers and speed.

▶ **3-minute demo video:** [youtu.be/zagX0zqtIeU](https://youtu.be/zagX0zqtIeU) (a real 1.16 M-variable run, the head-to-head with HiGHS, and all benchmark results)
📄 **Idea deck:** [`docs/SAMADHAN_SIH26119_Idea.pdf`](docs/SAMADHAN_SIH26119_Idea.pdf)

[![SAMADHAN demo video](docs/img/video_thumbnail.png)](https://youtu.be/zagX0zqtIeU)

## Results at a glance

All numbers are measured on one laptop (RTX 5050 Laptop GPU 8 GB), are reproducible with the scripts
below, and are stored in [`results/`](results).

| | What | Result |
|---|---|---|
| **LP** | Refinery planning LP, 1.16 M variables | **5.1× faster** than HiGHS at 1e-6 (cost within 0.0002 %), **22×** at 1e-4 |
| **LP** | Netlib (91 models) | 86 solved to 1e-4 relative KKT; MPS reader identical to HiGHS on 91/91 |
| **MILP** | MIPLIB 3 (58 models ≤ 1,500 rows, 60 s) | **26 proved optimal**, all correct; 3 of them HiGHS could not finish (nw04, pk1, mas76); HiGHS proves 46 |
| **QP** | Maros–Meszaros (129 convex QPs, 60 s) | **79 solved** to 1e-6 relative KKT (HiGHS 99); all 64 solved by both agree; 15 only SAMADHAN solves |
| **QP** | Refinery QP (convex cost curves) | 13k vars: 78× faster than the HiGHS QP solver; from 110k vars HiGHS does not finish in 600 s, SAMADHAN takes 1.5 s |

![LP scaling](docs/img/lp_scaling.png)

## Architecture

```mermaid
flowchart LR
    A["Model<br/>MPS / QPS / .mat / Python API"] --> B["Scaling<br/>Ruiz + Pock–Chambolle"]
    B --> C["GPU engine (LP + QP)<br/>restarted PDHG · adaptive steps<br/>CUDA graphs · PyTorch sparse CSR"]
    A --> D["C++ core (MILP)<br/>bounded dual simplex · DSE pricing<br/>branch-and-cut · Gomory cuts"]
    C --> E["Certified answer<br/>solution · duals · KKT error"]
    D --> E
```

## Results in detail

### LP — refinery supply-chain planning (MRPL use case)

Crude purchase, CDU capacity, sulfur blending, yields, dispatch and depot inventory. HiGHS is credited with
the faster of its dual simplex and interior point; SAMADHAN with the faster of its two GPU methods.

| Model | Variables | Nonzeros | HiGHS (best) | SAMADHAN accurate (1e-6) | SAMADHAN fast (1e-4) |
|---|---:|---:|---:|---:|---:|
| S  | 13,296    | 25,416    | 0.24 s | 2.2 s (0.1×)  | 0.7 s (0.3×) |
| M  | 109,536   | 214,092   | 8.5 s  | 12.6 s (0.7×) | 1.8 s (4.8×) |
| L  | 406,080   | 796,512   | 63.9 s | 30.7 s (2.1×) | 5.3 s (12×) |
| XL | 1,157,120 | 2,273,888 | 323 s  | 63.7 s (5.1×) | 15.0 s (22×) |

**Netlib:** 86/91 solved to 1e-4 relative KKT (CPU mode, median 1.7 s), 80 within 0.1 % of the known optimum.
The 5 unsolved (bnl1, greenbea, greenbeb, perold, pilot4) are ill-conditioned cases for the planned crossover.

### MILP — MIPLIB 3

![MIPLIB](docs/img/miplib.png)

58 MIPLIB 3 models (7 larger ones skipped: the prototype keeps a dense basis inverse), 60 s each, one thread
each, relative gap 1e-4. **SAMADHAN proves 26 optimal**, every one matching the known optimum;
HiGHS proves 46. SAMADHAN finishes **nw04** (87,482 columns) in 10.6 s, **pk1** and **mas76** where HiGHS runs
out of time, and is faster on air03, mod010, khb05250 and p0033. On the rest HiGHS is far ahead: it has
presolve, many cut families and strong primal heuristics that this prototype does not have yet.

### QP — Maros–Meszaros and refinery QP

![QP](docs/img/qp_maros.png)

The same GPU engine solves convex QPs (`0.5 x'Qx + c'x`) by adding the gradient `Qx` to the primal step.
**Maros–Meszaros** (129 convex QPs, CPU mode, 60 s, one thread each): SAMADHAN solves **79** to 1e-6
relative KKT and the HiGHS QP solver 99. All **64** solved by both agree to 1e-4. SAMADHAN solves **15** that
HiGHS does not (HiGHS reports a solve error, no status or a time-out, e.g. AUG2D, LASER, MOSARQP1, QSCTAP3),
and HiGHS crashes on STADAT1. HiGHS solves 35 that SAMADHAN does not finish in 60 s: ill-conditioned problems
where a first-order method needs many more iterations or a crossover step.

**Refinery QP** (every activity has a convex cost curve: crude supply, processing, freight congestion,
holding and shortage), GPU, 1e-6 relative KKT. HiGHS's QP solver is an active-set method:

| Model | Variables | HiGHS QP | SAMADHAN GPU (1e-6) | Cost check |
|---|---:|---:|---:|---|
| S | 13,296 | 86.3 s | 1.1 s (78×) | 0.00016 % from HiGHS |
| M | 109,536 | > 600 s (not solved) | 1.5 s (> 402×) | certified by KKT (1e-6) |
| L | 406,080 | > 1660 s (not solved) | 5.1 s (> 323×) | certified by KKT (1e-6) |
| XL | 1,157,120 | not attempted ¹ | 19.9 s (—) | certified by KKT (1e-6) |

¹ HiGHS was given 600 s per model; on L it stopped only after 1,660 s without a solution, so XL was not attempted.

## Honest limitations

* Small LPs are faster on HiGHS (CPU simplex); the GPU pays off from roughly 100 k variables.
* First-order methods reach 1e-4…1e-6 relative accuracy, not simplex vertices; crossover is on the roadmap.
* The MILP core has no presolve yet and only Gomory cuts, and uses a dense basis inverse (≤ ~1,500 rows).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128   # or /whl/cpu
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest                                  # correctness suite (~30 s)
.venv/Scripts/python -m samadhan demo --size XL --tol 1e-6      # 1.16M-variable refinery LP on the GPU
.venv/Scripts/python -m samadhan solve model.mps --tol 1e-6 --method graph
```

```python
from samadhan.mps import read_mps
from samadhan.pdlp import PDLP            # GPU LP / QP
from samadhan.core import solve_core      # C++ dual simplex / branch-and-cut (compiled on first use)

lp = read_mps("model.mps")
print(PDLP(lp, "cuda").solve(tol=1e-6).primal_obj)   # LP or QP (set lp.Q)
print(solve_core(lp, time_limit=60).obj)             # LP or MILP (integer markers in the MPS file)
```

The C++ core is compiled automatically with the zig toolchain (`pip install ziglang`), so no system
compiler is needed. CI builds it and runs the tests on Ubuntu and Windows.

## Reproduce every number

```bash
python bench_final.py                    # refinery LP table           -> results/final.json
python bench_milp.py                     # refinery MILPs vs HiGHS     -> results/milp.json
git clone https://github.com/coin-or-tools/Data-Netlib  data/netlib
python bench_netlib.py                   # Netlib                      -> results/netlib_cpu_tol0.0001.json
git clone https://github.com/coin-or-tools/Data-miplib3 data/miplib3
python bench_miplib.py                   # MIPLIB 3                    -> results/miplib.json
# Maros-Meszaros .mat files (< 300 KB) from github.com/qpsolvers/maros_meszaros_qpbenchmark -> data/maros/
python bench_qp.py maros                 # Maros-Meszaros QPs          -> results/qp_maros.json
python bench_qp.py refinery              # refinery QP                 -> results/qp_refinery.json
python docs/make_figures.py              # the figures above
```

## Project layout

```
samadhan/pdlp.py      GPU LP/QP engine (restarted PDHG, adaptive + CUDA-graph modes)
samadhan/core.py      ctypes binding for the C++ core, builds it with zig on first use
cpp/samadhan_core.cpp C++17 dual simplex + branch-and-cut (C ABI)
samadhan/mps.py       MPS reader (free / fixed format, RANGES, bounds, integer markers)
samadhan/qpdata.py    Maros-Meszaros .mat reader
samadhan/simplex.py   reference dense simplex + samadhan/milp.py reference B&B (pure Python)
samadhan/generate.py  refinery LP / QP / MILP generators
samadhan/baseline.py  HiGHS referee (LP, QP, MILP)
tests/                pytest suite, every component checked against HiGHS or a hand-computed optimum
bench_*.py, results/  benchmarks and their raw outputs;  docs/  deck and figures
```

## How it works

* **LP / QP (GPU).** Restarted PDHG (Applegate et al., NeurIPS 2021; Lu & Yang, cuPDLP 2023) with Ruiz and
  Pock–Chambolle scaling, adaptive step sizes, primal-weight updates and KKT-based restarts. A constant-step
  variant has no host synchronisation between checks, so 64 iterations are captured once as a CUDA graph and
  replayed with one launch. For QP the primal step uses `Qx + c − Kᵀy` with `τ ≤ 1/(‖Q‖/2 + σ‖K‖²)`
  (linearised PDHG, as in PDQP) and the Wolfe dual for the gap.
* **MILP (C++).** Every row gets a logical variable, so the simplex works on `[A −I]` with column bounds.
  Bounded dual simplex with exact dual steepest-edge weights and a Harris ratio test; the dense basis inverse
  is updated per pivot and refactorised every 100. Branch-and-bound dives from each node keep the
  factorisation (a bound change on a basic variable keeps the basis dual feasible); other nodes store their
  basis for a warm restart. Pseudocost branching, rounding heuristic, Gomory mixed-integer cuts at the root.

## Roadmap (SIH build phase)

1. Presolve, feasibility polishing and crossover to exact vertices.
2. Sparse LU with Forrest–Tomlin updates (lift the 1,500-row limit); MIR, cover and flow cuts; feasibility pump.
3. GPU LP relaxations inside branch-and-bound for very large MILPs; MIQP.
4. Native CUDA kernels for the PDHG loop; REST service for plant planning systems.

## Team and license

Team VIGHNAX (127364), SRM University, for Smart India Hackathon 2026. Licensed under [Apache-2.0](LICENSE).
