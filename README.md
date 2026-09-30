# SAMADHAN — sovereign GPU-accelerated optimization engine

Smart India Hackathon 2026 · PS **SIH26119** (MRPL, *Indigenous GPU-Accelerated Optimization Solver*) ·
Team **VIGHNAX** (127364)

SAMADHAN (समाधान, "solution") is an optimization solver core written from scratch: no CPLEX, Gurobi,
Xpress, HiGHS, CBC or SCIP code inside. HiGHS is used only as an outside referee for answers and speed.

## Results (one laptop: RTX 5050 Laptop GPU 8 GB, all CPU threads for HiGHS)

Refinery supply-chain planning LPs (crude purchase, CDU capacity, sulfur blending, yields, dispatch,
depot inventory). Each solver is credited with its faster method: HiGHS = best of dual simplex and
interior point; SAMADHAN = best of its two GPU methods.

| Model | Variables | Nonzeros | HiGHS (best) | SAMADHAN accurate (1e-6) | SAMADHAN fast (1e-4) |
|---|---:|---:|---:|---:|---:|
| S  | 13,296    | 25,416    | 0.24 s | 2.2 s (0.1×)  | 0.7 s (0.3×) |
| M  | 109,536   | 214,092   | 8.5 s  | 12.6 s (0.7×) | 1.8 s (4.8×) |
| L  | 406,080   | 796,512   | 63.9 s | 30.7 s (2.1×) | 5.3 s (12×) |
| XL | 1,157,120 | 2,273,888 | 323 s  | 63.7 s (5.1×) | 15.0 s (22×) |

Accurate mode reaches the HiGHS optimum to within 0.0002 % of cost; fast mode to within 0.07 %.
The GPU pays off as models grow; small models are faster on a CPU simplex, and SAMADHAN will route them there.

* **Netlib LP set:** see `results/netlib_cpu_tol0.0001.json` (own MPS reader, CPU mode, 1e-4 relative KKT).
* **MILP:** 10/10 refinery contract/refinery-selection MILPs proven optimal by our branch-and-bound on our
  own simplex, identical cost to HiGHS (`results/milp.json`).

## What is in this prototype

| Module | What it does |
|---|---|
| `samadhan/mps.py` | MPS reader: free format, true fixed-column format (names with spaces), RANGES, all bound types, integer markers |
| `samadhan/pdlp.py` | GPU LP engine: restarted primal-dual hybrid gradient (PDLP) with Ruiz + Pock-Chambolle scaling, adaptive steps, primal-weight updates, KKT restarts. A constant-step variant is replayed as CUDA graphs (one launch per 64 iterations) |
| `samadhan/simplex.py` | Dense two-phase primal simplex (exact vertices) for small LPs and MILP nodes |
| `samadhan/milp.py` | Best-first branch-and-bound, most-fractional branching, rounding heuristic |
| `samadhan/generate.py` | Refinery planning LP and refinery design MILP generators |
| `samadhan/baseline.py` | HiGHS wrapper, used **only** as the benchmark referee |

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install numpy scipy highspy
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m samadhan demo --size L            # generated refinery LP on the GPU
.venv/Scripts/python -m samadhan solve model.mps --tol 1e-6
.venv/Scripts/python bench_final.py S M L XL              # the table above
.venv/Scripts/python bench_milp.py                        # MILP check vs HiGHS
git clone https://github.com/coin-or-tools/Data-Netlib data/netlib
.venv/Scripts/python bench_netlib.py                      # Netlib LP set
```

## Roadmap (SIH build phase)

1. Presolve, feasibility polishing and dual-simplex crossover for exact vertex solutions.
2. Sparse revised dual simplex with LU updates (warm-started MILP nodes); auto-route small models to CPU.
3. MILP: cutting planes (Gomory, MIR, cover), node selection, feasibility pump and RINS, GPU LP for large
   relaxations; MIPLIB 2017 benchmark set.
4. QP: primal-dual QP (PDQP) on the same GPU kernels; then MIQP and NLP.
5. C++/CUDA kernels for the hot loop; REST service for plant planning systems.
