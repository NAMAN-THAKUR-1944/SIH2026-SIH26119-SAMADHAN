# Live demo run sheet (about 10 minutes)

Seven steps, each one command, in the order that tells the story: correct answers, then speed at scale on the
GPU, then exact vertices, then the MRPL-shaped MILPs, then the one-line integration. Times are from
[`results/`](../results) on the RTX 5050 laptop; the large-LP times are in the README's
[refinery LP table](../README.md#lp--refinery-supply-chain-planning-mrpl-use-case).

## Before the session

* Mains power **and a full battery**: a laptop that is charging runs everything 1.2–1.5× slower. Turn off sleep.
* Run every command below once beforehand. The first run compiles the C++ core and the CUDA kernels (both are
  cached), so the live run starts at once.
* The data folders are present: `data/miplib3` (for step 6). No internet is needed for any step.
* Close other GPU and CPU heavy programs. Keep a terminal with a large font, one window, no notifications.
* Fallback without a GPU: add `--device cpu` to steps 1 and 2 (step 2 then with `--size M`, about 10 s); steps
  3–7 run on the CPU anyway.

## Steps

| # | Show | Command | Expect |
|---|---|---|---|
| 1 | Every engine checked against HiGHS or a hand-worked optimum, feasibility recomputed from the answer | `python -m samadhan verify` | 12/12 PASS in about 10 s |
| 2 | A 1.16 M-variable refinery planning LP on the GPU | `python -m samadhan demo --size XL` | optimal at 1e-4; HiGHS needs minutes (README table) |
| 3 | An exact optimal vertex from the GPU answer (crossover) | `python -m samadhan demo --size M --crossover` | GPU engine 1–2 s, crossover about 3 s; a cold-start simplex needs 53 s |
| 4 | Refinery planning MILP: crude cargoes, crude units on or off | `python -m samadhan demo --model plan --size S` | proved optimal in about 2 s (HiGHS about 6 s) |
| 5 | Crude oil scheduling at the marine terminal | `python -m samadhan demo --model crude --size S --time-limit 20` | a schedule in 20 s; say honestly that HiGHS's are still somewhat cheaper here |
| 6 | A MIPLIB model HiGHS does not finish in 60 s | `python -m samadhan solve data/miplib3/nw04.gz` | 87,482 binaries, proved optimal in about 2.5 s |
| 7 | An existing PuLP model, one line changed | `python -m examples.pulp_crude_selection` | the same optimum as HiGHS, margin 10,220 k$ |

## What to say at each step

1. "We never trust the solver's own report: feasibility is recomputed from the solution vector, and HiGHS is
   the referee."
2. "First-order PDHG on the GPU, with our own CUDA kernels; the answer is certified by its KKT error."
3. "When an exact vertex is needed, the C++ simplex finishes from the GPU solution."
4. and 5. "Built the way a coastal refinery plans and schedules. Planning: faster than HiGHS. Scheduling:
   our presolve now reaches HiGHS's bound, but HiGHS's schedules are still cheaper on most instances."
6. "Branch-and-cut, presolve with probing, cuts and heuristics, all written from scratch in C++17."
7. "Nothing to rewrite: existing PuLP or Pyomo models switch solver with one line."

Questions that usually follow are answered in [FAQ.md](FAQ.md).
