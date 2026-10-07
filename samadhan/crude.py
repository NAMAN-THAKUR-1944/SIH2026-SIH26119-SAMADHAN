"""Crude oil scheduling MILP for a coastal refinery (the MRPL marine-terminal use case).

Discrete time (periods of 8 h). Crude arrives in vessels at a single mooring (SPM) and is unloaded into storage
tanks; tanks feed the crude distillation units (CDUs). Tanks are segregated by crude class (for example low- and
high-sulphur): a tank holds one class at a time and may change class only when it is down to its heel. This keeps
the model linear (no blending of properties) and is how sour and sweet crude are kept apart in practice.

Sets: vessels V, tanks K, CDUs U, crude classes G, periods T.
Variables
  xu[v,k,t]  binary  vessel v unloads into tank k in period t       fu[v,k,t] >= 0  volume unloaded
  xc[k,u,t]  binary  tank k feeds CDU u in period t                  fc[k,u,t] >= 0  volume charged
  y[k,g,t]   binary  tank k holds class g in period t
  inv[k,t]           tank inventory (heel <= inv <= capacity)
  sw[k,u,t]  >= 0    tank k starts feeding CDU u (changeover)
  dep[v]             period in which vessel v has finished unloading; dem[v] >= 0 demurrage periods
  short[u,t] >= 0    CDU feed below its minimum rate (starvation);   left[v] >= 0  crude left on board
Constraints
  unloading rate     fu <= Fu xu,  and xu = 0 before the vessel arrives
  one berth          sum_{v,k} xu[v,k,t] <= 1
  cargo              sum_{k,t} fu[v,k,t] + left[v] = vol[v]
  departure          dep[v] >= (t + 1) sum_k xu[v,k,t];   dem[v] >= dep[v] - arrival[v] - laytime[v]
  class of a tank    sum_g y[k,g,t] = 1;   xu[v,k,t] <= y[k,class(v),t];   xc[k,u,t] <= sum_{g allowed at u} y[k,g,t]
  class change       inv[k,t-1] <= heel[k] + cap[k] (1 - y[k,g,t-1] + y[k,g,t])        (only when nearly empty)
  charging           fc <= Fc xc;  sum_k xc[k,u,t] <= 1;  sum_u xc[k,u,t] <= 1
  receive or feed    sum_v xu[v,k,t] + sum_u xc[k,u,t] <= 1
  settling           xc[k,u,t] <= 1 - sum_v xu[v,k,t-1]     (brine settles one period after receipt)
  inventory          inv[k,t] = inv[k,t-1] + sum_v fu[v,k,t] - sum_u fc[k,u,t]
  CDU rate           sum_k fc[k,u,t] + short[u,t] >= Rmin[u];   sum_k fc[k,u,t] <= Rmax[u]
  changeover         sw[k,u,t] >= xc[k,u,t] - xc[k,u,t-1]
Objective: demurrage + changeovers + starvation + crude left on board (all as costs).
"""
import numpy as np
import scipy.sparse as sp

from .lp import LP

# Instance sizes for the README and benchmarks/mrpl.py (generated with seed=BENCH_SEED and more seeds)
CRUDE_SIZES = {
    "S": dict(V=3, K=6, U=2, T=21),     # one week, 8 h periods
    "M": dict(V=5, K=8, U=3, T=30),     # ten days
    "L": dict(V=8, K=12, U=3, T=42),    # two weeks
}


def crude_schedule(V=3, K=6, U=2, T=21, G=2, seed=0):
    rng = np.random.default_rng(seed)
    # ---- data (volumes in kbbl, rates per 8 h period)
    vclass = rng.integers(0, G, V)
    vol = rng.uniform(250, 600, V)
    arrival = np.sort(rng.integers(0, max(1, T - 8), V))
    laytime = np.full(V, 3)
    Fu = 160.0                                         # SPM pumping per period
    cap = rng.uniform(300, 500, K)
    heel = 0.05 * cap
    kclass0 = np.arange(K) % G                         # initial class of each tank
    inv0 = heel + rng.uniform(0.4, 0.8, K) * (cap - heel)
    Rmin = rng.uniform(28, 38, U)
    Rmax = Rmin * 1.6
    Fc = Rmax.max()
    allowed = np.ones((U, G), bool)
    allowed[0, G - 1] = False                          # CDU 1 cannot take the heaviest / sourest class
    c_dem, c_sw, c_short, c_left = 40.0, 5.0, 100.0, 200.0

    # ---- variable layout
    idx = {}
    names = []

    def block(name, shape, kind):
        start = len(names)
        size = int(np.prod(shape))
        names.extend([kind] * size)
        idx[name] = (start, shape)
        return start

    block("xu", (V, K, T), "B"); block("fu", (V, K, T), "C")
    block("xc", (K, U, T), "B"); block("fc", (K, U, T), "C")
    block("y", (K, G, T), "B"); block("inv", (K, T), "C"); block("sw", (K, U, T), "C")
    block("dep", (V,), "C"); block("dem", (V,), "C"); block("short", (U, T), "C"); block("left", (V,), "C")
    n = len(names)

    def var(name, *ix):
        start, shape = idx[name]
        return start + int(np.ravel_multi_index(ix, shape))

    lo, up, cost = np.zeros(n), np.full(n, np.inf), np.zeros(n)
    integer = np.array([k == "B" for k in names])
    up[integer] = 1.0
    for v in range(V):
        for k in range(K):
            for t in range(T):
                if t < arrival[v]:
                    up[var("xu", v, k, t)] = 0.0
                    up[var("fu", v, k, t)] = 0.0
        cost[var("dem", v)] = c_dem
        cost[var("left", v)] = c_left
        up[var("dep", v)] = T
    for k in range(K):
        for t in range(T):
            lo[var("inv", k, t)], up[var("inv", k, t)] = heel[k], cap[k]
            for u in range(U):
                cost[var("sw", k, u, t)] = c_sw
                up[var("sw", k, u, t)] = 1.0
    for u in range(U):
        for t in range(T):
            cost[var("short", u, t)] = c_short

    eq, ge = [], []                                     # (cols, vals, rhs):  eq  a x = b,  ge  a x >= b

    def row(lst, terms, rhs):
        cols = [c for c, _ in terms]
        vals = [a for _, a in terms]
        lst.append((cols, vals, rhs))

    for t in range(T):
        # one berth
        row(ge, [(var("xu", v, k, t), -1.0) for v in range(V) for k in range(K)], -1.0)
        for v in range(V):
            row(ge, [(var("dep", v), 1.0)] + [(var("xu", v, k, t), -(t + 1.0)) for k in range(K)], 0.0)
            for k in range(K):
                row(ge, [(var("xu", v, k, t), Fu), (var("fu", v, k, t), -1.0)], 0.0)
                row(ge, [(var("y", k, vclass[v], t), 1.0), (var("xu", v, k, t), -1.0)], 0.0)
        for k in range(K):
            row(eq, [(var("y", k, g, t), 1.0) for g in range(G)], 1.0)
            # receive or feed, settling
            recv = [(var("xu", v, k, t), -1.0) for v in range(V)]
            row(ge, recv + [(var("xc", k, u, t), -1.0) for u in range(U)], -1.0)
            for u in range(U):
                if t > 0:
                    row(ge, [(var("xc", k, u, t), -1.0)] + [(var("xu", v, k, t - 1), -1.0) for v in range(V)], -1.0)
                row(ge, [(var("xc", k, u, t), Fc), (var("fc", k, u, t), -1.0)], 0.0)
                row(ge, [(var("y", k, g, t), 1.0) for g in range(G) if allowed[u, g]] + [(var("xc", k, u, t), -1.0)],
                    0.0)
                prev = [(var("xc", k, u, t - 1), 1.0)] if t > 0 else []
                row(ge, [(var("sw", k, u, t), 1.0), (var("xc", k, u, t), -1.0)] + prev, 0.0)
            # inventory balance
            terms = [(var("inv", k, t), 1.0)] + [(var("fu", v, k, t), -1.0) for v in range(V)] + \
                [(var("fc", k, u, t), 1.0) for u in range(U)]
            if t > 0:
                row(eq, terms + [(var("inv", k, t - 1), -1.0)], 0.0)
            else:
                row(eq, terms, inv0[k])
            # class changes only near the heel:  inv[k,t-1] <= heel + cap (1 - y[g,t-1] + y[g,t])
            for g in range(G):
                if t > 0:
                    row(ge, [(var("inv", k, t - 1), -1.0), (var("y", k, g, t - 1), -cap[k]),
                             (var("y", k, g, t), cap[k])], heel[k] - cap[k])
        for u in range(U):
            fed = [(var("fc", k, u, t), 1.0) for k in range(K)]
            row(ge, fed + [(var("short", u, t), 1.0)], Rmin[u])
            row(ge, [(c, -a) for c, a in fed], -Rmax[u])
            row(ge, [(var("xc", k, u, t), -1.0) for k in range(K)], -1.0)
    for v in range(V):
        row(eq, [(var("fu", v, k, t), 1.0) for k in range(K) for t in range(T)] + [(var("left", v), 1.0)], vol[v])
        row(ge, [(var("dem", v), 1.0), (var("dep", v), -1.0)], -float(arrival[v] + laytime[v]))
    # initial tank classes
    for k in range(K):
        lo[var("y", k, kclass0[k], 0)] = 1.0

    def build(rows):
        rr, cc, vv = [], [], []
        for i, (cols, vals, _) in enumerate(rows):
            rr += [i] * len(cols); cc += cols; vv += vals
        return sp.csr_matrix((vv, (rr, cc)), shape=(len(rows), n))

    A = sp.vstack([build(eq), build(ge)]).tocsr()
    A.sum_duplicates()
    q = np.array([b for *_, b in eq] + [b for *_, b in ge], float)
    return LP(cost, A, q, len(eq), lo, up, name=f"crude_V{V}_K{K}_U{U}_T{T}_s{seed}", integer=integer)
