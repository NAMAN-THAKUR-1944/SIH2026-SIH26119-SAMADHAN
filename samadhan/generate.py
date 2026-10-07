"""Synthetic but structurally realistic refinery supply-chain planning LPs (the MRPL use case).

Sets: refineries R, crude grades C, products P, depots D, periods T.
Variables
  x[c,r,t]   crude c processed at refinery r in period t        (kbbl)
  y[p,r,t]   product p produced at r in t
  s[p,r,d,t] product p shipped from r to depot d in t
  I[p,d,t]   closing inventory at depot d                        (<= tank capacity)
  z[p,d,t]   unmet demand (penalised)
Constraints
  crude supply      sum_r x[c,r,t]                 <= S[c,t]
  CDU capacity      sum_c x[c,r,t]                 <= cap[r]
  sulfur blending   sum_c (sulfur_c - smax_r) x    <= 0
  yields            y[p,r,t] - sum_c Y[c,p,r] x    =  0
  dispatch          sum_d s[p,r,d,t]               <= y[p,r,t]
  depot balance     I[t-1] + sum_r s + z - I[t]    =  dem[p,d,t]
Objective: crude cost + freight (distance based) + holding + shortage penalty.
"""
import numpy as np
import scipy.sparse as sp

from .lp import LP

# Benchmark sizes used in the README, the deck and benchmarks/lp.py (always generated with seed=7).
REFINERY_SIZES = {
    "S":  dict(R=4,  C=8,  P=6, D=60,  T=6),    # 13k variables
    "M":  dict(R=8,  C=10, P=6, D=150, T=12),   # 110k
    "L":  dict(R=12, C=12, P=8, D=300, T=12),   # 406k
    "XL": dict(R=16, C=12, P=8, D=500, T=16),   # 1.16M
}
BENCH_SEED = 7


def refinery_lp(R=4, C=8, P=6, D=40, T=6, seed=0, quad=0.0):
    """quad > 0 turns it into a convex QP: crude price rises with volume (supply curve),
    cost_c * x + 0.5 * q_c * x^2 with q_c = quad * price / typical run size."""
    rng = np.random.default_rng(seed)
    # indexers
    nx, ny, ns, nI, nz = C * R * T, P * R * T, P * R * D * T, P * D * T, P * D * T
    ox, oy, os_, oI, oz = 0, nx, nx + ny, nx + ny + ns, nx + ny + ns + nI
    n = oz + nz
    X = lambda c, r, t: ox + (c * R + r) * T + t
    Yv = lambda p, r, t: oy + (p * R + r) * T + t
    S = lambda p, r, d, t: os_ + ((p * R + r) * D + d) * T + t
    I = lambda p, d, t: oI + (p * D + d) * T + t
    Z = lambda p, d, t: oz + (p * D + d) * T + t

    # data
    ref_xy = rng.uniform([8, 68], [32, 92], size=(R, 2))      # rough India lat/long box
    dep_xy = rng.uniform([8, 68], [32, 92], size=(D, 2))
    dist = np.linalg.norm(ref_xy[:, None, :] - dep_xy[None, :, :], axis=2) * 111.0  # km
    freight = 0.02 + 0.0009 * dist * rng.uniform(0.9, 1.1, size=dist.shape)          # $/bbl
    crude_price = rng.uniform(70, 95, size=(C, T))
    sulfur = rng.uniform(0.2, 3.0, size=C)
    smax = rng.uniform(1.2, 2.0, size=R)
    cap = rng.uniform(150, 400, size=R) * (D * P / 240.0)
    yields = rng.dirichlet(np.ones(P + 1) * 2, size=(C, R))[:, :, :P].transpose(0, 2, 1)  # C,P,R ; rest = loss/fuel
    base_dem = rng.uniform(2, 12, size=(P, D)) * (cap.sum() * 0.8 / (P * D * 7.0))
    season = 1 + 0.15 * np.sin(np.arange(T) / max(T, 1) * 2 * np.pi)
    dem = base_dem[:, :, None] * season[None, None, :] * rng.uniform(0.85, 1.15, size=(P, D, T))
    supply = rng.uniform(0.2, 0.5, size=(C, T)) * cap.sum()
    tank = base_dem * 1.5

    cost = np.zeros(n)
    for c in range(C):
        for r in range(R):
            cost[[X(c, r, t) for t in range(T)]] = crude_price[c]
    idx = np.arange(ns)
    d_i = (idx // T) % D; r_i = (idx // (T * D)) % R
    cost[os_:os_ + ns] = freight[r_i, d_i]
    cost[oI:oI + nI] = 0.15
    cost[oz:oz + nz] = 400.0

    l = np.zeros(n)
    u = np.full(n, np.inf)
    ii = np.arange(nI)
    u[oI:oI + nI] = tank[(ii // (D * T)) % P, (ii // T) % D]

    eq_r, eq_c, eq_v, eq_b = [], [], [], []
    ge_r, ge_c, ge_v, ge_b = [], [], [], []
    def add(rr, cc, vv, bb, rows, cols, vals, rhs):
        k = len(rhs)
        rows.extend([k] * len(cc)); cols.extend(cc); vals.extend(vv); rhs.append(bb)

    for t in range(T):
        for c in range(C):  # -sum_r x >= -S
            add(None, [X(c, r, t) for r in range(R)], [-1.0] * R, -supply[c, t], ge_r, ge_c, ge_v, ge_b)
        for r in range(R):
            add(None, [X(c, r, t) for c in range(C)], [-1.0] * C, -cap[r], ge_r, ge_c, ge_v, ge_b)
            add(None, [X(c, r, t) for c in range(C)], list(smax[r] - sulfur), 0.0, ge_r, ge_c, ge_v, ge_b)
            for p in range(P):
                add(None, [Yv(p, r, t)] + [X(c, r, t) for c in range(C)],
                    [1.0] + list(-yields[:, p, r]), 0.0, eq_r, eq_c, eq_v, eq_b)
                add(None, [Yv(p, r, t)] + [S(p, r, d, t) for d in range(D)],
                    [1.0] + [-1.0] * D, 0.0, ge_r, ge_c, ge_v, ge_b)
        for p in range(P):
            for d in range(D):
                cols = [S(p, r, d, t) for r in range(R)] + [Z(p, d, t), I(p, d, t)]
                vals = [1.0] * R + [1.0, -1.0]
                if t > 0:
                    cols.append(I(p, d, t - 1)); vals.append(1.0)
                add(None, cols, vals, dem[p, d, t], eq_r, eq_c, eq_v, eq_b)

    Keq = sp.csr_matrix((eq_v, (eq_r, eq_c)), shape=(len(eq_b), n))
    Kge = sp.csr_matrix((ge_v, (ge_r, ge_c)), shape=(len(ge_b), n))
    K = sp.vstack([Keq, Kge]).tocsr()
    q = np.array(eq_b + ge_b)
    lp = LP(cost, K, q, len(eq_b), l, u, name=f"refinery_R{R}_C{C}_P{P}_D{D}_T{T}")
    if quad > 0:
        # convex cost curves on every activity (Q positive definite):
        # crude supply curve, processing, freight congestion, quadratic holding, quadratic shortage
        qd = np.zeros(n)
        typ_crude, typ_prod = cap.mean() / C, cap.mean() / P
        typ_ship, typ_inv = base_dem.mean(), tank.mean()
        for c in range(C):
            for r in range(R):
                for t in range(T):
                    qd[X(c, r, t)] = quad * crude_price[c, t] / typ_crude
        qd[oy:oy + ny] = quad * 2.0 / typ_prod
        qd[os_:os_ + ns] = quad * cost[os_:os_ + ns] / typ_ship
        qd[oI:oI + nI] = quad * 0.15 / typ_inv
        qd[oz:oz + nz] = quad * 400.0 / typ_ship
        lp.Q = sp.diags(qd).tocsr()
        lp.name = f"refinery_qp_R{R}_C{C}_P{P}_D{D}_T{T}"
    return lp


def refinery_plan(R=3, C=6, P=4, D=12, T=6, seed=0):
    """Multi-period refinery planning MILP: the planning LP above with the discrete decisions of a real plan.

    Crude is bought in whole cargoes: n[c,t] in {0, ..., 4} parcels of crude c in period t, each of `parcel` kbbl
    with a fixed freight and port cost. A crude unit is either off or runs between its minimum and its maximum
    rate: on[r,t] binary, fixed running cost, start-up cost st[r,t] >= on[r,t] - on[r,t-1].
      crude bought      sum_r x[c,r,t] <= parcel * n[c,t]
      CDU modes         minrate_r on[r,t] <= sum_c x[c,r,t] <= cap_r on[r,t]
      sulfur, yields, dispatch, depot balance and costs as in refinery_lp.
    """
    rng = np.random.default_rng(seed)
    nx, ny, ns, nI, nz = C * R * T, P * R * T, P * R * D * T, P * D * T, P * D * T
    nn, non, nst = C * T, R * T, R * T
    ox, oy, os_, oI, oz = 0, nx, nx + ny, nx + ny + ns, nx + ny + ns + nI
    on_, oon, ost = oz + nz, oz + nz + nn, oz + nz + nn + non
    n = ost + nst
    X = lambda c, r, t: ox + (c * R + r) * T + t
    Yv = lambda p, r, t: oy + (p * R + r) * T + t
    S = lambda p, r, d, t: os_ + ((p * R + r) * D + d) * T + t
    I = lambda p, d, t: oI + (p * D + d) * T + t
    Z = lambda p, d, t: oz + (p * D + d) * T + t
    N = lambda c, t: on_ + c * T + t
    ON = lambda r, t: oon + r * T + t
    ST = lambda r, t: ost + r * T + t

    ref_xy = rng.uniform([8, 68], [32, 92], size=(R, 2))
    dep_xy = rng.uniform([8, 68], [32, 92], size=(D, 2))
    dist = np.linalg.norm(ref_xy[:, None, :] - dep_xy[None, :, :], axis=2) * 111.0
    freight = 0.02 + 0.0009 * dist * rng.uniform(0.9, 1.1, size=dist.shape)
    crude_price = rng.uniform(70, 95, size=(C, T))
    sulfur = rng.uniform(0.2, 3.0, size=C)
    smax = rng.uniform(1.2, 2.0, size=R)
    cap = rng.uniform(150, 400, size=R) * (D * P / 240.0)
    minrate = 0.5 * cap
    yields = rng.dirichlet(np.ones(P + 1) * 2, size=(C, R))[:, :, :P].transpose(0, 2, 1)
    base_dem = rng.uniform(2, 12, size=(P, D)) * (cap.sum() * 0.8 / (P * D * 7.0))
    season = 1 + 0.15 * np.sin(np.arange(T) / max(T, 1) * 2 * np.pi)
    dem = base_dem[:, :, None] * season[None, None, :] * rng.uniform(0.85, 1.15, size=(P, D, T))
    tank = base_dem * 1.5
    parcel = cap.sum() / (0.6 * C)                     # a cargo feeds the refineries for a fraction of a period
    cargo_cost = 2.0 * parcel                          # freight and port cost per cargo ($k)
    run_cost = 0.5 * cap                               # fixed cost of running a unit for a period
    start_cost = 4.0 * cap

    cost = np.zeros(n)
    for c in range(C):
        for r in range(R):
            cost[[X(c, r, t) for t in range(T)]] = crude_price[c]
    idx = np.arange(ns)
    cost[os_:os_ + ns] = freight[(idx // (T * D)) % R, (idx // T) % D]
    cost[oI:oI + nI] = 0.15
    cost[oz:oz + nz] = 400.0
    cost[on_:on_ + nn] = cargo_cost
    for r in range(R):
        for t in range(T):
            cost[ON(r, t)] = run_cost[r]
            cost[ST(r, t)] = start_cost[r]

    l = np.zeros(n)
    u = np.full(n, np.inf)
    ii = np.arange(nI)
    u[oI:oI + nI] = tank[(ii // (D * T)) % P, (ii // T) % D]
    u[on_:on_ + nn] = 4.0
    u[oon:] = 1.0
    integer = np.zeros(n, bool)
    integer[on_:ost] = True                            # cargoes and unit modes; start-ups follow from them

    eq_r, eq_c, eq_v, eq_b = [], [], [], []
    ge_r, ge_c, ge_v, ge_b = [], [], [], []

    def add(cols, vals, rhs, eq=False):
        rows, cc, vv, bb = (eq_r, eq_c, eq_v, eq_b) if eq else (ge_r, ge_c, ge_v, ge_b)
        rows.extend([len(bb)] * len(cols)); cc.extend(cols); vv.extend(vals); bb.append(rhs)

    for t in range(T):
        for c in range(C):                             # parcel n[c,t] - sum_r x >= 0
            add([N(c, t)] + [X(c, r, t) for r in range(R)], [parcel] + [-1.0] * R, 0.0)
        for r in range(R):
            run = [X(c, r, t) for c in range(C)]
            add(run + [ON(r, t)], [-1.0] * C + [cap[r]], 0.0)
            add(run + [ON(r, t)], [1.0] * C + [-minrate[r]], 0.0)
            add(run, list(smax[r] - sulfur), 0.0)
            if t > 0:                                  # (units are running when the plan starts)
                add([ST(r, t), ON(r, t), ON(r, t - 1)], [1.0, -1.0, 1.0], 0.0)
            for p in range(P):
                add([Yv(p, r, t)] + run, [1.0] + list(-yields[:, p, r]), 0.0, eq=True)
                add([Yv(p, r, t)] + [S(p, r, d, t) for d in range(D)], [1.0] + [-1.0] * D, 0.0)
        for p in range(P):
            for d in range(D):
                cols = [S(p, r, d, t) for r in range(R)] + [Z(p, d, t), I(p, d, t)]
                vals = [1.0] * R + [1.0, -1.0]
                if t > 0:
                    cols.append(I(p, d, t - 1)); vals.append(1.0)
                add(cols, vals, dem[p, d, t], eq=True)

    Keq = sp.csr_matrix((eq_v, (eq_r, eq_c)), shape=(len(eq_b), n))
    Kge = sp.csr_matrix((ge_v, (ge_r, ge_c)), shape=(len(ge_b), n))
    K = sp.vstack([Keq, Kge]).tocsr()
    q = np.array(eq_b + ge_b)
    return LP(cost, K, q, len(eq_b), l, u, name=f"refinery_plan_R{R}_C{C}_P{P}_D{D}_T{T}_s{seed}", integer=integer)


# Planning MILP sizes for the README and benchmarks/mrpl.py
PLAN_SIZES = {
    "S": dict(R=3, C=6, P=4, D=12, T=6),
    "M": dict(R=4, C=8, P=6, D=30, T=12),
    "L": dict(R=6, C=10, P=6, D=60, T=12),
}


def refinery_milp(R=3, C=6, P=3, D=5, seed=0):
    """Single-period refinery design/contract MILP.

    Binary y[c]: sign a term contract for crude c (fixed cost), binary z[r]: run refinery r (fixed cost).
    Continuous: crude runs x[c,r], products y[p,r], shipments s[p,r,d], shortage w[p,d].
      x[c,r] <= avail_c * y[c] ;  sum_c x[c,r] <= cap_r * z[r] ;  sulfur blend ;  yields ;  dispatch ;  demand
    """
    rng = np.random.default_rng(seed)
    nx, npr, ns, nw = C * R, P * R, P * R * D, P * D
    ox, op, os_, ow, oy, oz = 0, nx, nx + npr, nx + npr + ns, nx + npr + ns + nw, nx + npr + ns + nw + C
    n = oz + R
    X = lambda c, r: ox + c * R + r
    Pv = lambda p, r: op + p * R + r
    S = lambda p, r, d: os_ + (p * R + r) * D + d
    W = lambda p, d: ow + p * D + d

    ref_xy, dep_xy = rng.uniform(0, 10, (R, 2)), rng.uniform(0, 10, (D, 2))
    freight = 0.5 + 0.4 * np.linalg.norm(ref_xy[:, None] - dep_xy[None], axis=2)
    price = rng.uniform(60, 90, C)
    avail = rng.uniform(40, 120, C)
    sulfur = rng.uniform(0.3, 3.0, C)
    smax = rng.uniform(1.2, 2.0, R)
    cap = rng.uniform(100, 200, R)
    yields = rng.dirichlet(np.ones(P + 1) * 2, size=(C, R))[:, :, :P]  # C,R,P
    dem = rng.uniform(5, 25, (P, D))
    fix_c = rng.uniform(300, 900, C)
    fix_r = rng.uniform(1500, 3000, R)

    cost = np.zeros(n)
    for c in range(C):
        for r in range(R):
            cost[X(c, r)] = price[c]
    for p in range(P):
        for r in range(R):
            for d in range(D):
                cost[S(p, r, d)] = freight[r, d]
        for d in range(D):
            cost[W(p, d)] = 400.0
    cost[oy:oy + C] = fix_c
    cost[oz:oz + R] = fix_r
    l = np.zeros(n)
    u = np.full(n, np.inf)
    u[oy:] = 1.0
    integer = np.zeros(n, bool); integer[oy:] = True

    eq, ge = [], []  # (cols, vals, rhs)
    for c in range(C):
        for r in range(R):
            ge.append(([X(c, r), oy + c], [-1.0, avail[c]], 0.0))
    for r in range(R):
        ge.append(([X(c, r) for c in range(C)] + [oz + r], [-1.0] * C + [cap[r]], 0.0))
        ge.append(([X(c, r) for c in range(C)], list(smax[r] - sulfur), 0.0))
        for p in range(P):
            eq.append(([Pv(p, r)] + [X(c, r) for c in range(C)], [1.0] + list(-yields[:, r, p]), 0.0))
            ge.append(([Pv(p, r)] + [S(p, r, d) for d in range(D)], [1.0] + [-1.0] * D, 0.0))
    for p in range(P):
        for d in range(D):
            eq.append(([S(p, r, d) for r in range(R)] + [W(p, d)], [1.0] * (R + 1), dem[p, d]))

    def build(rows):
        rr, cc, vv = [], [], []
        for i, (cols, vals, _) in enumerate(rows):
            rr += [i] * len(cols); cc += cols; vv += vals
        return sp.csr_matrix((vv, (rr, cc)), shape=(len(rows), n))
    K = sp.vstack([build(eq), build(ge)]).tocsr()
    q = np.array([b for *_, b in eq] + [b for *_, b in ge])
    return LP(cost, K, q, len(eq), l, u, name=f"refinery_milp_R{R}_C{C}_P{P}_D{D}_s{seed}", integer=integer)
