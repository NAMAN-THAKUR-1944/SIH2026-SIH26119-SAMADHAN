"""Crude selection for one month, written in PuLP and solved by SAMADHAN. The only SAMADHAN-specific line is the
solve call; everything else is ordinary PuLP. The numbers are illustrative.

    pip install pulp            (and, to compare with HiGHS, highspy)
    python -m examples.pulp_crude_selection          (from the repository root)

Crude comes in whole cargoes (integer), the crude unit either runs between its minimum and maximum rate or is
shut down (binary), the slate must meet a sulfur limit and a diesel commitment, and the margin is maximised.
"""
import pulp

from samadhan.pulp_solver import SAMADHAN

# crude: price $/bbl, sulfur %, yields of naphtha, diesel, fuel oil (fractions of the crude run)
CRUDES = {
    "Arab Light": (82.0, 1.8, (0.22, 0.38, 0.40)),
    "Basrah Medium": (79.5, 2.9, (0.19, 0.35, 0.46)),
    "Bonny Light": (86.0, 0.15, (0.26, 0.45, 0.29)),
    "Murban": (85.0, 0.8, (0.27, 0.42, 0.31)),
}
PRODUCTS = ("naphtha", "diesel", "fuel oil")
PRICE = {"naphtha": 80.0, "diesel": 108.0, "fuel oil": 70.0}          # $/bbl
DIESEL_COMMITMENT = 700.0                                              # kbbl
CARGO = 500.0                                                          # kbbl per cargo, at most 3 per crude
CDU_MIN, CDU_MAX, CDU_COST = 1000.0, 2500.0, 1500.0                    # kbbl, kbbl, k$ per month while running
SULFUR_MAX = 1.6                                                       # % (volume average of the slate)


def build():
    prob = pulp.LpProblem("crude_selection", pulp.LpMaximize)
    cargoes = {c: prob.add_variable(f"cargoes_{i}", lowBound=0, upBound=3, cat="Integer") for i, c in enumerate(CRUDES)}
    run = prob.add_variable("cdu_running", lowBound=0, upBound=1, cat="Binary")
    crude = {c: CARGO * cargoes[c] for c in CRUDES}                    # kbbl processed
    total = pulp.lpSum(crude.values())
    made = {p: pulp.lpSum(CRUDES[c][2][k] * crude[c] for c in CRUDES) for k, p in enumerate(PRODUCTS)}

    prob += (pulp.lpSum(PRICE[p] * made[p] for p in PRODUCTS)          # k$ = $/bbl x kbbl
             - pulp.lpSum(CRUDES[c][0] * crude[c] for c in CRUDES) - CDU_COST * run)
    prob += total <= CDU_MAX * run
    prob += total >= CDU_MIN * run
    prob += pulp.lpSum(CRUDES[c][1] * crude[c] for c in CRUDES) <= SULFUR_MAX * total
    prob += made["diesel"] >= DIESEL_COMMITMENT
    return prob, cargoes, run


def main():
    prob, cargoes, run = build()
    stats = prob.solve(SAMADHAN(msg=False, timeLimit=30))               # <- the one SAMADHAN line
    print(f"SAMADHAN: {stats.status.name}, margin {pulp.value(prob.objective):,.1f} k$")
    for c, v in cargoes.items():
        if v.varValue:
            print(f"  {c:14s} {v.varValue:.0f} cargo(es)")
    print(f"  crude unit running: {'yes' if run.varValue else 'no'}")

    try:
        other, *_ = build()
        other.solve(pulp.HiGHS(msg=False))
        print(f"HiGHS:    margin {pulp.value(other.objective):,.1f} k$ (referee)")
    except Exception as e:                                             # noqa: BLE001 (HiGHS is optional here)
        print(f"(HiGHS not available for comparison: {e})")


if __name__ == "__main__":
    main()
