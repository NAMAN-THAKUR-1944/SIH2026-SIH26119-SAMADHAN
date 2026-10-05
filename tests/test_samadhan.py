"""Correctness tests: every SAMADHAN component is checked against a hand-computed optimum or against HiGHS.

    python -m pytest -q
"""
import dataclasses
from pathlib import Path

import pytest
import torch

from samadhan.baseline import solve_highs
from samadhan.generate import REFINERY_SIZES, refinery_lp, refinery_milp
from samadhan.milp import solve_milp
from samadhan.mps import read_mps
from samadhan.pdlp import PDLP
from samadhan.simplex import solve_lp_dense

TINY = Path(__file__).with_name("tiny.mps")
TINY_MILP = Path(__file__).with_name("tiny_milp.mps")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def test_mps_reader_tiny():
    lp = read_mps(TINY)
    assert lp.K.shape == (5, 3) and lp.n_eq == 1
    assert solve_highs(lp)["obj"] == pytest.approx(-36.0)  # max 3x + 5y -> x = 2, y = 6


def test_mps_integer_default_bounds():
    """MARKER integer columns without any BOUNDS record are binary; any bound record keeps the usual defaults
    (the convention of MIPLIB and HiGHS)."""
    import numpy as np
    lp = read_mps(Path(__file__).with_name("int_bounds.mps"))
    inf = np.inf
    assert list(zip(lp.l, lp.u)) == [(0, 1), (2, inf), (0, 5), (-1, inf), (-inf, inf), (0, inf)]
    assert list(lp.integer) == [True] * 5 + [False]


def test_simplex_tiny():
    status, x, obj = solve_lp_dense(read_mps(TINY))
    assert status == "optimal" and obj == pytest.approx(-36.0) and x[:2] == pytest.approx([2.0, 6.0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("adaptive", [True, False])
def test_pdlp_tiny(device, adaptive):
    r = PDLP(read_mps(TINY), device=device).solve(tol=1e-8, adaptive=adaptive)
    assert r.status == "optimal" and r.primal_obj == pytest.approx(-36.0, rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_pdlp_refinery_matches_highs(device):
    lp = refinery_lp(R=3, C=5, P=4, D=20, T=4, seed=1)
    ref = solve_highs(lp)["obj"]
    r = PDLP(lp, device=device).solve(tol=1e-6, max_iter=200_000)
    assert r.status == "optimal" and rel(r.primal_obj, ref) < 1e-5


def test_simplex_matches_highs_on_relaxation():
    lp = dataclasses.replace(refinery_milp(seed=3), integer=None)
    status, _, obj = solve_lp_dense(lp)
    assert status == "optimal" and rel(obj, solve_highs(lp)["obj"]) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_milp_matches_highs(seed):
    lp = refinery_milp(seed=seed)
    r = solve_milp(lp)
    assert r.status == "optimal" and rel(r.obj, solve_highs(lp)["obj"]) < 1e-9


# ---------------------------------------------------------------- C++ core (dual simplex + branch-and-cut)
def test_core_lp_matches_highs():
    from samadhan.core import solve_core
    lp = refinery_lp(R=3, C=5, P=4, D=20, T=4, seed=1)
    r = solve_core(lp)
    assert r.status == "optimal" and rel(r.obj, solve_highs(lp)["obj"]) < 1e-9


def test_core_tiny():
    from samadhan.core import solve_core
    r = solve_core(read_mps(TINY))
    assert r.status == "optimal" and r.obj == pytest.approx(-36.0) and r.x[:2] == pytest.approx([2.0, 6.0])


@pytest.mark.parametrize("seed", range(5))
def test_core_milp_matches_highs(seed):
    from samadhan.core import solve_core
    lp = refinery_milp(R=4, C=8, P=3, D=6, seed=seed)
    r = solve_core(lp, time_limit=60)
    assert r.status == "optimal" and rel(r.obj, solve_highs(lp)["obj"]) < 1e-9


@pytest.mark.parametrize("features", [0, 1, 2, 4, 8, 16, 32, 63])
def test_core_milp_every_feature(features):
    """Each branch-and-cut feature (pump, diving, covers, reliability branching, propagation, c-MIR cuts) on its own
    and all together must reach the same proven optimum."""
    from samadhan.core import solve_core
    from samadhan.verify import violation
    for seed in range(3):
        lp = refinery_milp(R=4, C=8, P=3, D=6, seed=seed)
        r = solve_core(lp, time_limit=60, features=features)
        assert r.status == "optimal" and rel(r.obj, solve_highs(lp)["obj"]) < 1e-9 and violation(lp, r.x) < 1e-6


def test_core_milp_without_cuts():
    from samadhan.core import solve_core
    lp = refinery_milp(seed=1)
    r = solve_core(lp, cut_rounds=0)
    assert r.status == "optimal" and rel(r.obj, solve_highs(lp)["obj"]) < 1e-9


def test_core_tiny_milp_integer_optimum():
    """The LP relaxation gives x = 3, y = 1.5 (-21); the integer optimum is x = 4, y = 0 (-20)."""
    from samadhan.core import solve_core
    lp = read_mps(TINY_MILP)
    assert lp.integer is not None and lp.integer.all()
    r = solve_core(lp)
    assert r.status == "optimal" and r.obj == pytest.approx(-20.0) and r.x == pytest.approx([4.0, 0.0])


def test_core_lp_beyond_dense_limit():
    """2,544 rows: needs the sparse LU (the old dense basis inverse stopped at about 1,500 rows)."""
    from samadhan.core import solve_core
    lp = refinery_lp(seed=7, **REFINERY_SIZES["S"])
    r = solve_core(lp)
    assert r.status == "optimal" and rel(r.obj, solve_highs(lp)["obj"]) < 1e-9


def test_core_lp_status_cases():
    import numpy as np
    import scipy.sparse as sp

    from samadhan.core import solve_core
    from samadhan.lp import LP
    inf = np.inf
    # unbounded ray (t, t):  min -x - y,  x - y >= -1,  x, y >= 0
    lp = LP(np.array([-1.0, -1.0]), sp.csr_matrix([[1.0, -1.0]]), np.array([-1.0]), 0, np.zeros(2), np.full(2, inf))
    assert solve_core(lp).status == "unbounded"
    # free variables:  min x + y,  x - y = 0,  x + y >= 2   ->  (1, 1)
    lp = LP(np.ones(2), sp.csr_matrix([[1.0, -1.0], [1.0, 1.0]]), np.array([0.0, 2.0]), 1, np.full(2, -inf),
            np.full(2, inf))
    r = solve_core(lp)
    assert r.status == "optimal" and r.obj == pytest.approx(2.0) and r.x == pytest.approx([1.0, 1.0])
    # infeasible:  x >= 3  and  x <= 1
    lp = LP(np.ones(1), sp.csr_matrix([[1.0], [-1.0]]), np.array([3.0, -1.0]), 0, np.zeros(1), np.full(1, inf))
    assert solve_core(lp).status == "infeasible"


def test_presolve_reductions():
    """Each reduction fires on a small model, and the postsolved point is the true optimum of the original."""
    import numpy as np
    import scipy.sparse as sp

    from samadhan.lp import LP
    from samadhan.presolve import presolve
    from samadhan.verify import violation
    inf = np.inf
    # x0 fixed (l = u = 2); row 1 singleton 2 x1 >= 2 (-> x1 >= 1); row 2 redundant x1 + x2 >= -5;
    # row 3 forcing  -x3 - x4 >= 0 with x3, x4 >= 0 (-> x3 = x4 = 0); x5 in no row (empty column, cost -1 -> u = 3)
    K = sp.csr_matrix(np.array([[1.0, 1, 1, 0, 0, 0],      # eq:  x0 + x1 + x2 = 6
                                [0, 2, 0, 0, 0, 0],
                                [0, 1, 1, 0, 0, 0],
                                [0, 0, 0, -1, -1, 0]]))
    lp = LP(np.array([1.0, 1, 2, 1, 1, -1]), K, np.array([6.0, 2, -5, 0]), 1, np.array([2.0, 0, 0, 0, 0, 0]),
            np.array([2.0, inf, inf, inf, inf, 3]))
    P = presolve(lp)
    st = P.stats
    assert P.status == "reduced" and st["fixed_cols"] >= 1 and st["singleton_rows"] >= 1
    # x3, x4 (forcing row) and x5 (empty column) may instead be removed earlier by dual fixing
    assert st["redundant_rows"] >= 1 and st["forcing_rows"] + st["empty_cols"] + st["dual_fixed"] >= 2
    assert P.lp.K.shape[1] <= 2
    ref = solve_highs(lp)["obj"]                       # x = (2, 4, 0, 0, 0, 3): 2 + 4 - 3 = 3
    r = solve_highs(P.lp)
    assert r["obj"] == pytest.approx(ref) == pytest.approx(3.0)
    from samadhan.core import solve_core
    rc = solve_core(lp)
    assert rc.status == "optimal" and rc.obj == pytest.approx(3.0) and violation(lp, rc.x) < 1e-9


def test_presolve_free_singleton_and_dual_fixing():
    """min x1 + x2 + 2y  s.t.  x1 + x2 - s = 3 (s in [0, 10], implied free: x in [2, 4] gives s in [1, 5]),
    x2 + y <= 10 (y has no down-lock and cost 2: fixed at 0)  ->  x = (2, 2), s = 1, y = 0, value 4."""
    import numpy as np
    import scipy.sparse as sp

    from samadhan.lp import LP
    from samadhan.presolve import presolve
    from samadhan.verify import violation
    K = sp.csr_matrix(np.array([[1.0, 1.0, -1.0, 0.0],     # eq: x1 + x2 - s = 3
                                [0.0, -1.0, 0.0, -1.0]]))  # ge: -x2 - y >= -10
    lp = LP(np.array([1.0, 1.0, 0.0, 2.0]), K, np.array([3.0, -10.0]), 1, np.array([2.0, 2.0, 0.0, 0.0]),
            np.array([4.0, 4.0, 10.0, 5.0]))
    P = presolve(lp)
    assert P.stats["free_singletons"] >= 1 and P.stats["dual_fixed"] >= 1
    from samadhan.core import solve_core
    r = solve_core(lp)
    assert r.status == "optimal" and r.obj == pytest.approx(4.0) and violation(lp, r.x) < 1e-9
    assert r.x == pytest.approx([2.0, 2.0, 1.0, 0.0])


def test_presolve_detects_infeasible():
    import numpy as np
    import scipy.sparse as sp

    from samadhan.lp import LP
    from samadhan.presolve import presolve
    # x + y >= 5 with 0 <= x, y <= 2: activity bound 4 < 5
    lp = LP(np.ones(2), sp.csr_matrix([[1.0, 1.0]]), np.array([5.0]), 0, np.zeros(2), np.full(2, 2.0))
    assert presolve(lp).status == "infeasible"


@pytest.mark.parametrize("seed", range(3))
def test_presolve_keeps_milp_optimum(seed):
    from samadhan.core import solve_core
    from samadhan.verify import violation
    lp = refinery_milp(R=4, C=8, P=3, D=6, seed=seed)
    a, b = solve_core(lp, presolve=True), solve_core(lp, presolve=False)
    assert a.status == b.status == "optimal" and rel(a.obj, b.obj) < 1e-9 and violation(lp, a.x) < 1e-6


@pytest.mark.parametrize("start", ["gpu", "zeros"])
def test_crossover_gives_exact_vertex(start):
    """GPU solution (1e-4) or a poor starting point -> optimal vertex, equal to HiGHS to 1e-9."""
    import numpy as np

    from samadhan.core import crossover
    from samadhan.verify import violation
    lp = refinery_lp(R=3, C=5, P=4, D=20, T=4, seed=1)
    x0 = PDLP(lp, device="cpu").solve(tol=1e-4).x if start == "gpu" else np.zeros(lp.K.shape[1])
    r = crossover(lp, x0)
    assert r.status == "optimal" and rel(r.obj, solve_highs(lp)["obj"]) < 1e-9 and violation(lp, r.x) < 1e-9


def test_gpu_solve_with_crossover():
    from samadhan.pdlp import solve
    lp = refinery_lp(R=3, C=5, P=4, D=20, T=4, seed=1)
    r = solve(lp, device="cpu", tol=1e-4, crossover=True)
    assert r.vertex and r.status == "optimal" and rel(r.primal_obj, solve_highs(lp)["obj"]) < 1e-9


@pytest.mark.parametrize("crossover", [False, True])
def test_ipm_matches_highs(crossover):
    """Interior point alone reaches 1e-8; with crossover the vertex equals the HiGHS optimum to 1e-9."""
    from samadhan.ipm import solve_ipm
    from samadhan.verify import violation
    lp = refinery_lp(R=3, C=5, P=4, D=20, T=4, seed=1)
    r = solve_ipm(lp, crossover=crossover)
    ref = solve_highs(lp)["obj"]
    assert r.status == "optimal" and r.vertex == crossover
    assert rel(r.primal_obj, ref) < (1e-9 if crossover else 1e-6) and violation(lp, r.x) < 1e-6


def test_ipm_free_and_upper_bounded_columns():
    """min x + y - z  s.t.  x - y = 0,  x + y >= 2,  z <= 3 (only an upper bound),  x, y free  ->  -1."""
    import numpy as np
    import scipy.sparse as sp

    from samadhan.ipm import solve_ipm
    from samadhan.lp import LP
    inf = np.inf
    lp = LP(np.array([1.0, 1.0, -1.0]), sp.csr_matrix([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]), np.array([0.0, 2.0]), 1,
            np.array([-inf, -inf, -inf]), np.array([inf, inf, 3.0]))
    r = solve_ipm(lp, crossover=True)
    assert r.status == "optimal" and r.primal_obj == pytest.approx(-1.0) and r.x == pytest.approx([1.0, 1.0, 3.0])


def test_sparse_lu_symmetric_mode():
    """Diagonal-pivot (LDL') mode on a symmetric positive definite matrix with a wide diagonal range."""
    import numpy as np
    import scipy.sparse as sp

    from samadhan.core import SparseLU
    rng = np.random.default_rng(1)
    A = sp.random(60, 200, density=0.05, random_state=rng)
    M = (A @ sp.diags(10.0 ** rng.uniform(-6, 6, 200)) @ A.T + 1e-8 * sp.identity(60)).tocsc()
    b = rng.normal(size=60)
    lu = SparseLU(M, symmetric=True)
    assert lu.ok and lu.rank == 60
    x = lu.solve(b)
    assert np.linalg.norm(M @ x - b) <= 1e-9 * (np.linalg.norm(M.toarray(), np.inf) * np.linalg.norm(x) + 1)


def test_sparse_lu_matches_numpy():
    """C++ sparse LU: solves with B and B', a product-form column replacement, and rank detection."""
    import numpy as np
    import scipy.sparse as sp

    from samadhan.core import lu_check
    rng = np.random.default_rng(0)

    def backward_ok(A, v, rhs):
        return np.linalg.norm(A @ v - rhs, np.inf) <= 1e-10 * len(rhs) * (
            np.linalg.norm(A, np.inf) * np.linalg.norm(v, np.inf) + np.linalg.norm(rhs, np.inf))

    for _ in range(150):
        m = int(rng.integers(1, 40))
        B = sp.random(m, m, density=rng.uniform(0.05, 0.4), random_state=rng).toarray()
        for j in range(m):
            if rng.random() < 0.4:                       # logical (slack) columns, as in a simplex basis
                B[:, j] = 0.0
                B[rng.integers(m), j] = -1.0
        B += np.diag(rng.normal(size=m)) * (rng.random(m) < 0.7)
        b, d = rng.normal(size=m), rng.normal(size=m)
        rank, x, y = lu_check(B, b, d)
        if np.linalg.matrix_rank(B) < m:
            assert rank < m
            continue
        assert rank == m and backward_ok(B, x, b) and backward_ok(B.T, y, d)
        r, col = int(rng.integers(m)), rng.normal(size=m)
        B2 = B.copy()
        B2[:, r] = col
        if np.linalg.cond(B2) < 1e8:
            _, x2, y2 = lu_check(B, b, d, replace=(r, col))
            assert backward_ok(B2, x2, b) and backward_ok(B2.T, y2, d)


def test_cli_routes_milp_to_core(capsys):
    from samadhan.__main__ import main
    assert main(["solve", str(TINY_MILP), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "C++ core" in out and "-20" in out


def test_cli_verify(capsys):
    from samadhan.__main__ import main
    assert main(["verify", "--device", "cpu"]) == 0
    assert "12/12 checks passed" in capsys.readouterr().out


def test_cli_lp_on_cpu(capsys):
    from samadhan.__main__ import main
    assert main(["solve", str(TINY), "--engine", "gpu", "--device", "cpu", "--tol", "1e-8", "--quiet"]) == 0
    assert "GPU engine" in capsys.readouterr().out


def test_cli_auto_engine():
    """auto: MILP and small LP -> C++ core, large LP and QP -> GPU engine."""
    import dataclasses

    from samadhan.__main__ import pick_engine
    assert pick_engine(read_mps(TINY)) == "core" and pick_engine(read_mps(TINY_MILP)) == "core"
    assert pick_engine(refinery_lp(seed=7, **REFINERY_SIZES["M"])) == "gpu"
    assert pick_engine(dataclasses.replace(read_mps(TINY), Q=None)) == "core"
    assert pick_engine(refinery_lp(R=3, C=5, P=4, D=20, T=4, seed=1, quad=0.3)) == "gpu"


# ---------------------------------------------------------------- QP (restarted PDHG with a quadratic term)
@pytest.mark.parametrize("device", DEVICES)
def test_qp_refinery_matches_highs(device):
    lp = refinery_lp(R=3, C=5, P=4, D=20, T=4, seed=1, quad=0.3)
    ref = solve_highs(lp)["obj"]
    r = PDLP(lp, device=device).solve(tol=1e-6, max_iter=2_000_000)
    assert r.status == "optimal" and rel(r.primal_obj, ref) < 1e-5


def test_maros_reader_infinity_encoding(tmp_path):
    """Maros-Meszaros files encode infinity as +-1e20, sometimes as -9.999999999999998e19: both must be infinite."""
    import numpy as np
    import scipy.io as sio
    import scipy.sparse as sp

    from samadhan.qpdata import read_maros
    f = tmp_path / "t.mat"
    sio.savemat(f, dict(P=sp.csc_matrix(np.eye(2)), q=np.zeros((2, 1)), r=np.zeros((1, 1)),
                        A=sp.csc_matrix(np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])),
                        l=np.array([[-9.999999999999998e19], [0.0], [0.0]]), u=np.array([[4.0], [1e20], [3.0]])))
    lp = read_maros(f)
    assert lp.K.shape == (1, 2) and lp.q[0] == -4.0              # only  x1 + x2 <= 4  remains as a row
    assert list(lp.l) == [0.0, 0.0] and lp.u[0] == np.inf and lp.u[1] == 3.0


def test_qp_maros_meszaros_hs21():
    """HS21 from the Maros-Meszaros set, written inline: min 0.01 x1^2 + x2^2 - 100, 10 x1 - x2 >= 10."""
    import numpy as np
    import scipy.sparse as sp

    from samadhan.lp import LP
    lp = LP(np.zeros(2), sp.csr_matrix([[10.0, -1.0]]), np.array([10.0]), 0, np.array([2.0, -50.0]),
            np.array([50.0, 50.0]), obj_const=-100.0, Q=sp.csr_matrix(np.diag([0.02, 2.0])))
    r = PDLP(lp, device="cpu").solve(tol=1e-8, max_iter=1_000_000)
    assert r.status == "optimal" and r.primal_obj == pytest.approx(-99.96, abs=1e-5)
