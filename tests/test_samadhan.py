"""Correctness tests: every SAMADHAN component is checked against a hand-computed optimum or against HiGHS.

    python -m pytest -q
"""
import dataclasses
from pathlib import Path

import pytest
import torch

from samadhan.baseline import solve_highs
from samadhan.generate import refinery_lp, refinery_milp
from samadhan.milp import solve_milp
from samadhan.mps import read_mps
from samadhan.pdlp import PDLP
from samadhan.simplex import solve_lp_dense

TINY = Path(__file__).with_name("tiny.mps")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def test_mps_reader_tiny():
    lp = read_mps(TINY)
    assert lp.K.shape == (5, 3) and lp.n_eq == 1
    assert solve_highs(lp)["obj"] == pytest.approx(-36.0)  # max 3x + 5y -> x = 2, y = 6


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


def test_core_milp_without_cuts():
    from samadhan.core import solve_core
    lp = refinery_milp(seed=1)
    r = solve_core(lp, cut_rounds=0)
    assert r.status == "optimal" and rel(r.obj, solve_highs(lp)["obj"]) < 1e-9


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
