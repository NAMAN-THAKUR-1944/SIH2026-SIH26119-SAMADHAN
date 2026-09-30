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
