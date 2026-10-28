"""Reference construction (spec D12) and the European self-convergence check (AC4)."""
from __future__ import annotations

import pytest

from example.case_studies.localvol_transition_quad import reference
from example.case_studies.localvol_transition_quad.adapters import environment, production_vanilla
from example.case_studies.localvol_transition_quad.contract import vanilla_contract
from example.case_studies.localvol_transition_quad.engine import OUTPUTS, LocalVolQuad
from example.case_studies.localvol_transition_quad.grid import GridConfig
from example.case_studies.localvol_transition_quad.kernels import CTMCKernel
from example.case_studies.localvol_transition_quad.market import load_fixture


def _rung(**values):
    base = dict.fromkeys(OUTPUTS, 0.0)
    base.update(values)
    return {"values": base}


def test_assess_requires_both_pde_step_and_quad_gap_within_a_fifth_budget():
    case = {"pde": {"a": _rung(pv=0.9, theta=1e-4), "b": _rung(pv=0.9 + 1e-5, theta=1e-4)},
            "quad": {"q": _rung(pv=0.9 + 1e-5 + 1.9e-5, theta=1e-4)}}
    verdict = reference.assess(case)
    assert verdict["pde_converged"] and verdict["quad_confirms"]
    case["quad"]["q"] = _rung(pv=0.9 + 1e-5 + 2.1e-5, theta=1e-4)
    assert not reference.assess(case)["quad_confirms"]
    case["pde"]["b"] = _rung(pv=0.9, theta=1e-4 + 3e-6)        # theta floor 1e-5: a 1/5 step is 2e-6
    assert not reference.assess(case)["pde_converged"]


def test_budget_matches_spec_floors():
    assert reference.budget("pv", 0.9) == 1e-4
    assert reference.budget("gamma", -3.5) == pytest.approx(0.035)
    assert reference.budget("theta", 1e-4) == 1e-5


@pytest.mark.slow
@pytest.mark.parametrize("maturity", [0.5, 1.0])
@pytest.mark.parametrize("strike_m", [0.9, 1.0, 1.1])
def test_ac4_european_puts_self_converge_to_refined_production_pde(maturity, strike_m):
    m = load_fixture()
    c = vanilla_contract(m.spot, maturity, strike_m=strike_m, option="put")
    from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
    pde = LocalVolPDESolver(params=reference.pde_params(3200, 32), local_vol_surface=m.surface).price(
        production_vanilla(c), environment(m)) / m.spot
    gaps = [abs(LocalVolQuad(CTMCKernel(), GridConfig(cells=200, half_width_std=6.0, subdivide=s))
                .price(c, m)["values"]["pv"] - pde) for s in (1, 3, 9)]
    assert gaps[0] > gaps[1] > gaps[2], gaps
    assert gaps[2] < 1e-5
