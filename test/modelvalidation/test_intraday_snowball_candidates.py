"""QUAD V2 and PDE intraday candidates: normal-API values, per-output statuses, three-level axes, identities."""
import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.builders import intraday_snowball
from quantark.modelvalidation.candidate import candidate_identity, convergence_evidence
from quantark.modelvalidation.evidence import identity_hash
from quantark.modelvalidation.registry import get_builder
from quantark.modelvalidation.study import CaseSpec
from quantark.modelvalidation.yaml_loader import _ensure_builtin_builders

from test_intraday_common import CONTEXT, ENV, PRODUCT

QUANTITIES = ("pv", "desk_delta", "desk_gamma", "desk_theta", "point_delta", "point_gamma")
ON_BARRIER = CaseSpec(name="on_ki_barrier_at_close", environment_params={"spot": 75.0},
                      context_params={"valuation": "2026-09-10T15:00:00+08:00"},
                      expected={"point_delta": "undefined", "point_gamma": "undefined", "desk_theta": "undefined"})


def _candidate(name, env=ENV, **params):
    _ensure_builtin_builders()
    return get_builder(name, kind="candidate")(environment_params=env, product_params=PRODUCT, quantities=QUANTITIES,
                                               params=params, context_params=CONTEXT)


def test_builders_require_a_context():
    _ensure_builtin_builders()
    with pytest.raises(ValidationError, match="context"):
        get_builder("equity.snowball.intraday.quad_v2", kind="candidate")(
            environment_params=ENV, product_params=PRODUCT, quantities=("pv",), params={})


def test_quad_records_its_resolved_quadrature_desk_bump_and_axes():
    quad = _candidate("equity.snowball.intraday.quad_v2", order=8, cells_per_sd=2.0)
    assert quad.name() == "equity.snowball.intraday.quad_v2"
    params = quad.params()
    assert params["engine"] == "SnowballQuadEngineV2" and params["desk_bump"] == 0.01
    assert params["grid"]["cells_per_sd"] == 2.0 and params["grid"]["order"] == 8 and params["grid"]["domain_sd"] == 11.0
    assert "backend" not in params["grid"] and "max_nodes" not in params["grid"]
    assert params["convergence_axes"] == {"cells_per_sd": [2.0, 4.0, 8.0], "order": [8, 12, 16], "domain_sd": [11.0, 13.0, 15.0]}
    refined = _candidate("equity.snowball.intraday.quad_v2", order=8, cells_per_sd=4.0, label="cells4")
    assert refined.name() == "equity.snowball.intraday.quad_v2.cells4" and "label" not in refined.params()


def test_identities_cover_the_clock_the_market_and_the_implementation():
    """Review R1: a same-name context change and a study-level market change both move the identity."""
    quad = _candidate("equity.snowball.intraday.quad_v2")
    early = CaseSpec(name="near_ki", environment_params={"spot": 75.2})
    late = CaseSpec(name="near_ki", environment_params={"spot": 75.2}, context_params={"valuation": "2026-09-10T14:59:59+08:00"})
    assert identity_hash(candidate_identity(quad, early, schema=2)) != identity_hash(candidate_identity(quad, late, schema=2))
    high_vol = _candidate("equity.snowball.intraday.quad_v2", env={**ENV, "vol": 0.35})
    assert identity_hash(candidate_identity(quad, early, schema=2)) != identity_hash(candidate_identity(high_vol, early, schema=2))
    pde = _candidate("equity.snowball.intraday.pde")
    assert len(quad.fingerprint()) == 64 and quad.fingerprint() != pde.fingerprint()


def test_quad_evaluates_every_quantity_with_three_levels_on_three_axes():
    result = _candidate("equity.snowball.intraday.quad_v2").evaluate(CaseSpec(name="ordinary"))
    assert set(result.values) == set(QUANTITIES) and all(result.status(q) == "ok" for q in QUANTITIES)
    assert {axis.name: len(axis.levels) for axis in result.convergence} == {"cells_per_sd": 3, "order": 3, "domain_sd": 3}
    for axis in result.convergence:
        target = next(level for level in axis.levels if level.is_target)
        assert target.values == result.values                       # the shipped output, never replaced by a finer level
        assert axis.name in target.settings or "cells_per_sd" in target.settings
    for quantity in QUANTITIES:
        evidence = convergence_evidence(result, quantity)
        assert evidence.complete and evidence.envelope is not None and evidence.envelope >= 0.0


def test_on_the_barrier_at_the_close_the_point_derivatives_and_theta_are_undefined():
    result = _candidate("equity.snowball.intraday.quad_v2").evaluate(ON_BARRIER)
    assert result.status("point_delta") == "undefined" and result.status("point_gamma") == "undefined"
    assert result.status("desk_theta") == "undefined"
    assert "point_delta" not in result.values and result.status("pv") == "ok" and result.status("desk_delta") == "ok"
    assert "discontinuity" in result.reasons["point_delta"]


def test_pde_axes_double_the_geometry_it_actually_solved_on():
    pde = _candidate("equity.snowball.intraday.pde", accuracy="standard")
    assert pde.name() == "equity.snowball.intraday.pde"
    assert pde.params()["grid"]["points"] == 400 and pde.params()["engine"] == "SnowballPDESolver"
    result = pde.evaluate(CaseSpec(name="ordinary"))
    assert set(result.values) == set(QUANTITIES)
    axes = {axis.name: axis for axis in result.convergence}
    assert set(axes) == {"space", "time", "placement"} and axes["placement"].kind == "placement"
    solved = next(level for level in axes["space"].levels if level.is_target).settings
    assert solved["points"] >= 400                                   # the route's achieved grid, not the request
    finer = [level.settings["points"] for level in axes["space"].levels if not level.is_target]
    assert finer == [2 * solved["points"], 4 * solved["points"]]
    assert [lv.settings["steps_per_day"] for lv in axes["time"].levels if not lv.is_target] == [
        2 * solved["steps_per_day"], 4 * solved["steps_per_day"]]
    assert [lv.settings["points"] - solved["points"] for lv in axes["placement"].levels if not lv.is_target] == [1, 2, 3]


def test_a_level_beyond_the_memory_cap_is_skipped_and_leaves_the_axis_incomplete(monkeypatch):
    monkeypatch.setattr(intraday_snowball, "LADDER_MAX_GRID_CELLS", 1)     # nothing finer than the target fits
    result = _candidate("equity.snowball.intraday.pde").evaluate(CaseSpec(name="ordinary"))
    evidence = convergence_evidence(result, "desk_gamma")
    assert not evidence.complete and set(evidence.missing) == {"space", "time", "placement"}


def test_evaluate_target_is_the_shipped_output_without_the_axes():
    quad = _candidate("equity.snowball.intraday.quad_v2")
    full, target = quad.evaluate(CaseSpec(name="ordinary")), quad.evaluate_target(CaseSpec(name="ordinary"))
    assert target.values == full.values and target.convergence == ()


def test_terminated_claim_reports_zero_spot_greeks_ok():
    case = CaseSpec(name="terminated", product_params={"settlement_lag_days": 30},
                    context_params={"fixings": [{"date": "2026-08-17", "level": 104.0}]})
    result = _candidate("equity.snowball.intraday.quad_v2").evaluate(case)
    assert result.values["pv"] > 0.0 and result.values["point_delta"] == 0.0 and result.values["desk_gamma"] == 0.0
