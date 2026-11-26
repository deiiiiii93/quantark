"""QUAD V1 statistics on the forward route (0.16.0): every outcome column off one adjoint forward pass.

The default statistics read the knock-out units and the terminal maturity / knocked-in settlement columns off the
march's adjoint forward pass instead of marching one backward column per outcome (the 0.13.0 cost: one row per
knock-out record). The stacked route stays for a march the adjoint cannot transpose, and as the reference here:

* AC1: forward and stacked agree within the adjoint identity (knock-out and terminal probabilities 1e-12, cash
  1e-10 on a 100 notional); the knock-in path fields are bitwise.
* AC1b (spec D7): the Phoenix coupon table and its EXPIRY payment buckets come off the same pass (extra dual rows),
  within the same bounds; the forward march carries O(1) backward rows (the terminal rows and the EXPIRY ``W``).
* AC3: a refused transpose re-runs the stacked march and gives its numbers bitwise.
* price() is untouched (the statistics' ``pv`` is ``price()`` on both routes).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.core.statistics.adjoint import AdjointUnavailable
from quantark.asset.equity.engine.quad import v1_statistics

from test_asset.equity.engine_quad.test_quad_v1_event_statistics import ADJOINT_CASES, adjoint_case
from test_asset.equity.engine_quad.test_quad_v1_exact_mass import CASES as EXACT_MASS
from test_asset.equity.engine_quad.test_quad_v1_exact_mass import _env as exact_env

PROBABILITY_TOL = 1e-12
TERMINAL_TOL = 1e-12
CASH_TOL = 1e-10
#: fields whose numbers come from the outcome columns (compared to tolerance); everything else is bitwise. The Phoenix
#: coupon table (probability, cash, EXPIRY earned) and its ledger buckets come off the forward pass too (spec D7).
OUTCOME_PATHS = ("observations", "maturity", "ki_settlement", "ledger", "total_discounted_cash", "expected_lifetime",
                 "probability_residual", "reconciliation_error", "knock_in.expected_loss",
                 # derived from the knock-out probabilities: P(KI and no KO) = P(KI settlement), coupon survival
                 "knock_in.probability_no_ko", "coupons", "total_cash_to_pay")


def _stats(engine, product, market, route):
    engine._v1_statistics_route = route
    return engine.calculate_event_stats(product, market)


def _near(a, b, path, tol):
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape, path
        if a.dtype.kind in "fc":
            np.testing.assert_allclose(a, b, rtol=0.0, atol=tol, err_msg=path)
        else:
            assert np.array_equal(a, b), path
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol, f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, path


def _tol(path):
    if path.endswith(("discounted_cash", "cash_to_pay", "expected_loss", "total_discounted_cash", "total_cash_to_pay",
                      "reconciliation_error")):
        return CASH_TOL
    if path.startswith(("maturity", "ki_settlement", "probability_residual", "expected_lifetime")):
        return TERMINAL_TOL
    return PROBABILITY_TOL


def compare(a, b, path="stats"):
    """``a`` (forward) vs ``b`` (stacked): outcome fields within the adjoint identity, the rest bitwise."""
    assert type(a) is type(b), path
    if dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            compare(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, Mapping):
        assert a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)) and not isinstance(a, str):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(a, (np.ndarray, float)):
        outcome = any(path.startswith(f"stats.{p}") for p in OUTCOME_PATHS)
        if outcome:
            _near(a, b, path, _tol(path[len("stats."):]))
        elif isinstance(a, float):
            assert (math.isnan(a) and math.isnan(b)) or a == b, f"{path}: {a!r} != {b!r}"
        else:
            assert a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), path
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize("name", ADJOINT_CASES)
def test_forward_route_matches_the_stacked_route(name):
    engine, product, market = adjoint_case(name)
    forward = _stats(engine, product, market, "forward")
    _assert_coupons_forward(engine)
    stacked = _stats(type(engine)(engine.params), product, market, "stacked")
    assert forward.pv == stacked.pv == type(engine)(engine.params).price(product, market)
    compare(forward, stacked)


def _assert_coupons_forward(engine):
    """A Phoenix statistics call on the forward route read its coupons off the pass (no stacked re-run, spec D7)."""
    from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine

    if isinstance(engine, PhoenixQuadEngine):
        assert engine._last_v1_march.extension is not None, "the forward route fell back to the stacked march"


@pytest.mark.parametrize("name", ["sb_transition_dki", "sb_dki", "sb_cki", "ph_mem_expiry_dki", "ph_nomem_instant_dki",
                                  "kr_cki", "kr_dki", "carried", "ki_at_valuation_cki"])
def test_forward_route_matches_on_the_lattice_readouts(name):
    """The exact first step switched off: the transition and the interpolation readouts feed the pass (checkpoint)."""
    engine, product, market = adjoint_case(name)
    other = type(engine)(engine.params)
    engine._FIRST_STEP_S_F = other._FIRST_STEP_S_F = 0.0
    forward = _stats(engine, product, market, "forward")
    stacked = _stats(other, product, market, "stacked")
    assert engine._last_v1_march.first_step is None
    compare(forward, stacked)


def test_auto_converge_keeps_the_route():
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.asset.equity.param import QuadParams

    _, product, market = adjoint_case("sb_dki")
    engine = SnowballQuadEngine(QuadParams(grid_points=401, auto_converge=True))
    engine._v1_statistics_route = "stacked"
    engine.calculate_event_stats(product, market)
    assert engine._last_v1_march.columns[("ko", 0)] != 0.0
    forward = SnowballQuadEngine(QuadParams(grid_points=401, auto_converge=True))
    compare(forward.calculate_event_stats(product, market), engine.calculate_event_stats(product, market))


@pytest.mark.parametrize("name", ["daily_ko_1y_vol05", "quarter_hour_ki_8001"])
def test_forward_route_matches_on_corrected_local_steps(name):
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.asset.equity.param import QuadParams

    build, vol, params, *_ = EXACT_MASS[name]
    product, market = build(), exact_env(vol)
    forward = _stats(SnowballQuadEngine(params or QuadParams()), product, market, "forward")
    stacked = _stats(SnowballQuadEngine(params or QuadParams()), product, market, "stacked")
    compare(forward, stacked)


@pytest.mark.parametrize("name", ["sb_cki", "ph_mem_expiry_cki", "kr_dki"])
def test_a_refused_transpose_reruns_the_stacked_march(name, monkeypatch):
    engine, product, market = adjoint_case(name)

    def refuse(log):
        raise AdjointUnavailable("undefined: test refusal")

    monkeypatch.setattr(v1_statistics, "forward_pass", refuse)
    forward = _stats(engine, product, market, "forward")
    stacked = _stats(type(engine)(engine.params), product, market, "stacked")
    assert forward.unavailable["knock_in.probability"] == "undefined: test refusal"
    compare_bitwise(forward, stacked)


def compare_bitwise(a, b, path="stats"):
    assert type(a) is type(b), path
    if dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            compare_bitwise(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, Mapping):
        assert a.keys() == b.keys(), path
        for k in a:
            compare_bitwise(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)) and not isinstance(a, str):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare_bitwise(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), path
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or a == b, f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_the_route_switch_refuses_an_unknown_route():
    from quantark.util.exceptions import ValidationError

    engine, product, market = adjoint_case("sb_cki")
    engine._v1_statistics_route = "sideways"
    with pytest.raises(ValidationError, match="route"):
        engine.calculate_event_stats(product, market)


# --- spec D7: the Phoenix coupon columns off the forward pass ---------------------------------------------------------


def _phoenix_case(name):
    """Phoenix cases beyond ``ADJOINT_CASES``: decided / carried coupons, ``disable_ko_after_ki``, knocked in now, and
    the 1y daily-observation books (memory Phoenix is capped at 50 observations by ``price``)."""
    from test_asset.equity.engine_quad.test_quad_v1_event_statistics import OBS, env, ph, phoenix
    from quantark.util.enum import CouponPayType

    pay = {"instant": CouponPayType.INSTANT, "expiry": CouponPayType.EXPIRY}
    parts = name.split("_")
    if parts[0] == "decided":                  # decided_<mem|nomem>_<instant|expiry>_<paid|missed>
        memory, paid = parts[1] == "mem", parts[3] == "paid"
        barriers = [80.0 if paid else 110.0] + [80.0, 108.0, 80.0, 80.0]
        return ph()(), phoenix(memory, pay[parts[2]], "dki", ko_dates=(0.0, *OBS), barriers=barriers), env()
    if parts[0] == "arrears":                  # arrears_<instant|expiry>: the contract carries initial arrears
        product = phoenix(True, pay[parts[1]], "cki")
        product.coupon_config = dataclasses.replace(product.coupon_config, initial_coupon_arrears=0.03)
        return ph()(), product, env()
    if parts[0] == "disable":                  # disable_<cki|dki>
        return ph()(), phoenix(True, CouponPayType.EXPIRY, parts[1], disable=True), env()
    if parts[0] == "kinow":                    # knocked in at the valuation (continuous KI above the spot)
        return ph()(), phoenix(True, pay[parts[1]], "cki"), env(spot=65.0)
    if parts[0] == "daily":                    # daily_<nomem|mem>_<instant|expiry>: 252 (memory: 50) observations
        memory = parts[1] == "mem"
        n = 50 if memory else 252
        dates = [k / n for k in range(1, n + 1)]
        return ph()(), phoenix(memory, pay[parts[2]], "cki", ko_dates=dates, barriers=[80.0] * n), env()
    return adjoint_case(name)


PHOENIX_ADJOINT = [name for name in ADJOINT_CASES if name.startswith("ph_") or name == "carried"]
PHOENIX_EXTRA = ["decided_mem_instant_paid", "decided_mem_expiry_paid", "decided_nomem_instant_paid",
                 "decided_nomem_expiry_paid", "decided_mem_instant_missed", "decided_mem_expiry_missed",
                 "arrears_instant", "arrears_expiry", "disable_cki", "disable_dki", "kinow_instant", "kinow_expiry"]
PHOENIX_DAILY = ["daily_nomem_instant", "daily_nomem_expiry", "daily_mem_expiry"]


def _backward_rows(engine, product, market, route, monkeypatch):
    """The most rows the statistics march diffuses backward at once (``_diffuse_fft`` on a stacked surface)."""
    rows = []
    original = type(engine)._diffuse_fft

    def spy(self, values, *args, **kwargs):
        if np.ndim(values) == 2:
            rows.append(np.shape(values)[0])
        return original(self, values, *args, **kwargs)

    monkeypatch.setattr(type(engine), "_diffuse_fft", spy)
    _stats(engine, product, market, route)
    monkeypatch.undo()
    return max(rows)


@pytest.mark.parametrize("name", PHOENIX_ADJOINT + ["decided_mem_expiry_missed", "decided_nomem_instant_paid"])
def test_forward_route_marches_O1_backward_rows(name, monkeypatch):
    """D7: on the forward route a Phoenix march carries the 4 terminal rows and, under EXPIRY, the one termination
    row ``W`` -- never a row per observation (the stacked route carries the per-observation coupon columns)."""
    from quantark.util.enum import CouponPayType

    engine, product, market = _phoenix_case(name)
    expiry = product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
    assert _backward_rows(engine, product, market, "forward", monkeypatch) == 4 + int(expiry)
    n_ko = len(product.resolve_ko_observations(market))
    assert _backward_rows(type(engine)(engine.params), product, market, "stacked", monkeypatch) > 4 + n_ko


@pytest.mark.parametrize("name", PHOENIX_EXTRA)
def test_forward_route_matches_on_phoenix_coupon_cases(name):
    """AC1b: decided and carried coupons, ``disable_ko_after_ki``, knocked in at the valuation."""
    engine, product, market = _phoenix_case(name)
    forward = _stats(engine, product, market, "forward")
    _assert_coupons_forward(engine)
    stacked = _stats(type(engine)(engine.params), product, market, "stacked")
    assert forward.pv == stacked.pv
    compare(forward, stacked)


@pytest.mark.parametrize("name", ["ph_mem_instant_cki", "ph_nomem_expiry_dki", "ph_mem_expiry_cki", "ph_nodal_dki",
                                  "carried", "decided_mem_expiry_missed", "decided_nomem_instant_paid"])
def test_phoenix_coupons_match_on_the_lattice_readouts(name):
    """AC1b with the exact first step switched off (the interpolation readout feeds the pass)."""
    engine, product, market = _phoenix_case(name)
    other = type(engine)(engine.params)
    engine._FIRST_STEP_S_F = other._FIRST_STEP_S_F = 0.0
    forward = _stats(engine, product, market, "forward")
    assert engine._last_v1_march.first_step is None
    _assert_coupons_forward(engine)
    stacked = _stats(other, product, market, "stacked")
    compare(forward, stacked)


@pytest.mark.parametrize("name", ["ph_mem_expiry_cki", "ph_nomem_expiry_dki"])
def test_a_refused_branch_map_reruns_the_stacked_march(name, monkeypatch):
    """A coupon observation's branch map that is not local in the probe band refuses the pass (spec D7, D2): the call
    re-runs the stacked march and gives its numbers bitwise."""
    from quantark.asset.equity.engine.quad import v1_coupons_forward

    def refuse(event, n, time):
        raise AdjointUnavailable("undefined: test refusal")

    engine, product, market = _phoenix_case(name)
    monkeypatch.setattr(v1_coupons_forward, "probe_event", refuse)
    forward = _stats(engine, product, market, "forward")
    assert engine._last_v1_march.extension is None
    compare_bitwise(forward, _stats(type(engine)(engine.params), product, market, "stacked"))


@pytest.mark.parametrize("name", PHOENIX_DAILY)
def test_forward_route_matches_on_daily_phoenix(name):
    """AC1b on the daily-observation books the forward route is for."""
    engine, product, market = _phoenix_case(name)
    forward = _stats(engine, product, market, "forward")
    _assert_coupons_forward(engine)
    stacked = _stats(type(engine)(engine.params), product, market, "stacked")
    compare(forward, stacked)
