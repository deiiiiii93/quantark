"""AC19 — the risk stack dispatches on the pillar-curve protocol, so a ZeroCurve
(bootstrapped or hand-built) is stressed, bumped and rolled like an
InterpolatedRateCurve, and keeps its interpolation, extrapolation and clock anchor.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.param import ContinuousDividendYield, CurveInterpolation, FlatRateCurve, FlatVolSurface, SpotQuote, ZeroCurve
from quantark.param.extrapolation import RateExtrapolation
from quantark.param.node_roles import NodeRole
from quantark.param.rrf import (
    CubicSplineRateCurve,
    InterpolatedRateCurve,
    LinearRateCurve,
    LogLinearRateCurve,
    ParallelShiftRateCurve,
    is_flat_rate_curve,
    is_pillar_curve,
)
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.sacva.exposure.asof import equity_asof_env
from quantark.sacva.exposure.curves import ForwardRateCurve, key_rate_bumped_curve
from quantark.sacva.exposure.simulator import MonteCarloExposureConfig, MonteCarloExposureEngine
from quantark.stresstest.credit.credit_stress_applicator import CreditStressApplicator
from quantark.stresstest.fx.fx_stress_applicator import FXStressApplicator
from quantark.stresstest.scenario.scenario import Stress
from quantark.stresstest.scenario.scenario_builder import ScenarioBuilder
from quantark.stresstest.stress.stress_applicator import StressApplicator
from quantark.stresstest.stress.stress_types import StressLevel, StressType
from quantark.util.calendar import DayCountConvention
from quantark.util.exceptions import ValidationError
from quantark.var.fx.revaluation import _shift_curve as var_shift_curve

PILLARS = [(1.0, 0.030), (2.0, 0.034), (5.0, 0.040), (10.0, 0.045), (30.0, 0.050)]
VD = datetime(2026, 11, 5)
SCHEMES = list(CurveInterpolation)
ROLES = tuple([NodeRole.CALIBRATED] * len(PILLARS))


def _zc(scheme=CurveInterpolation.MONOTONE_CONVEX, node_roles=None):
    return ZeroCurve(PILLARS, interpolation=scheme, extrapolation=RateExtrapolation.FLAT_ZERO_RATE,
                     node_roles=node_roles, last_observable_tenor=30.0, valuation_date=VD)


def _same_config(a: ZeroCurve, b: ZeroCurve) -> None:
    assert type(b) is ZeroCurve
    assert b.interpolation is a.interpolation and b.extrapolation is a.extrapolation
    assert b.valuation_date == a.valuation_date and b.last_observable_tenor == a.last_observable_tenor


def _env(curve):
    return PricingEnvironment(rate_curve=curve, valuation_date=VD, spot_quote=SpotQuote(spot=100.0, asset_name="A"),
                              vol_surface=FlatVolSurface(0.2), div_yield=ContinuousDividendYield(0.01),
                              day_count_convention=DayCountConvention.CALENDAR_DAYS)


# --------------------------------------------------------------------------- the protocol


def test_is_pillar_curve_and_is_flat_rate_curve():
    for c in (LinearRateCurve(PILLARS), LogLinearRateCurve(PILLARS), CubicSplineRateCurve(PILLARS), _zc()):
        assert is_pillar_curve(c) and not is_flat_rate_curve(c)
    assert not is_pillar_curve(FlatRateCurve(0.03)) and not is_pillar_curve(None)
    assert not is_pillar_curve(ParallelShiftRateCurve(LinearRateCurve(PILLARS), 1e-4))
    assert is_flat_rate_curve(FlatRateCurve(0.03))
    assert is_flat_rate_curve(ParallelShiftRateCurve(ParallelShiftRateCurve(FlatRateCurve(0.03), 1e-4), 1e-4))
    assert not is_flat_rate_curve(ParallelShiftRateCurve(LinearRateCurve(PILLARS), 1e-4))
    assert not is_flat_rate_curve(None)


def test_with_pillars_keeps_class_and_metadata_drops_roles_on_count_change():
    base = CubicSplineRateCurve(PILLARS, node_roles=ROLES, last_observable_tenor=30.0)
    same = base.with_pillars([(t, r + 1e-4) for t, r in PILLARS])
    assert type(same) is CubicSplineRateCurve and same.node_roles == ROLES and same.last_observable_tenor == 30.0
    assert same.rates == pytest.approx([r + 1e-4 for _, r in PILLARS])
    more = base.with_pillars(PILLARS + [(40.0, 0.05)])
    assert type(more) is CubicSplineRateCurve and more.node_roles is None and more.last_observable_tenor == 30.0
    zc = _zc(node_roles=ROLES)
    assert zc.with_pillars(PILLARS).node_roles == ROLES
    assert zc.with_pillars(PILLARS + [(40.0, 0.05)]).node_roles is None
    assert InterpolatedRateCurve.with_pillars is CubicSplineRateCurve.with_pillars  # inherited, not overridden


# --------------------------------------------------------------------------- stress


@pytest.mark.parametrize("scheme", SCHEMES)
def test_equity_stress_parallel_and_key_rate_keep_the_zero_curve(scheme):
    zc = _zc(scheme, node_roles=ROLES)
    portfolio = Portfolio(portfolio_name="t", pricing_environments={"U": _env(zc)})

    parallel = ScenarioBuilder().name("r+100").rate_stress(0.01, stress_type=StressType.ABSOLUTE).build()
    out = StressApplicator.apply_scenario_to_portfolio(portfolio, parallel)["U"].rate_curve
    _same_config(zc, out)
    assert out.node_roles == ROLES
    assert out.rates == pytest.approx([r + 0.01 for _, r in PILLARS], abs=1e-15)

    key = ScenarioBuilder().name("5Y+10").key_rate_stress(0.001, tenor_bucket="5Y", stress_type=StressType.ABSOLUTE).build()
    out = StressApplicator.apply_scenario_to_portfolio(portfolio, key)["U"].rate_curve
    _same_config(zc, out)
    assert dict(out.pillars)[5.0] == pytest.approx(0.040 + 0.001, abs=1e-15)
    for t in (1.0, 2.0, 10.0, 30.0):
        assert dict(out.pillars)[t] == dict(PILLARS)[t]

    added = ScenarioBuilder().name("7Y+10").key_rate_stress(0.001, tenor_bucket="7Y", stress_type=StressType.ABSOLUTE).build()
    out = StressApplicator.apply_scenario_to_portfolio(portfolio, added)["U"].rate_curve
    _same_config(zc, out)
    assert len(out.pillars) == len(PILLARS) + 1 and out.node_roles is None   # a new pillar: roles no longer per pillar
    assert dict(out.pillars)[7.0] == pytest.approx(zc.get_rate(7.0) + 0.001, abs=1e-15)


def test_fx_and_credit_stress_keep_the_zero_curve_and_refuse_wrappers():
    zc = _zc(CurveInterpolation.LOG_CUBIC_DF)
    stress = Stress("domestic_rate", StressType.ABSOLUTE, 0.02, StressLevel.PORTFOLIO)
    for shifted in (FXStressApplicator._shift_curve(zc, stress), CreditStressApplicator._shift_curve(zc, stress)):
        _same_config(zc, shifted)
        assert shifted.rates == pytest.approx([r + 0.02 for _, r in PILLARS], abs=1e-15)
    wrapper = ParallelShiftRateCurve(LinearRateCurve(PILLARS), 1e-4)   # neither flat nor a pillar curve
    with pytest.raises(ValidationError):
        FXStressApplicator._shift_curve(wrapper, stress)
    with pytest.raises(ValidationError):
        CreditStressApplicator._shift_curve(wrapper, stress)


def test_fx_var_revaluation_bumps_the_zero_curve():
    zc = _zc(CurveInterpolation.CUBIC_ZERO)
    assert var_shift_curve(zc, 0.0) is zc
    out = var_shift_curve(zc, 1e-4)
    _same_config(zc, out)
    assert out.rates == pytest.approx([r + 1e-4 for _, r in PILLARS], abs=1e-15)


# --------------------------------------------------------------------------- SA-CVA


def test_sacva_key_rate_bump_keeps_the_zero_curve():
    zc = _zc(CurveInterpolation.LINEAR_ZERO)
    out = key_rate_bumped_curve(zc, 5.0, 1e-4)
    _same_config(zc, out)
    assert dict(out.pillars)[5.0] == pytest.approx(0.040 + 1e-4, abs=1e-15)
    for t in (1.0, 2.0, 10.0, 30.0):
        assert dict(out.pillars)[t] == dict(PILLARS)[t]
    with pytest.raises(ValidationError):
        key_rate_bumped_curve(FlatRateCurve(0.03), 5.0, 1e-4)
    with pytest.raises(ValidationError):
        key_rate_bumped_curve(ParallelShiftRateCurve(zc, 1e-4), 5.0, 1e-4)


def test_sacva_asof_rolls_every_non_flat_curve_and_leaves_flat_ones():
    zc = _zc()
    out = equity_asof_env(_env(zc), 101.0, 1.0)
    assert isinstance(out.rate_curve, ForwardRateCurve) and out.rate_curve.base_curve is zc
    assert out.rate_curve.get_discount_factor(1.0) == pytest.approx(
        zc.get_discount_factor(2.0) / zc.get_discount_factor(1.0), rel=1e-15)
    # a shifted interpolated curve is a term structure too (it used to be skipped)
    shifted = ParallelShiftRateCurve(LinearRateCurve(PILLARS), 1e-4)
    assert isinstance(equity_asof_env(_env(shifted), 101.0, 1.0).rate_curve, ForwardRateCurve)
    # flat, and a parallel shift of flat: the forward curve is the curve itself
    flat = FlatRateCurve(0.03)
    assert equity_asof_env(_env(flat), 101.0, 1.0).rate_curve is flat
    sflat = ParallelShiftRateCurve(flat, 1e-4)
    assert equity_asof_env(_env(sflat), 101.0, 1.0).rate_curve is sflat


def test_sacva_term_structure_drift_activates_for_the_zero_curve():
    eng = MonteCarloExposureEngine(MonteCarloExposureConfig(num_paths=100, n_steps=4, seed=1))
    times = np.linspace(0.0, 2.0, 5)
    env = _env(_zc(CurveInterpolation.LOG_LINEAR_DF))
    step_rates, step_divs = eng._step_forwards(["A"], {"A": env}, times)
    assert step_rates is not None and step_divs is not None
    df = np.array([env.get_discount_factor(float(t)) for t in times])
    assert step_rates[0] == pytest.approx(np.diff(-np.log(df)) / np.diff(times), rel=1e-14)
    assert np.all(step_divs[0] == 0.01)
    for flat in (FlatRateCurve(0.03), ParallelShiftRateCurve(FlatRateCurve(0.03), 1e-4)):
        assert eng._step_forwards(["A"], {"A": _env(flat)}, times) == (None, None)
