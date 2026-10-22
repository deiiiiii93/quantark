"""SA-CVA interest-rate delta on a term-structure reporting curve (#3).

Exact for vanilla analytic trades: the spot distribution at each node uses the integrated
forward drift (term-structure GBM), the as-of repricer rolls the curve forward, and each
SA-CVA vertex pillar is bumped 1bp and the CVA re-run. Stateful/FX trades raise; a flat
curve produces no per-tenor factors.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.european_vanilla_option import (
    EuropeanVanillaOption,
)
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv.pricing_environment import PricingEnvironment
from quantark.sacva.exposure.curves import (
    IR_DELTA_TENORS,
    ForwardRateCurve,
    key_rate_bumped_curve,
)
from quantark.sacva.exposure.paths import StatePathGenerator
from quantark.sacva.exposure.simulator import (
    MonteCarloExposureConfig,
    MonteCarloExposureEngine,
)
from quantark.sacva.models.enums import CreditQuality, RiskClass, RiskType
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.credit_curve import PillarHazardCurve
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.trade import CVATrade
from quantark.sacva.portfolio.trade_portfolio import CVATradePortfolio
from quantark.sacva.sacva_engine import SACVAEngine
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import ObservationType, OptionType
from quantark.util.exceptions import ValidationError

VAL = datetime(2026, 6, 17)
EXP_3Y = datetime(2029, 6, 16)
PILLARS = [(1.0, 0.030), (2.0, 0.034), (5.0, 0.040), (10.0, 0.045), (30.0, 0.050)]


def _term_curve():
    return LinearRateCurve([(t, r) for t, r in PILLARS])


def _term_env(spot=100.0, vol=0.25, div=0.0, asset="ACME"):
    return PricingEnvironment(
        rate_curve=_term_curve(), valuation_date=VAL,
        spot_quote=SpotQuote(spot=spot, asset_name=asset),
        vol_surface=FlatVolSurface(vol), div_yield=ContinuousDividendYield(div),
        day_count_convention=DayCountConvention.CALENDAR_DAYS)


def _call(strike=100.0):
    return EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL,
                                 exercise_date=EXP_3Y)


def _cp(trade, name="CP"):
    return Counterparty(
        name, [NettingSet("n", [trade])],
        PillarHazardCurve([0.5, 1.0, 3.0, 5.0, 10.0], [0.02] * 5, recovery_rate=0.4),
        2, CreditQuality.IG)


def _engine(seed=11, n_steps=12, num_paths=8000):
    return SACVAEngine(exposure_engine=MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=num_paths, n_steps=n_steps, seed=seed)))


# --- curve helpers ----------------------------------------------------------

def test_forward_rate_curve_matches_base_forward():
    c = _term_curve()
    t0 = 2.0
    fwd = ForwardRateCurve(c, t0)
    for tau in (0.5, 1.0, 3.0):
        expected = c.get_discount_factor(t0 + tau) / c.get_discount_factor(t0)
        assert fwd.get_discount_factor(tau) == pytest.approx(expected, rel=1e-12)
    # flat base curve: the forward curve is numerically identical
    flat = FlatRateCurve(0.03)
    ff = ForwardRateCurve(flat, 1.5)
    assert ff.get_discount_factor(2.0) == pytest.approx(flat.get_discount_factor(2.0))


def test_key_rate_bump_moves_only_target_pillar():
    c = _term_curve()
    bumped = key_rate_bumped_curve(c, 5.0, 1e-4)
    pre = dict(c.pillars)
    post = dict(bumped.pillars)
    assert post[5.0] == pytest.approx(pre[5.0] + 1e-4)
    for t in (1.0, 2.0, 10.0, 30.0):
        assert post[t] == pytest.approx(pre[t])


def test_key_rate_bump_flat_curve_raises():
    with pytest.raises(ValidationError):
        key_rate_bumped_curve(FlatRateCurve(0.03), 5.0, 1e-4)


# --- exact term-structure forward drift -------------------------------------

def test_term_structure_paths_match_curve_forward():
    # E[S(t_j)] under the simulated measure must equal the deterministic forward
    # S0*exp(R(t_j) - Q(t_j)) (R=-ln DF, Q=q*t) -> validates per-step integrated-forward
    # drift (a flat rate(horizon) drift would miss the curve's term structure).
    env = _term_env(spot=100.0, div=0.015)
    times = np.linspace(0.0, 3.0, 13)
    dt = np.diff(times)
    df = np.array([env.get_discount_factor(float(t)) for t in times])
    step_rates = (np.diff(-np.log(df)) / dt)[None, :]
    q = 0.015
    step_divs = np.full((1, len(dt)), q)
    paths = StatePathGenerator(
        keys=["ACME"], spots=[100.0], vols=[0.25], rates=[0.0], divs=[0.0],
        corr=[[1.0]], grid_times=times, num_paths=200000, seed=3,
        step_rates=step_rates, step_divs=step_divs).generate()["ACME"]
    for j in range(1, len(times)):
        fwd = 100.0 * np.exp(-np.log(df[j]) - q * times[j])
        assert paths[:, j].mean() == pytest.approx(fwd, rel=5e-3), f"node {times[j]}"


def test_term_curve_t0_exposure_equals_price():
    trade = CVATrade("eq", _call(), BlackScholesEngine(), _term_env(), quantity=10.0,
                     trade_currency="USD", equity_bucket=5)
    price0 = BlackScholesEngine().price(trade.product, trade.env)
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=40000, n_steps=12, seed=4)).compute(_cp(trade))
    assert profile.epe_discounted[0] == pytest.approx(10.0 * price0, rel=4e-3)
    assert np.all(np.isfinite(profile.epe_discounted))


# --- end-to-end IR delta ----------------------------------------------------

def test_ir_delta_sensitivities_produced():
    trade = CVATrade("eq", _call(), BlackScholesEngine(), _term_env(), quantity=50.0,
                     trade_currency="USD", equity_bucket=5)
    result = _engine(num_paths=12000).compute(
        CVATradePortfolio([_cp(trade)], [], reporting_currency="USD"))
    ir = [s for s in result.sensitivities
          if s.risk_class is RiskClass.INTEREST_RATE and s.risk_type is RiskType.DELTA]
    tenors = sorted(s.tenor for s in ir)
    assert tenors == list(IR_DELTA_TENORS)        # one per vertex pillar
    assert all(s.currency == "USD" and s.bucket == 0 for s in ir)
    assert all(np.isfinite(s.s_cva) for s in ir)
    # a long equity call's CVA rises with rates (higher forward) -> positive near-maturity
    near = [s.s_cva for s in ir if s.tenor in (1.0, 2.0)]
    assert sum(near) > 0.0
    assert "INTEREST_RATE:DELTA" in result.by_risk_class
    assert result.by_risk_class["INTEREST_RATE:DELTA"] > 0.0


def _flat_equity_trade():
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03), valuation_date=VAL,
        spot_quote=SpotQuote(spot=100.0, asset_name="ACME"),
        vol_surface=FlatVolSurface(0.25), div_yield=ContinuousDividendYield(0.0),
        day_count_convention=DayCountConvention.CALENDAR_DAYS)
    return CVATrade("eq", _call(), BlackScholesEngine(), env, quantity=50.0,
                    trade_currency="USD", equity_bucket=5)


def test_flat_curve_with_ir_delta_enabled_raises():
    # IR delta ON (default) + flat reporting curve -> raise (no silent omission of IR risk)
    port = CVATradePortfolio([_cp(_flat_equity_trade())], [], reporting_currency="USD")
    with pytest.raises(ValidationError, match="flat"):
        _engine().compute(port)


def test_flat_curve_ir_delta_opt_out():
    # explicit opt-out: no IR sensitivities, equity/credit capital still produced
    port = CVATradePortfolio([_cp(_flat_equity_trade())], [], reporting_currency="USD")
    eng = SACVAEngine(exposure_engine=MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=8000, n_steps=12, seed=11)),
        include_ir_delta=False)
    result = eng.compute(port)
    assert [s for s in result.sensitivities
            if s.risk_class is RiskClass.INTEREST_RATE] == []
    assert result.total_capital > 0.0


def test_ir_delta_missing_vertex_raises():
    # a term curve lacking a SA-CVA vertex pillar (no 30y) must raise, not silently skip
    curve = LinearRateCurve([(1.0, 0.03), (2.0, 0.034), (5.0, 0.04), (10.0, 0.045)])
    env = PricingEnvironment(
        rate_curve=curve, valuation_date=VAL,
        spot_quote=SpotQuote(spot=100.0, asset_name="ACME"),
        vol_surface=FlatVolSurface(0.25), div_yield=ContinuousDividendYield(0.0),
        day_count_convention=DayCountConvention.CALENDAR_DAYS)
    trade = CVATrade("eq", _call(), BlackScholesEngine(), env, quantity=50.0,
                     trade_currency="USD", equity_bucket=5)
    with pytest.raises(ValidationError, match="missing"):
        _engine(num_paths=4000).compute(
            CVATradePortfolio([_cp(trade)], [], reporting_currency="USD"))


def test_ir_delta_extra_pillar_raises():
    # a curve with a non-regulatory pillar (3y) whose CVA dependence would be unreported
    curve = LinearRateCurve([(1.0, 0.03), (2.0, 0.034), (3.0, 0.037), (5.0, 0.04),
                             (10.0, 0.045), (30.0, 0.05)])
    env = PricingEnvironment(
        rate_curve=curve, valuation_date=VAL,
        spot_quote=SpotQuote(spot=100.0, asset_name="ACME"),
        vol_surface=FlatVolSurface(0.25), div_yield=ContinuousDividendYield(0.0),
        day_count_convention=DayCountConvention.CALENDAR_DAYS)
    trade = CVATrade("eq", _call(), BlackScholesEngine(), env, quantity=50.0,
                     trade_currency="USD", equity_bucket=5)
    with pytest.raises(ValidationError, match="extra pillars"):
        _engine(num_paths=4000).compute(
            CVATradePortfolio([_cp(trade)], [], reporting_currency="USD"))


def test_ir_delta_mixed_curves_raise():
    # one term-curve trade + one flat-curve trade -> ambiguous shared reporting curve
    t1 = CVATrade("eq1", _call(), BlackScholesEngine(), _term_env(), quantity=10.0,
                  trade_currency="USD", equity_bucket=5)
    flat_env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03), valuation_date=VAL,
        spot_quote=SpotQuote(spot=100.0, asset_name="ACME"),
        vol_surface=FlatVolSurface(0.25), div_yield=ContinuousDividendYield(0.0),
        day_count_convention=DayCountConvention.CALENDAR_DAYS)
    t2 = CVATrade("eq2", _call(), BlackScholesEngine(), flat_env, quantity=10.0,
                  trade_currency="USD", equity_bucket=5)
    cp = Counterparty(
        "CP", [NettingSet("n", [t1, t2])],
        PillarHazardCurve([0.5, 1.0, 3.0, 5.0, 10.0], [0.02] * 5, recovery_rate=0.4),
        2, CreditQuality.IG)
    # rejected either at base-exposure market-consistency or the IR shared-curve fence
    with pytest.raises(ValidationError,
                       match="share one reporting discount curve|flat or absent|single shared"):
        _engine(num_paths=4000).compute(
            CVATradePortfolio([cp], [], reporting_currency="USD"))


def test_ir_delta_stateful_trade_raises():
    # a snowball (single-rate QUAD engine) with a term curve -> IR delta deferred (raise)
    cfg = BarrierConfig(
        ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.5, 1.0, 1.5, 2.0], ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    snow = SnowballOption(initial_price=100.0, strike=100.0, barrier_config=cfg,
                          contract_multiplier=1.0, maturity=2.0, is_reverse=False)
    trade = CVATrade("snow", snow, SnowballQuadEngine(params=QuadParams(grid_points=301)),
                     _term_env(), quantity=1.0, trade_currency="USD", equity_bucket=5)
    with pytest.raises(ValidationError, match="IR delta is deferred for stateful"):
        _engine(num_paths=4000).compute(
            CVATradePortfolio([_cp(trade)], [], reporting_currency="USD"))
