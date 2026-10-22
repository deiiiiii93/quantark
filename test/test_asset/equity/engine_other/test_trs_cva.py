"""Gate 7 tests: equity TRS in Basel SA-CVA (MAR50 SBA).

A single-period TRS is a delta-one equity trade whose value is Markovian in
(spot, time). ``EquitySwapPosition.to_cva_trade`` maps it to a ``CVATrade`` priced
by the closed-form as-of repricer (``TRSCVARepricer``), which flows through
Counterparty -> NettingSet -> MonteCarloExposureEngine -> RegulatoryCVAEngine ->
the SBA calculator to SA-CVA capital.

The repricer's core correctness claim — that V(S,t) reproduces the realized
cashflow engine exactly — is asserted directly against ``TotalReturnSwapEngine``.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.util.calendar.business_calendar import Calendar
from quantark.util.calendar import DayCountConvention
from quantark.util.exceptions import ValidationError
from quantark.asset.equity.product.swap import (
    AssetParams,
    FixLegParams,
    FloatLegParams,
    PricingParams,
    TRSParams,
    OneAssetTotalReturnSwap,
    OneAssetTotalReturnSwapDualCcy,
)
from quantark.asset.equity.product.swap.trs_params import AccrualType
from quantark.asset.equity.engine.cashflow.total_return_swap_engine import (
    TotalReturnSwapEngine,
)
from quantark.asset.equity.engine.cashflow.trs_cva_repricer import (
    TRSCVARepricer,
    build_trs_cva_components,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.portfolio import EquitySwapPosition
from quantark.sacva.exposure.simulator import (
    MonteCarloExposureConfig,
    MonteCarloExposureEngine,
)
from quantark.sacva.models.enums import CreditQuality
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.credit_curve import PillarHazardCurve
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.trade_portfolio import CVATradePortfolio
from quantark.sacva.sacva_engine import SACVAEngine

START, END = "2024-01-02", "2024-12-31"
VAL = "2024-01-02"
S0, NOTIONAL = 100.0, 1_000_000.0
REG_TENORS = [0.5, 1.0, 3.0, 5.0, 10.0]


def _flat_path() -> pd.Series:
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    return pd.Series([S0] * len(idx), index=idx)


def _make_trs(
    long: bool = True,
    accrual_type: AccrualType = AccrualType.NOTIONAL,
    redemption_events=None,
    asset_price_precision: int = 2,
) -> OneAssetTotalReturnSwap:
    from quantark.asset.equity.product.swap.trs_params import EventParams

    calendar = Calendar(name="DemoCalendar")
    float_dir = 1 if long else -1
    events = EventParams(redemption_events=redemption_events or [])
    return OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="TRS_CVA",
            asset=AssetParams("IDX", S0, _flat_path(), asset_price_precision=asset_price_precision),
            fix_leg=FixLegParams(
                rate=0.05, notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date=END, payment_calendar=calendar,
                direction=-float_dir, accrual_type=accrual_type,
            ),
            float_leg=FloatLegParams(
                notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date=END, payment_calendar=calendar,
                direction=float_dir,
            ),
            events=events,
            pricing=PricingParams(valuation_date=VAL, output_mode="spot"),
        )
    )


def _env(spot: float = S0, vol: float = 0.25) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(0.0),
        valuation_date=datetime(2024, 1, 2),
        day_count_convention=DayCountConvention.CALENDAR_DAYS,
    )


def _counterparty(position: EquitySwapPosition, env: PricingEnvironment, *, bucket=2):
    trade = position.to_cva_trade(env, equity_bucket=bucket)
    curve = PillarHazardCurve(tenors=REG_TENORS, hazards=[0.03] * 5, recovery_rate=0.4)
    return Counterparty(name="CP", netting_sets=[NettingSet("n1", [trade])],
                        credit_curve=curve, bucket=bucket,
                        credit_quality=CreditQuality.IG)


# --- core correctness: repricer == realized engine ------------------------

def test_repricer_matches_realized_engine_at_pivots():
    """V(S,t) reproduces the realized cashflow engine's present value exactly.

    Uses a non-flat ramp path and high asset-price precision (the realized engine
    rounds prices to ``asset_price_precision``; the repricer uses full precision).
    """
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    ramp = pd.Series([S0 + 20.0 * i / (len(idx) - 1) for i in range(len(idx))], index=idx)
    calendar = Calendar(name="C")

    def make(val):
        return TRSParams(
            contract_id="T", asset=AssetParams("IDX", S0, ramp, asset_price_precision=10),
            fix_leg=FixLegParams(rate=0.05, notional=NOTIONAL, initial_notional=NOTIONAL,
                                 start_date=START, end_date=END, payment_calendar=calendar,
                                 direction=-1),
            float_leg=FloatLegParams(notional=NOTIONAL, initial_notional=NOTIONAL,
                                     start_date=START, end_date=END,
                                     payment_calendar=calendar, direction=1),
            pricing=PricingParams(valuation_date=val, output_mode="spot"))

    engine = TotalReturnSwapEngine()
    repricer = TRSCVARepricer(make(START))
    for val in ["2024-02-01", "2024-04-15", "2024-07-01", "2024-10-15"]:
        eng_pv = float(engine.price(make(val), precision=10).iloc[-1]["present_value"])
        rep_v = repricer.value_at(float(ramp[val]), val)
        assert rep_v == pytest.approx(eng_pv, abs=1e-4)


def test_short_swap_flips_value_sign():
    """A short TRS (receive financing, pay total return) has the opposite spot sign."""
    long_rep = TRSCVARepricer(_make_trs(long=True, asset_price_precision=10).params)
    short_rep = TRSCVARepricer(_make_trs(long=False, asset_price_precision=10).params)
    # at spot above S0 the long swap's equity term is positive, the short's negative
    lv = long_rep.value_at(120.0, "2024-06-28")
    sv = short_rep.value_at(120.0, "2024-06-28")
    assert (lv - long_rep.value_at(100.0, "2024-06-28")) > 0
    assert (sv - short_rep.value_at(100.0, "2024-06-28")) < 0


# --- wiring ---------------------------------------------------------------

def test_to_cva_trade_builds_equity_trade():
    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=1.0)
    trade = pos.to_cva_trade(_env(), equity_bucket=5)
    assert trade.equity_bucket == 5
    assert trade.quantity == 1.0
    assert trade.product.get_maturity(_env()) > 0.0
    # unit price is the per-contract MtM (delta-one), here ~one day's financing
    assert trade.engine.price(trade.product, _env()) < 0.0


# --- end-to-end SA-CVA capital --------------------------------------------

def test_trs_portfolio_to_capital_end_to_end():
    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=1.0)
    portfolio = CVATradePortfolio(
        counterparties=[_counterparty(pos, _env(), bucket=2)], hedges=[])
    eng = SACVAEngine(exposure_engine=MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=8000, n_steps=16, seed=11)),
        include_ir_delta=False)  # flat-curve TRS book: IR delta intentionally omitted
    result = eng.compute(portfolio)

    assert result.total_capital > 0.0
    assert result.delta_capital > 0.0           # counterparty credit-spread + equity delta
    assert result.counterparty_cva["CP"] > 0.0
    assert "CP" in result.exposure_profiles


def test_cva_rises_with_exposure_vol():
    """A delta-one TRS has no vega payoff, but its COUNTERPARTY exposure grows with
    the underlying's volatility (wider potential future exposure), so CVA rises."""
    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=1.0)
    cfg = MonteCarloExposureConfig(num_paths=12000, n_steps=16, seed=7)
    lo = MonteCarloExposureEngine(cfg).compute(_counterparty(pos, _env(vol=0.15)))
    hi = MonteCarloExposureEngine(cfg).compute(_counterparty(pos, _env(vol=0.45)))
    # higher vol -> higher discounted EPE at the mid of the profile
    assert hi.epe_discounted[len(hi.times) // 2] > lo.epe_discounted[len(lo.times) // 2]


# --- guards: non-Markovian / matured variants raise -----------------------

def test_matured_swap_has_no_cva_exposure():
    # Legs end 2024-01-10 but the valuation date is 2024-01-30 (state MATURED): a
    # matured swap carries no future counterparty exposure, so mapping must raise.
    calendar = Calendar(name="C")
    matured = OneAssetTotalReturnSwap(TRSParams(
        contract_id="M", asset=AssetParams("IDX", S0, _flat_path()),
        fix_leg=FixLegParams(rate=0.05, notional=NOTIONAL, initial_notional=NOTIONAL,
                             start_date=START, end_date="2024-01-10",
                             payment_calendar=calendar, direction=-1),
        float_leg=FloatLegParams(notional=NOTIONAL, initial_notional=NOTIONAL,
                                 start_date=START, end_date="2024-01-10",
                                 payment_calendar=calendar, direction=1),
        pricing=PricingParams(valuation_date="2024-01-30", output_mode="spot")))
    pos = EquitySwapPosition(product=matured, quantity=1.0)
    with pytest.raises(ValidationError):
        pos.to_cva_trade(_env(), equity_bucket=2)


def test_env_valuation_after_maturity_raises():
    # Product built ACTIVE (val=START), but to_cva_trade is handed an environment
    # whose valuation date is past the contract end — maturity must be judged
    # against the supplied env, not the product's stale stored state.
    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=1.0)
    late = PricingEnvironment(
        spot_quote=SpotQuote(spot=S0, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(0.0),
        valuation_date=datetime(2025, 6, 1),  # past END 2024-12-31
        day_count_convention=DayCountConvention.CALENDAR_DAYS,
    )
    with pytest.raises(ValidationError):
        pos.to_cva_trade(late, equity_bucket=2)


def test_forward_starting_swap_deferred():
    # env valuation date before the contract start: a forward-starting swap's
    # pre-inception value is a forward exposure the realized repricer can't express.
    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=1.0)
    early = PricingEnvironment(
        spot_quote=SpotQuote(spot=S0, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(0.0),
        valuation_date=datetime(2023, 12, 1),  # before START 2024-01-02
        day_count_convention=DayCountConvention.CALENDAR_DAYS,
    )
    with pytest.raises(ValidationError):
        pos.to_cva_trade(early, equity_bucket=2)


def test_market_value_accrual_without_optin_raises():
    """Market-value financing is path-dependent: the exact (default) path rejects it."""
    pos = EquitySwapPosition(
        product=_make_trs(long=True, accrual_type=AccrualType.MARKET_VALUE), quantity=1.0)
    with pytest.raises(ValidationError):
        pos.to_cva_trade(_env(), equity_bucket=2)  # allow_approx_financing defaults False


# --- Phase B: opt-in approximate pathwise exposure (market-value financing) ---

def test_pathwise_engine_required_for_market_value_trs():
    """A market-value-financing TRS routed through the Markovian engine fails loudly
    (its repricer.price raises) rather than silently dropping financing path-dependence."""
    from quantark.asset.equity.engine.cashflow.trs_cva_repricer import (
        build_trs_pathwise_cva_components,
    )
    trs = _make_trs(long=True, accrual_type=AccrualType.MARKET_VALUE)
    product, engine = build_trs_pathwise_cva_components(trs.params)
    with pytest.raises(ValidationError):
        engine.price(product, _env())  # Markovian value-surface entry point is barred


def test_pathwise_value_equals_realized_mtm_at_t0():
    """At the first grid node the pathwise value collapses to the realized MtM today."""
    from quantark.asset.equity.engine.cashflow.trs_cva_repricer import (
        build_trs_pathwise_cva_components,
    )
    from quantark.asset.equity.engine.cashflow.trs_valuation import TRSValuationEngine

    trs = _make_trs(long=True, accrual_type=AccrualType.MARKET_VALUE)
    _, engine = build_trs_pathwise_cva_components(trs.params)
    env = _env(spot=108.0)
    import numpy as np
    times = np.array([0.0, 0.25, 0.5, 0.75])
    spots = np.full((5, 4), 108.0)  # all paths start at env.spot
    vals = engine.value_paths(None, spots, times, env)
    v_today = TRSValuationEngine(trs.params).mark_to_market(env)
    assert vals[:, 0] == pytest.approx(v_today)  # t0 column == realized MtM


def test_pathwise_value_zeroes_after_maturity():
    """A matured TRS netted on a longer grid must stop contributing exposure."""
    from quantark.asset.equity.engine.cashflow.trs_cva_repricer import (
        build_trs_pathwise_cva_components,
    )
    import numpy as np

    trs = _make_trs(long=True, accrual_type=AccrualType.MARKET_VALUE)
    _, engine = build_trs_pathwise_cva_components(trs.params)
    env = _env(spot=100.0)
    # swap matures ~0.997y from START; grid runs out to 2y
    times = np.array([0.0, 0.5, 0.99, 1.5, 2.0])
    spots = np.full((4, 5), 110.0)
    vals = engine.value_paths(None, spots, times, env)
    assert np.any(vals[:, 2] != 0.0)          # node within life carries value
    assert np.all(vals[:, 3] == 0.0)          # past maturity -> zero
    assert np.all(vals[:, 4] == 0.0)


def test_pathwise_trs_portfolio_to_capital_end_to_end():
    from quantark.asset.equity.engine.cashflow.trs_cva_exposure import (
        TRSPathwiseExposureEngine,
    )
    from quantark.sacva.exposure.simulator import MonteCarloExposureConfig

    pos = EquitySwapPosition(
        product=_make_trs(long=True, accrual_type=AccrualType.MARKET_VALUE), quantity=1.0)
    trade = pos.to_cva_trade(_env(), equity_bucket=2, allow_approx_financing=True)
    curve = PillarHazardCurve(tenors=REG_TENORS, hazards=[0.03] * 5, recovery_rate=0.4)
    cp = Counterparty(name="CP", netting_sets=[NettingSet("n1", [trade])],
                      credit_curve=curve, bucket=2, credit_quality=CreditQuality.IG)
    eng = SACVAEngine(
        exposure_engine=TRSPathwiseExposureEngine(
            MonteCarloExposureConfig(num_paths=8000, n_steps=16, seed=11)),
        include_ir_delta=False)
    result = eng.compute(CVATradePortfolio(counterparties=[cp], hedges=[]))
    assert result.total_capital > 0.0
    assert result.counterparty_cva["CP"] > 0.0


def test_pathwise_engine_leaves_notional_trs_unchanged():
    """The pathwise engine subclass delegates non-pathwise trades to the base engine,
    so a NOTIONAL TRS gives the same EPE under both engines (common random numbers)."""
    from quantark.asset.equity.engine.cashflow.trs_cva_exposure import (
        TRSPathwiseExposureEngine,
    )
    from quantark.sacva.exposure.simulator import MonteCarloExposureConfig

    pos = EquitySwapPosition(product=_make_trs(long=True), quantity=1.0)
    cfg = MonteCarloExposureConfig(num_paths=6000, n_steps=12, seed=9)
    base = MonteCarloExposureEngine(cfg).compute(_counterparty(pos, _env()))
    sub = TRSPathwiseExposureEngine(cfg).compute(_counterparty(pos, _env()))
    assert sub.epe_discounted == pytest.approx(base.epe_discounted)


def test_pathwise_builder_rejects_notional():
    from quantark.asset.equity.engine.cashflow.trs_cva_repricer import (
        build_trs_pathwise_cva_components,
    )
    with pytest.raises(ValidationError):
        build_trs_pathwise_cva_components(_make_trs(long=True).params)  # NOTIONAL accrual


def test_intermediate_redemption_routes_to_stateful():
    redm = [{"date": "2024-06-28", "redeem_notional": 400_000.0,
             "redeem_price": 100.0, "redeem_fee_rate": 0.0,
             "redeem_settle_option": ["asset", "int"]}]
    trs = _make_trs(long=True, redemption_events=redm)
    with pytest.raises(ValidationError):
        build_trs_cva_components(trs.params)


def test_dual_currency_trs_deferred():
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    path = pd.Series([S0] * len(idx), index=idx)
    calendar = Calendar(name="C")
    dual = OneAssetTotalReturnSwapDualCcy(
        TRSParams(
            contract_id="D", asset=AssetParams("IDX", S0, path),
            fix_leg=FixLegParams(rate=0.05, notional=NOTIONAL, initial_notional=NOTIONAL,
                                 start_date=START, end_date=END,
                                 payment_calendar=calendar, direction=-1),
            float_leg=FloatLegParams(notional=NOTIONAL, initial_notional=NOTIONAL,
                                     start_date=START, end_date=END,
                                     payment_calendar=calendar, direction=1),
            pricing=PricingParams(valuation_date=VAL, output_mode="spot")),
        asset_ccy="USD", settle_ccy="USD")
    pos = EquitySwapPosition(product=dual, quantity=1.0)
    with pytest.raises(ValidationError):
        pos.to_cva_trade(_env(), equity_bucket=2)
