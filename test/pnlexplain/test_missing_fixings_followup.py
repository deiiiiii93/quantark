"""Follow-up rulings on missing past fixings (user decisions 2026-10-09, after review of w5b-fills-unbumped).

1. PnL explain: ``pv_t1`` is the t1 snapshot's OWN value, so consecutive daily explains chain; a filled missing
   fixing that is re-decided between t0 and t1 shows as its own ``FIXING_REFILL`` row. Shocked states keep t0's fill.
2. A fixing at exactly t = 0 (today, unprinted) is LIVE: only strictly past fixings are frozen as history; a t = 0
   fixing moves with spot bumps and shocks, as on main.
Plus reviewer minors: the roll check runs before the freeze, the spot ladder validates spots before freezing, the
surface-shock pipeline warns once, the replay carry reprices keep the day's fill.
"""

import math
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from quantark.asset.equity.engine.analytical import (
    AsianOptionAnalyticalEngine,
    RangeAccrualAnalyticalEngine,
)
from quantark.asset.equity.product.option import (
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    RangeAccrualOption,
)
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.fx.engine.analytical import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
    FxRangeAccrualOption,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.util.enum import AsianStrikeType, AveragingType, OptionType
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning, freeze_missing_fixings

VAL = datetime(2025, 1, 1)


def _count_missing(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, sum(issubclass(w.category, MissingFixingWarning) for w in caught)


def _eq_env(spot, date=VAL):
    return PricingEnvironment(
        valuation_date=date,
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
    )


def _ra_dated(outcome=None, *, records=None):
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.05),
        exercise_date=VAL + timedelta(days=365),
        initial_date=VAL - timedelta(days=92),
        observation_records=records
        or [
            RangeAccrualObservationRecord(observation_date=VAL - timedelta(days=61), observed_in_range=True),
            RangeAccrualObservationRecord(
                observation_date=VAL - timedelta(days=31), weight=2.0, observed_in_range=outcome
            ),
            RangeAccrualObservationRecord(observation_date=VAL + timedelta(days=181)),
            RangeAccrualObservationRecord(observation_date=VAL + timedelta(days=365)),
        ],
    )


# =============================================================================
# 1. PnL explain: daily explains chain; the re-fill is its own row
# =============================================================================


def _explain_chain(product_of, spots, quantity=2.0):
    from quantark.pnlexplain import explain
    from quantark.pnlexplain.base import ExplainMethod
    from quantark.pnlexplain.config import PnLExplainConfig
    from quantark.pnlexplain.equity.snapshot import ValuationSnapshot

    engine = RangeAccrualAnalyticalEngine()
    config = PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR))
    snaps = [
        ValuationSnapshot(product_of(), engine, _eq_env(s, VAL + timedelta(days=k)), date=VAL + timedelta(days=k),
                          quantity=quantity)
        for k, s in enumerate(spots)
    ]
    return snaps, [explain(a, b, config=config) for a, b in zip(snaps, snaps[1:])]


# W5b's warning for _ra_dated()'s record #1 (dated VAL - 31 days), left without an outcome
_MISSING_31D = r"RangeAccrualOption: 1 past fixing without observed_in_range \(#1 2024-12-01\)"


def _row(result, factor):
    rows = [r for r in result.rows if r.factor is factor]
    assert len(rows) <= 1
    return rows[0] if rows else None


def test_daily_explains_chain_and_the_refill_is_attributed():
    from quantark.pnlexplain.base import ExplainMethod, Factor

    spots = (109.5, 111.0, 111.5)
    (snaps, results), _ = _count_missing(_explain_chain, _ra_dated, spots)
    engine = RangeAccrualAnalyticalEngine()
    with pytest.warns(MissingFixingWarning, match=_MISSING_31D) as caught:   # each snapshot's own value fills #1
        own = [2.0 * engine.price(s.product, s.pricing_env) for s in snaps]
    assert len(caught) == len(snaps) == 3
    for k, res in enumerate(results):
        assert res.pv_t0.total == own[k] and res.pv_t1.total == own[k + 1]  # each endpoint is its own value
        assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-10)
        assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-10)
    assert results[0].pv_t1.total == results[1].pv_t0.total
    assert math.fsum(r.total_pnl for r in results) == pytest.approx(own[-1] - own[0], abs=1e-12)

    # Day 1: decided in range at 109.5 (t0), out of range at 111 (t1): the re-fill row carries exactly that.
    t1_env = snaps[1].pricing_env
    refill_1 = 2.0 * (engine.price(_ra_dated(False), t1_env) - engine.price(_ra_dated(True), t1_env))
    assert _row(results[0], Factor.FIXING_REFILL).pnl == pytest.approx(refill_1, abs=1e-12)
    assert refill_1 < -1.0
    # Day 2: out of range at both ends, nothing re-filled.
    assert _row(results[1], Factor.FIXING_REFILL).pnl == 0.0


def test_explain_market_rows_keep_the_t0_fill():
    """Shocked / intermediate states price the t0 decision: every market row equals the pair recorded at t0."""
    from quantark.pnlexplain.base import MARKET_FACTORS, Factor

    with pytest.warns(MissingFixingWarning, match=_MISSING_31D) as caught:
        _, (got,) = _explain_chain(_ra_dated, (109.5, 111.0))
    assert len(caught) == 2     # the pair's t0 freeze, and pv_t1 = the t1 snapshot's own fill
    _, (recorded,) = _explain_chain(lambda: _ra_dated(True), (109.5, 111.0))
    pick = lambda res: [(r.factor, r.method, r.pnl) for r in res.rows if r.factor in MARKET_FACTORS]  # noqa: E731
    assert pick(got) == pick(recorded)
    assert got.pv_alive_t1 == recorded.pv_alive_t1
    assert _row(got, Factor.LIFECYCLE_EVENT).pnl == _row(recorded, Factor.LIFECYCLE_EVENT).pnl
    assert _row(recorded, Factor.FIXING_REFILL) is None  # nothing missing, no row


def test_explain_roll_check_runs_before_the_freeze():
    """A t1 contract that is not the t0 contract rolled gets the roll-check message, not a fill error."""
    from quantark.pnlexplain import explain
    from quantark.pnlexplain.equity.snapshot import ValuationSnapshot

    engine = RangeAccrualAnalyticalEngine()
    t1 = VAL + timedelta(days=1)
    other = _ra_dated(records=[
        RangeAccrualObservationRecord(observation_date=VAL - timedelta(days=20)),
        RangeAccrualObservationRecord(observation_date=VAL + timedelta(days=365)),
    ])
    s0 = ValuationSnapshot(_ra_dated(), engine, _eq_env(109.5), date=VAL)
    s1 = ValuationSnapshot(other, engine, _eq_env(111.0, t1), date=t1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        with pytest.raises(ValidationError, match="contract replacement is not a time step"):
            explain(s0, s1)


# =============================================================================
# 2. A fixing at t = 0 is live
# =============================================================================


def _asian(**kw):
    return AsianOption(strike=100.0, option_type=OptionType.CALL, asian_strike_type=AsianStrikeType.FIXED,
                       averaging_type=AveragingType.ARITHMETIC, **kw)


def _asian_today_dated(first_price=None):
    return _asian(exercise_date=datetime(2026, 1, 1), observation_records=[
        AsianObservationRecord(observation_date=VAL - timedelta(days=31), observed_price=first_price),
        AsianObservationRecord(observation_date=VAL),
        AsianObservationRecord(observation_date=datetime(2025, 7, 2)),
        AsianObservationRecord(observation_date=datetime(2026, 1, 1)),
    ])


def _fx_env(spot, date=datetime(2026, 6, 15)):
    return FxPricingEnvironment(valuation_date=date, spot_quote=SpotQuote(spot=spot),
                                domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                                vol_surface=FlatVolSurface(volatility=0.10))


def _fx_today(numeric):
    if numeric:
        records = [FxRangeAccrualObservationRecord(observation_time=t) for t in (0.0, 0.5, 1.0)]
        kw = {"maturity": 1.0}
    else:
        d = datetime(2026, 6, 15)
        records = [FxRangeAccrualObservationRecord(observation_date=x)
                   for x in (d, datetime(2026, 12, 15), datetime(2027, 6, 15))]
        kw = {"expiry_date": datetime(2027, 6, 15), "initial_date": d}
    return FxRangeAccrualOption(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04),
        currency_pair=CurrencyPair("EUR", "USD"), observation_records=records, **kw,
    )


TODAY_CASES = {
    "ra_numeric": lambda: (RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.05),
        maturity=1.0, tenor=1.0,
        observation_records=[RangeAccrualObservationRecord(observation_time=t) for t in (0.0, 0.5, 1.0)],
    ), _eq_env(109.5)),
    "ra_dated": lambda: (_ra_dated(records=[
        RangeAccrualObservationRecord(observation_date=VAL),
        RangeAccrualObservationRecord(observation_date=VAL + timedelta(days=181)),
        RangeAccrualObservationRecord(observation_date=VAL + timedelta(days=365)),
    ]), _eq_env(109.5)),
    "asian_numeric": lambda: (_asian(maturity=1.0, observation_records=[
        AsianObservationRecord(observation_time=t) for t in (0.0, 0.5, 1.0)]), _eq_env(100.0)),
    "asian_dated": lambda: (_asian(exercise_date=datetime(2026, 1, 1), observation_records=[
        AsianObservationRecord(observation_date=d) for d in (VAL, datetime(2025, 7, 2), datetime(2026, 1, 1))]),
        _eq_env(100.0)),
    "fx_numeric": lambda: (_fx_today(True), _fx_env(1.2995)),
    "fx_dated": lambda: (_fx_today(False), _fx_env(1.2995)),
}


@pytest.mark.parametrize("case", sorted(TODAY_CASES))
def test_a_fixing_today_is_not_frozen(case):
    product, env = TODAY_CASES[case]()
    frozen, count = _count_missing(freeze_missing_fixings, product, env)
    assert frozen is product and count == 0


def _fd_delta_gamma(engine, product, env, h):
    """Central difference on engine.price over spot-bumped copies of the UNFROZEN product (main's semantics)."""
    def at(factor):
        bumped = deepcopy(env)
        bumped.spot_quote.spot *= factor
        return engine.price(product, bumped)

    up, base, down = at(1 + h), engine.price(product, env), at(1 - h)
    s = env.spot
    return (up - down) / (2 * s * h), (up - 2 * base + down) / (s * h) ** 2


@pytest.mark.parametrize("case", ["ra_numeric", "ra_dated", "asian_numeric", "asian_dated"])
def test_a_fixing_today_moves_with_the_equity_bump(case):
    product, env = TODAY_CASES[case]()
    engine = RangeAccrualAnalyticalEngine() if case.startswith("ra") else AsianOptionAnalyticalEngine()
    calc = GreeksCalculator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        got = calc.calculate(product, env, engine, greeks=["delta", "gamma"])
        delta, gamma = _fd_delta_gamma(engine, product, env, calc._bump_config.spot_bump)
    assert got["delta"] == pytest.approx(delta, rel=1e-12)
    assert got["gamma"] == pytest.approx(gamma, rel=1e-9)


@pytest.mark.parametrize("case", ["fx_numeric", "fx_dated"])
def test_a_fixing_today_moves_with_the_fx_bump(case):
    product, env = TODAY_CASES[case]()
    engine = FxRangeAccrualAnalyticalEngine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        got = BaseFxEngine.calculate_greeks(engine, product, env)
        delta, gamma = _fd_delta_gamma(engine, product, env, engine.params.spot_bump)
    assert got["delta"] == pytest.approx(delta, rel=1e-12)
    assert got["gamma"] == pytest.approx(gamma, rel=1e-9)


def test_asian_numeric_records_equal_the_legacy_schedule_with_a_fixing_today():
    calc = GreeksCalculator()
    env = _eq_env(100.0)
    legacy = calc.calculate(_asian(maturity=1.0, observation_times=[0.0, 0.5, 1.0]), env,
                            AsianOptionAnalyticalEngine(), greeks=["price", "delta", "gamma"])
    records = calc.calculate(TODAY_CASES["asian_numeric"]()[0], env, AsianOptionAnalyticalEngine(),
                             greeks=["price", "delta", "gamma"])
    assert records == legacy


def test_strictly_past_fixing_is_frozen_while_todays_stays_live():
    calc = GreeksCalculator()
    env = _eq_env(100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        got = calc.calculate(_asian_today_dated(), env, AsianOptionAnalyticalEngine(), greeks=["delta", "gamma"])
        want = calc.calculate(_asian_today_dated(100.0), env, AsianOptionAnalyticalEngine(),
                              greeks=["delta", "gamma"])
    assert got == want


# =============================================================================
# Reviewer minors
# =============================================================================


def test_spot_ladder_validates_spots_before_freezing():
    from quantark.asset.equity.engine.mc import RangeAccrualMCEngine

    engine = RangeAccrualMCEngine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="positive and finite"):
            engine.calculate_spot_greeks_curve(_ra_dated(), _eq_env(109.5), [100.0, -1.0])
    assert not [w for w in caught if issubclass(w.category, MissingFixingWarning)]


@pytest.mark.parametrize("mode", ["FROZEN", "RECALIBRATE"])
def test_surface_shock_pipeline_warns_once(mode):
    from dcn_fixtures import synthetic_cleaned_set

    from quantark.asset.equity.riskmeasures.surface_shock_pipeline import SurfaceShockMode, run_surface_shock_pipeline
    from quantark.util.calendar import DayCountConvention

    cleaned, rate_curve, carry_curve, _ = synthetic_cleaned_set()
    val = datetime(2023, 1, 3)
    spot = 6000.0

    def builder(surface):
        env = PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=surface, rate_curve=rate_curve,
                                 div_yield=None, valuation_date=val, day_count_convention=DayCountConvention.ACT_365)
        env.div_yield = carry_curve.to_dividend_yield(rate_curve)
        return env

    class Analytic:
        def price(self, product, env):
            return RangeAccrualAnalyticalEngine().price(product, env)

    product = RangeAccrualOption(
        initial_price=6000.0,
        range_config=RangeAccrualConfig(upper_barrier=6600.0, lower_barrier=5400.0, accrual_rate=0.05),
        exercise_date=val + timedelta(days=365), initial_date=val - timedelta(days=60),
        observation_records=[
            RangeAccrualObservationRecord(observation_date=val - timedelta(days=30)),
            RangeAccrualObservationRecord(observation_date=val + timedelta(days=365)),
        ],
    )
    _, count = _count_missing(
        run_surface_shock_pipeline, product, builder, cleaned, spot, rate_curve, carry_curve,
        model="local_vol", mode=getattr(SurfaceShockMode, mode), engine_factory=lambda model, artifact: Analytic(),
    )
    assert count == 1


def test_replay_carry_reprices_keep_the_days_fill():
    """The carry closures reprice the day's contract at moved spots: its missing past fixing stays at the day's spot."""
    from quantark.backtest.replay.engine import _book_price_callback
    from quantark.backtest.replay.product_replay import ProductReplay

    env = _eq_env(109.5)
    engine = RangeAccrualAnalyticalEngine()
    replay = SimpleNamespace(underlying="X")
    want = engine.price(_ra_dated(True), _eq_env(111.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        single = ProductReplay.carry_price_callback(replay, _ra_dated(), env, engine=engine)
        book = _book_price_callback([(2.0, replay, engine, _ra_dated(), 0.0, 0.0)], env)
        assert single(111.0, env.div_yield) == want
        assert book(111.0, env.div_yield) == 2.0 * want


# =============================================================================
# 3. Run fixings in PnL explain (spec 2026-10-09 run fixings §4, X5): a fixing that passes inside the step
# =============================================================================
import run_fixings_fixtures as RF  # noqa: E402
from quantark.util.numerical import is_close  # noqa: E402


def _rf_explain(product_t0, day_t0, product_t1, day_t1):
    from quantark.pnlexplain import explain
    from quantark.pnlexplain.base import ExplainMethod
    from quantark.pnlexplain.config import PnLExplainConfig
    from quantark.pnlexplain.equity.snapshot import ValuationSnapshot

    engine = RangeAccrualAnalyticalEngine()
    e0, e1 = RF.eq_env(day_t0), RF.eq_env(day_t1)
    s0 = ValuationSnapshot(product_t0, engine, e0, date=e0.valuation_date, quantity=1.0)
    s1 = ValuationSnapshot(product_t1, engine, e1, date=e1.valuation_date, quantity=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        return explain(s0, s1, config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR)))


def _close6(a, b):
    return is_close(a, b, rel_tol=0.0, abs_tol=5e-7)


def _rf_ra(form, day, outcomes=None):
    """The §2.1 range accrual as held on run day ``day``: the dated contract, or its numeric twin aged by the run
    clock (ACT/365 calendar days) to that day."""
    if form == "dated":
        return RF.ra_dated(outcomes)
    return RF.ra_numeric(outcomes).with_time_elapsed(day / 365.0)


FORMS = pytest.mark.parametrize("form", ["dated", "numeric"])


@FORMS
@pytest.mark.parametrize("t1_outcomes, pv_t1, refill", [
    ({}, 2.744374, -1.905067),          # P1
    ({0: True}, 4.649441, 0.0),         # P2
    ({0: False}, 2.744374, -1.905067),  # P2b
], ids=["P1", "P2", "P2b"])
def test_x5_a_fixing_dated_t0_is_fixed_at_s0_for_the_pair(form, t1_outcomes, pv_t1, refill):
    from quantark.pnlexplain.base import ExplainMethod, Factor

    res = _rf_explain(_rf_ra(form, 4), 4, _rf_ra(form, 10, t1_outcomes), 10)
    alive = RF.quiet_price(RangeAccrualAnalyticalEngine(), RF.ra_dated({0: True}), RF.eq_env(10))
    assert is_close(res.pv_alive_t1.total, alive, rel_tol=0.0, abs_tol=1e-12) and _close6(alive, 4.649441)
    assert _close6(res.pv_t1.total, pv_t1) and _close6(res.pv_t0.total, 5.707299)
    row = _row(res, Factor.FIXING_REFILL)
    assert row is not None and _close6(row.pnl, refill)
    assert is_close(row.pnl, res.pv_t1.total - res.pv_alive_t1.total - _row(res, Factor.LIFECYCLE_EVENT).pnl,
                    rel_tol=0.0, abs_tol=1e-12)
    for m in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR):
        assert res.reconcile(m) == pytest.approx(0.0, abs=1e-10)


@FORMS
@pytest.mark.parametrize("t1_outcomes", [{}, {0: True}], ids=["P3", "P3b"])
def test_x5_a_fixing_strictly_inside_the_step_is_fixed_at_s0(form, t1_outcomes):
    from quantark.pnlexplain.base import Factor

    res = _rf_explain(_rf_ra(form, 3), 3, _rf_ra(form, 5, t1_outcomes), 5)
    assert _close6(res.pv_t1.total, 5.592668) and _row(res, Factor.FIXING_REFILL).pnl == 0.0


def test_x5_p4_missing_at_t0_is_unchanged():
    from quantark.pnlexplain import explain
    from quantark.pnlexplain.base import ExplainMethod, Factor
    from quantark.pnlexplain.config import PnLExplainConfig
    from quantark.pnlexplain.equity.snapshot import ValuationSnapshot

    def prod():
        return RangeAccrualOption(
            initial_price=100.0, range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0,
                                                                 accrual_rate=0.10),
            exercise_date=RF.START + timedelta(days=365), initial_date=RF.START - timedelta(days=10),
            observation_records=[RangeAccrualObservationRecord(observation_date=RF.START - timedelta(days=3))]
            + [RangeAccrualObservationRecord(observation_date=d) for d in RF.fixing_dates()[2:]])
    engine = RangeAccrualAnalyticalEngine()
    e0, e1 = RF.eq_env(0, spot=105.0), RF.eq_env(6, spot=111.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        res = explain(ValuationSnapshot(prod(), engine, e0, date=e0.valuation_date, quantity=1.0),
                      ValuationSnapshot(prod(), engine, e1, date=e1.valuation_date, quantity=1.0),
                      config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR)))
    assert _close6(_row(res, Factor.FIXING_REFILL).pnl, -2.380029)


@FORMS
def test_x5_changing_a_recorded_fixing_is_still_refused(form):
    with pytest.raises(ValidationError, match="contract replacement is not a time step"):
        _rf_explain(_rf_ra(form, 5, {0: True}), 5, _rf_ra(form, 10, {0: False}), 10)


@FORMS
def test_x5_an_outcome_on_a_future_record_is_still_refused(form):
    with pytest.raises(ValidationError, match="contract replacement is not a time step"):
        _rf_explain(_rf_ra(form, 4), 4, _rf_ra(form, 10, {1: True}), 10)   # #1 is dated day 18


@FORMS
def test_a_fixing_dated_t1_stays_live_in_the_pair(form):
    """Review Focus 5."""
    from quantark.pnlexplain.base import Factor

    res = _rf_explain(_rf_ra(form, 3), 3, _rf_ra(form, 4), 4)
    assert _row(res, Factor.FIXING_REFILL) is None
    assert res.pv_alive_t1.total == pytest.approx(res.pv_t1.total, abs=1e-12)


def test_x5_numeric_participants_roll_through_the_check():
    from quantark.pnlexplain.base import Factor

    day = 1.0 / 365.0
    t0 = RF.ra_numeric()
    for _ in range(3):
        t0 = t0.with_time_elapsed(day)
    t1 = t0
    for _ in range(2):
        t1 = t1.with_time_elapsed(day)
    res = _rf_explain(t0, 3, t1, 5)                                             # #0 (day 4) passes inside
    assert is_close(_row(res, Factor.FIXING_REFILL).pnl, 0.0, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize("d0, d1, t1_outcomes", [(4, 10, {}), (4, 10, {0: False}), (3, 5, {}), (3, 4, {})],
                         ids=["P1", "P2b", "P3", "t1-live"])
def test_x5_the_numeric_twin_explains_as_the_dated_contract(d0, d1, t1_outcomes):
    dated = _rf_explain(_rf_ra("dated", d0), d0, _rf_ra("dated", d1, t1_outcomes), d1)
    numeric = _rf_explain(_rf_ra("numeric", d0), d0, _rf_ra("numeric", d1, t1_outcomes), d1)
    key = lambda res: [(r.factor, r.method, r.term) for r in res.rows]  # noqa: E731
    assert key(numeric) == key(dated)
    for a, b in zip(numeric.rows, dated.rows):
        assert is_close(a.pnl, b.pnl, rel_tol=1e-12, abs_tol=1e-12), (a.factor, a.pnl, b.pnl)
    for name in ("pv_t0", "pv_alive_t1", "pv_t1"):
        assert is_close(getattr(numeric, name).total, getattr(dated, name).total, rel_tol=1e-12)
