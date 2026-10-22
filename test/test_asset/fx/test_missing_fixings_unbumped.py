"""A filled missing past fixing is history: every bump and scenario revaluation keeps it at the BASE spot.

User decision 2026-10-09 ("the past fixings need to remain unbumped"; "history doesn't move" under scenario
shocks too). A past fixing without a recorded outcome is decided once, at the base environment's spot, and every
bumped or shocked repricing reuses that decision. Parity: the Greeks / shocked values of a contract with missing
past fixings equal those of the same contract with the fixings recorded explicitly at the base spot. The warning
fires once per base valuation, never once per bump.
"""

import warnings
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import (
    AsianOptionAnalyticalEngine,
    RangeAccrualAnalyticalEngine,
)
from quantark.asset.equity.engine.mc import AsianOptionMCEngine, RangeAccrualMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import (
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    RangeAccrualOption,
)
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.fx.engine.analytical import (
    FxForeignRangeAccrualAnalyticalEngine,
    FxQuantoRangeAccrualAnalyticalEngine,
    FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.asset.fx.engine.mc import FxRangeAccrualMCEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxForeignRangeAccrualOption,
    FxQuantoRangeAccrualOption,
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
    FxRangeAccrualOption,
)
from quantark.asset.fx.riskmeasures import FxGreeksCalculator
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.equity.position import EquityPosition
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.priceenv.fx_pricing_environment import FxQuantoMarketData, QuantoConversionOrientation
from quantark.util.enum import AsianStrikeType, AveragingType, GreeksCalculationMode, OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.fixings import MissingFixingWarning, freeze_missing_fixings

VAL = datetime(2025, 1, 1)
# One 1% spot bump from 109.5 crosses the 110 upper barrier: a fill that followed the bumped spot would flip.
EQ_SPOT = 109.5


def _missing_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [w for w in caught if issubclass(w.category, MissingFixingWarning)]


def _assert_same(got: dict, want: dict):
    assert set(got) == set(want)
    diffs = {k: abs(got[k] - want[k]) for k in want}
    assert all(d == 0.0 for d in diffs.values()), diffs


# =============================================================================
# Equity range accrual: numeric and dated twins
# =============================================================================


def _eq_env(spot=EQ_SPOT):
    return PricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
    )


def _ra_numeric(outcome=None):
    records = [
        RangeAccrualObservationRecord(observation_time=-0.2, weight=1.0, observed_in_range=True),
        RangeAccrualObservationRecord(observation_time=-0.1, weight=2.0, observed_in_range=outcome),
        RangeAccrualObservationRecord(observation_time=0.5, weight=1.0),
        RangeAccrualObservationRecord(observation_time=1.0, weight=1.0),
    ]
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.05),
        maturity=1.0,
        tenor=1.25,
        observation_records=records,
    )


def _ra_dated(outcome=None):
    records = [
        RangeAccrualObservationRecord(observation_date=datetime(2024, 11, 1), observed_in_range=True),
        RangeAccrualObservationRecord(observation_date=datetime(2024, 12, 1), weight=2.0, observed_in_range=outcome),
        RangeAccrualObservationRecord(observation_date=datetime(2025, 7, 1)),
        RangeAccrualObservationRecord(observation_date=datetime(2026, 1, 1)),
    ]
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.05),
        exercise_date=datetime(2026, 1, 1),
        initial_date=datetime(2024, 10, 1),
        observation_records=records,
    )


RA_FORMS = pytest.mark.parametrize("make_ra", [_ra_numeric, _ra_dated], ids=["numeric", "dated"])
RA_ENGINES = pytest.mark.parametrize(
    "make_engine",
    [
        RangeAccrualAnalyticalEngine,
        lambda: RangeAccrualMCEngine(params=MCParams(num_paths=2000, seed=7), method=MonteCarloMethod.PSEUDO),
    ],
    ids=["analytical", "mc"],
)
EQ_GREEKS = ["delta", "gamma", "vega", "theta", "rho", "dividend_rho", "vanna", "speed", "delta_q", "charm"]


@RA_FORMS
@RA_ENGINES
def test_ra_calculator_greeks_equal_explicit_base_fill(make_ra, make_engine):
    env = _eq_env()
    calc = GreeksCalculator()
    greeks, caught = _missing_warnings(calc.calculate, make_ra(), env, make_engine(), greeks=EQ_GREEKS)
    assert len(caught) == 1  # once for the base valuation, not once per bump
    explicit, explicit_caught = _missing_warnings(
        calc.calculate, make_ra(outcome=True), env, make_engine(), greeks=EQ_GREEKS
    )
    assert explicit_caught == []
    _assert_same(greeks, explicit)


@RA_FORMS
def test_ra_mc_engine_bump_greeks_equal_explicit_base_fill(make_ra):
    engine = RangeAccrualMCEngine(params=MCParams(num_paths=2000, seed=7), method=MonteCarloMethod.PSEUDO)
    greeks, caught = _missing_warnings(engine.calculate_greeks, make_ra(), _eq_env())
    assert len(caught) == 1
    _assert_same(greeks, engine.calculate_greeks(make_ra(outcome=True), _eq_env()))


@RA_FORMS
@pytest.mark.parametrize("spot", [109.2, 109.9, 110.1, 110.8])
def test_ra_no_delta_jump_from_flipping_fills_near_the_barrier(make_ra, spot):
    """Each bump straddles 110; the delta of the filled contract is that of the contract with the base outcome."""
    calc = GreeksCalculator()
    engine = RangeAccrualAnalyticalEngine()
    greeks, _ = _missing_warnings(calc.calculate, make_ra(), _eq_env(spot), engine, greeks=["delta", "gamma"])
    explicit = calc.calculate(make_ra(outcome=spot <= 110.0), _eq_env(spot), engine, greeks=["delta", "gamma"])
    _assert_same(greeks, explicit)


@RA_FORMS
def test_ra_spot_ladder_keeps_the_base_fill(make_ra):
    engine = RangeAccrualMCEngine(params=MCParams(num_paths=1000, seed=7), method=MonteCarloMethod.PSEUDO)
    rows, caught = _missing_warnings(engine.calculate_spot_greeks_curve, make_ra(), _eq_env(), [105.0, 112.0])
    assert len(caught) == 1
    assert rows == engine.calculate_spot_greeks_curve(make_ra(outcome=True), _eq_env(), [105.0, 112.0])


@RA_FORMS
def test_ra_bucketed_greeks_keep_the_base_fill(make_ra):
    calc = GreeksCalculator()
    engine = RangeAccrualAnalyticalEngine()
    got, caught = _missing_warnings(calc.calculate_bucketed_greeks, make_ra(), _eq_env(), engine)
    assert len(caught) == 1
    assert got == calc.calculate_bucketed_greeks(make_ra(outcome=True), _eq_env(), engine)


@RA_FORMS
def test_ra_position_trade_greeks_keep_the_base_fill(make_ra):
    calc = GreeksCalculator()

    def position(product):
        return EquityPosition(
            product=product,
            quantity=3.0,
            entry_price=1.0,
            underlying="X",
            engine=RangeAccrualAnalyticalEngine(),
            entry_timestamp=VAL,
        )

    trade, caught = _missing_warnings(position(make_ra()).get_trade_greeks, _eq_env(), calc)
    assert len(caught) == 1
    _assert_same(trade, position(make_ra(outcome=True)).get_trade_greeks(_eq_env(), calc))
    risk, caught = _missing_warnings(position(make_ra()).get_trade_risk, _eq_env(), calc)
    assert len(caught) == 1
    explicit = position(make_ra(outcome=True)).get_trade_risk(_eq_env(), calc)
    _assert_same(risk.product, explicit.product)
    _assert_same(risk.total, explicit.total)


@RA_FORMS
def test_ra_freeze_records_base_fill_without_touching_the_product(make_ra):
    product = make_ra()
    frozen, caught = _missing_warnings(freeze_missing_fixings, product, _eq_env(111.0))
    assert len(caught) == 1
    assert product.observation_records[1].observed_in_range is None  # the user's product is never mutated
    assert frozen is not product
    assert frozen.observation_records[1].observed_in_range is False
    again, caught = _missing_warnings(freeze_missing_fixings, frozen, _eq_env(105.0))
    assert again is frozen and caught == []  # nothing left to fill
    complete = make_ra(outcome=True)
    assert freeze_missing_fixings(complete, _eq_env()) is complete


# =============================================================================
# Equity Asian
# =============================================================================

ASIAN_SPOT = 103.0


def _asian(first_price=None):
    return AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        maturity=1.0,
        observation_records=[
            AsianObservationRecord(observation_time=-0.2, observed_price=95.0),
            AsianObservationRecord(observation_time=-0.1, observed_price=first_price),
            AsianObservationRecord(observation_time=0.5),
            AsianObservationRecord(observation_time=1.0),
        ],
    )


def _asian_dated(first_price=None):
    return AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        exercise_date=datetime(2026, 1, 1),
        observation_records=[
            AsianObservationRecord(observation_date=datetime(2024, 11, 1), observed_price=95.0),
            AsianObservationRecord(observation_date=datetime(2024, 12, 1), observed_price=first_price),
            AsianObservationRecord(observation_date=datetime(2025, 7, 1)),
            AsianObservationRecord(observation_date=datetime(2026, 1, 1)),
        ],
    )


ASIAN_FORMS = pytest.mark.parametrize("make_asian", [_asian, _asian_dated], ids=["numeric", "dated"])
ASIAN_ENGINES = pytest.mark.parametrize(
    "make_engine",
    [
        AsianOptionAnalyticalEngine,
        lambda: AsianOptionMCEngine(params=MCParams(num_paths=2000, seed=7), method=MonteCarloMethod.PSEUDO),
    ],
    ids=["analytical", "mc"],
)


@ASIAN_FORMS
@ASIAN_ENGINES
def test_asian_calculator_greeks_equal_explicit_base_fill(make_asian, make_engine):
    env = _eq_env(ASIAN_SPOT)
    calc = GreeksCalculator()
    greeks, caught = _missing_warnings(calc.calculate, make_asian(), env, make_engine(), greeks=EQ_GREEKS)
    assert len(caught) == 1
    explicit = calc.calculate(make_asian(first_price=ASIAN_SPOT), env, make_engine(), greeks=EQ_GREEKS)
    _assert_same(greeks, explicit)


@ASIAN_FORMS
@ASIAN_ENGINES
def test_asian_engine_bump_greeks_equal_explicit_base_fill(make_asian, make_engine):
    engine = make_engine()
    greeks, caught = _missing_warnings(engine.calculate_greeks, make_asian(), _eq_env(ASIAN_SPOT))
    assert len(caught) == 1
    _assert_same(greeks, make_engine().calculate_greeks(make_asian(first_price=ASIAN_SPOT), _eq_env(ASIAN_SPOT)))


# =============================================================================
# FX range accrual (+ foreign, quanto): numeric and dated twins
# =============================================================================

# One 0.1% FX spot bump from 1.2995 crosses the 1.30 upper barrier.
FX_SPOT = 1.2995
FX_VAL = datetime(2026, 6, 15)


def _fx_env(spot=FX_SPOT):
    return FxPricingEnvironment(
        valuation_date=FX_VAL,
        spot_quote=SpotQuote(spot=spot),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
        quanto=FxQuantoMarketData(
            settlement_curve=FlatRateCurve(rate=0.05),
            quanto_vol=0.12,
            correlation=0.3,
            conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC,
        ),
    )


def _fx_kwargs(cls):
    return {"quanto_fx_rate": 110.0, "settlement_ccy": "JPY"} if cls is FxQuantoRangeAccrualOption else {}


def _fx_numeric(cls=FxRangeAccrualOption, outcome=None):
    return cls(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04),
        currency_pair=CurrencyPair("EUR", "USD"),
        maturity=1.0,
        observation_records=[
            FxRangeAccrualObservationRecord(observation_time=-0.25, observed_in_range=True),
            FxRangeAccrualObservationRecord(observation_time=-0.1, weight=2.0, observed_in_range=outcome),
            FxRangeAccrualObservationRecord(observation_time=0.5),
            FxRangeAccrualObservationRecord(observation_time=1.0),
        ],
        **_fx_kwargs(cls),
    )


def _fx_dated(cls=FxRangeAccrualOption, outcome=None):
    return cls(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04),
        currency_pair=CurrencyPair("EUR", "USD"),
        expiry_date=datetime(2027, 6, 15),
        initial_date=datetime(2026, 3, 15),
        observation_records=[
            FxRangeAccrualObservationRecord(observation_date=datetime(2026, 4, 15), observed_in_range=True),
            FxRangeAccrualObservationRecord(
                observation_date=datetime(2026, 5, 15), weight=2.0, observed_in_range=outcome
            ),
            FxRangeAccrualObservationRecord(observation_date=datetime(2026, 12, 15)),
            FxRangeAccrualObservationRecord(observation_date=datetime(2027, 6, 15)),
        ],
        **_fx_kwargs(cls),
    )


FX_FORMS = pytest.mark.parametrize("make_fx", [_fx_numeric, _fx_dated], ids=["numeric", "dated"])
FX_CASES = pytest.mark.parametrize(
    "cls, make_engine",
    [
        (FxRangeAccrualOption, FxRangeAccrualAnalyticalEngine),
        (FxForeignRangeAccrualOption, FxForeignRangeAccrualAnalyticalEngine),
        (FxQuantoRangeAccrualOption, FxQuantoRangeAccrualAnalyticalEngine),
        (FxRangeAccrualOption, lambda: FxRangeAccrualMCEngine(num_paths=2000, seed=7)),
    ],
    ids=["analytical", "foreign", "quanto", "mc"],
)


@FX_FORMS
@FX_CASES
def test_fx_engine_greeks_equal_explicit_base_fill(make_fx, cls, make_engine):
    greeks, caught = _missing_warnings(make_engine().calculate_greeks, make_fx(cls), _fx_env())
    assert len(caught) == 1  # once for the base valuation, not once per bump
    explicit, explicit_caught = _missing_warnings(
        make_engine().calculate_greeks, make_fx(cls, outcome=True), _fx_env()
    )
    assert explicit_caught == []
    _assert_same(greeks, explicit)


@FX_FORMS
@pytest.mark.parametrize("mode", [GreeksCalculationMode.BUMP, GreeksCalculationMode.AUTO])
def test_fx_calculator_greeks_equal_explicit_base_fill(make_fx, mode):
    calc = FxGreeksCalculator()
    engine = FxRangeAccrualAnalyticalEngine()
    greeks, caught = _missing_warnings(calc.calculate, make_fx(), _fx_env(), engine, mode=mode)
    assert len(caught) == 1
    _assert_same(greeks, calc.calculate(make_fx(outcome=True), _fx_env(), engine, mode=mode))


@FX_FORMS
@pytest.mark.parametrize("spot", [1.2990, 1.2999, 1.3001, 1.3010])
def test_fx_no_delta_jump_from_flipping_fills_near_the_barrier(make_fx, spot):
    engine = FxRangeAccrualAnalyticalEngine()
    greeks, _ = _missing_warnings(BaseFxEngine.calculate_greeks, engine, make_fx(), _fx_env(spot))
    explicit = BaseFxEngine.calculate_greeks(engine, make_fx(outcome=spot <= 1.30), _fx_env(spot))
    _assert_same(greeks, explicit)


@FX_FORMS
def test_fx_freeze_records_base_fill_without_touching_the_product(make_fx):
    product = make_fx()
    frozen, caught = _missing_warnings(freeze_missing_fixings, product, _fx_env(1.31))
    assert len(caught) == 1
    assert product.observation_records[1].observed_in_range is None
    assert frozen.observation_records[1].observed_in_range is False
    assert freeze_missing_fixings(frozen, _fx_env(1.2)) is frozen
