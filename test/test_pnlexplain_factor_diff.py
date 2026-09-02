"""Factor coordinate and FactorMoves (spec §5.3)."""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import DeltaOneEngine
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote, TermStructureVolSurface
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain.base import Factor, MARKET_FACTORS
from quantark.pnlexplain.equity.coordinate import resolve_coordinate
from quantark.pnlexplain.equity.factor_diff import build_factor_moves, validate_pair
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import OptionType
from quantark.util.enum.deltaone_enums import DeltaOneType
from quantark.util.exceptions import NumericalError, ValidationError

FRI = datetime(2026, 6, 26)
MON = datetime(2026, 6, 29)
# MODEL is detected by engine identity (spec §5.1): both snapshots share one engine object.
ENG = BlackScholesEngine()
D1 = DeltaOneEngine()


def _env(spot, vol, rate, div, date, calendar=None, **kw):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div), valuation_date=date, calendar=calendar, **kw)


def _call(m):
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=m)


def test_moves_read_at_t1_coordinate_and_changed_set():
    cal = create_calendar(CalendarType.CHINA_SSE)
    e0 = _env(100.0, TermStructureVolSurface(times=[0.5, 1.0], vols=[0.20, 0.25]), 0.03, 0.01, FRI, cal)
    e1 = _env(102.0, TermStructureVolSurface(times=[0.5, 1.0], vols=[0.21, 0.26]), 0.03, 0.02, MON, cal)
    p0, p1 = _call(1.0), _call(1.0 - 3 / 365)
    s0 = ValuationSnapshot(p0, ENG, e0, date=FRI)
    s1 = ValuationSnapshot(p1, ENG, e1, date=MON)
    coord = resolve_coordinate(p0, 100.0, p1, e1)
    assert coord.reference_strike == 100.0
    assert coord.tenor_t1 == pytest.approx(1.0 - 3 / 365)
    assert coord.applicable == frozenset(MARKET_FACTORS)
    mv = build_factor_moves(s0, s1, coord, engine_alive_t1=s1.engine, lifecycle_changed=False)
    t1 = coord.tenor_t1
    assert mv.vol_t0 == pytest.approx(e0.get_vol(100.0, t1))
    assert mv.vol_t1 == pytest.approx(e1.get_vol(100.0, t1))
    assert mv.d_vol == pytest.approx(mv.vol_t1 - mv.vol_t0)
    assert mv.d_rate == 0.0 and mv.d_div == pytest.approx(0.01)
    assert mv.calendar_days == 3 and mv.trading_days == 1
    assert Factor.RATE not in mv.changed and Factor.DIVIDEND in mv.changed
    assert Factor.TIME in mv.changed and Factor.MODEL not in mv.changed
    assert mv.display(Factor.SPOT) == {"spot_return": pytest.approx(0.02)}
    assert mv.display(Factor.TIME) == {"days": 3.0, "trading_days": 1.0}
    assert "trading_days" not in build_factor_moves(
        ValuationSnapshot(p0, ENG, _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, FRI), date=FRI),
        ValuationSnapshot(p1, ENG, _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON), date=MON),
        coord, engine_alive_t1=None, lifecycle_changed=False).display(Factor.TIME)
    # two distinct but equivalent engine objects ARE a model change (identity rule)
    other = build_factor_moves(s0, s1, coord, engine_alive_t1=BlackScholesEngine(), lifecycle_changed=False)
    assert Factor.MODEL in other.changed


def test_delta_one_coordinates():
    e0 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, FRI)
    e1 = _env(101.0, FlatVolSurface(0.2), 0.03, 0.01, MON)
    spot = SpotInstrument(underlying="X", deltaone_type=DeltaOneType.STOCK)
    c = resolve_coordinate(spot, 100.0, spot, e1)
    assert c.applicable == frozenset({Factor.SPOT, Factor.MODEL}) and c.tenor_t1 is None
    fut = Futures(underlying="X", multiplier=300.0, maturity=0.5)
    fut1 = Futures(underlying="X", multiplier=300.0, maturity=0.5 - 3 / 365)
    c2 = resolve_coordinate(fut, 100.0, fut1, e1)
    assert Factor.VOL not in c2.applicable and Factor.BASIS in c2.applicable
    s0 = ValuationSnapshot(spot, D1, e0, date=FRI)
    s1 = ValuationSnapshot(spot, D1, e1, date=MON)
    mv = build_factor_moves(s0, s1, c, engine_alive_t1=s1.engine, lifecycle_changed=False)
    assert mv.vol_t0 is None and mv.d_vol is None and mv.display(Factor.VOL) == {}
    assert Factor.VOL not in mv.changed


def test_expiry_day_drops_market_factors():
    e1 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON)
    c = resolve_coordinate(_call(3 / 365), 100.0, _call(1e-8), e1)
    assert c.applicable == frozenset({Factor.TIME, Factor.SPOT, Factor.MODEL})


def test_validate_pair_rejects_order_clock_quantity_currency():
    cal = create_calendar(CalendarType.CHINA_SSE)
    e0 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, FRI, cal)
    e1 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON, cal)
    s0 = ValuationSnapshot(_call(1.0), BlackScholesEngine(), e0, date=FRI)
    s1 = ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e1, date=MON)
    validate_pair(s0, s1)
    with pytest.raises(ValidationError):
        validate_pair(s1, s0)
    e_us = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON, create_calendar(CalendarType.US))
    with pytest.raises(ValidationError):
        validate_pair(s0, ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e_us, date=MON))
    e_bd = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON, cal,
                day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=244)
    with pytest.raises(ValidationError):
        validate_pair(s0, ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e_bd, date=MON))
    with pytest.raises(ValidationError):
        validate_pair(s0, ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e1, date=MON, quantity=2.0))
    with pytest.raises(ValidationError):
        validate_pair(ValuationSnapshot(_call(1.0), BlackScholesEngine(), e0, date=FRI, currency="CNY"),
                      ValuationSnapshot(_call(1.0 - 3 / 365), BlackScholesEngine(), e1, date=MON, currency="USD"))


def test_non_finite_sample_raises():
    class InfSurface(FlatVolSurface):
        def get_vol(self, strike, time_to_maturity, spot=None):
            return float("inf")

    e0 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, FRI)
    e1 = _env(100.0, InfSurface(0.2), 0.03, 0.01, MON)
    p0, p1 = _call(1.0), _call(1.0 - 3 / 365)
    coord = resolve_coordinate(p0, 100.0, p1, e1)
    with pytest.raises(NumericalError):
        build_factor_moves(ValuationSnapshot(p0, ENG, e0, date=FRI), ValuationSnapshot(p1, ENG, e1, date=MON),
                           coord, engine_alive_t1=ENG, lifecycle_changed=False)


def test_tenor_detects_date_expiry_and_reraises_other_validation_errors():
    e1 = _env(100.0, FlatVolSurface(0.2), 0.03, 0.01, MON)
    expired = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=FRI)
    c = resolve_coordinate(expired, 100.0, expired, e1)          # valued on/after expiry
    assert c.tenor_t1 == 0.0 and Factor.VOL not in c.applicable and Factor.SPOT in c.applicable

    class Broken(EuropeanVanillaOption):
        def get_maturity(self, pricing_env=None):
            raise ValidationError("malformed maturity")

    with pytest.raises(ValidationError, match="malformed"):
        resolve_coordinate(_call(1.0), 100.0,
                           Broken(strike=100.0, option_type=OptionType.CALL, maturity=1.0), e1)
