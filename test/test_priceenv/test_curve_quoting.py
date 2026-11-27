"""Cash-curve quoting basis on BUSINESS_DAYS environments (0.15.0 spec, AC1-AC7c).

A BUSINESS_DAYS environment integrates r, q and the futures basis over trading-year time. A curve quoted per
calendar year must be wrapped onto the trading clock; the environment now knows which curves are which.
"""
import dataclasses
from copy import deepcopy
from datetime import datetime

import pytest

from quantark.param import SpotQuote
from quantark.param.basis import FlatBasisYield, TradingClockBasisYield, ZeroBasis
from quantark.param.div import TradingClockDividendYield
from quantark.param.div.dividend_yield import ContinuousDividendYield, NoDividend, TermStructureDividendYield
from quantark.param.rrf import TradingClockRateCurve
from quantark.param.rrf.rate_curve import FlatRateCurve, LinearRateCurve
from quantark.param.vol import FlatVolSurface
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, DateKnotTimeMap, TradingClock
from quantark.util.exceptions import MarketDataError, ValidationError

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2029))
VAL = datetime(2026, 2, 9)
HORIZON = datetime(2027, 6, 9)
CAL_Q, TRD_Q = CurveQuotingBasis.CALENDAR_YEAR, CurveQuotingBasis.TRADING_YEAR
FIELDS = ("rate_curve", "div_yield", "basis_yield")


def _natives():
    return {"rate_curve": FlatRateCurve(0.02), "div_yield": ContinuousDividendYield(0.01),
            "basis_yield": FlatBasisYield(0.013)}


def _env(convention=DayCountConvention.BUSINESS_DAYS, calendar=CAL, **kw):
    base = dict(valuation_date=VAL, spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2),
                day_count_convention=convention, bus_days_in_year=244, calendar=calendar)
    base.update(_natives())
    base.update(kw)
    return PricingEnvironment(**base)


def _declared(q, **kw):
    return dict(rate_curve_quoting=q, div_yield_quoting=q, basis_yield_quoting=q, **kw)


def _map(anchor=VAL, horizon=HORIZON):
    return BusinessTimeMap(TradingClock(CAL, 244), anchor, horizon)


def _env_map(anchor=VAL, horizon=HORIZON):
    """The map the environment builds for its CALENDAR_YEAR curves (run 8 D2: business-date knots)."""
    return DateKnotTimeMap(TradingClock(CAL, 244), anchor, horizon)


WRAPPERS = {"rate_curve": TradingClockRateCurve, "div_yield": TradingClockDividendYield,
            "basis_yield": TradingClockBasisYield}


# --- AC1: undeclared non-zero native refuses ---------------------------------------------------------------------

@pytest.mark.parametrize("field", FIELDS)
def test_undeclared_nonzero_native_refuses(field):
    others = {f: CAL_Q for f in FIELDS if f != field}
    kw = {f"{f}_quoting": q for f, q in others.items()}
    with pytest.raises(ValidationError) as err:
        _env(trading_clock_horizon=HORIZON, **kw)
    msg = str(err.value)
    assert field in msg and "CALENDAR_YEAR" in msg and "TRADING_YEAR" in msg and "trading_clock_horizon" in msg


def test_undeclared_term_structure_refuses():
    with pytest.raises(ValidationError):
        _env(rate_curve=FlatRateCurve(0.0),
             div_yield=TermStructureDividendYield(times=[0.5, 1.0], yields=[0.0, 0.0]), basis_yield=None)


# --- AC2: exact zero is exempt -----------------------------------------------------------------------------------

@pytest.mark.parametrize("rate,div,basis", [
    (FlatRateCurve(0.0), ContinuousDividendYield(0.0), FlatBasisYield(0.0)),
    (FlatRateCurve(0.0), NoDividend(), ZeroBasis()),
    (FlatRateCurve(0.0), None, None),
])
def test_exact_zero_curves_need_no_declaration(rate, div, basis):
    env = _env(rate_curve=rate, div_yield=div, basis_yield=basis)
    assert env.rate_curve is rate and env.div_yield is div and env.basis_yield is basis


def test_zero_flags_are_exact():
    assert FlatRateCurve(0.0).is_identically_zero() and not FlatRateCurve(1e-300).is_identically_zero()
    assert ContinuousDividendYield(0.0).is_identically_zero() and NoDividend().is_identically_zero()
    assert not ContinuousDividendYield(1e-300).is_identically_zero()
    assert not LinearRateCurve(pillars=[(0.5, 0.0), (1.0, 0.0)]).is_identically_zero()


def test_duck_typed_curve_counts_as_non_zero():
    class Duck:
        def get_yield(self, t):
            return 0.0
    with pytest.raises(ValidationError):
        _env(rate_curve=FlatRateCurve(0.0), div_yield=Duck(), basis_yield=None)
    env = _env(rate_curve=FlatRateCurve(0.0), div_yield=Duck(), basis_yield=None, div_yield_quoting=TRD_Q)
    assert isinstance(env.div_yield, Duck)


# --- AC3: TRADING_YEAR ---------------------------------------------------------------------------------------------

def test_trading_year_keeps_native_objects():
    n = _natives()
    env = _env(**n, **_declared(TRD_Q))
    for f in FIELDS:
        assert getattr(env, f) is n[f]


def test_trading_year_on_wrapped_curve_refuses():
    with pytest.raises(ValidationError):
        _env(rate_curve=TradingClockRateCurve(FlatRateCurve(0.02), _map()), rate_curve_quoting=TRD_Q,
             div_yield_quoting=TRD_Q, basis_yield_quoting=TRD_Q)


# --- AC4: CALENDAR_YEAR wraps --------------------------------------------------------------------------------------

def test_calendar_year_needs_horizon():
    with pytest.raises(ValidationError, match="trading_clock_horizon"):
        _env(**_declared(CAL_Q))


def test_calendar_year_wraps_on_env_map():
    n = _natives()
    env = _env(**n, **_declared(CAL_Q, trading_clock_horizon=HORIZON))
    for f in FIELDS:
        w = getattr(env, f)
        assert type(w) is WRAPPERS[f] and w.inner is n[f]
        assert w.time_map.anchor_date == VAL and w.time_map.horizon_date == HORIZON
        assert w.clock.calendar is CAL and w.clock.days_per_year == 244
    # one map shared by every wrapper
    assert env.rate_curve.time_map is env.div_yield.time_map is env.basis_yield.time_map


def test_calendar_year_on_none_leaves_none():
    env = _env(div_yield=None, basis_yield=None, **_declared(CAL_Q, trading_clock_horizon=HORIZON))
    assert env.div_yield is None and env.basis_yield is None


# --- AC5: CALENDAR_YEAR on an existing wrapper ---------------------------------------------------------------------

def test_matching_wrapper_is_kept():
    # run 8 D3: "matching" is the env's own map -- a DateKnotTimeMap at its anchor, horizon and holiday convention
    w = TradingClockRateCurve(FlatRateCurve(0.02), _env_map())
    env = _env(rate_curve=w, **_declared(CAL_Q, trading_clock_horizon=HORIZON))
    assert env.rate_curve is w


def test_declared_wrapper_on_a_business_time_map_moves_onto_the_env_map():
    # run 8 D3: the declaration owns the map (before run 8 a matching BusinessTimeMap wrapper was kept as is)
    w = TradingClockRateCurve(FlatRateCurve(0.02), _map())
    env = _env(rate_curve=w, **_declared(CAL_Q, trading_clock_horizon=HORIZON))
    assert env.rate_curve.inner is w.inner and type(env.rate_curve.time_map) is DateKnotTimeMap
    assert env.rate_curve.time_map is env.div_yield.time_map


def test_replace_valuation_date_re_anchors():
    env = _env(**_declared(CAL_Q, trading_clock_horizon=HORIZON))
    d2 = datetime(2026, 3, 2)
    env2 = dataclasses.replace(env, valuation_date=d2)
    for f in FIELDS:
        w0, w2 = getattr(env, f), getattr(env2, f)
        assert w2.time_map.anchor_date == d2 and w2.inner is w0.inner
    assert env.rate_curve.time_map.anchor_date == VAL  # source untouched


def test_mismatched_clock_still_refuses():
    other = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2029))
    w = TradingClockRateCurve(FlatRateCurve(0.02), BusinessTimeMap(TradingClock(other, 244), VAL, HORIZON))
    with pytest.raises(ValidationError):
        _env(rate_curve=w, **_declared(CAL_Q, trading_clock_horizon=HORIZON))


# --- AC6 (owner D7): undeclared hand wrappers re-anchor on their own horizon ----------------------------------------

def test_undeclared_hand_wrapper_on_anchor_is_kept():
    w = TradingClockRateCurve(FlatRateCurve(0.02), _map())
    env = _env(rate_curve=w, div_yield=None, basis_yield=None)
    assert env.rate_curve is w


def test_undeclared_hand_wrapper_off_anchor_re_anchors_at_construction():
    own_horizon = datetime(2027, 3, 1)
    w = TradingClockRateCurve(FlatRateCurve(0.02), _map(anchor=datetime(2026, 1, 5), horizon=own_horizon))
    env = _env(rate_curve=w, div_yield=None, basis_yield=None)
    assert env.rate_curve.inner is w.inner
    assert env.rate_curve.time_map.anchor_date == VAL and env.rate_curve.time_map.horizon_date == own_horizon
    hand = TradingClockRateCurve(FlatRateCurve(0.02), _map(horizon=own_horizon))
    for u in (1 / 244.0, 0.3, 1.0):
        assert env.get_discount_factor(u) == hand.get_discount_factor(u)


@pytest.mark.parametrize("convention", [DayCountConvention.BUSINESS_DAYS, DayCountConvention.CALENDAR_DAYS])
def test_undeclared_hand_wrapper_re_anchors_on_assignment(convention):
    w = TradingClockDividendYield(ContinuousDividendYield(0.01), _map())
    env = _env(convention=convention, calendar=CAL, rate_curve=FlatRateCurve(0.0), div_yield=w, basis_yield=None)
    d2 = datetime(2026, 3, 2)
    env.valuation_date = d2
    assert env.div_yield.time_map.anchor_date == d2 and env.div_yield.time_map.horizon_date == HORIZON
    assert env.div_yield.inner is w.inner


def test_undeclared_hand_wrapper_past_its_horizon_refuses():
    w = TradingClockRateCurve(FlatRateCurve(0.02), _map(horizon=datetime(2026, 6, 1)))
    env = _env(rate_curve=w, div_yield=None, basis_yield=None)
    with pytest.raises(ValidationError, match="rate_curve"):
        env.valuation_date = datetime(2026, 7, 1)
    assert env.rate_curve is w and env.valuation_date == VAL


def test_off_business_horizon_refusal_does_not_advise_declaring():
    w = TradingClockRateCurve(FlatRateCurve(0.02), _map(horizon=datetime(2026, 6, 1)))
    env = _env(convention=DayCountConvention.CALENDAR_DAYS, rate_curve=w, div_yield=None, basis_yield=None)
    with pytest.raises(ValidationError) as err:
        env.valuation_date = datetime(2026, 7, 1)
    assert "later horizon" in str(err.value) and "trading_clock_horizon" not in str(err.value)


# --- AC7d (owner D19): the vol wrapper re-anchors too -------------------------------------------------------------

def _vol_env(surface, convention=DayCountConvention.CALENDAR_DAYS, **kw):
    return _env(convention=convention, rate_curve=FlatRateCurve(0.0), div_yield=None, basis_yield=None,
                vol_surface=surface, **kw)


def test_vol_wrapper_on_anchor_is_kept():
    from quantark.param.vol import TradingClockVolSurface
    s = TradingClockVolSurface(FlatVolSurface(0.2), _map())
    assert _vol_env(s).vol_surface is s


@pytest.mark.parametrize("convention", [DayCountConvention.CALENDAR_DAYS, DayCountConvention.BUSINESS_DAYS])
def test_vol_wrapper_off_anchor_re_anchors_at_construction(convention):
    from quantark.param.vol import TradingClockVolSurface
    s = TradingClockVolSurface(FlatVolSurface(0.2), _map(anchor=datetime(2026, 1, 5)))
    env = _vol_env(s, convention=convention)
    assert env.vol_surface.inner is s.inner
    assert env.vol_surface.time_map.anchor_date == VAL and env.vol_surface.time_map.horizon_date == HORIZON


def test_vol_wrapper_re_anchors_on_assignment():
    from quantark.param.vol import TradingClockVolSurface
    env = _vol_env(TradingClockVolSurface(FlatVolSurface(0.2), _map()))
    d2 = datetime(2026, 2, 14)
    env.valuation_date = d2
    fresh = TradingClockVolSurface(FlatVolSurface(0.2), _map(anchor=d2))
    for t in (1 / 365.0, 10 / 365.0, 0.5):
        assert env.vol_surface.get_vol(100.0, t, 100.0) == fresh.get_vol(100.0, t, 100.0)


def test_vol_wrapper_past_its_horizon_refuses():
    from quantark.param.vol import TradingClockVolSurface
    s = TradingClockVolSurface(FlatVolSurface(0.2), _map(horizon=datetime(2026, 6, 1)))
    env = _vol_env(s)
    with pytest.raises(ValidationError, match="vol_surface"):
        env.valuation_date = datetime(2026, 7, 1)
    assert env.vol_surface is s and env.valuation_date == VAL


def test_sticky_view_keeps_its_shock_when_re_anchored():
    from quantark.param.vol import TradingClockVolSurface
    from quantark.param.vol.sticky import _StickyMoneynessView
    view = _StickyMoneynessView(TradingClockVolSurface(FlatVolSurface(0.2), _map(anchor=datetime(2026, 1, 5))),
                                100.0, 90.0)
    env = _vol_env(view)
    assert isinstance(env.vol_surface, _StickyMoneynessView)
    assert env.vol_surface.time_map.anchor_date == VAL and env.vol_surface._ratio == view._ratio


# --- AC7: non-BUSINESS_DAYS -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("conv", [DayCountConvention.CALENDAR_DAYS, DayCountConvention.ACT_365])
def test_calendar_axis_calendar_year_is_noop(conv):
    n = _natives()
    env = _env(convention=conv, calendar=None, **n, **_declared(CAL_Q))
    for f in FIELDS:
        assert getattr(env, f) is n[f]


def test_calendar_axis_trading_year_refuses():
    with pytest.raises(ValidationError):
        _env(convention=DayCountConvention.CALENDAR_DAYS, rate_curve_quoting=TRD_Q)


def test_axis_switch_unwraps_env_owned_wrappers():
    n = _natives()
    env = _env(**n, **_declared(CAL_Q, trading_clock_horizon=HORIZON))
    cal_env = dataclasses.replace(env, day_count_convention=DayCountConvention.CALENDAR_DAYS)
    for f in FIELDS:
        assert getattr(cal_env, f) is n[f]


def test_bad_quoting_type_refuses():
    with pytest.raises(MarketDataError):
        _env(rate_curve_quoting="calendar_year")


# --- AC7b: calendar / horizon edges ---------------------------------------------------------------------------------

def test_calendar_year_without_calendar_refuses():
    with pytest.raises(ValidationError, match="calendar"):
        PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=VAL,
                           day_count_convention=DayCountConvention.BUSINESS_DAYS, calendar=None,
                           rate_curve_quoting=CAL_Q, trading_clock_horizon=HORIZON)


@pytest.mark.parametrize("rate", [
    pytest.param(lambda: TradingClockRateCurve(FlatRateCurve(0.02), _map()), id="matching-wrapper"),
    pytest.param(lambda: FlatRateCurve(0.0), id="zero-native"),
])
def test_calendar_year_without_calendar_refuses_up_front(rate):
    """D14 at construction, not at the first theta bump: no map need be built for the refusal."""
    with pytest.raises(ValidationError, match="calendar"):
        PricingEnvironment(rate_curve=rate(), valuation_date=VAL, day_count_convention=DayCountConvention.BUSINESS_DAYS,
                           calendar=None, bus_days_in_year=244, rate_curve_quoting=CAL_Q,
                           div_yield_quoting=CAL_Q, trading_clock_horizon=HORIZON)


def test_failed_reassignment_leaves_env_unchanged():
    env = _env(**_declared(CAL_Q, trading_clock_horizon=HORIZON))
    before = {f: getattr(env, f) for f in FIELDS}
    with pytest.raises(ValidationError):
        env.valuation_date = datetime(2027, 7, 1)
    assert env.valuation_date == VAL
    for f in FIELDS:
        assert getattr(env, f) is before[f]


def test_flat_subclass_is_not_assumed_zero():
    class Overridden(FlatRateCurve):
        def get_rate(self, t):
            return 0.05
    assert not Overridden(0.0).is_identically_zero()

    class OverriddenQ(ContinuousDividendYield):
        pass

    class OverriddenB(FlatBasisYield):
        pass
    assert not OverriddenQ(0.0).is_identically_zero()
    assert not OverriddenB(0.0).is_identically_zero()


def test_valuation_at_or_past_horizon_refuses_naming_field():
    with pytest.raises(ValidationError, match="trading_clock_horizon"):
        _env(valuation_date=HORIZON, **_declared(CAL_Q, trading_clock_horizon=HORIZON))
    env = _env(**_declared(CAL_Q, trading_clock_horizon=HORIZON))
    with pytest.raises(ValidationError, match="trading_clock_horizon"):
        env.valuation_date = datetime(2027, 7, 1)


def test_time_of_day_anchor_matches_hand_wrap():
    vt = datetime(2026, 2, 9, 14, 55)
    env = _env(valuation_date=vt, **_declared(CAL_Q, trading_clock_horizon=HORIZON))
    # run 8 D6: the hand recipe is the env's DateKnotTimeMap (knots from the midnight-normalised anchor, like the
    # engines' axis); on the 0.15.0 BusinessTimeMap recipe u = 1.0 read 0.9797154068310359, now 0.979715406831036
    hand = TradingClockRateCurve(FlatRateCurve(0.02), _env_map(anchor=vt))
    for u in (1 / 244.0, 37 / 244.0, 1.0):
        assert env.rate_curve.get_discount_factor(u) == hand.get_discount_factor(u)


# --- AC7c: re-anchor on assignment ---------------------------------------------------------------------------------

def test_assignment_re_anchors_declared_wrappers():
    env = _env(**_declared(CAL_Q, trading_clock_horizon=HORIZON))
    d2 = datetime(2026, 3, 2)
    ref = dataclasses.replace(env, valuation_date=d2)
    moved = deepcopy(env)
    moved.valuation_date = d2
    for f in FIELDS:
        w = getattr(moved, f)
        assert w.time_map.anchor_date == d2
    for u in (1 / 244.0, 0.3, 1.0):
        assert moved.get_discount_factor(u) == ref.get_discount_factor(u)
        assert moved.get_div_yield(u) == ref.get_div_yield(u)
        assert moved.get_basis_yield(u) == ref.get_basis_yield(u)


def test_assignment_on_trading_year_env_changes_nothing():
    n = _natives()
    env = _env(**n, **_declared(TRD_Q))
    env.valuation_date = datetime(2026, 3, 2)
    for f in FIELDS:
        assert getattr(env, f) is n[f]


def test_deepcopy_keeps_wrappers_and_identity():
    env = _env(**_declared(CAL_Q, trading_clock_horizon=HORIZON))
    c = deepcopy(env)
    assert c.rate_curve.clock.calendar is c.calendar
