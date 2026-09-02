"""ValuationSnapshot, value identity, fingerprints (spec §5.1-5.3, §8)."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.lifecycle.cashflows import RealizedCashflow, ValuationPoint
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState, BarrierLifecycleState
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain.equity.fingerprints import (
    calendars_equal, check_contract_roll, contract_fingerprint, lifecycle_fingerprint,
)
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal, value
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import OptionType
from quantark.util.exceptions import NumericalError, ValidationError

D0 = datetime(2026, 6, 26)


def _env(spot=100.0, date=D0, calendar=None):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=date, calendar=calendar,
    )


def _call(maturity=1.0):
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=maturity)


def test_snapshot_validation():
    env = _env()
    ok = ValuationSnapshot(product=_call(), engine=BlackScholesEngine(), pricing_env=env, date=D0)
    assert ok.point == ValuationPoint(date=D0)
    with pytest.raises(ValidationError):  # date != env date
        ValuationSnapshot(_call(), BlackScholesEngine(), env, date=D0 + timedelta(days=1))
    with pytest.raises(ValidationError):  # intraday
        env2 = _env(date=D0 + timedelta(hours=3))
        ValuationSnapshot(_call(), BlackScholesEngine(), env2, date=D0 + timedelta(hours=3))
    with pytest.raises(ValidationError):  # aware
        env3 = _env(date=D0.replace(tzinfo=timezone.utc))
        ValuationSnapshot(_call(), BlackScholesEngine(), env3, date=D0.replace(tzinfo=timezone.utc))
    with pytest.raises(ValidationError):  # zero quantity
        ValuationSnapshot(_call(), BlackScholesEngine(), env, date=D0, quantity=0.0)
    with pytest.raises(ValidationError):  # date-based point must equal date
        ValuationSnapshot(_call(), BlackScholesEngine(), env, date=D0,
                          valuation_point=ValuationPoint(date=D0 + timedelta(days=1)))


def test_value_identity_contingent_plus_ledger():
    env = _env()
    engine = BlackScholesEngine()
    snap = ValuationSnapshot(_call(), engine, env, date=D0, quantity=3.0)
    vb = value(snap)
    assert vb.contingent_mtm == pytest.approx(3.0 * engine.price(_call(), env))
    assert vb.pending_receivable_pv == 0.0 and vb.paid_cash == 0.0

    state = AutocallableLifecycleState()
    state.mark_ko(D0 - timedelta(days=5), cashflow=30.0,
                  settlement_date=D0 + timedelta(days=2))
    term = ValuationSnapshot(_call(), engine, env, date=D0, quantity=3.0, lifecycle_state=state)
    assert is_terminal(state)
    vb2 = value(term)
    assert vb2.contingent_mtm == 0.0
    df = env.get_discount_factor(2.0 / 365.0)
    assert vb2.pending_receivable_pv == pytest.approx(30.0 * df, rel=1e-9)
    # after payment the same ledger is paid cash
    env_later = _env(date=D0 + timedelta(days=3))
    vb3 = value(term, pricing_env=env_later, valuation_point=ValuationPoint(date=D0 + timedelta(days=3)))
    assert vb3.pending_receivable_pv == 0.0 and vb3.paid_cash == pytest.approx(30.0)


def test_calendar_semantic_equality():
    a = create_calendar(CalendarType.CHINA_SSE)
    assert calendars_equal(a, deepcopy(a))
    assert calendars_equal(None, None)
    assert not calendars_equal(a, None)
    assert not calendars_equal(a, create_calendar(CalendarType.US))


def test_contract_fingerprint_and_roll():
    p0 = _call(maturity=1.0)
    p1 = _call(maturity=1.0 - 3 / 365)
    assert contract_fingerprint(p0) == contract_fingerprint(p1)          # maturity is rolled
    check_contract_roll(p0, p1, calendar_days=3)                          # ok
    with pytest.raises(ValidationError):
        check_contract_roll(p0, _call(maturity=1.0 - 2 / 365), calendar_days=3)
    with pytest.raises(ValidationError):                                  # different strike
        check_contract_roll(p0, EuropeanVanillaOption(strike=105.0, option_type=OptionType.CALL,
                                                      maturity=1.0 - 3 / 365), calendar_days=3)
    # a product carrying the tracker's KI flag differs only by a rolled field
    p2 = deepcopy(p1)
    setattr(p2, "_otc_lifecycle_knocked_in", True)
    assert contract_fingerprint(p2) == contract_fingerprint(p1)


def test_value_rejects_non_finite_price():
    class NanEngine:
        def price(self, product, env):
            return float("nan")

    snap = ValuationSnapshot(_call(), NanEngine(), _env(), date=D0)
    with pytest.raises(NumericalError):
        value(snap)


def test_lifecycle_fingerprint_is_computed_and_sensitive():
    assert lifecycle_fingerprint(None) == ("v1", None)
    s = AutocallableLifecycleState()
    f0 = lifecycle_fingerprint(s)
    assert f0[0] == "v1" and f0[1] == "AutocallableLifecycleState"
    assert lifecycle_fingerprint(deepcopy(s)) == f0
    s.mark_ki(D0)
    assert lifecycle_fingerprint(s) != f0
    s2 = AutocallableLifecycleState()
    s2.coupon_memory_count = 2                    # a plain int field also counts
    assert lifecycle_fingerprint(s2) != f0
    b = BarrierLifecycleState()
    b.ledger.register(RealizedCashflow(
        cashflow_id="x", event_type=LifecycleEventType.COUPON, amount=1.0,
        determination_date=D0, payment_date=D0))
    assert lifecycle_fingerprint(b) != lifecycle_fingerprint(BarrierLifecycleState())


def test_lifecycle_fingerprint_ignores_the_trackers_clock_stamp():
    """Trackers re-stamp ``valuation_point`` on every observation; that is not an event."""
    from quantark.asset.equity.lifecycle.state import BarrierLifecycleState
    a = BarrierLifecycleState()
    b = BarrierLifecycleState()
    a.valuation_point = ValuationPoint(date=D0)
    b.valuation_point = ValuationPoint(date=D0 + timedelta(days=1))
    assert lifecycle_fingerprint(a) == lifecycle_fingerprint(b)
    b.knocked_in = True
    assert lifecycle_fingerprint(a) != lifecycle_fingerprint(b)
