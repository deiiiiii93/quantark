from datetime import datetime, time, timedelta
from math import exp, log, sqrt

import pytest
from scipy.stats import norm

from quantark.asset.equity.engine.analytical import DigitalOptionAnalyticalEngine
from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.events import EventPhase
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.timestamp import SECONDS_PER_YEAR, calendar_year_fraction
from quantark.util.enum.option_enums import OptionType
from intraday.conftest import SHANGHAI, digital, flat_env

EXPIRY = datetime(2026, 9, 15)
CLOSE = datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI)
ENGINE = DigitalOptionAnalyticalEngine()


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _ctx(sse_sessions, profile, ts, spot=100.0, phase=EventPhase.BEFORE, vol=0.20, r=0.03, q=0.01, **kw):
    req = IntradayValuationRequest(product=kw.pop("product", digital(EXPIRY)), pricing_env=flat_env(ts, spot=spot, vol=vol, r=r, q=q),
                                   session_calendar=sse_sessions, variance_profile=profile, event_phase=phase, **kw)
    return resolve_context(req)


def _price(ctx):
    return route_for(ctx, ENGINE).price(ctx, ENGINE)


def test_one_second_before_close_matches_closed_form_with_profile_variance(sse_sessions, desk):
    ctx = _ctx(sse_sessions, desk, CLOSE - timedelta(seconds=1), spot=100.05)
    out = _price(ctx)
    T = 1.0 / SECONDS_PER_YEAR
    W = 0.04 * (0.35 / 244) / 7200.0                    # one second of the afternoon session
    carry = (0.03 - 0.01) * T
    d2 = (log(100.05 / 100.0) + carry - 0.5 * W) / sqrt(W)
    assert out.contingent_pv == pytest.approx(exp(-0.03 * T) * norm.cdf(d2), rel=1e-10)
    assert out.method == "analytical_bs_effective_variance" and out.numerical["total_variance"] == pytest.approx(W, rel=1e-12)


def test_lunch_break_is_deterministic_forward_payoff(sse_sessions):
    p = VarianceProfile.sessions_only(sse_sessions, 244)
    ts = datetime(2026, 9, 15, 12, 0, tzinfo=SHANGHAI)
    # expiry at the 15:00 close: the afternoon session carries variance, so this is NOT zero-variance -> closed form
    assert _price(_ctx(sse_sessions, p, ts, spot=99.999)).method == "analytical_bs_effective_variance"
    # expiring at 13:00 (contract fixing time) valued at 12:00 under sessions_only: zero variance
    out0 = _price(_ctx(sse_sessions, p, ts, spot=99.999, fixing_time_of_day=time(13, 0)))
    T = calendar_year_fraction(ts, datetime(2026, 9, 15, 13, 0, tzinfo=SHANGHAI))
    T_pay = calendar_year_fraction(ts, CLOSE)            # a date-only payment is deemed made at the close
    F = 99.999 * exp((0.03 - 0.01) * T)
    assert out0.method == "deterministic_zero_variance"
    assert out0.contingent_pv == pytest.approx((1.0 if F > 100.0 else 0.0) * exp(-0.03 * T_pay), rel=1e-12)
    out_put = _price(_ctx(sse_sessions, p, ts, spot=99.999, fixing_time_of_day=time(13, 0),
                          product=digital(EXPIRY, option_type=OptionType.PUT)))
    assert out_put.contingent_pv == pytest.approx((1.0 if F < 100.0 else 0.0) * exp(-0.03 * T_pay), rel=1e-12)
    # a fixing after the close is paid at its own determination, never before it
    late = _ctx(sse_sessions, p, ts, spot=99.999, fixing_time_of_day=time(15, 30))
    assert late.timeline.terminal().payment_timestamp == late.timeline.terminal().timestamp


def test_before_at_expiry_is_intrinsic_with_the_contracts_strict_inequality(sse_sessions, desk):
    out = _price(_ctx(sse_sessions, desk, CLOSE, spot=101.0, phase=EventPhase.BEFORE))
    assert out.contingent_pv == 1.0 and out.method == "deterministic_zero_variance"
    out_eq = _price(_ctx(sse_sessions, desk, CLOSE, spot=100.0, phase=EventPhase.BEFORE))
    assert out_eq.contingent_pv == 0.0 and any("equal" in r for r in out_eq.records)


def test_engine_path_uses_exactly_the_profile_variance_through_the_wrapped_surface(sse_sessions, desk):
    ts = datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI)
    ctx = _ctx(sse_sessions, desk, ts, spot=100.3)
    out = _price(ctx)
    T = ctx.numerical.maturity_tau
    W = float(ctx.pricing_env.vol_surface.total_variance(100.0, T, 100.3))
    # 10:00 -> 11:30 (1.5h of the 2h morning), lunch, afternoon: 0.35*0.75 + 0.05 + 0.35 of 1/244
    assert W == pytest.approx(0.04 * (0.35 * 0.75 + 0.05 + 0.35) / 244, rel=1e-12)
    d2 = (log(100.3 / 100.0) + 0.02 * T - 0.5 * W) / sqrt(W)
    assert out.contingent_pv == pytest.approx(exp(-0.03 * T) * norm.cdf(d2), rel=1e-10)
