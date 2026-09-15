from datetime import datetime, timedelta
from math import exp

import pytest

from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.intraday import VarianceProfile, resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.roll import ShiftedDividendYield, ShiftedRateCurve, ShiftedTradingVolSurface, next_event_after, roll_context
from quantark.intraday.timestamp import calendar_year_fraction as yf
from quantark.param import ContinuousDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.param.vol import TermStructureVolSurface
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, dated_snowball, flat_env

T0 = datetime(2026, 3, 16)


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _req(sse_calendar, sse_sessions, profile, ts):
    return IntradayValuationRequest(product=dated_snowball(sse_calendar, T0), pricing_env=flat_env(ts), session_calendar=sse_sessions,
                                    variance_profile=profile)


def test_shifted_curves_preserve_absolute_schedule():
    rc = LinearRateCurve(pillars=[(0.25, 0.01), (1.0, 0.03), (2.0, 0.04)])
    s = ShiftedRateCurve(rc, 0.25)
    for t in (0.1, 0.5, 1.2):
        assert s.get_discount_factor(t) == pytest.approx(rc.get_discount_factor(t + 0.25) / rc.get_discount_factor(0.25), rel=1e-14)
        assert s.parallel_shifted(1e-4).get_discount_factor(t) == pytest.approx(s.get_discount_factor(t) * exp(-1e-4 * t), rel=1e-12)
    d = ShiftedDividendYield(ContinuousDividendYield(0.02), 0.25)
    assert d.get_yield(0.7) == pytest.approx(0.02, rel=1e-12)
    inner = TermStructureVolSurface(times=[0.1, 0.5, 1.0], vols=[0.25, 0.20, 0.18])
    v = ShiftedTradingVolSurface(inner, 0.2)
    u = 0.3
    w_abs = inner.get_vol(100.0, u + 0.2, 100.0) ** 2 * (u + 0.2) - inner.get_vol(100.0, 0.2, 100.0) ** 2 * 0.2
    assert v.get_vol(100.0, u, 100.0) ** 2 * u == pytest.approx(w_abs, rel=1e-12)
    assert v.quad_v2_deterministic_variance is True


def test_frozen_roll_matches_direct_valuation_and_keeps_assumptions(sse_calendar, sse_sessions, desk):
    ts = datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI)
    ctx = resolve_context(_req(sse_calendar, sse_sessions, desk, ts))
    later = ts + timedelta(hours=4)
    rolled = roll_context(ctx, later)
    assert rolled.reconstruction.assumptions == ctx.reconstruction.assumptions and rolled.provisional
    fresh = resolve_context(_req(sse_calendar, sse_sessions, desk, later))
    engine = SnowballQuadEngineV2()
    frozen = route_for(rolled, engine).price(rolled, engine).contingent_pv
    direct = route_for(fresh, engine).price(fresh, engine).contingent_pv
    assert frozen == pytest.approx(direct, rel=1e-12)          # flat market: frozen roll == fresh environment
    X = datetime(2026, 10, 20, 15, 0, tzinfo=SHANGHAI)
    w_old = ctx.pricing_env.vol_surface.total_variance(100.0, yf(ts, X), 100.0)
    w_mid = ctx.pricing_env.vol_surface.total_variance(100.0, yf(ts, later), 100.0)
    w_new = rolled.pricing_env.vol_surface.total_variance(100.0, yf(later, X), 100.0)
    assert w_old == pytest.approx(w_mid + w_new, rel=1e-12)


def test_frozen_roll_of_a_term_market_keeps_forward_quantities(sse_calendar, sse_sessions, desk):
    from quantark.priceenv import PricingEnvironment
    from quantark.param import SpotQuote
    ts = datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI)
    env = PricingEnvironment(rate_curve=LinearRateCurve(pillars=[(0.25, 0.01), (1.0, 0.03)]), valuation_date=ts,
                             spot_quote=SpotQuote(100.0, timestamp=ts),
                             vol_surface=TermStructureVolSurface(times=[0.1, 0.5, 1.0], vols=[0.25, 0.20, 0.18]),
                             div_yield=ContinuousDividendYield(0.01))
    ctx = resolve_context(IntradayValuationRequest(product=dated_snowball(sse_calendar, T0), pricing_env=env,
                                                   session_calendar=sse_sessions, variance_profile=desk))
    later = ts + timedelta(hours=4)
    rolled = roll_context(ctx, later)
    X = datetime(2026, 12, 20, 15, 0, tzinfo=SHANGHAI)
    df_ratio = ctx.pricing_env.get_discount_factor(yf(ts, X)) / ctx.pricing_env.get_discount_factor(yf(ts, later))
    assert rolled.pricing_env.get_discount_factor(yf(later, X)) == pytest.approx(df_ratio, rel=1e-12)
    w_old = ctx.pricing_env.vol_surface.total_variance(100.0, yf(ts, X), 100.0)
    w_mid = ctx.pricing_env.vol_surface.total_variance(100.0, yf(ts, later), 100.0)
    assert rolled.pricing_env.vol_surface.total_variance(100.0, yf(later, X), 100.0) == pytest.approx(w_old - w_mid, rel=1e-10)


def test_roll_through_an_event_is_refused(sse_calendar, sse_sessions, desk):
    ctx = resolve_context(_req(sse_calendar, sse_sessions, desk, datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI)))
    nxt = next_event_after(ctx, ctx.valuation_timestamp)
    with pytest.raises(ValidationError, match="crosses"):
        roll_context(ctx, nxt.timestamp + timedelta(seconds=1))
    at = roll_context(ctx, nxt.timestamp)                       # landing exactly on the event: BEFORE by default
    assert at.phase.value == "before" and min(at.numerical.event_taus.values()) == 0.0
    with pytest.raises(ValidationError, match="outcome"):
        roll_context(ctx, nxt.timestamp, phase="after")
