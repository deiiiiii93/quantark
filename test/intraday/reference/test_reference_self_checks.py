"""Self-checks of the independent Gaussian reference (closed forms, quadrature, exact-bridge MC)."""
from datetime import datetime, timedelta
from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from intraday.conftest import SHANGHAI, dated_snowball, digital, flat_env
from intraday.reference.gaussian_reference import (
    PLJ, barrier_zero_carry, expect, reference_digital, reference_snowball, splice,
)

T0 = datetime(2026, 3, 16)


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _snow_ctx(sse_calendar, sse_sessions, profile, ts, **kw):
    return resolve_context(IntradayValuationRequest(product=dated_snowball(sse_calendar, T0), pricing_env=flat_env(ts, spot=kw.pop("spot", 100.0)),
                                                    session_calendar=sse_sessions, variance_profile=profile, **kw))


def test_expectation_and_its_derivatives_match_quadrature():
    x = np.array([-0.4, -0.1, 0.0, 0.05, 0.3])
    fn = PLJ(x, np.array([2.0, 1.0, 3.0, 2.5, 0.5]), np.array([0.0, 0.0, 1.5, 0.0, 0.0]))   # jump of +1.5 at 0.0
    mu, v = 0.02, 0.01
    s = sqrt(v)

    def integrand(y, k):
        z = (y - mu) / s
        weight = (1.0, z / s, (z * z - 1.0) / v)[k]
        return float(fn(y)[0]) * weight * norm.pdf(z) / s

    breaks = list(x)
    for k in range(3):
        expected = sum(quad(integrand, a, b, args=(k,), limit=200)[0] for a, b in zip([-3.0] + breaks, breaks + [3.0]))
        got = expect(fn, np.array([mu]), v, 1.0, derivatives=True)[k][0]
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-10), k


def test_splice_is_an_exact_jump():
    x = np.linspace(-1.0, 1.0, 5)
    below, above = PLJ.sample(x, np.zeros(5)), PLJ.sample(x, np.full(5, 7.0))
    s = splice(below, above, 2)
    assert s(np.array([-1e-12, 0.0, 0.5])).tolist() == [0.0, 7.0, 7.0]
    assert expect(s, np.array([0.0]), 0.04, 1.0)[0][0] == pytest.approx(3.5, abs=1e-15)


def test_reference_reproduces_a_single_period_snowball_in_closed_form(sse_calendar, sse_sessions, desk):
    kos = [e for e in _snow_ctx(sse_calendar, sse_sessions, desk, datetime(2026, 9, 15, tzinfo=SHANGHAI)).timeline.events
           if e.kind is EventKind.KO]
    ts = kos[10].timestamp + timedelta(hours=1)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:11])
    ctx = _snow_ctx(sse_calendar, sse_sessions, desk, ts, fixings=fixings, event_phase="after")
    ref = reference_snowball(ctx)
    T = ctx.numerical.maturity_tau
    W = float(ctx.pricing_env.vol_surface.total_variance(100.0, T, 100.0))
    R = -log(ctx.pricing_env.get_discount_factor(T))
    Q = ctx.pricing_env.get_div_yield(T) * T
    B = kos[11].barrier
    d = (log(100.0 / B) + R - Q - 0.5 * W) / sqrt(W)
    prod = ctx.numerical.product
    y = np.linspace(log(20.0), log(B), 400001)
    pdf = norm.pdf((y - log(100.0) - (R - Q - 0.5 * W)) / sqrt(W)) / sqrt(W)
    payoff = np.array([prod.get_maturity_payoff_v1(exp(v), ctx.pricing_env) if exp(v) <= 75.0
                       else prod.get_maturity_payoff_v0(exp(v), ctx.pricing_env) for v in y])
    expected = exp(-R) * (kos[11].cash * norm.cdf(d) + float(np.trapezoid(payoff * pdf, y)))
    assert ref.price == pytest.approx(expected, abs=5e-7) and ref.uncertainty_price < 2e-7


def test_reference_digital_is_the_closed_form(sse_sessions, desk):
    ts = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    ctx = resolve_context(IntradayValuationRequest(product=digital(datetime(2026, 9, 15)), pricing_env=flat_env(ts, spot=100.05),
                                                   session_calendar=sse_sessions, variance_profile=desk))
    ref = reference_digital(ctx)
    T = ctx.numerical.maturity_tau
    W = float(ctx.pricing_env.vol_surface.total_variance(100.0, T, 100.05))
    d2 = (log(100.05 / 100.0) + 0.02 * T - 0.5 * W) / sqrt(W)
    # a date-only payment is deemed made at the 15:00 close, which is also the expiry here
    assert ref.price == pytest.approx(exp(-0.03 * T) * norm.cdf(d2), rel=1e-12)
    assert ref.delta == pytest.approx(exp(-0.03 * T) * norm.pdf(d2) / (100.05 * sqrt(W)), rel=1e-9)


def test_reference_converges_and_reports_uncertainty(sse_calendar, sse_sessions, desk):
    ctx = _snow_ctx(sse_calendar, sse_sessions, desk, datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI))
    a = reference_snowball(ctx, points=(1001, 2001, 4001))
    b = reference_snowball(ctx, points=(2001, 4001, 8001))
    tol = 3.0 * max(a.uncertainty_price, b.uncertainty_price, 1e-12)
    assert abs(a.price - b.price) <= tol
    assert b.uncertainty_price <= a.uncertainty_price


def test_barrier_zero_carry_in_out_parity_and_exact_bridge_mc():
    S, K, u, sigma = 100.0, 100.0, 0.5, 0.25
    for is_call in (True, False):
        for is_up, H in ((True, 115.0), (False, 85.0)):
            vanilla = barrier_zero_carry(S, K, H, u, sigma, is_call=is_call, is_up=is_up, is_knock_out=True) + \
                barrier_zero_carry(S, K, H, u, sigma, is_call=is_call, is_up=is_up, is_knock_out=False)
            sw = sigma * sqrt(u)
            d1 = (log(S / K) + 0.5 * sw * sw) / sw
            bs = S * norm.cdf(d1) - K * norm.cdf(d1 - sw) if is_call else K * norm.cdf(sw - d1) - S * norm.cdf(-d1)
            assert vanilla == pytest.approx(bs, rel=1e-12)
    # exact-in-distribution MC: Gaussian log steps + zero-carry bridge crossing probabilities
    rng = np.random.default_rng(20260915)
    n_paths, n_steps = 200_000, 64
    dt = u / n_steps
    for is_call, is_up, H, rebate in ((True, True, 115.0, 2.0), (False, False, 85.0, 1.0), (True, False, 90.0, 0.0)):
        z = rng.standard_normal((n_paths, n_steps))
        logs = np.log(S) + np.cumsum(-0.5 * sigma * sigma * dt + sigma * sqrt(dt) * z, axis=1)
        logs = np.concatenate([np.full((n_paths, 1), np.log(S)), logs], axis=1)
        a, b = logs[:, :-1] - np.log(H), logs[:, 1:] - np.log(H)
        crossed = (a * b <= 0.0)
        p_bridge = np.where(crossed, 1.0, np.exp(-2.0 * a * b / (sigma * sigma * dt)))
        survive = np.prod(1.0 - p_bridge, axis=1)
        ST = np.exp(logs[:, -1])
        payoff = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
        mc = payoff * survive + rebate * (1.0 - survive)
        closed = barrier_zero_carry(S, K, H, u, sigma, is_call=is_call, is_up=is_up, is_knock_out=True, rebate=rebate)
        assert abs(mc.mean() - closed) <= 4.0 * mc.std(ddof=1) / sqrt(n_paths), (is_call, is_up, H, mc.mean(), closed)
