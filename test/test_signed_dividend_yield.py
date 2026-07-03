"""Signed dividend/carry yields (negative implied carry from futures)."""
import math

import pytest

from quantark.param.div.dividend_yield import (
    ContinuousDividendYield,
    TermStructureDividendYield,
)
from quantark.util.exceptions import ValidationError


def test_continuous_accepts_negative_within_bound():
    assert ContinuousDividendYield(-0.05).get_yield(1.0) == -0.05


def test_continuous_rejects_beyond_symmetric_bound():
    with pytest.raises(ValidationError):
        ContinuousDividendYield(-0.25)
    with pytest.raises(ValidationError):
        ContinuousDividendYield(0.25)


def test_term_structure_accepts_negative_nodes():
    ts = TermStructureDividendYield(times=[0.1, 0.5], yields=[-0.02, 0.03])
    assert ts.get_yield(0.1) == pytest.approx(-0.02)
    assert ts.get_yield(0.05) == pytest.approx(-0.02)  # flat extrapolation


def test_term_structure_rejects_magnitude_over_one():
    with pytest.raises(ValidationError):
        TermStructureDividendYield(times=[0.1, 0.5], yields=[-1.5, 0.03])


def test_term_structure_rejects_non_finite():
    with pytest.raises(ValidationError):
        TermStructureDividendYield(times=[0.1, 0.5], yields=[math.nan, 0.03])


# --- Task 5: clamps removed from bump wrappers / greeks calculator ---

import numpy as np
from datetime import datetime
from scipy import stats

from quantark.asset.equity.report.term_structure import (
    BucketedDividendYield,
    ShiftedDividendYield,
)
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


def _env_q0():
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(0.20),
        div_yield=ContinuousDividendYield(0.0),
    )


def test_shifted_yield_goes_negative_without_clamp():
    shifted = ShiftedDividendYield(base=ContinuousDividendYield(0.0), shift=-0.01)
    assert shifted.get_yield(1.0) == pytest.approx(-0.01)


def test_bucketed_yield_goes_negative_without_clamp():
    bucketed = BucketedDividendYield(
        base=ContinuousDividendYield(0.0),
        bucket_start=0.0, bucket_end=1.0, bump=-0.01,
    )
    assert bucketed.get_yield(0.5) == pytest.approx(-0.01)


def test_wrappers_reject_non_finite_or_oversized_bumps():
    with pytest.raises(ValidationError):
        ShiftedDividendYield(base=ContinuousDividendYield(0.0), shift=math.nan)
    with pytest.raises(ValidationError):
        ShiftedDividendYield(base=ContinuousDividendYield(0.0), shift=-1.5)
    with pytest.raises(ValidationError):
        BucketedDividendYield(
            base=ContinuousDividendYield(0.0),
            bucket_start=0.0, bucket_end=1.0, bump=2.0,
        )


def test_central_dividend_rho_at_q_zero_matches_analytical():
    """At q=0 the old clamp broke the down bump; central FD must now match BS."""
    env = _env_q0()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    engine = BlackScholesEngine()
    base = engine.price(option, env)

    bump = 0.0001
    pv = {}
    for sign in (+1.0, -1.0):
        e = _env_q0()
        e.div_yield = ShiftedDividendYield(
            base=ContinuousDividendYield(0.0), shift=sign * bump
        )
        pv[sign] = engine.price(option, e)
    fd_rhoq_per_1pct = (pv[1.0] - pv[-1.0]) / (2 * bump) * 0.01

    S, K, T, r, q, vol = 100.0, 100.0, 1.0, 0.03, 0.0, 0.20
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    analytical = -S * T * np.exp(-q * T) * stats.norm.cdf(d1) / 100.0
    assert fd_rhoq_per_1pct == pytest.approx(analytical, rel=1e-4)
    assert pv[-1.0] != pytest.approx(base, abs=1e-12)  # down bump really applied


def test_calculate_numerical_delta_q_at_q_zero_is_finite_and_central():
    env = _env_q0()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    engine = BlackScholesEngine()
    out = GreeksCalculator().calculate_numerical_delta_q(option, env, engine)
    # dDelta/dq for a call: exp(-qT) * (-T*N(d1) - n(d1)*sqrt(T)/vol), q=0
    S, K, T, r, vol = 100.0, 100.0, 1.0, 0.03, 0.20
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    expected = -T * stats.norm.cdf(d1) - stats.norm.pdf(d1) * np.sqrt(T) / vol
    assert out == pytest.approx(expected, rel=1e-3)
