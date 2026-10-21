"""Tsiveriotis-Fernandes binomial tree for convertible bonds (spec D3-D7, AC3-AC6)."""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy.stats import norm

from quantark.asset.bond.engine.tree.convertible import (
    ConvertibleBondTFTreeEngine,
    ConvertibleBondTreeParams,
)
from quantark.asset.bond.product.convertible import (
    CallScheduleEntry,
    ConvertibleBond,
    PutScheduleEntry,
)
from quantark.param.quote import SpotQuote
from quantark.param.rrf import FlatRateCurve
from quantark.param.vol import FlatVolSurface
from quantark.param.vol.vol_surface import VolatilitySurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import PaymentFrequency
from quantark.util.exceptions import ValidationError


class SteppedVolSurface(VolatilitySurface):
    """vol1 up to ``switch`` years, vol2 after, total variance continuous."""

    def __init__(self, vol1: float, vol2: float, switch: float):
        self.vol1, self.vol2, self.switch = vol1, vol2, switch

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        t = time_to_maturity
        if t <= self.switch:
            return self.vol1
        return math.sqrt((self.vol1**2 * self.switch + self.vol2**2 * (t - self.switch)) / t)


class ShrinkingVarianceSurface(VolatilitySurface):
    """Total variance rises to t = 1 then falls: calendar arbitrage."""

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        t = time_to_maturity
        w = 0.09 * min(t, 1.0) - 0.03 * max(t - 1.0, 0.0)
        return math.sqrt(max(w, 1e-12) / t) if t > 0 else 0.3


def _env(valuation, spot, vol, rate):
    surface = vol if isinstance(vol, VolatilitySurface) else FlatVolSurface(volatility=vol)
    return PricingEnvironment(
        valuation_date=valuation,
        spot_quote=SpotQuote(spot=spot),
        vol_surface=surface,
        rate_curve=FlatRateCurve(rate=rate),
    )


def _tree(env, n=500):
    return ConvertibleBondTFTreeEngine(env, ConvertibleBondTreeParams(num_steps=n))


# Sheet terms (spec D10): 5% annual A365F, 150bp spread, q = 19.1219% on the bond.
SHEET_ENV = _env(datetime(2025, 10, 27), 111.0, 0.182784, 0.0209)


def _sheet_bond(ratio):
    return ConvertibleBond(
        issue_date=datetime(2024, 12, 2),
        maturity_date=datetime(2026, 12, 2),
        face_value=100.0,
        coupon_rate=0.05,
        conversion_ratio=ratio,
        conversion_price=100.0 / ratio,
        payment_frequency=PaymentFrequency.ANNUAL,
        day_count_convention=DayCountConvention.ACT_365,
        conversion_start_date=datetime(2025, 3, 3),
        conversion_end_date=datetime(2026, 12, 2),
        credit_spread=0.015,
        continuous_dividend_yield=0.191219,
    )


DEMO_VALUATION = datetime(2024, 6, 1)
DEMO_MATURITY = datetime(2029, 1, 1)


def _demo_bond(**kwargs):
    terms = dict(
        issue_date=datetime(2024, 1, 1),
        maturity_date=DEMO_MATURITY,
        face_value=100.0,
        coupon_rate=0.025,
        conversion_ratio=10.0,
        credit_spread=0.02,
        hazard_rate=0.01,
        recovery_rate=0.4,
    )
    terms.update(kwargs)
    return ConvertibleBond(**terms)


# --------------------------------------------------------------------------- AC3


@pytest.mark.parametrize(
    "bond, env",
    [
        pytest.param(_sheet_bond(100.0 / 110.0), SHEET_ENV, id="sheet-0.909"),
        pytest.param(_sheet_bond(1.1), SHEET_ENV, id="sheet-1.1"),
        pytest.param(
            _demo_bond(call_schedule=[CallScheduleEntry(datetime(2026, 1, 1), 105.0)]),
            _env(DEMO_VALUATION, 12.0, 0.30, 0.05),
            id="demo-callable",
        ),
    ],
)
def test_components_add_up_and_probability_is_bounded(bond, env):
    result = _tree(env).price_with_details(bond)
    assert abs(result.dirty_price - (result.equity_component + result.bond_component)) <= 1e-10
    assert 0.0 <= result.conversion_probability <= 1.0
    assert result.price == pytest.approx(
        result.dirty_price - bond.calculate_accrued_interest(env.valuation_date), abs=1e-12
    )


# --------------------------------------------------------------------------- AC4


def test_no_conversion_prices_the_risky_straight_bond():
    # Conversion window closed before the valuation date: only the cash flows remain.
    bond = _demo_bond(conversion_start_date=datetime(2024, 1, 1), conversion_end_date=datetime(2024, 3, 1))
    env = _env(DEMO_VALUATION, 12.0, 0.30, 0.05)
    result = _tree(env, 300).price_with_details(bond)
    risky = 0.05 + bond.credit_spread
    expected = sum(
        cf.amount * math.exp(-risky * (cf.payment_date - DEMO_VALUATION).days / 365.0)
        for cf in bond.get_cashflows(DEMO_VALUATION)
    )
    assert result.dirty_price == pytest.approx(expected, rel=1e-8)
    assert result.equity_component == 0.0
    assert result.conversion_probability == 0.0


# --------------------------------------------------------------------------- AC5


def test_flat_vol_gives_a_uniform_grid():
    engine = _tree(SHEET_ENV, 400)
    T = _sheet_bond(1.1).time_to_maturity(SHEET_ENV.valuation_date)
    times = engine._time_grid(T)
    assert times.shape == (401,)
    assert np.array_equal(times[:-1], np.arange(400) * (T / 400))
    assert times[-1] == T


def test_term_structure_vol_gives_equal_variance_steps():
    env = _env(SHEET_ENV.valuation_date, 111.0, SteppedVolSurface(0.15, 0.35, 0.5), 0.0209)
    engine = _tree(env, 400)
    T = _sheet_bond(1.1).time_to_maturity(env.valuation_date)
    times = engine._time_grid(T)
    w = np.array([env.get_vol(111.0, t) ** 2 * t if t > 0 else 0.0 for t in times])
    assert times[-1] == T
    np.testing.assert_allclose(np.diff(w), w[-1] / 400, rtol=1e-9)
    assert np.diff(times)[0] > np.diff(times)[-1]  # low vol early ⇒ longer early steps


def test_shrinking_total_variance_is_rejected():
    env = _env(SHEET_ENV.valuation_date, 111.0, ShrinkingVarianceSurface(), 0.0209)
    with pytest.raises(ValidationError):
        _tree(env, 100).price(_sheet_bond(1.1))


@pytest.mark.parametrize(
    "vol", [pytest.param(0.25, id="flat"), pytest.param(SteppedVolSurface(0.20, 0.30, 1.0), id="stepped")]
)
def test_european_limit_matches_black(vol):
    # Conversion on the maturity day only, no coupons, no spread:
    # value = face·e^{-rT} + ratio·Call(K = face/ratio) on total variance w(T).
    valuation, maturity = datetime(2025, 1, 1), datetime(2027, 1, 1)
    bond = ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=maturity,
        face_value=100.0,
        coupon_rate=0.0,
        conversion_ratio=1.0,
        conversion_start_date=maturity,
        conversion_end_date=maturity,
        continuous_dividend_yield=0.02,
    )
    r, q, S = 0.03, 0.02, 100.0
    env = _env(valuation, S, vol, r)
    T = bond.time_to_maturity(valuation)
    w = env.get_vol(S, T) ** 2 * T
    d1 = (math.log(S / 100.0) + (r - q) * T + 0.5 * w) / math.sqrt(w)
    d2 = d1 - math.sqrt(w)
    call = S * math.exp(-q * T) * norm.cdf(d1) - 100.0 * math.exp(-r * T) * norm.cdf(d2)
    expected = 100.0 * math.exp(-r * T) + call
    assert _tree(env, 2000).price_with_details(bond).dirty_price == pytest.approx(expected, abs=0.02)


# --------------------------------------------------------------------------- AC6

PUT_ENV = _env(DEMO_VALUATION, 8.0, 0.30, 0.05)  # parity 80: the put at 100 is in the money


@pytest.mark.parametrize("n", [7, 8, 200, 201])
def test_put_between_nodes_is_exercised(n):
    with_put = _demo_bond(put_schedule=[PutScheduleEntry(datetime(2026, 1, 1), 100.0)])
    engine = _tree(PUT_ENV, n)
    assert engine.price(with_put) - engine.price(_demo_bond()) > 1.5


def test_put_in_last_step_is_applied_at_maturity():
    engine = _tree(PUT_ENV, 200)
    with_put = _demo_bond(put_schedule=[PutScheduleEntry(DEMO_MATURITY - timedelta(days=2), 110.0)])
    assert engine.price(with_put) - engine.price(_demo_bond()) > 3.0
