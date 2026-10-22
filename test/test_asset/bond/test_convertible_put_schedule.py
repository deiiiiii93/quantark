"""Put dates on convertible-bond time grids (spec D11, AC10).

Every convertible engine used to apply a put only on a step whose whole-day date equalled the put date, so a
put falling between two steps was silently dropped. ``put_prices_by_step`` maps each put onto the first step at or
after it, and the engines apply puts by step index (the maturity step included).
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.bond.schedule import put_prices_by_step
from quantark.asset.bond.engine.pde.convertible import (
    ConvertibleBondJumpDiffusionEngine,
    ConvertibleBondPDEParams,
    ConvertibleBondTFEngine,
)
from quantark.asset.bond.engine.tree.convertible import (
    ConvertibleBondBinomialEngine,
    ConvertibleBondTreeParams,
    ConvertibleBondTrinomialEngine,
)
from quantark.asset.bond.product.convertible import ConvertibleBond, PutScheduleEntry
from quantark.param.quote import SpotQuote
from quantark.param.rrf import FlatRateCurve
from quantark.param.vol import FlatVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import ConvertibleBondTrinomialVolScheme

VALUATION = datetime(2024, 6, 1)
MATURITY = datetime(2029, 1, 1)
PUT_DATE = datetime(2026, 1, 1)  # a semi-annual coupon date


def _bond(puts=()):
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=MATURITY,
        face_value=100.0,
        coupon_rate=0.025,
        conversion_ratio=10.0,
        credit_spread=0.02,
        hazard_rate=0.01,
        recovery_rate=0.4,
        put_schedule=[PutScheduleEntry(put_date=d, put_price=p) for d, p in puts],
    )


@pytest.fixture(scope="module")
def env():
    # Spot 8 (parity 80): the put at 100 is deep in the money.
    return PricingEnvironment(
        valuation_date=VALUATION,
        spot_quote=SpotQuote(spot=8.0),
        vol_surface=FlatVolSurface(volatility=0.30),
        rate_curve=FlatRateCurve(rate=0.05),
    )


T_YEARS = (MATURITY - VALUATION).days / 365.0
T_PUT = (PUT_DATE - VALUATION).days / 365.0


# --------------------------------------------------------------------------- helper


class TestPutPricesByStep:
    def test_put_between_steps_maps_to_next_step(self):
        steps = np.linspace(0.0, 1.0, 11)
        bond = _bond([(VALUATION + timedelta(days=40), 101.0)])  # t = 40/365 = 0.1096
        assert put_prices_by_step(bond, VALUATION, steps) == {2: 101.0}

    def test_put_on_a_step_maps_to_that_step(self):
        steps = np.arange(0, 366) / 365.0  # one step per day
        bond = _bond([(VALUATION + timedelta(days=40), 101.0)])
        assert put_prices_by_step(bond, VALUATION, steps) == {40: 101.0}

    def test_put_today_maps_to_step_zero(self):
        steps = np.linspace(0.0, 1.0, 11)
        assert put_prices_by_step(_bond([(VALUATION, 99.0)]), VALUATION, steps) == {0: 99.0}

    def test_past_put_is_dropped(self):
        steps = np.linspace(0.0, 1.0, 11)
        bond = _bond([(VALUATION - timedelta(days=1), 99.0)])
        assert put_prices_by_step(bond, VALUATION, steps) == {}

    def test_two_puts_on_one_step_keep_the_larger(self):
        steps = np.linspace(0.0, 1.0, 11)
        bond = _bond([(VALUATION + timedelta(days=38), 101.0), (VALUATION + timedelta(days=40), 103.0)])
        assert put_prices_by_step(bond, VALUATION, steps) == {2: 103.0}

    def test_put_inside_last_step_maps_to_maturity(self):
        steps = np.linspace(0.0, T_YEARS, 201)
        bond = _bond([(MATURITY - timedelta(days=2), 110.0)])
        assert put_prices_by_step(bond, VALUATION, steps) == {200: 110.0}

    def test_valuation_time_of_day_uses_calendar_days(self):
        # Intraday valuation: a put on the maturity date still lands on the last step.
        valuation = VALUATION + timedelta(hours=15)
        T = (MATURITY - valuation).days / 365.0
        steps = np.linspace(0.0, T, 101)
        bond = _bond([(MATURITY, 110.0)])
        assert put_prices_by_step(bond, valuation, steps) == {100: 110.0}


# --------------------------------------------------------------------------- engines


def _gs(env, n):
    return ConvertibleBondBinomialEngine(env, ConvertibleBondTreeParams(num_steps=n))


def _hw(env, scheme):
    return ConvertibleBondTrinomialEngine(
        env, ConvertibleBondTreeParams(num_steps=200, trinomial_vol_scheme=scheme)
    )


def _jd(env, nt):
    return ConvertibleBondJumpDiffusionEngine(env, ConvertibleBondPDEParams(num_time_steps=nt))


def _tf(env, nt):
    return ConvertibleBondTFEngine(env, ConvertibleBondPDEParams(num_time_steps=nt))


ENGINE_GRIDS = [
    pytest.param(lambda e: _gs(e, 200), lambda e: _gs(e, 201), id="gs-200"),
    pytest.param(lambda e: _gs(e, 365), lambda e: _gs(e, 366), id="gs-365"),
    pytest.param(lambda e: _jd(e, 500), lambda e: _jd(e, 501), id="jd-500"),
    pytest.param(lambda e: _tf(e, 500), lambda e: _tf(e, 501), id="tf-pde-500"),
] + [
    pytest.param(lambda e, s=s: _hw(e, s), None, id=f"hw-{s.value}")
    for s in ConvertibleBondTrinomialVolScheme
]


@pytest.mark.parametrize("make, make_next", ENGINE_GRIDS)
def test_put_between_steps_is_exercised(env, make, make_next):
    with_put, no_put = _bond([(PUT_DATE, 100.0)]), _bond()
    engine = make(env)
    put_value = engine.price(with_put) - engine.price(no_put)
    # A dropped put is worth exactly 0; applied it is worth 1.4 (hazard models) to 2.4 (GS).
    assert put_value > 1.0
    if make_next is not None:
        # One more step moves every node: the put (and the coupon paid on the put date) must not flicker.
        engine_next = make_next(env)
        put_value_next = engine_next.price(with_put) - engine_next.price(no_put)
        assert abs(put_value_next - put_value) < 0.15


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda e: _gs(e, 200), id="gs"),
        pytest.param(lambda e: _jd(e, 500), id="jd"),
        pytest.param(lambda e: _tf(e, 500), id="tf-pde"),
    ]
    + [pytest.param(lambda e, s=s: _hw(e, s), id=f"hw-{s.value}") for s in ConvertibleBondTrinomialVolScheme],
)
def test_put_in_last_step_is_applied_at_maturity(env, make):
    engine = make(env)
    with_put = _bond([(MATURITY - timedelta(days=2), 110.0)])
    # Put 110 against redemption at 100 two days before maturity: ~10 x risky discount x P(parity < 110) = 4.3-4.9.
    assert engine.price(with_put) - engine.price(_bond()) > 3.0


# --------------------------------------------------------------------------- one clock for puts and coupons
# A coupon enters the step (t_i, t_{i+1}] it falls in; a put lands on the first step t_k >= t_put. On grids
# whose nodes sit an ulp from a coupon date, a tolerance in either rule split the two (CP1 review): the coupon
# paid on the put date was lost, or a coupon fell between two steps and was never paid.

ULP_VALUATION = datetime(2024, 10, 22)
ULP_MATURITY = datetime(2027, 1, 1)
ULP_PUT = datetime(2025, 7, 1)  # a coupon date; at N = 267 a step sits an ulp below it


def _ulp_bond(put=False):
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=ULP_MATURITY,
        face_value=100.0,
        coupon_rate=0.05,
        conversion_ratio=10.0,
        credit_spread=0.02,
        put_schedule=[PutScheduleEntry(put_date=ULP_PUT, put_price=100.0)] if put else [],
    )


@pytest.fixture(scope="module")
def ulp_env():
    return PricingEnvironment(
        valuation_date=ULP_VALUATION,
        spot_quote=SpotQuote(spot=8.0),
        vol_surface=FlatVolSurface(volatility=0.30),
        rate_curve=FlatRateCurve(rate=0.05),
    )


def test_put_on_a_coupon_date_keeps_the_coupon_on_every_grid(ulp_env):
    from quantark.asset.bond.engine.tree.convertible import ConvertibleBondTFTreeEngine

    values = []
    for n in (266, 267, 268):
        for engine in (_gs(ulp_env, n), ConvertibleBondTFTreeEngine(ulp_env, ConvertibleBondTreeParams(num_steps=n))):
            values.append(engine.price(_ulp_bond(put=True)) - engine.price(_ulp_bond()))
    assert max(values[0::2]) - min(values[0::2]) < 0.05  # GS
    assert max(values[1::2]) - min(values[1::2]) < 0.05  # TF tree


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda e, n: _gs(e, n), id="gs"),
        pytest.param(
            lambda e, n: ConvertibleBondTrinomialEngine(e, ConvertibleBondTreeParams(num_steps=n)), id="hw"
        ),
        pytest.param(lambda e, n: _jd(e, n), id="jd"),
        pytest.param(lambda e, n: _tf(e, n), id="tf-pde"),
    ],
)
def test_no_coupon_is_dropped_between_steps(ulp_env, make):
    # 265 and 267 steps used to drop a coupon (GS 105.29 against ~107.4 on neighbouring grids)
    prices = [make(ulp_env, n).price(_ulp_bond()) for n in (263, 264, 265, 266, 267)]
    assert max(prices) - min(prices) < 0.3


def test_hw_put_value_is_continuous_in_n(env):
    with_put, no_put = _bond([(PUT_DATE, 100.0)]), _bond()
    values = []
    for n in (200, 201):
        engine = ConvertibleBondTrinomialEngine(env, ConvertibleBondTreeParams(num_steps=n))
        values.append(engine.price(with_put) - engine.price(no_put))
    assert abs(values[1] - values[0]) < 0.15


# The last step ends at T itself: N * (T / N) can sit an ulp below T, and a coupon paid at maturity then fell
# into no window (final review: GS 100.72 -> 98.28 on a 168-day bond at the default 200 steps).


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda e, n: _gs(e, n), id="gs"),
        pytest.param(
            lambda e, n: ConvertibleBondTrinomialEngine(e, ConvertibleBondTreeParams(num_steps=n)), id="hw"
        ),
        pytest.param(lambda e, n: _jd(e, n), id="jd"),
        pytest.param(lambda e, n: _tf(e, n), id="tf-pde"),
    ],
)
@pytest.mark.parametrize("days", [168, 571])
def test_maturity_coupon_is_paid_on_every_grid(make, days):
    env = PricingEnvironment(
        valuation_date=ULP_MATURITY - timedelta(days=days),
        spot_quote=SpotQuote(spot=8.0),
        vol_surface=FlatVolSurface(volatility=0.30),
        rate_curve=FlatRateCurve(rate=0.05),
    )
    prices = [make(env, n).price(_ulp_bond()) for n in (199, 200, 201)]
    assert max(prices) - min(prices) < 0.3
