"""Bootstrappers, carry builder and reprice_quotes (spec 2026-11-05, T6).

Covers AC6, AC7, AC8, AC9, the builder parts of AC12, AC14 and the curve-anchor
half of AC11.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.param import (
    BondQuote,
    CarryCurve,
    CurveInterpolation,
    DepositQuote,
    FlatRateCurve,
    FuturesQuote,
    OptionPairQuote,
    SwapQuote,
    ZeroCurve,
)
from quantark.param.curve.bootstrap import (
    PILLAR_MAXITER,
    PILLAR_RTOL,
    PILLAR_XTOL,
    SWEEP_TOLERANCE,
    BootstrapConfig,
    bootstrap_discount_curve,
    bootstrap_projection_curve,
    build_carry_curve,
    reprice_quotes,
)
from quantark.param.extrapolation import CarryExtrapolation, RateExtrapolation
from quantark.param.node_roles import NodeRole
from quantark.param.index import REPO_7D, SHIBOR_3M
from quantark.util.enum import PaymentFrequency
from quantark.util.exceptions import NumericalError, ValidationError

SCHEMES = list(CurveInterpolation)
DEPOSITS = [
    DepositQuote(rate=0.0180, tenor_years=1 / 12, label="1M"),
    DepositQuote(rate=0.0190, tenor_years=0.25, label="3M"),
    DepositQuote(rate=0.0205, tenor_years=0.5, label="6M"),
]
SWAPS = [
    SwapQuote(fixed_rate=0.0220, index=SHIBOR_3M, tenor_years=1.0, label="1Y"),
    SwapQuote(fixed_rate=0.0245, index=SHIBOR_3M, tenor_years=2.0, label="2Y"),
    SwapQuote(fixed_rate=0.0262, index=SHIBOR_3M, tenor_years=3.0, label="3Y"),
    SwapQuote(fixed_rate=0.0285, index=SHIBOR_3M, tenor_years=5.0, label="5Y"),
    SwapQuote(fixed_rate=0.0298, index=SHIBOR_3M, tenor_years=7.0, label="7Y"),
    SwapQuote(fixed_rate=0.0306, index=SHIBOR_3M, tenor_years=10.0, label="10Y"),
]
AC7_QUOTES = DEPOSITS + SWAPS
VD = datetime(2026, 11, 5)


def test_named_constants():
    assert SWEEP_TOLERANCE == 1e-14
    assert PILLAR_XTOL == 1e-16
    assert PILLAR_RTOL == 4 * np.finfo(float).eps
    assert PILLAR_MAXITER == 200
    cfg = BootstrapConfig()
    assert cfg.interpolation is CurveInterpolation.LOG_LINEAR_DF
    assert cfg.extrapolation is RateExtrapolation.FLAT_FORWARD_RATE
    assert cfg.max_sweeps == 50
    with pytest.raises(ValidationError):
        BootstrapConfig(max_sweeps=0)
    with pytest.raises(ValidationError):
        BootstrapConfig(interpolation="linear_zero")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- AC6


@pytest.mark.parametrize("scheme", SCHEMES)
def test_deposits_only_pin_discount_factors(scheme):
    curve = bootstrap_discount_curve(DEPOSITS, BootstrapConfig(interpolation=scheme))
    assert isinstance(curve, ZeroCurve) and curve.interpolation is scheme
    for q in DEPOSITS:
        assert curve.get_discount_factor(q.tenor_years) == pytest.approx(
            1.0 / (1.0 + q.rate * q.tenor_years), rel=1e-15
        )
    assert curve.tenors == [q.tenor_years for q in DEPOSITS]


# --------------------------------------------------------------------------- AC7


@pytest.mark.parametrize("scheme", SCHEMES)
def test_single_curve_bootstrap_reprices_every_quote(scheme):
    curve = bootstrap_discount_curve(AC7_QUOTES, BootstrapConfig(interpolation=scheme))
    residuals = reprice_quotes(AC7_QUOTES, curve)
    assert len(residuals) == len(AC7_QUOTES)
    assert max(abs(r) for r in residuals) <= 1e-12
    assert curve.tenors == [q.tenor_years for q in AC7_QUOTES]
    assert curve.node_roles == (NodeRole.CALIBRATED,) * len(AC7_QUOTES)
    assert curve.last_observable_tenor == 10.0
    assert curve.valuation_date is None
    assert curve.extrapolation is RateExtrapolation.FLAT_FORWARD_RATE
    # unsorted input gives the same curve and residuals in the caller's order
    shuffled = [AC7_QUOTES[4], AC7_QUOTES[0], AC7_QUOTES[8], AC7_QUOTES[2], AC7_QUOTES[6],
                AC7_QUOTES[1], AC7_QUOTES[7], AC7_QUOTES[3], AC7_QUOTES[5]]
    curve2 = bootstrap_discount_curve(shuffled, BootstrapConfig(interpolation=scheme))
    assert curve2.tenors == curve.tenors
    assert curve2.rates == pytest.approx(curve.rates, abs=1e-15)
    assert max(abs(r) for r in reprice_quotes(shuffled, curve2)) <= 1e-12


def test_bootstrap_is_deterministic_and_scheme_dependent():
    a = bootstrap_discount_curve(AC7_QUOTES)
    b = bootstrap_discount_curve(AC7_QUOTES)
    assert a.pillars == b.pillars
    cubic = bootstrap_discount_curve(AC7_QUOTES, BootstrapConfig(interpolation=CurveInterpolation.CUBIC_ZERO))
    # pillars pinned by deposits agree; swap pillars differ because the interpolant
    # between them differs
    assert cubic.rates[:3] == pytest.approx(a.rates[:3], abs=1e-15)
    assert any(abs(x - y) > 1e-7 for x, y in zip(cubic.rates[3:], a.rates[3:]))


# --------------------------------------------------------------------------- AC8


def test_dual_curve_projection_bootstrap():
    discount = FlatRateCurve(0.02)
    projection = bootstrap_projection_curve(AC7_QUOTES, discount)
    residuals = reprice_quotes(AC7_QUOTES, discount, projection)
    assert max(abs(r) for r in residuals) <= 1e-12
    assert any(abs(projection.get_rate(t) - discount.get_rate(t)) > 1e-6 for t in projection.tenors)
    # against the single-curve result the projection curve reproduces it
    single = bootstrap_discount_curve(AC7_QUOTES)
    again = bootstrap_projection_curve(AC7_QUOTES, single)
    assert again.tenors == single.tenors
    assert again.rates == pytest.approx(single.rates, abs=1e-12)
    # a projection curve is per index
    mixed = AC7_QUOTES + [SwapQuote(fixed_rate=0.03, index=REPO_7D, tenor_years=12.0)]
    with pytest.raises(ValidationError):
        bootstrap_projection_curve(mixed, discount)
    with pytest.raises(ValidationError):
        bootstrap_projection_curve(
            [BondQuote(coupon_rate=0.03, clean_price=100.0, payment_frequency=PaymentFrequency.ANNUAL, tenor_years=2.0)],
            discount,
        )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_dual_curve_every_scheme(scheme):
    discount = bootstrap_discount_curve(DEPOSITS + SWAPS[:3])
    projection = bootstrap_projection_curve(AC7_QUOTES, discount, BootstrapConfig(interpolation=scheme))
    assert max(abs(r) for r in reprice_quotes(AC7_QUOTES, discount, projection)) <= 1e-12


# --------------------------------------------------------------------------- AC9


BONDS = [
    BondQuote(coupon_rate=0.025, clean_price=100.4, payment_frequency=PaymentFrequency.SEMI_ANNUAL, tenor_years=2.0, label="2Y"),
    BondQuote(coupon_rate=0.030, clean_price=100.9, payment_frequency=PaymentFrequency.SEMI_ANNUAL, tenor_years=5.0, label="5Y"),
    BondQuote(coupon_rate=0.032, clean_price=99.6, payment_frequency=PaymentFrequency.SEMI_ANNUAL, tenor_years=10.0, label="10Y"),
]
AC9_SWAPS = [SWAPS[0], SWAPS[2], SWAPS[4]]  # 1Y, 3Y, 7Y


@pytest.mark.parametrize("scheme", SCHEMES)
def test_mixed_deposit_swap_bond_bootstrap(scheme):
    quotes = DEPOSITS + AC9_SWAPS + BONDS
    curve = bootstrap_discount_curve(quotes, BootstrapConfig(interpolation=scheme))
    residuals = reprice_quotes(quotes, curve)
    for q, r in zip(quotes, residuals):
        tol = 1e-10 if isinstance(q, BondQuote) else 1e-12
        assert abs(r) <= tol, (q.label, r)
    assert curve.tenors == sorted(q.pillar_time(None) for q in quotes)


def test_duplicate_pillar_raises_naming_both():
    dup = DEPOSITS + AC9_SWAPS + BONDS + [SwapQuote(fixed_rate=0.03, index=SHIBOR_3M, tenor_years=5.0, label="5Y-swap")]
    with pytest.raises(ValidationError) as excinfo:
        bootstrap_discount_curve(dup)
    assert "5Y" in str(excinfo.value) and "5Y-swap" in str(excinfo.value)
    with pytest.raises(ValidationError):
        SwapQuote(fixed_rate=0.03, index=SHIBOR_3M, tenor_years=1.1)


# --------------------------------------------------------------------------- AC12 (builder)


def test_carry_curve_from_futures_matches_implied_yields():
    rc = bootstrap_discount_curve(AC7_QUOTES)
    fut_curve = IndexFuturesCurve(
        underlying="CSI1000", spot=6000.0,
        quotes=[
            IndexFuturesQuote("IM2612", 0.12, 5940.0, 200.0),
            IndexFuturesQuote("IM2703", 0.37, 5880.0, 200.0),
            IndexFuturesQuote("IM2706", 0.62, 5820.0, 200.0),
            IndexFuturesQuote("IM2709", 0.87, 5770.0, 200.0),
        ],
    )
    quotes = FuturesQuote.from_index_futures_curve(fut_curve)
    carry = build_carry_curve(6000.0, quotes, rc)
    assert isinstance(carry, CarryCurve)
    assert carry.tenors == [0.12, 0.37, 0.62, 0.87]
    assert carry.node_roles == (NodeRole.CALIBRATED,) * 4 and carry.last_observable_tenor == 0.87
    view = carry.to_dividend_yield(rc)
    for T, q in zip(carry.tenors, fut_curve.implied_yields(rc)):
        assert view.get_yield(T) == pytest.approx(q, abs=1e-14)
    for T, F in zip(carry.tenors, [5940.0, 5880.0, 5820.0, 5770.0]):
        assert carry.forward(6000.0, T) == pytest.approx(F, rel=1e-15)
    assert carry.interpolation is CurveInterpolation.LOG_LINEAR_DF
    assert carry.extrapolation is CarryExtrapolation.FLAT_FORWARD_CARRY


def test_carry_curve_from_option_pairs_and_futures():
    rc = FlatRateCurve(0.02)
    spot = 6000.0
    pairs = [
        OptionPairQuote(strike=6000.0, call_price=310.0, put_price=360.0, tenor_years=1.0, label="1Y"),
        OptionPairQuote(strike=5800.0, call_price=520.0, put_price=410.0, tenor_years=2.0, label="2Y"),
    ]
    futures = [FuturesQuote(price=5950.0, tenor_years=0.25), FuturesQuote(price=5905.0, tenor_years=0.5)]
    carry = build_carry_curve(spot, futures + pairs, rc, interpolation=CurveInterpolation.CUBIC_ZERO,
                              extrapolation=CarryExtrapolation.ZERO_FORWARD_CARRY)
    assert carry.tenors == [0.25, 0.5, 1.0, 2.0]
    for p in pairs:
        T = p.tenor_years
        df = rc.get_discount_factor(T)
        F = spot * np.exp(carry.carry(T))
        assert p.call_price - p.put_price == pytest.approx(df * (F - p.strike), abs=1e-14 * spot)
    assert carry.interpolation is CurveInterpolation.CUBIC_ZERO
    assert carry.extrapolation is CarryExtrapolation.ZERO_FORWARD_CARRY
    with pytest.raises(ValidationError):
        build_carry_curve(spot, pairs + [FuturesQuote(price=5900.0, tenor_years=1.0)], rc)  # duplicate expiry
    with pytest.raises(ValidationError):
        build_carry_curve(spot, [OptionPairQuote(strike=100.0, call_price=0.0, put_price=500.0, tenor_years=1.0)], rc)
    with pytest.raises(ValidationError):
        build_carry_curve(0.0, futures, rc)
    with pytest.raises(ValidationError):
        build_carry_curve(spot, DEPOSITS, rc)  # not carry quotes
    with pytest.raises(ValidationError):
        build_carry_curve(spot, futures, rc, interpolation=CurveInterpolation.MONOTONE_CONVEX)


# --------------------------------------------------------------------------- AC14


def test_failure_modes():
    with pytest.raises(NumericalError):
        bootstrap_discount_curve(AC7_QUOTES, BootstrapConfig(max_sweeps=1))
    with pytest.raises(ValidationError):
        bootstrap_discount_curve([])
    with pytest.raises(ValidationError):
        DepositQuote(rate=-3.0, tenor_years=1.0)


# --------------------------------------------------------------------------- AC11 (curve anchors) and dated runs


def test_dated_run_stamps_valuation_date_and_checks_anchors():
    dated_quotes = [
        DepositQuote(rate=0.018, maturity_date=datetime(2026, 12, 7)),
        DepositQuote(rate=0.019, maturity_date=datetime(2027, 2, 5)),
        DepositQuote(rate=0.0205, maturity_date=datetime(2027, 5, 5)),
        SwapQuote(fixed_rate=0.022, index=SHIBOR_3M, maturity_date=datetime(2027, 11, 5)),
        SwapQuote(fixed_rate=0.0245, index=SHIBOR_3M, maturity_date=datetime(2028, 11, 6)),
        SwapQuote(fixed_rate=0.0285, index=SHIBOR_3M, maturity_date=datetime(2031, 11, 5)),
    ]
    curve = bootstrap_discount_curve(dated_quotes, valuation_date=VD)
    assert curve.valuation_date == VD
    assert max(abs(r) for r in reprice_quotes(dated_quotes, curve, valuation_date=VD)) <= 1e-12
    assert curve.parallel_shifted(1e-4).valuation_date == VD and curve.with_pillars(curve.pillars).valuation_date == VD
    # anchors: a differently dated curve is refused, a FlatRateCurve passes, a numeric curve in a dated run is refused
    other = ZeroCurve(curve.pillars, valuation_date=datetime(2026, 11, 6))
    with pytest.raises(ValidationError):
        reprice_quotes(dated_quotes, other, valuation_date=VD)
    with pytest.raises(ValidationError):
        bootstrap_projection_curve(dated_quotes, other, valuation_date=VD)
    assert max(abs(r) for r in reprice_quotes(dated_quotes, FlatRateCurve(0.02), curve, valuation_date=VD)) < 1.0
    with pytest.raises(ValidationError):
        reprice_quotes(AC7_QUOTES, curve)  # dated curve in a numeric run
    with pytest.raises(ValidationError):
        build_carry_curve(6000.0, [FuturesQuote(price=5900.0, tenor_years=0.5)], curve)
    carry = build_carry_curve(6000.0, [FuturesQuote(price=5900.0, expiry_date=datetime(2027, 3, 19))], curve,
                              valuation_date=VD)
    assert carry.valuation_date == VD
    # clock mixing inside a run
    with pytest.raises(ValidationError):
        bootstrap_discount_curve(dated_quotes + DEPOSITS[:1], valuation_date=VD)
    with pytest.raises(ValidationError):
        bootstrap_discount_curve(dated_quotes)
    with pytest.raises(ValidationError):
        bootstrap_discount_curve(AC7_QUOTES, valuation_date=VD)
