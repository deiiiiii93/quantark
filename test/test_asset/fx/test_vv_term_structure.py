"""
Tests for the term-structure Vanna-Volga surface and its integration with the
FX range accrual engines.
"""

import math
from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import (
    FxForeignRangeAccrualAnalyticalEngine,
    FxQuantoRangeAccrualAnalyticalEngine,
    FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.product import (
    CurrencyPair,
    FxForeignRangeAccrualOption,
    FxQuantoRangeAccrualOption,
    FxRangeAccrualConfig,
    FxRangeAccrualOption,
)
from quantark.param import FlatRateCurve, SpotQuote
from quantark.param.vol.vannavolga import (
    DeltaConvention,
    FXEnv,
    SmileQuotes,
    TermStructureVannaVolgaVolSurface,
    VannaVolgaVolSurface,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.priceenv.fx_pricing_environment import (
    FxQuantoMarketData,
    QuantoConversionOrientation,
)
from quantark.util.enum import FxRangeAccrualMethod
from quantark.util.exceptions import ValidationError

SPOT, RD, RF = 1.20, 0.05, 0.03
K = 1.25


def slice_at(tau, atm, rr=-0.02, bf=0.004, spot=SPOT):
    return VannaVolgaVolSurface(
        FXEnv(spot=spot, rd=RD, rf=RF, tau=tau),
        SmileQuotes(sigma_atm=atm, rr25=rr, bf25_2vol=bf),
        DeltaConvention.SPOT,
    )


def term_structure():
    return TermStructureVannaVolgaVolSurface(
        [
            slice_at(0.25, 0.090, rr=-0.030, bf=0.006),
            slice_at(1.00, 0.110, rr=-0.020, bf=0.004),
            slice_at(2.00, 0.120, rr=-0.010, bf=0.003),
        ]
    )


# ---------------------------------------------------------------------------
# Interpolation correctness
# ---------------------------------------------------------------------------


class TestInterpolation:
    def test_reproduces_each_slice_at_quoted_tenor(self):
        ts = term_structure()
        for s in ts.slices:
            tau = s.env.tau
            assert ts.get_vol(K, tau, SPOT) == pytest.approx(
                s.get_vol(K, tau, SPOT), rel=1e-12
            )

    def test_total_variance_interpolation_between_tenors(self):
        ts = term_structure()
        t = 0.5
        s0, s1 = ts.slices[0], ts.slices[1]
        w0 = s0.get_vol(K, 0.25, SPOT) ** 2 * 0.25
        w1 = s1.get_vol(K, 1.0, SPOT) ** 2 * 1.0
        w = w0 + (w1 - w0) * (t - 0.25) / (1.0 - 0.25)
        assert ts.get_vol(K, t, SPOT) == pytest.approx(math.sqrt(w / t), rel=1e-12)

    def test_flat_vol_extrapolation_outside(self):
        ts = term_structure()
        assert ts.get_vol(K, 0.05, SPOT) == pytest.approx(
            ts.slices[0].get_vol(K, 0.25, SPOT), rel=1e-12
        )
        assert ts.get_vol(K, 9.0, SPOT) == pytest.approx(
            ts.slices[-1].get_vol(K, 2.0, SPOT), rel=1e-12
        )

    def test_skew_carried_per_tenor(self):
        # Short tenor has a steeper risk-reversal, so its put-call vol gap at
        # symmetric strikes should exceed the long tenor's.
        ts = term_structure()
        short_gap = ts.get_vol(1.10, 0.25, SPOT) - ts.get_vol(1.40, 0.25, SPOT)
        long_gap = ts.get_vol(1.10, 2.0, SPOT) - ts.get_vol(1.40, 2.0, SPOT)
        assert abs(short_gap) > abs(long_gap)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_slices_rejected(self):
        with pytest.raises(ValidationError):
            TermStructureVannaVolgaVolSurface([])

    def test_non_increasing_tenors_rejected(self):
        with pytest.raises(ValidationError):
            TermStructureVannaVolgaVolSurface(
                [slice_at(1.0, 0.10), slice_at(1.0, 0.11)]
            )

    def test_mismatched_spot_rejected(self):
        with pytest.raises(ValidationError):
            TermStructureVannaVolgaVolSurface(
                [slice_at(0.5, 0.10, spot=1.20), slice_at(1.0, 0.11, spot=1.25)]
            )

    def test_single_slice_allowed(self):
        ts = TermStructureVannaVolgaVolSurface([slice_at(1.0, 0.10)])
        # Degenerate: flat in time, returns that slice everywhere.
        assert ts.get_vol(K, 0.3, SPOT) == pytest.approx(
            ts.get_vol(K, 5.0, SPOT), rel=1e-12
        )


# ---------------------------------------------------------------------------
# Calendar-arbitrage guard
# ---------------------------------------------------------------------------


class TestCalendarArbitrageGuard:
    def test_valid_rising_atm_surface_passes(self):
        # Constructing term_structure() (rising ATM) must not raise.
        assert term_structure().tenors == [0.25, 1.0, 2.0]

    def test_decreasing_total_variance_rejected(self):
        # 30% vol at 0.5y then 10% vol at 1y: w drops 0.045 -> 0.010.
        with pytest.raises(ValidationError, match="[Cc]alendar arbitrage"):
            TermStructureVannaVolgaVolSurface(
                [slice_at(0.5, 0.30), slice_at(1.0, 0.10)]
            )

    def test_bypass_flag_allows_arbitrageable_data(self):
        ts = TermStructureVannaVolgaVolSurface(
            [slice_at(0.5, 0.30), slice_at(1.0, 0.10)],
            check_calendar_arbitrage=False,
        )
        assert ts.tenors == [0.5, 1.0]

    def test_flat_total_variance_allowed(self):
        # sigma^2 * tau constant across tenors (sigma ~ 1/sqrt(tau)) is the
        # zero-forward-variance boundary and must be accepted (within tol).
        s0 = slice_at(1.0, 0.12, rr=0.0, bf=0.0)
        s1 = slice_at(4.0, 0.06, rr=0.0, bf=0.0)  # 0.12^2*1 == 0.06^2*4
        ts = TermStructureVannaVolgaVolSurface([s0, s1])
        assert ts.get_vol(SPOT, 1.0, SPOT) == pytest.approx(0.12, rel=1e-6)

    def test_derived_surfaces_skip_recheck(self):
        # reanchor/with_scaled_quotes return surfaces with the check disabled
        # (they derive from an already-validated surface).
        ts = term_structure()
        assert ts.reanchor(1.25, lambda t: RD, lambda t: RF).check_calendar_arbitrage is False
        assert ts.with_scaled_quotes(1.1).check_calendar_arbitrage is False


# ---------------------------------------------------------------------------
# Re-anchoring and vega scaling (used by Greeks)
# ---------------------------------------------------------------------------


class TestReanchorAndScale:
    def test_reanchor_moves_spot_preserves_quotes(self):
        ts = term_structure()
        bumped = ts.reanchor(1.25, lambda tau: RD, lambda tau: RF)
        assert all(s.env.spot == 1.25 for s in bumped.slices)
        assert [s.quotes.sigma_atm for s in bumped.slices] == [
            s.quotes.sigma_atm for s in ts.slices
        ]

    def test_scaled_quotes_scale_all_slices(self):
        ts = term_structure()
        scaled = ts.with_scaled_quotes(1.10)
        for orig, new in zip(ts.slices, scaled.slices):
            assert new.quotes.sigma_atm == pytest.approx(orig.quotes.sigma_atm * 1.10)
            assert new.quotes.rr25 == pytest.approx(orig.quotes.rr25 * 1.10)


# ---------------------------------------------------------------------------
# Range accrual integration
# ---------------------------------------------------------------------------


def base_env(vol_surface, quanto=None):
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 15),
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=RD),
        foreign_curve=FlatRateCurve(rate=RF),
        vol_surface=vol_surface,
        quanto=quanto,
    )


def accrual(maturity=2.0, n=24):
    return FxRangeAccrualOption(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(
            upper_barrier=1.35, lower_barrier=1.05, accrual_rate=0.04
        ),
        currency_pair=CurrencyPair("EUR", "USD"),
        maturity=maturity,
        num_observations=n,
    )


class TestRangeAccrualIntegration:
    @pytest.mark.parametrize(
        "method",
        [FxRangeAccrualMethod.DIGITAL_COMBINATION, FxRangeAccrualMethod.CALL_PUT_SPREAD],
    )
    def test_price_and_greeks(self, method):
        env = base_env(term_structure())
        engine = FxRangeAccrualAnalyticalEngine(method=method)
        price = engine.price(accrual(), env)
        assert price > 0.0
        greeks = engine.calculate_greeks(accrual(), env)
        for key in ("price", "delta", "gamma", "vega", "theta"):
            assert key in greeks and math.isfinite(greeks[key])

    def test_differs_from_single_tenor_anchor(self):
        # The term structure rising from 9% (short) to 12% (long) must price
        # differently than a single 1y smile applied flat across all fixings.
        ts = term_structure()
        single = VannaVolgaVolSurface(
            FXEnv(spot=SPOT, rd=RD, rf=RF, tau=1.0),
            SmileQuotes(sigma_atm=0.110, rr25=-0.020, bf25_2vol=0.004),
            DeltaConvention.SPOT,
        )
        engine = FxRangeAccrualAnalyticalEngine()
        p_ts = engine.price(accrual(), base_env(ts))
        p_single = engine.price(accrual(), base_env(single))
        assert abs(p_ts - p_single) > 1.0

    def test_quanto_and_foreign_consume_term_structure(self):
        ts = term_structure()
        quanto_md = FxQuantoMarketData(
            settlement_curve=FlatRateCurve(rate=RD),
            quanto_vol=0.12,
            correlation=0.3,
            conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC,
        )
        cfg = FxRangeAccrualConfig(
            upper_barrier=1.35, lower_barrier=1.05, accrual_rate=0.04
        )
        common = dict(
            notional=1_000_000.0,
            range_config=cfg,
            currency_pair=CurrencyPair("EUR", "USD"),
            maturity=2.0,
            num_observations=24,
        )
        quanto = FxQuantoRangeAccrualOption(quanto_fx_rate=110.0, **common)
        foreign = FxForeignRangeAccrualOption(**common)

        qp = FxQuantoRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(quanto, base_env(ts, quanto=quanto_md))
        fp = FxForeignRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(foreign, base_env(ts))
        assert qp > 0.0 and fp > 0.0
