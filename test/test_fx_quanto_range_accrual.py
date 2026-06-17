"""
Tests for the FX quanto range accrual product and analytical engine.

The decisive correctness gate is the reduction property: with zero correlation
and a settlement curve equal to the domestic curve, the quanto coupon must
equal ``quanto_fx_rate`` times the (non-quanto) range accrual coupon.
"""

from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import (
    FxQuantoRangeAccrualAnalyticalEngine,
    FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxQuantoRangeAccrualOption,
    FxRangeAccrualConfig,
    FxRangeAccrualOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.vol.vannavolga import (
    DeltaConvention,
    FXEnv,
    SmileQuotes,
    VannaVolgaVolSurface,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.priceenv.fx_pricing_environment import (
    FxQuantoMarketData,
    QuantoConversionOrientation,
)
from quantark.util.enum import FxRangeAccrualMethod
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import MarketDataError, PricingError, ValidationError

VAL = datetime(2026, 6, 15)
SPOT, RD, RF, VOL, T = 1.20, 0.05, 0.03, 0.10, 1.0
NOTIONAL = 1_000_000.0
XQ = 110.0
LOWER, UPPER, RATE = 1.10, 1.30, 0.04

BOTH_METHODS = [
    FxRangeAccrualMethod.DIGITAL_COMBINATION,
    FxRangeAccrualMethod.CALL_PUT_SPREAD,
]


def quanto_data(correlation, quanto_vol=0.12, settle_rate=RD, orientation=None):
    return FxQuantoMarketData(
        settlement_curve=FlatRateCurve(rate=settle_rate),
        quanto_vol=quanto_vol,
        correlation=correlation,
        conversion_orientation=(
            orientation or QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC
        ),
    )


def env(correlation=0.0, vol_surface=None, **quanto_overrides):
    return FxPricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=RD),
        foreign_curve=FlatRateCurve(rate=RF),
        vol_surface=vol_surface or FlatVolSurface(volatility=VOL),
        quanto=quanto_data(correlation, **quanto_overrides),
    )


def make_options(**cfg_overrides):
    cfg_kwargs = dict(upper_barrier=UPPER, lower_barrier=LOWER, accrual_rate=RATE)
    cfg_kwargs.update(cfg_overrides)
    cfg = FxRangeAccrualConfig(**cfg_kwargs)
    common = dict(
        notional=NOTIONAL,
        range_config=cfg,
        currency_pair=CurrencyPair("EUR", "USD"),
        maturity=T,
        num_observations=12,
    )
    vanilla = FxRangeAccrualOption(**common)
    quanto = FxQuantoRangeAccrualOption(quanto_fx_rate=XQ, settlement_ccy="JPY", **common)
    return vanilla, quanto


# ---------------------------------------------------------------------------
# Reduction property
# ---------------------------------------------------------------------------


class TestQuantoReduction:
    @pytest.mark.parametrize("method", BOTH_METHODS)
    def test_zero_corr_same_curve_reduces_to_scaled_vanilla(self, method):
        vanilla, quanto = make_options()
        e = env(correlation=0.0)
        vp = FxRangeAccrualAnalyticalEngine(method=method).price(vanilla, e)
        qp = FxQuantoRangeAccrualAnalyticalEngine(method=method).price(quanto, e)
        assert qp == pytest.approx(XQ * vp, rel=1e-6)

    def test_methods_agree_under_flat_vol(self):
        _, quanto = make_options()
        e = env(correlation=0.4)
        dig = FxQuantoRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        ).price(quanto, e)
        spr = FxQuantoRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(quanto, e)
        assert spr == pytest.approx(dig, rel=1e-4)


# ---------------------------------------------------------------------------
# Correlation / drift effect
# ---------------------------------------------------------------------------


class TestQuantoDrift:
    def test_correlation_sign_moves_value(self):
        _, quanto = make_options()
        engine = FxQuantoRangeAccrualAnalyticalEngine()
        pos = engine.price(quanto, env(correlation=0.5))
        neg = engine.price(quanto, env(correlation=-0.5))
        zero = engine.price(quanto, env(correlation=0.0))
        assert pos != pytest.approx(neg, rel=1e-6)
        # Zero correlation sits between the two signed cases.
        assert min(pos, neg) < zero < max(pos, neg)

    def test_orientation_flips_drift(self):
        _, quanto = make_options()
        engine = FxQuantoRangeAccrualAnalyticalEngine()
        settle = engine.price(
            quanto,
            env(
                correlation=0.5,
                orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC,
            ),
        )
        domestic = engine.price(
            quanto,
            env(
                correlation=0.5,
                orientation=QuantoConversionOrientation.DOMESTIC_PER_SETTLEMENT,
            ),
        )
        # Opposite drift signs straddle the zero-correlation value.
        zero = engine.price(quanto, env(correlation=0.0))
        assert (settle - zero) * (domestic - zero) < 0


# ---------------------------------------------------------------------------
# Settlement-curve discounting
# ---------------------------------------------------------------------------


class TestSettlementDiscounting:
    def test_higher_settlement_rate_discounts_more(self):
        _, quanto = make_options()
        engine = FxQuantoRangeAccrualAnalyticalEngine()
        low = engine.price(quanto, env(correlation=0.0, settle_rate=0.01))
        high = engine.price(quanto, env(correlation=0.0, settle_rate=0.08))
        assert high < low


# ---------------------------------------------------------------------------
# Vanna-Volga
# ---------------------------------------------------------------------------


class TestQuantoVannaVolga:
    def _smile_env(self, correlation, quotes):
        surface = VannaVolgaVolSurface(
            FXEnv(spot=SPOT, rd=RD, rf=RF, tau=T), quotes, DeltaConvention.SPOT
        )
        return env(correlation=correlation, vol_surface=surface)

    def test_methods_diverge_under_skewed_smile(self):
        _, quanto = make_options()
        smile = SmileQuotes(sigma_atm=VOL, rr25=-0.02, bf25_2vol=0.004)
        e = self._smile_env(0.3, smile)
        dig = FxQuantoRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        ).price(quanto, e)
        spr = FxQuantoRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(quanto, e)
        assert abs(spr - dig) > 1.0


# ---------------------------------------------------------------------------
# Greeks, validation, plumbing
# ---------------------------------------------------------------------------


class TestQuantoGreeksAndValidation:
    def test_greeks_present(self):
        _, quanto = make_options()
        greeks = FxQuantoRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).calculate_greeks(quanto, env(correlation=0.4))
        for key in ("price", "delta", "gamma", "vega", "theta"):
            assert key in greeks

    def test_two_level_enum_method(self):
        engine = FxQuantoRangeAccrualAnalyticalEngine(
            method=EngineType.ANALYTICAL(FxRangeAccrualMethod.CALL_PUT_SPREAD)
        )
        assert engine.method == FxRangeAccrualMethod.CALL_PUT_SPREAD

    def test_missing_quanto_data_raises(self):
        _, quanto = make_options()
        bare = FxPricingEnvironment(
            valuation_date=VAL,
            spot_quote=SpotQuote(spot=SPOT),
            domestic_curve=FlatRateCurve(rate=RD),
            foreign_curve=FlatRateCurve(rate=RF),
            vol_surface=FlatVolSurface(volatility=VOL),
        )
        with pytest.raises(MarketDataError):
            FxQuantoRangeAccrualAnalyticalEngine().price(quanto, bare)

    def test_rejects_non_quanto_product(self):
        vanilla, _ = make_options()
        with pytest.raises(PricingError):
            FxQuantoRangeAccrualAnalyticalEngine().price(vanilla, env())

    def test_negative_quanto_rate_rejected(self):
        cfg = FxRangeAccrualConfig(upper_barrier=UPPER, lower_barrier=LOWER)
        with pytest.raises(ValidationError):
            FxQuantoRangeAccrualOption(
                notional=NOTIONAL,
                range_config=cfg,
                quanto_fx_rate=-1.0,
                maturity=T,
                num_observations=12,
            )

    def test_payoff_converts_at_quanto_rate(self):
        _, quanto = make_options()
        total = quanto.get_total_weights()
        payoff = quanto.get_payoff(SPOT, in_range_weights=total, total_weights=total)
        # Full in-range coupon, converted to settlement currency.
        assert payoff == pytest.approx(NOTIONAL * RATE * XQ)
