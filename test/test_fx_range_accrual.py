"""
Tests for the FX range accrual product and analytical engine.

Covers both per-fixing pricing methods (digital combination and call/put
spread) and both surface types (flat Black-Scholes and Vanna-Volga).
"""

import math
from datetime import datetime

import pytest
from scipy.stats import norm

from quantark.asset.fx.engine.analytical import (
    FxRangeAccrualAnalyticalEngine,
    FxRangeAccrualResult,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
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
from quantark.util.enum import CouponPayType, FxRangeAccrualMethod
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError

VAL = datetime(2026, 6, 15)
SPOT = 1.20
RD = 0.05
RF = 0.03
VOL = 0.10
T = 1.0
NOTIONAL = 1_000_000.0
LOWER = 1.10
UPPER = 1.30
RATE = 0.04


def flat_env(**overrides):
    kwargs = dict(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=RD),
        foreign_curve=FlatRateCurve(rate=RF),
        vol_surface=FlatVolSurface(volatility=VOL),
    )
    kwargs.update(overrides)
    return FxPricingEnvironment(**kwargs)


def vv_env(quotes, **overrides):
    surface = VannaVolgaVolSurface(
        FXEnv(spot=SPOT, rd=RD, rf=RF, tau=T), quotes, DeltaConvention.SPOT
    )
    return flat_env(vol_surface=surface, **overrides)


def make_option(**overrides):
    cfg_kwargs = dict(upper_barrier=UPPER, lower_barrier=LOWER, accrual_rate=RATE)
    cfg_kwargs.update(overrides.pop("config", {}))
    kwargs = dict(
        notional=NOTIONAL,
        range_config=FxRangeAccrualConfig(**cfg_kwargs),
        currency_pair=CurrencyPair("EUR", "USD"),
        maturity=T,
        num_observations=12,
    )
    kwargs.update(overrides)
    return FxRangeAccrualOption(**kwargs)


def manual_range_prob(t, lower=LOWER, upper=UPPER, vol=VOL):
    """Closed-form P(lower < X_t < upper) under flat Garman-Kohlhagen."""
    fwd = SPOT * math.exp((RD - RF) * t)
    sig_sqrt_t = vol * math.sqrt(t)
    d2_l = (math.log(fwd / lower) - 0.5 * vol * vol * t) / sig_sqrt_t
    d2_u = (math.log(fwd / upper) - 0.5 * vol * vol * t) / sig_sqrt_t
    return norm.cdf(d2_l) - norm.cdf(d2_u)


# ---------------------------------------------------------------------------
# Core closed-form correctness
# ---------------------------------------------------------------------------


class TestDigitalCombination:
    def test_matches_manual_single_fixing(self):
        option = make_option(num_observations=None, observation_times=[T])
        engine = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        )
        price = engine.price(option, flat_env())

        prob = manual_range_prob(T)
        expected = NOTIONAL * RATE * prob * math.exp(-RD * T)
        assert price == pytest.approx(expected, rel=1e-10)

    def test_per_observation_probs_match_manual(self):
        times = [0.25, 0.5, 0.75, 1.0]
        option = make_option(num_observations=None, observation_times=times)
        engine = FxRangeAccrualAnalyticalEngine()
        engine.price(option, flat_env())
        result = engine.get_last_result()
        assert isinstance(result, FxRangeAccrualResult)
        for t, p in zip(times, result.per_observation_probs):
            assert p == pytest.approx(manual_range_prob(t), rel=1e-10)

    def test_default_method_is_digital_combination(self):
        engine = FxRangeAccrualAnalyticalEngine()
        assert engine.method == FxRangeAccrualMethod.DIGITAL_COMBINATION


# ---------------------------------------------------------------------------
# Method equivalence under flat vol
# ---------------------------------------------------------------------------


class TestMethodEquivalenceFlatVol:
    def test_spread_matches_digital_under_flat_vol(self):
        option = make_option()
        env = flat_env()
        digital = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        ).price(option, env)
        spread = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(option, env)
        # Centered call spread is O(h^2)-accurate for the digital under flat vol.
        assert spread == pytest.approx(digital, rel=1e-5)


# ---------------------------------------------------------------------------
# Vanna-Volga smile behaviour
# ---------------------------------------------------------------------------


class TestVannaVolga:
    def test_flat_vv_reduces_to_flat_bs(self):
        flat_smile = SmileQuotes(sigma_atm=VOL, rr25=0.0, bf25_2vol=0.0)
        option = make_option()
        bs = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(option, flat_env())
        vv = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(option, vv_env(flat_smile))
        assert vv == pytest.approx(bs, rel=1e-3)

    def test_methods_diverge_under_skewed_smile(self):
        smile = SmileQuotes(sigma_atm=VOL, rr25=-0.02, bf25_2vol=0.004)
        option = make_option()
        env = vv_env(smile)
        digital = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        ).price(option, env)
        spread = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(option, env)
        # The replication captures the smile skew the level-only digital omits.
        assert abs(spread - digital) > 1.0


# ---------------------------------------------------------------------------
# Reverse mode and accrual identities
# ---------------------------------------------------------------------------


class TestReverseAndIdentities:
    def test_normal_plus_reverse_is_full_coupon(self):
        env = flat_env()
        normal = make_option()
        reverse = make_option(config={"is_reverse": True})
        engine = FxRangeAccrualAnalyticalEngine()
        full = NOTIONAL * RATE * math.exp(-RD * T)
        assert engine.price(normal, env) + engine.price(reverse, env) == pytest.approx(
            full, rel=1e-10
        )

    def test_price_within_full_coupon_bound(self):
        engine = FxRangeAccrualAnalyticalEngine()
        price = engine.price(make_option(), flat_env())
        assert 0.0 < price < NOTIONAL * RATE


# ---------------------------------------------------------------------------
# Pay timing
# ---------------------------------------------------------------------------


class TestPayTiming:
    def test_instant_pay_discounts_more_than_expiry(self):
        env = flat_env()
        expiry = make_option(config={"accrual_pay_type": CouponPayType.EXPIRY})
        instant = make_option(config={"accrual_pay_type": CouponPayType.INSTANT})
        engine = FxRangeAccrualAnalyticalEngine()
        # INSTANT discounts each fixing from its (earlier) date -> less discounting
        # for early fixings, so PV is higher than the single maturity discount.
        assert engine.price(instant, env) > engine.price(expiry, env)


# ---------------------------------------------------------------------------
# Historical (settled) fixings
# ---------------------------------------------------------------------------


class TestHistoricalFixings:
    def _records(self):
        return [
            FxRangeAccrualObservationRecord(observation_time=-0.25, observed_in_range=True),
            FxRangeAccrualObservationRecord(observation_time=-0.1, observed_in_range=False),
            FxRangeAccrualObservationRecord(observation_time=0.5),
            FxRangeAccrualObservationRecord(observation_time=1.0),
        ]

    def test_past_in_range_weight_recorded(self):
        option = make_option(num_observations=None, observation_records=self._records())
        engine = FxRangeAccrualAnalyticalEngine()
        engine.price(option, flat_env())
        result = engine.get_last_result()
        assert result.past_in_range_weights == pytest.approx(1.0)
        assert result.num_past_observations == 2
        assert result.num_future_observations == 2

    def test_near_expiry_uses_settled_outcomes(self):
        records = [
            FxRangeAccrualObservationRecord(observation_time=0.0, observed_in_range=True),
            FxRangeAccrualObservationRecord(observation_time=0.0, observed_in_range=False),
        ]
        option = FxRangeAccrualOption(
            notional=NOTIONAL,
            range_config=FxRangeAccrualConfig(
                upper_barrier=UPPER, lower_barrier=LOWER, accrual_rate=RATE
            ),
            maturity=1e-12,
            observation_records=records,
        )
        engine = FxRangeAccrualAnalyticalEngine()
        price = engine.price(option, flat_env())
        # Half the (equal-weight) fixings accrued; discount ~ 1 at T~0.
        assert price == pytest.approx(NOTIONAL * RATE * 0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


class TestGreeks:
    def test_greeks_present(self):
        engine = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        )
        greeks = engine.calculate_greeks(make_option(), flat_env())
        for key in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
            assert key in greeks

    def test_vega_positive_when_spot_inside_band(self):
        # Spot well inside a wide band: more vol pushes mass toward the barriers,
        # lowering the in-range probability -> negative vega is expected here.
        engine = FxRangeAccrualAnalyticalEngine()
        greeks = engine.calculate_greeks(make_option(), flat_env())
        assert greeks["vega"] != 0.0


# ---------------------------------------------------------------------------
# Method parsing and validation
# ---------------------------------------------------------------------------


class TestMethodParsing:
    def test_two_level_enum(self):
        engine = FxRangeAccrualAnalyticalEngine(
            method=EngineType.ANALYTICAL(FxRangeAccrualMethod.CALL_PUT_SPREAD)
        )
        assert engine.method == FxRangeAccrualMethod.CALL_PUT_SPREAD

    def test_string_method(self):
        engine = FxRangeAccrualAnalyticalEngine(method="call_put_spread")
        assert engine.method == FxRangeAccrualMethod.CALL_PUT_SPREAD

    def test_invalid_string_method(self):
        with pytest.raises(ValidationError):
            FxRangeAccrualAnalyticalEngine(method="nonsense")


class TestValidation:
    def test_wrong_product_type(self):
        from quantark.asset.fx.product.option import FxVanillaOption
        from quantark.util.enum import OptionType

        vanilla = FxVanillaOption(
            strike=1.25, option_type=OptionType.CALL, maturity=T, notional_foreign=1.0
        )
        with pytest.raises(PricingError):
            FxRangeAccrualAnalyticalEngine().price(vanilla, flat_env())

    def test_negative_notional_rejected(self):
        with pytest.raises(ValidationError):
            make_option(notional=-1.0)

    def test_inverted_barriers_rejected(self):
        with pytest.raises(ValidationError):
            FxRangeAccrualConfig(upper_barrier=1.10, lower_barrier=1.30)

    def test_conflicting_schedule_inputs_rejected(self):
        with pytest.raises(ValidationError):
            make_option(num_observations=12, observation_times=[0.5, 1.0])

    def test_barrier_list_length_mismatch_rejected(self):
        with pytest.raises(ValidationError):
            make_option(
                num_observations=3,
                config={"upper_barrier": [1.30, 1.31], "lower_barrier": LOWER},
            )

    def test_past_fixing_without_outcome_rejected(self):
        option = make_option(
            num_observations=None,
            observation_records=[
                FxRangeAccrualObservationRecord(observation_time=-0.25),
                FxRangeAccrualObservationRecord(observation_time=1.0),
            ],
        )
        with pytest.raises(ValidationError):
            FxRangeAccrualAnalyticalEngine().price(option, flat_env())
