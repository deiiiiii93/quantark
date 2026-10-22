"""
Tests for the FX range accrual Monte Carlo engine.

The primary gate is cross-validation against the analytical range-digital
decomposition under a flat (Black-Scholes) surface, where the two must agree to
within Monte Carlo error for both pay types and the full feature set (weights,
per-fixing barriers, settled fixings, reverse mode). The engine is a
constant-volatility Garman-Kohlhagen model, so the comparison is exact only
under a flat surface.
"""

import math
from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.engine.mc import (
    FxRangeAccrualMCEngine,
    FxRangeAccrualMCResult,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxForeignRangeAccrualOption,
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
    FxRangeAccrualOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import CouponPayType, FxRangeAccrualMethod
from quantark.util.exceptions import PricingError, ValidationError

VAL = datetime(2026, 6, 15)
SPOT, RD, RF, VOL, T = 1.20, 0.05, 0.03, 0.10, 1.0
NOTIONAL, LOWER, UPPER, RATE = 1_000_000.0, 1.10, 1.30, 0.04

# Paths / tolerance for the cross-validation gate. ~5 sigma keeps the suite
# both meaningful and non-flaky with a fixed seed.
PATHS = 200_000
SEED = 12345
NSIG = 5.0


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


def mc_engine(**overrides):
    kwargs = dict(num_paths=PATHS, seed=SEED)
    kwargs.update(overrides)
    return FxRangeAccrualMCEngine(**kwargs)


def assert_within_mc_error(mc_price, analytical, std_error, nsig=NSIG):
    assert std_error > 0.0
    z = abs(mc_price - analytical) / std_error
    assert z <= nsig, (
        f"MC {mc_price:.4f} vs analytical {analytical:.4f} "
        f"({z:.2f} sigma, se={std_error:.4f})"
    )


# ---------------------------------------------------------------------------
# Cross-validation against the analytical engine (flat surface)
# ---------------------------------------------------------------------------


class TestAnalyticalAgreement:
    @pytest.mark.parametrize("pay_type", [CouponPayType.EXPIRY, CouponPayType.INSTANT])
    def test_matches_analytical_flat(self, pay_type):
        option = make_option(config={"accrual_pay_type": pay_type})
        env = flat_env()
        analytical = FxRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        ).price(option, env)
        eng = mc_engine()
        mc = eng.price(option, env)
        assert_within_mc_error(mc, analytical, eng.get_last_result().std_error)

    def test_matches_with_weights(self):
        records = [
            FxRangeAccrualObservationRecord(observation_time=(i + 1) / 12.0, weight=w)
            for i, w in enumerate([1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1])
        ]
        option = make_option(observation_records=records, num_observations=None)
        env = flat_env()
        analytical = FxRangeAccrualAnalyticalEngine().price(option, env)
        eng = mc_engine()
        mc = eng.price(option, env)
        assert_within_mc_error(mc, analytical, eng.get_last_result().std_error)

    def test_matches_per_fixing_barriers(self):
        n = 12
        uppers = [1.28 + 0.005 * i for i in range(n)]
        lowers = [1.12 - 0.005 * i for i in range(n)]
        option = make_option(
            config={"upper_barrier": uppers, "lower_barrier": lowers}
        )
        env = flat_env()
        analytical = FxRangeAccrualAnalyticalEngine().price(option, env)
        eng = mc_engine()
        mc = eng.price(option, env)
        assert_within_mc_error(mc, analytical, eng.get_last_result().std_error)

    def test_matches_reverse_mode(self):
        option = make_option(config={"is_reverse": True})
        env = flat_env()
        analytical = FxRangeAccrualAnalyticalEngine().price(option, env)
        eng = mc_engine()
        mc = eng.price(option, env)
        assert_within_mc_error(mc, analytical, eng.get_last_result().std_error)

    def test_matches_with_settled_past_fixings(self):
        # Valuation midway through the schedule, with two settled fixings.
        records = [
            FxRangeAccrualObservationRecord(
                observation_time=-0.20, weight=1.0, observed_in_range=True
            ),
            FxRangeAccrualObservationRecord(
                observation_time=-0.10, weight=1.0, observed_in_range=False
            ),
        ] + [
            FxRangeAccrualObservationRecord(observation_time=0.1 * (i + 1), weight=1.0)
            for i in range(8)
        ]
        option = make_option(
            observation_records=records, num_observations=None, maturity=0.8
        )
        env = flat_env()
        analytical = FxRangeAccrualAnalyticalEngine().price(option, env)
        eng = mc_engine()
        mc = eng.price(option, env)
        assert_within_mc_error(mc, analytical, eng.get_last_result().std_error)

    def test_matches_non_flat_curves(self):
        # Term-structure-shaped forward via differing flat curves still matches:
        # the simulation anchors each fixing on its own parity forward.
        env = flat_env(
            domestic_curve=FlatRateCurve(rate=0.06),
            foreign_curve=FlatRateCurve(rate=0.01),
        )
        option = make_option()
        analytical = FxRangeAccrualAnalyticalEngine().price(option, env)
        eng = mc_engine()
        mc = eng.price(option, env)
        assert_within_mc_error(mc, analytical, eng.get_last_result().std_error)


# ---------------------------------------------------------------------------
# Determinism / variance reduction
# ---------------------------------------------------------------------------


class TestDeterminismAndVR:
    def test_fixed_seed_is_deterministic(self):
        option, env = make_option(), flat_env()
        p1 = mc_engine().price(option, env)
        p2 = mc_engine().price(option, env)
        assert p1 == p2

    def test_qmc_reduces_std_error(self):
        option, env = make_option(), flat_env()
        pseudo = mc_engine(use_qmc=False)
        pseudo.price(option, env)
        quasi = mc_engine(use_qmc=True)
        quasi.price(option, env)
        assert (
            quasi.get_last_result().std_error
            < pseudo.get_last_result().std_error
        )

    def test_result_fields(self):
        option, env = make_option(), flat_env()
        eng = mc_engine()
        eng.price(option, env)
        res = eng.get_last_result()
        assert isinstance(res, FxRangeAccrualMCResult)
        assert res.num_paths == PATHS
        assert res.num_future_observations == 12
        assert res.num_past_observations == 0
        assert 0.0 < res.in_range_ratio_mean < 1.0
        assert res.sigma == pytest.approx(VOL)


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------


class TestDegenerate:
    def test_all_past_fixings_deterministic(self):
        records = [
            FxRangeAccrualObservationRecord(
                observation_time=-0.5, weight=1.0, observed_in_range=True
            ),
            FxRangeAccrualObservationRecord(
                observation_time=-0.25, weight=1.0, observed_in_range=False
            ),
        ]
        option = make_option(
            observation_records=records, num_observations=None, maturity=0.5
        )
        env = flat_env()
        eng = mc_engine()
        mc = eng.price(option, env)
        analytical = FxRangeAccrualAnalyticalEngine().price(option, env)
        # No simulation: exact match, zero std error.
        assert eng.get_last_result().num_paths == 0
        assert mc == pytest.approx(analytical, rel=1e-12)

    def test_expired_product_matches_realised_coupon(self):
        records = [
            FxRangeAccrualObservationRecord(
                observation_time=0.0, weight=1.0, observed_in_range=True
            ),
            FxRangeAccrualObservationRecord(
                observation_time=0.0, weight=1.0, observed_in_range=False
            ),
        ]
        # maturity ~ 0 (is_zero) triggers the deterministic near-expiry branch.
        option = make_option(
            observation_records=records, num_observations=None, maturity=1e-12
        )
        env = flat_env()
        mc = mc_engine().price(option, env)
        analytical = FxRangeAccrualAnalyticalEngine().price(option, env)
        assert mc == pytest.approx(analytical, rel=1e-12)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


class TestGreeks:
    def test_greeks_finite_and_signed(self):
        option, env = make_option(), flat_env()
        greeks = mc_engine().calculate_greeks(option, env)
        for key in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
            assert key in greeks and math.isfinite(greeks[key])
        # Spot is centered in the band: more vol or a spot move toward a barrier
        # lowers the in-range probability, so vega < 0 and gamma < 0.
        assert greeks["vega"] < 0.0
        assert greeks["gamma"] < 0.0


# ---------------------------------------------------------------------------
# Product guard
# ---------------------------------------------------------------------------


class TestProductGuard:
    def test_rejects_non_range_accrual(self):
        from quantark.asset.fx.product.option import FxVanillaOption
        from quantark.util.enum import OptionType

        vanilla = FxVanillaOption(
            strike=1.20,
            option_type=OptionType.CALL,
            maturity=1.0,
            notional_foreign=1.0,
        )
        with pytest.raises(PricingError):
            mc_engine().price(vanilla, flat_env())

    def test_rejects_foreign_subclass(self):
        foreign = FxForeignRangeAccrualOption(
            notional=NOTIONAL,
            range_config=FxRangeAccrualConfig(
                upper_barrier=UPPER, lower_barrier=LOWER, accrual_rate=RATE
            ),
            currency_pair=CurrencyPair("EUR", "USD"),
            maturity=T,
            num_observations=12,
        )
        with pytest.raises(PricingError, match="own MC engine"):
            mc_engine().price(foreign, flat_env())
