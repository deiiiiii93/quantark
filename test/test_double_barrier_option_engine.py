"""
Tests for DoubleBarrierOptionAnalyticalEngine.

Validation baselines from Haug, Table 4-15 (Ikeda & Kuintomo 1992).
S = 100, X = 100, r = 0.1, b = 0.1 (cost of carry = risk-free, so q = 0).
"""

import pytest
from datetime import datetime

from quantark.asset.equity.engine.analytical import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.product.option import DoubleBarrierOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.option_enums import DoubleBarrierType, ObservationType
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield


def make_env(spot: float, vol: float, rate: float, div: float = 0.0) -> PricingEnvironment:
    """Factory for pricing environments used in tests."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        rate_curve=FlatRateCurve(rate),
        div_yield=ContinuousDividendYield(div),
        valuation_date=datetime(2024, 1, 1),
    )


class TestContinuousObservationBenchmarks:
    """Table 4-15 continuous observation benchmark cases."""

    @pytest.mark.parametrize(
        "L,U,delta1,delta2,T,sigma,expected",
        [
            # T = 0.25, sigma = 0.15
            (50, 150, 0, 0, 0.25, 0.15, 4.3515),
            (60, 140, 0, 0, 0.25, 0.15, 4.3505),
            (70, 130, 0, 0, 0.25, 0.15, 4.3139),
            (80, 120, 0, 0, 0.25, 0.15, 3.7516),
            (90, 110, 0, 0, 0.25, 0.15, 1.2055),
            # T = 0.25, sigma = 0.25
            (50, 150, 0, 0, 0.25, 0.25, 6.1644),
            (60, 140, 0, 0, 0.25, 0.25, 5.8500),
            (70, 130, 0, 0, 0.25, 0.25, 4.8293),
            (80, 120, 0, 0, 0.25, 0.25, 2.6387),
            (90, 110, 0, 0, 0.25, 0.25, 0.3098),
            # T = 0.5, sigma = 0.15
            (50, 150, 0, 0, 0.50, 0.15, 6.9853),
            (60, 140, 0, 0, 0.50, 0.15, 6.8082),
            (70, 130, 0, 0, 0.50, 0.15, 5.9697),
            (80, 120, 0, 0, 0.50, 0.15, 3.5805),
            (90, 110, 0, 0, 0.50, 0.15, 0.5537),
            # T = 0.5, sigma = 0.25
            (50, 150, 0, 0, 0.50, 0.25, 7.9336),
            (60, 140, 0, 0, 0.50, 0.25, 6.3383),
            (70, 130, 0, 0, 0.50, 0.25, 4.0004),
            (80, 120, 0, 0, 0.50, 0.25, 1.5098),
            (90, 110, 0, 0, 0.50, 0.25, 0.0441),
            # Curvature cases: delta1=-0.1, delta2=0.1, T=0.25
            (50, 150, -0.1, 0.1, 0.25, 0.15, 4.3514),
            (60, 140, -0.1, 0.1, 0.25, 0.15, 4.3478),
            (70, 130, -0.1, 0.1, 0.25, 0.15, 4.2558),
            (80, 120, -0.1, 0.1, 0.25, 0.15, 3.2953),
            (90, 110, -0.1, 0.1, 0.25, 0.15, 0.5887),
            (50, 150, -0.1, 0.1, 0.25, 0.25, 6.0997),
            (60, 140, -0.1, 0.1, 0.25, 0.25, 5.6351),
            (70, 130, -0.1, 0.1, 0.25, 0.25, 4.3291),
            (80, 120, -0.1, 0.1, 0.25, 0.25, 1.9868),
            (90, 110, -0.1, 0.1, 0.25, 0.25, 0.1016),
            # Curvature cases: delta1=0.1, delta2=-0.1, T=0.25
            (50, 150, 0.1, -0.1, 0.25, 0.15, 4.3515),
            (60, 140, 0.1, -0.1, 0.25, 0.15, 4.3512),
            (70, 130, 0.1, -0.1, 0.25, 0.15, 4.3382),
            (80, 120, 0.1, -0.1, 0.25, 0.15, 4.0428),
            (90, 110, 0.1, -0.1, 0.25, 0.15, 1.9229),
            (50, 150, 0.1, -0.1, 0.25, 0.25, 6.2040),
            (60, 140, 0.1, -0.1, 0.25, 0.25, 5.9998),
            (70, 130, 0.1, -0.1, 0.25, 0.25, 5.2358),
            (80, 120, 0.1, -0.1, 0.25, 0.25, 3.2872),
            (90, 110, 0.1, -0.1, 0.25, 0.25, 0.6451),
        ],
    )
    def test_call_up_out_down_out(self, L, U, delta1, delta2, T, sigma, expected):
        """Call up-and-out-down-and-out benchmark validation."""
        env = make_env(spot=100.0, vol=sigma, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=float(U),
            lower_barrier=float(L),
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=T,
            observation_type=ObservationType.CONTINUOUS,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()

        # For non-flat boundaries we need to pass delta1/delta2 into the engine.
        # The public API does not expose them yet; they are internal kwargs to
        # _price_continuous.  We call the internal helper directly for curvature cases.
        if delta1 != 0.0 or delta2 != 0.0:
            price = engine._price_continuous(
                product=option,
                pricing_env=env,
                S=100.0,
                K=100.0,
                T=T,
                r=0.1,
                q=0.0,
                sigma=sigma,
                L=float(L),
                U=float(U),
                multiplier=1.0,
                delta1=delta1,
                delta2=delta2,
            )
        else:
            price = engine.price(option, env)

        assert price == pytest.approx(expected, abs=1e-3)


class TestPutContinuousObservation:
    """Put continuous observation sanity checks."""

    def test_put_knock_out_basic(self):
        env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.PUT,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        price = engine.price(option, env)
        assert price >= 0.0
        # Put KO should be cheaper than vanilla put
        from quantark.asset.equity.engine.analytical import BlackScholesEngine
        vanilla_opt = engine._create_vanilla(option)
        vanilla_price = BlackScholesEngine().price(vanilla_opt, env)
        assert price < vanilla_price


class TestKnockInParity:
    """Knock-in = Vanilla - Knock-out parity."""

    def test_call_knock_in_parity(self):
        env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
        ko = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        ki = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_IN,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        ko.validate()
        ki.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        ko_price = engine.price(ko, env)
        ki_price = engine.price(ki, env)
        vanilla_opt = engine._create_vanilla(ko)
        from quantark.asset.equity.engine.analytical import BlackScholesEngine
        vanilla_price = BlackScholesEngine().price(vanilla_opt, env)
        assert ki_price == pytest.approx(vanilla_price - ko_price, abs=1e-4)

    def test_put_knock_in_parity(self):
        env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
        ko = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.PUT,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        ki = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.PUT,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_IN,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        ko.validate()
        ki.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        ko_price = engine.price(ko, env)
        ki_price = engine.price(ki, env)
        vanilla_opt = engine._create_vanilla(ko)
        from quantark.asset.equity.engine.analytical import BlackScholesEngine
        vanilla_price = BlackScholesEngine().price(vanilla_opt, env)
        assert ki_price == pytest.approx(vanilla_price - ko_price, abs=1e-4)


class TestExpiryObservation:
    """Expiry-only observation tests."""

    def test_expiry_call_ko(self):
        env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.EXPIRY,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        price = engine.price(option, env)
        # Expiry KO should be >= continuous KO because less chance of hitting
        cont_option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        cont_option.validate()
        cont_price = engine.price(cont_option, env)
        assert price >= cont_price

    def test_expiry_put_ko(self):
        env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.PUT,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.EXPIRY,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        price = engine.price(option, env)
        assert price >= 0.0


class TestDailyObservation:
    """Daily observation (discrete with barrier shift) tests."""

    def test_daily_call_ko_shifted(self):
        from quantark.asset.equity.product.option.observation_schedule import (
            ObservationSchedule,
            ObservationRecord,
        )
        from quantark.util.enum import ObservationAggregation

        env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
        schedule = ObservationSchedule(
            records=[
                ObservationRecord(observation_time=t / 252.0)
                for t in range(1, 64)  # ~3 months of daily observations
            ],
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        )
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.DISCRETE,
            observation_schedule=schedule,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        price = engine.price(option, env)
        # Daily KO should be >= continuous KO because discrete monitoring is "easier"
        cont_option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        cont_option.validate()
        cont_price = engine.price(cont_option, env)
        assert price >= cont_price


class TestEdgeCases:
    """Boundary and edge-case tests."""

    def test_zero_maturity_inside_barriers_ko(self):
        env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=1e-12,
            observation_type=ObservationType.CONTINUOUS,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        price = engine.price(option, env)
        assert price == pytest.approx(max(100.0 - 100.0, 0.0), abs=1e-10)

    def test_zero_maturity_outside_barriers_ko(self):
        env = make_env(spot=160.0, vol=0.25, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=1e-12,
            observation_type=ObservationType.CONTINUOUS,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        price = engine.price(option, env)
        assert price == pytest.approx(0.0, abs=1e-10)

    def test_spot_already_outside_ko(self):
        env = make_env(spot=160.0, vol=0.25, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        price = engine.price(option, env)
        assert price == 0.0

    def test_spot_already_outside_ki(self):
        env = make_env(spot=160.0, vol=0.25, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_IN,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        price = engine.price(option, env)
        vanilla_opt = engine._create_vanilla(option)
        from quantark.asset.equity.engine.analytical import BlackScholesEngine
        vanilla_price = BlackScholesEngine().price(vanilla_opt, env)
        assert price == pytest.approx(vanilla_price, abs=1e-10)

    def test_zero_vol_ko_inside(self):
        env = make_env(spot=100.0, vol=0.0001, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        # Very low vol path stays inside; price approximates discounted payoff
        price = engine.price(option, env)
        assert price >= 0.0

    def test_strike_at_barrier_rejected(self):
        env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=150.0,  # equal to upper barrier
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
        )
        option.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        from quantark.util.exceptions import ValidationError
        with pytest.raises(ValidationError):
            engine.price(option, env)

    def test_contract_multiplier_scaling(self):
        env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
        option1 = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
            contract_multiplier=1.0,
        )
        option2 = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=150.0,
            lower_barrier=50.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.25,
            observation_type=ObservationType.CONTINUOUS,
            contract_multiplier=10_000.0,
        )
        option1.validate()
        option2.validate()
        engine = DoubleBarrierOptionAnalyticalEngine()
        p1 = engine.price(option1, env)
        p2 = engine.price(option2, env)
        assert p2 == pytest.approx(p1 * 10_000.0, rel=1e-10)
