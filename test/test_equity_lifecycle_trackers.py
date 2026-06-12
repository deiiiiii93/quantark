"""Tests for the shared equity lifecycle core (state, events, trackers)."""

from datetime import datetime

import pandas as pd

from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.numerical import almost_equal, is_close


def make_env(spot, vol=0.20, rate=0.03, valuation_date=datetime(2026, 1, 5)):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        valuation_date=valuation_date,
    )


class TestLifecyclePackage:
    def test_event_and_state_importable(self):
        from quantark.asset.equity.lifecycle import (
            AutocallableLifecycleState,
            BarrierLifecycleState,
            LifecycleEvent,
            LifecycleEventType,
        )

        state = AutocallableLifecycleState()
        assert state.alive and not state.knocked_in

        bstate = BarrierLifecycleState()
        assert bstate.alive and not bstate.knocked_out

        event = LifecycleEvent(
            event_type=LifecycleEventType.KNOCK_OUT,
            date=datetime(2026, 2, 4),
            spot=105.0,
        )
        assert event.event_type.value == "KO"
        assert event.cashflow == 0.0
        assert isinstance(hash(event), int)

    def test_backtest_state_is_reexport(self):
        from quantark.asset.equity.lifecycle import AutocallableLifecycleState as Shared
        from quantark.backtest.otc.state import AutocallableLifecycleState as Legacy

        assert Shared is Legacy


def make_snowball(maturity=1.0, ko_barrier=103.0, ki_barrier=70.0,
                  contract_multiplier=1.0):
    from quantark.asset.equity.product.option import SnowballOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig

    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=ko_barrier,
            ko_rate=0.15,
            ko_observation_dates=[i / 12.0 for i in range(1, 13)],
            ki_barrier=ki_barrier,
            ki_continuous=True,
        ),
        maturity=maturity,
        contract_multiplier=contract_multiplier,
    )


START = pd.Timestamp("2026-01-05")
FIRST_KO_OBS = START + pd.Timedelta(days=int(round(365 / 12)))


class TestAutocallableLifecycleTracker:
    def _tracker(self, product, quantity=2.0):
        from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker

        return AutocallableLifecycleTracker(
            product=product, quantity=quantity, start_date=START
        )

    def test_no_event_before_first_observation(self):
        product = make_snowball()
        tracker = self._tracker(product)
        env = make_env(spot=105.0)
        live = tracker.product_for_lifecycle()
        events = tracker.observe(START + pd.Timedelta(days=5), live, env, 105.0)
        assert events == []
        assert tracker.lifecycle.alive

    def test_ko_event_on_first_observation(self):
        from quantark.asset.equity.lifecycle import LifecycleEventType

        product = make_snowball()
        tracker = self._tracker(product, quantity=2.0)
        env = make_env(spot=105.0)
        live = tracker.product_for_lifecycle()
        events = tracker.observe(FIRST_KO_OBS, live, env, 105.0)

        assert len(events) == 1
        event = events[0]
        assert event.event_type is LifecycleEventType.KNOCK_OUT
        assert event.observation_index == 0
        assert event.terminates_position
        assert tracker.lifecycle.knocked_out and not tracker.lifecycle.alive

        profile = product.get_ko_observation_profile(
            make_env(spot=105.0, valuation_date=START.to_pydatetime())
        )
        expected = 2.0 * float(profile["payoffs"][0])
        assert almost_equal(event.cashflow, expected)
        assert almost_equal(tracker.lifecycle.realized_cashflows, expected)

    def test_continuous_ki_sets_flag_and_pricing_product(self):
        from quantark.asset.equity.lifecycle import LifecycleEventType

        product = make_snowball()
        tracker = self._tracker(product)
        env = make_env(spot=65.0)
        date = START + pd.Timedelta(days=3)
        live = tracker.product_for_lifecycle()
        events = tracker.observe(date, live, env, 65.0)

        assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_IN]
        assert events[0].metadata.get("monitoring") == "daily_close"
        assert tracker.lifecycle.knocked_in and tracker.lifecycle.alive

        priced = tracker.product_for_pricing(date, env)
        assert getattr(priced, "_otc_lifecycle_knocked_in") is True
        # float maturity decays by elapsed calendar time
        assert is_close(priced.maturity, 1.0 - 3.0 / 365.0)
        # original product untouched
        assert is_close(product.maturity, 1.0)

    def test_maturity_settlement(self):
        from quantark.asset.equity.lifecycle import LifecycleEventType

        product = make_snowball(ko_barrier=200.0)  # never KO
        tracker = self._tracker(product, quantity=3.0)
        env = make_env(spot=100.0)
        live = tracker.product_for_lifecycle()
        maturity_date = START + pd.Timedelta(days=365)
        event = tracker.settle_maturity_if_due(maturity_date, live, env, 100.0)

        assert event is not None
        assert event.event_type is LifecycleEventType.MATURITY
        assert event.terminates_position
        assert tracker.lifecycle.matured and not tracker.lifecycle.alive
        expected = 3.0 * float(live.get_payoff(100.0, env, knocked_in=False))
        assert almost_equal(event.cashflow, expected)

    def test_phoenix_coupon_event(self):
        # NOTE: get_ki_observation_profile requires an ObservationSchedule for
        # discrete KI, which create_standard_phoenix does not build from
        # ki_observation_dates alone.  Using ki_barrier=None avoids the KI
        # schedule path entirely (_scheduled_records("ki") returns [] when
        # has_ki_barrier is False); the coupon detection is independent of KI.
        from quantark.asset.equity.lifecycle import LifecycleEventType
        from quantark.asset.equity.product.option import PhoenixOption
        from quantark.asset.equity.product.option.phoenix_config import (
            CouponBarrierConfig,
        )
        from quantark.asset.equity.product.option.snowball_config import (
            AccrualConfig,
            AirbagConfig,
            BarrierConfig,
            PayoffConfig,
        )

        phoenix = PhoenixOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=BarrierConfig(
                ko_barrier=103.0,
                ko_rate=0.15,
                ko_observation_dates=[i / 12.0 for i in range(1, 13)],
            ),
            coupon_config=CouponBarrierConfig(coupon_barrier=85.0, coupon_rate=0.01),
            payoff_config=PayoffConfig(rebate_rate=0.15),
            accrual_config=AccrualConfig(),
            airbag_config=AirbagConfig(),
            maturity=1.0,
            contract_multiplier=1.0,
        )
        tracker = self._tracker(phoenix, quantity=1.0)
        env = make_env(spot=100.0)
        live = tracker.product_for_lifecycle()
        events = tracker.observe(FIRST_KO_OBS, live, env, 100.0)

        coupon_events = [
            e for e in events if e.event_type is LifecycleEventType.COUPON
        ]
        assert len(coupon_events) == 1
        assert not coupon_events[0].terminates_position
        assert tracker.lifecycle.alive
        expected = 1.0 * float(phoenix.get_coupon_payoff(0))
        assert almost_equal(coupon_events[0].cashflow, expected)


class TestBarrierLifecycleTracker:
    def _tracker(self, product, quantity=1.0):
        from quantark.asset.equity.lifecycle import BarrierLifecycleTracker

        return BarrierLifecycleTracker(
            product=product, quantity=quantity, start_date=START
        )

    def test_up_out_call_knock_out_pays_rebate_at_hit(self):
        from quantark.asset.equity.lifecycle import LifecycleEventType
        from quantark.asset.equity.product.option import BarrierOption
        from quantark.util.enum import BarrierType, OptionType

        product = BarrierOption(
            strike=100.0, option_type=OptionType.CALL,
            barrier=110.0, barrier_type=BarrierType.UP_OUT,
            maturity=1.0, rebate=2.0, pay_at_hit=True,
        )
        tracker = self._tracker(product, quantity=3.0)
        env = make_env(spot=112.0)

        assert tracker.observe(START + pd.Timedelta(days=1), env, 108.0) == []

        events = tracker.observe(START + pd.Timedelta(days=2), env, 112.0)
        assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]
        assert events[0].terminates_position
        assert almost_equal(events[0].cashflow, 3.0 * 2.0 * product.contract_multiplier)
        assert tracker.state.knocked_out and not tracker.state.alive

    def test_up_out_ko_pay_at_expiry_rebate_is_discounted(self):
        from quantark.asset.equity.lifecycle import LifecycleEventType
        from quantark.asset.equity.product.option import BarrierOption
        from quantark.util.enum import BarrierType, OptionType
        from quantark.util.numerical import safe_exp

        product = BarrierOption(
            strike=100.0, option_type=OptionType.CALL,
            barrier=110.0, barrier_type=BarrierType.UP_OUT,
            maturity=1.0, rebate=2.0, pay_at_hit=False,
        )
        tracker = self._tracker(product)
        env = make_env(spot=112.0, rate=0.03)
        hit_date = START + pd.Timedelta(days=10)

        events = tracker.observe(hit_date, env, 112.0)
        assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]
        remaining = 1.0 - 10.0 / 365.0
        expected = 2.0 * product.contract_multiplier * safe_exp(
            -env.get_rate(remaining) * remaining
        )
        assert almost_equal(events[0].cashflow, expected)

    def test_down_in_put_knock_in_substitutes_european(self):
        from quantark.asset.equity.lifecycle import LifecycleEventType
        from quantark.asset.equity.product.option import (
            BarrierOption,
            EuropeanVanillaOption,
        )
        from quantark.util.enum import BarrierType, OptionType

        product = BarrierOption(
            strike=100.0, option_type=OptionType.PUT,
            barrier=90.0, barrier_type=BarrierType.DOWN_IN,
            maturity=1.0,
        )
        tracker = self._tracker(product)
        env = make_env(spot=88.0)
        date = START + pd.Timedelta(days=10)

        events = tracker.observe(date, env, 88.0)
        assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_IN]
        assert not events[0].terminates_position
        assert tracker.state.knocked_in and tracker.state.alive

        priced = tracker.product_for_pricing(date, env)
        assert isinstance(priced, EuropeanVanillaOption)
        assert is_close(priced.strike, 100.0)
        assert is_close(priced.maturity, 1.0 - 10.0 / 365.0)
        assert tracker.engine_for_pricing() is not None

    def test_knock_in_not_hit_expiry_pays_rebate(self):
        from quantark.asset.equity.lifecycle import LifecycleEventType
        from quantark.asset.equity.product.option import BarrierOption
        from quantark.util.enum import BarrierType, OptionType

        product = BarrierOption(
            strike=100.0, option_type=OptionType.PUT,
            barrier=90.0, barrier_type=BarrierType.DOWN_IN,
            maturity=10.0 / 365.0, rebate=1.5,
        )
        tracker = self._tracker(product, quantity=2.0)
        env = make_env(spot=95.0)

        events = tracker.observe(START + pd.Timedelta(days=10), env, 95.0)
        assert [e.event_type for e in events] == [LifecycleEventType.EXPIRY]
        assert events[0].terminates_position
        assert almost_equal(events[0].cashflow, 2.0 * 1.5 * product.contract_multiplier)

    def test_one_touch_hit_pays_rebate_and_terminates(self):
        from quantark.asset.equity.lifecycle import LifecycleEventType
        from quantark.asset.equity.product.option.one_touch_option import (
            OneTouchOption,
        )
        from quantark.util.enum import BarrierDirection

        product = OneTouchOption(
            barrier=105.0, barrier_direction=BarrierDirection.UP,
            maturity=1.0, rebate=1.0, payment_at_hit=True,
        )
        tracker = self._tracker(product, quantity=10.0)
        env = make_env(spot=106.0)

        events = tracker.observe(START + pd.Timedelta(days=5), env, 106.0)
        assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]
        assert events[0].terminates_position
        assert almost_equal(events[0].cashflow, 10.0 * product.get_payoff(106.0, touched=True))

    def test_exercise_date_product_not_immediately_expired(self):
        from quantark.asset.equity.product.option import BarrierOption
        from quantark.util.enum import BarrierType, OptionType

        product = BarrierOption(
            strike=100.0, option_type=OptionType.PUT,
            barrier=90.0, barrier_type=BarrierType.DOWN_IN,
            exercise_date=(START + pd.Timedelta(days=180)).to_pydatetime(),
        )
        tracker = self._tracker(product)
        env = make_env(spot=95.0)

        # well before exercise_date, no barrier hit -> no events at all
        assert tracker.observe(START + pd.Timedelta(days=5), env, 95.0) == []
        assert tracker.state.alive

        # on/after exercise_date -> EXPIRY settles
        events = tracker.observe(START + pd.Timedelta(days=180), env, 95.0)
        assert [e.event_type.value for e in events] == ["EXPIRY"]

    def test_sharkfin_ko_pays_knock_out_rebate(self):
        from quantark.asset.equity.lifecycle import LifecycleEventType
        from quantark.asset.equity.product.option.single_sharkfin_option import (
            SingleSharkfinOption,
        )
        from quantark.util.enum import OptionType
        from quantark.util.numerical import safe_exp

        product = SingleSharkfinOption(
            strike=100.0, option_type=OptionType.CALL, barrier=110.0,
            maturity=1.0, knock_out_rebate=0.5,
            # pay_at_hit defaults to False => discount applies
        )
        tracker = self._tracker(product)
        env = make_env(spot=111.0, rate=0.03)
        hit_date = START + pd.Timedelta(days=5)

        events = tracker.observe(hit_date, env, 111.0)
        assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]
        assert events[0].terminates_position
        # pay_at_hit=False => PV-discount the barrier payoff
        remaining = 1.0 - 5.0 / 365.0
        expected = product.get_barrier_payoff() * safe_exp(-env.get_rate(remaining) * remaining)
        assert almost_equal(events[0].cashflow, expected)
