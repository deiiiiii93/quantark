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

        profile = product.get_ko_observation_profile(make_env(spot=105.0))
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
