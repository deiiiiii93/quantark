"""Lifecycle event handling in the dynamic scenario engine."""

from datetime import datetime

import pandas as pd
import pytest

from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.numerical import almost_equal, is_close, is_zero

VAL_DATE = datetime(2026, 1, 5)


def make_env(spot=100.0, vol=0.20, rate=0.03):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        valuation_date=VAL_DATE,
    )


def make_snowball(maturity=1.0, ko_barrier=103.0, ki_barrier=70.0):
    from quantark.asset.equity.product.option import SnowballOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig

    return SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=ko_barrier, ko_rate=0.15,
            ko_observation_dates=[i / 12.0 for i in range(1, 13)],
            ki_barrier=ki_barrier, ki_continuous=True,
        ),
        maturity=maturity, contract_multiplier=1.0,
    )


def make_portfolio(product, engine, spot=100.0):
    from quantark.portfolio import Portfolio

    env = make_env(spot=spot)
    portfolio = Portfolio(
        portfolio_name="lifecycle-test",
        pricing_environments={"IDX": env},
    )
    portfolio.add_position(
        product=product, quantity=1.0, entry_price=0.0,
        underlying="IDX", engine=engine,
    )
    return portfolio


class TestLifecycleManager:
    def test_register_attaches_trackers_by_product_type(self):
        from quantark.asset.equity.engine.quad import SnowballQuadEngine
        from quantark.dynamicscenario.lifecycle_manager import LifecycleManager

        portfolio = make_portfolio(make_snowball(), SnowballQuadEngine())
        manager = LifecycleManager(base_date=VAL_DATE)
        manager.register_positions(portfolio)
        assert manager.num_tracked == 1

    def test_ko_reset_snowball_warns_and_is_untracked(self):
        from quantark.asset.equity.engine.quad import SnowballQuadEngine
        from quantark.asset.equity.product.option.ko_reset_snowball_option import (
            KnockOutResetSnowballOption,
        )
        from quantark.asset.equity.product.option.snowball_config import BarrierConfig
        from quantark.dynamicscenario.lifecycle_manager import LifecycleManager

        snowball = make_snowball()
        # post_barrier_config is required; provide a minimal valid one
        post_config = BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15,
            ko_observation_dates=[i / 12.0 for i in range(1, 13)],
        )
        ko_reset = KnockOutResetSnowballOption(
            initial_price=100.0, strike=100.0,
            barrier_config=snowball.barrier_config,
            post_barrier_config=post_config,
            maturity=1.0, contract_multiplier=1.0,
        )
        portfolio = make_portfolio(ko_reset, SnowballQuadEngine())
        manager = LifecycleManager(base_date=VAL_DATE)
        with pytest.warns(UserWarning, match="KO-reset"):
            manager.register_positions(portfolio)
        assert manager.num_tracked == 0

    def test_process_day_settles_snowball_ko_to_cash(self):
        from quantark.asset.equity.engine.quad import SnowballQuadEngine
        from quantark.dynamicscenario.lifecycle_manager import LifecycleManager

        portfolio = make_portfolio(make_snowball(), SnowballQuadEngine(), spot=105.0)
        manager = LifecycleManager(base_date=VAL_DATE)
        manager.register_positions(portfolio)

        # Day 5: above KO barrier but before first observation -> no event
        snapshots = manager.process_day(portfolio, day_index=5, day_date=None)
        assert snapshots == []
        assert len(portfolio) == 1

        # Day 30: first monthly KO observation date
        ko_day = int(round(365 / 12))
        snapshots = manager.process_day(portfolio, day_index=ko_day, day_date=None)
        assert len(snapshots) == 1
        assert snapshots[0].event_type == "KO"
        assert snapshots[0].terminates_position
        assert len(portfolio) == 0
        assert manager.realized_cash > 0.0
        assert almost_equal(manager.realized_cash, snapshots[0].cashflow)

    def test_process_day_sets_ki_flag_and_decays_product(self):
        from quantark.asset.equity.engine.quad import SnowballQuadEngine
        from quantark.dynamicscenario.lifecycle_manager import LifecycleManager

        portfolio = make_portfolio(make_snowball(), SnowballQuadEngine(), spot=65.0)
        manager = LifecycleManager(base_date=VAL_DATE)
        manager.register_positions(portfolio)

        snapshots = manager.process_day(portfolio, day_index=10, day_date=None)
        assert [s.event_type for s in snapshots] == ["KI"]
        assert len(portfolio) == 1
        assert is_zero(manager.realized_cash)

        position = next(iter(portfolio.positions.values()))
        assert getattr(position.product, "_otc_lifecycle_knocked_in") is True
        assert is_close(position.product.maturity, 1.0 - 10.0 / 365.0)


class TestEngineLifecycleIntegration:
    def _run(self, product, engine, path, spot=100.0, handle=True):
        from quantark.dynamicscenario import (
            DynamicScenarioConfig,
            DynamicScenarioEngine,
        )

        portfolio = make_portfolio(product, engine, spot=spot)
        config = DynamicScenarioConfig(
            calculate_greeks=False, handle_lifecycle_events=handle
        )
        scenario_engine = DynamicScenarioEngine(config)
        return scenario_engine.run(portfolio, path)

    def test_snowball_ko_mid_path_settles_to_cash(self):
        from quantark.asset.equity.engine.quad import SnowballQuadEngine
        from quantark.dynamicscenario import PathLibrary

        path = PathLibrary.consecutive_rally(days=35, daily_pct=0.02)
        path.start_date = VAL_DATE
        results = self._run(make_snowball(), SnowballQuadEngine(), path)

        ko_day = int(round(365 / 12))
        ko_result = results.day_results[ko_day]
        assert [e.event_type for e in ko_result.lifecycle_events] == ["KO"]
        assert ko_result.realized_cash > 0.0
        # From the KO day onward the portfolio is pure cash
        for day in results.day_results[ko_day:]:
            assert len(day.positions) == 0
            assert is_close(day.portfolio_value, ko_result.realized_cash)
        assert is_close(results.final_value, ko_result.realized_cash)

        events_df = results.get_lifecycle_events()
        assert len(events_df) == 1
        assert events_df.iloc[0]["event_type"] == "KO"

    def test_snowball_ki_changes_subsequent_pricing(self):
        from copy import deepcopy

        from quantark.asset.equity.engine.quad import SnowballQuadEngine
        from quantark.dynamicscenario import PathLibrary

        engine = SnowballQuadEngine()
        # daily_pct=-0.04 applies -4% per day (PERCENTAGE stress: spot*(1-0.04));
        # vol_change_pct=0.0 keeps vol flat so the expected env is easy to construct.
        path = PathLibrary.consecutive_decline(days=10, daily_pct=-0.04, vol_change_pct=0.0)
        path.start_date = VAL_DATE
        results = self._run(make_snowball(), engine, path)

        # spot on day k close = 100 * 0.96^(k+1); first close <= 70 is day index 8
        ki_day = 8
        assert [e.event_type for e in results.day_results[ki_day].lifecycle_events] == ["KI"]

        # Final-day value equals directly pricing the decayed, flagged product.
        # We use AutocallableLifecycleTracker.product_for_pricing to construct the
        # exact product the engine sees (including barrier-schedule time-shift).
        import pandas as pd
        from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
        from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState

        final_day = results.day_results[-1]
        assert len(final_day.positions) == 1

        start_ts = pd.Timestamp(VAL_DATE).normalize()
        final_ts = start_ts + pd.Timedelta(days=9)
        expected_env = make_env(spot=100.0 * 0.96 ** 10)
        expected_env.valuation_date = final_day.date or VAL_DATE

        tracker = AutocallableLifecycleTracker(
            product=make_snowball(),
            quantity=1.0,
            lifecycle=AutocallableLifecycleState(knocked_in=True),
            start_date=start_ts,
        )
        expected_product = tracker.product_for_pricing(final_ts, expected_env)
        expected_value = engine.price(expected_product, expected_env)
        assert is_close(
            final_day.positions[0].market_value, expected_value, rel_tol=1e-6
        )

    def test_lifecycle_disabled_reproduces_inert_behavior(self):
        from quantark.asset.equity.engine.quad import SnowballQuadEngine
        from quantark.dynamicscenario import PathLibrary

        path = PathLibrary.consecutive_rally(days=35, daily_pct=0.02)
        path.start_date = VAL_DATE
        product = make_snowball()
        results = self._run(product, SnowballQuadEngine(), path, handle=False)

        for day in results.day_results:
            assert day.lifecycle_events == []
            assert is_zero(day.realized_cash)
        assert len(results.day_results[-1].positions) == 1
        # the original product object was never mutated
        assert not hasattr(product, "_otc_lifecycle_knocked_in")
        assert is_close(product.maturity, 1.0)


class TestConfigAndResults:
    def test_config_flag_defaults_on(self):
        from quantark.dynamicscenario import DynamicScenarioConfig

        config = DynamicScenarioConfig()
        assert config.handle_lifecycle_events is True
        assert config.get_summary()["handle_lifecycle_events"] is True

    def test_day_result_lifecycle_fields_default_empty(self):
        from quantark.dynamicscenario.results.dynamic_results import DayResult

        day = DayResult(day_index=0)
        assert day.lifecycle_events == []
        assert day.realized_cash == 0.0

    def test_lifecycle_event_snapshot_roundtrip(self):
        from quantark.dynamicscenario.results.dynamic_results import (
            LifecycleEventSnapshot,
        )

        snap = LifecycleEventSnapshot(
            position_id="p1", underlying="IDX", product_type="SnowballOption",
            event_type="KO", date=datetime(2026, 2, 4), observation_index=0,
            spot=105.0, barrier=103.0, payoff=1.25, cashflow=2.5,
            terminates_position=True,
        )
        data = snap.to_dict()
        assert data["event_type"] == "KO"
        assert data["position_id"] == "p1"
        assert data["terminates_position"] is True

    def test_get_lifecycle_events_dataframe_columns(self):
        from quantark.dynamicscenario.results.dynamic_results import (
            DayResult,
            DynamicScenarioResults,
            LifecycleEventSnapshot,
        )

        snap = LifecycleEventSnapshot(
            position_id="p1", underlying="IDX", product_type="SnowballOption",
            event_type="KO", date=datetime(2026, 2, 4), observation_index=0,
            spot=105.0, barrier=103.0, payoff=1.25, cashflow=2.5,
            terminates_position=True,
        )
        day = DayResult(day_index=3, lifecycle_events=[snap])
        results = DynamicScenarioResults(
            path_name="t", baseline_value=0.0, final_value=0.0,
            day_results=[day],
        )
        df = results.get_lifecycle_events()
        assert list(df.columns)[:2] == ["day_index", "date"]
        assert "event_date" in df.columns
        assert df.iloc[0]["day_index"] == 3


class TestBarrierFamilyIntegration:
    def test_phoenix_coupon_books_cash_position_survives(self):
        from quantark.asset.equity.engine.quad import PhoenixQuadEngine
        from quantark.asset.equity.product.option.phoenix_helpers import (
            create_standard_phoenix,
        )
        from quantark.dynamicscenario import PathBuilder

        phoenix = create_standard_phoenix(
            initial_price=100.0, strike=100.0, maturity=1.0,
            ko_barrier=103.0, ki_barrier=70.0,
            coupon_barrier=85.0, coupon_rate=0.01, num_observations=12,
        )
        # 32-day flat path: no spot/vol/rate changes — spot stays at 100.0
        path = PathBuilder(num_days=32, name="Flat").build()
        path.start_date = VAL_DATE

        runner = TestEngineLifecycleIntegration()
        results = runner._run(phoenix, PhoenixQuadEngine(), path)

        # First observation is at t = 1/12 year ≈ day 30 (round(365/12))
        coupon_day = int(round(365 / 12))
        events = results.day_results[coupon_day].lifecycle_events
        assert [e.event_type for e in events] == ["COUPON"]
        assert not events[0].terminates_position
        # get_coupon_payoff(0) with no dates → dcf=1.0, so payoff = 100*1.0*0.01*1.0 = 1.0
        expected_coupon = float(phoenix.get_coupon_payoff(0))
        assert almost_equal(events[0].cashflow, expected_coupon)
        # Position must still be alive after coupon
        final_day = results.day_results[-1]
        assert len(final_day.positions) == 1
        # realized_cash accumulates the coupon cashflow (no KO in this path)
        assert is_close(final_day.realized_cash, expected_coupon)

    def test_barrier_ki_substitution_prices_as_european(self):
        from quantark.asset.equity.engine.analytical import (
            BarrierAnalyticalEngine,
            BlackScholesEngine,
        )
        from quantark.asset.equity.product.option import (
            BarrierOption,
            EuropeanVanillaOption,
        )
        from quantark.dynamicscenario import PathLibrary
        from quantark.util.enum import BarrierType, OptionType

        product = BarrierOption(
            strike=100.0, option_type=OptionType.PUT,
            barrier=90.0, barrier_type=BarrierType.DOWN_IN, maturity=1.0,
        )
        # consecutive_decline requires NEGATIVE daily_pct to produce a declining spot;
        # vol_change_pct=0.0 keeps vol flat so the BlackScholes comparison is exact.
        # spot on day k = 100 * 0.97^(k+1); first close <= 90 is day index 3 (100*0.97^4 ≈ 88.53)
        path = PathLibrary.consecutive_decline(days=6, daily_pct=-0.03, vol_change_pct=0.0)
        path.start_date = VAL_DATE

        runner = TestEngineLifecycleIntegration()
        results = runner._run(product, BarrierAnalyticalEngine(), path)

        ki_day = 3
        events = results.day_results[ki_day].lifecycle_events
        assert [e.event_type for e in events] == ["KI"]

        # After KI the lifecycle manager replaces position.product with EuropeanVanillaOption
        # and switches the engine to BlackScholesEngine.
        snapshot = results.day_results[ki_day].positions[0]
        assert snapshot.product_type == "EuropeanVanillaOption"

        # The tracker computes remaining maturity as 1.0 - elapsed(3 days) = 1.0 - 3/365.
        # The pricing env on day 3 has spot = 100 * 0.97^4 (day-3 close, i.e. 4th multiplicative
        # factor), vol unchanged at 0.20, rate at 0.03.
        ki_spot = 100.0 * (0.97 ** 4)
        remaining_maturity = 1.0 - 3.0 / 365.0
        expected_env = make_env(spot=ki_spot)
        ki_date = VAL_DATE + pd.Timedelta(days=ki_day)
        expected_env.valuation_date = ki_date.to_pydatetime() if hasattr(ki_date, "to_pydatetime") else ki_date
        expected = BlackScholesEngine().price(
            EuropeanVanillaOption(
                strike=100.0, option_type=OptionType.PUT,
                maturity=remaining_maturity,
            ),
            expected_env,
        )
        # snapshot.market_value = BlackScholesEngine().price(European, env) * quantity (=1.0)
        assert is_close(snapshot.market_value, expected, rel_tol=1e-6)

    def test_barrier_ko_rebate_settles_at_hit(self):
        from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
        from quantark.asset.equity.product.option import BarrierOption
        from quantark.dynamicscenario import PathLibrary
        from quantark.util.enum import BarrierType, OptionType

        product = BarrierOption(
            strike=100.0, option_type=OptionType.CALL,
            barrier=110.0, barrier_type=BarrierType.UP_OUT,
            maturity=1.0, rebate=2.0, pay_at_hit=True,
        )
        # vol_change_pct=0.0 keeps vol flat; spot on day k = 100 * 1.03^(k+1)
        # first close >= 110 is day index 3 (100*1.03^4 ≈ 112.55)
        path = PathLibrary.consecutive_rally(days=6, daily_pct=0.03, vol_change_pct=0.0)
        path.start_date = VAL_DATE

        runner = TestEngineLifecycleIntegration()
        results = runner._run(product, BarrierAnalyticalEngine(), path)

        ko_day = 3
        events = results.day_results[ko_day].lifecycle_events
        assert [e.event_type for e in events] == ["KO"]
        # rebate * contract_multiplier (=1.0) = 2.0; cashflow = quantity(1.0) * payoff(2.0)
        assert almost_equal(
            results.day_results[ko_day].realized_cash,
            2.0 * product.contract_multiplier,
        )
        # From KO day onward: no live positions; portfolio_value = realized_cash
        for day in results.day_results[ko_day:]:
            assert len(day.positions) == 0
            assert is_close(day.portfolio_value, day.realized_cash)

    def test_one_touch_hit_terminates_with_rebate(self):
        from quantark.asset.equity.engine.analytical import OneTouchAnalyticalEngine
        from quantark.asset.equity.product.option.one_touch_option import (
            OneTouchOption,
        )
        from quantark.dynamicscenario import PathLibrary
        from quantark.util.enum import BarrierDirection

        product = OneTouchOption(
            barrier=105.0, barrier_direction=BarrierDirection.UP,
            maturity=1.0, rebate=1.0, payment_at_hit=True,
        )
        # vol_change_pct=0.0 keeps vol flat; spot on day k = 100 * 1.02^(k+1)
        # first close >= 105 is day index 2 (100*1.02^3 ≈ 106.12)
        path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02, vol_change_pct=0.0)
        path.start_date = VAL_DATE

        runner = TestEngineLifecycleIntegration()
        results = runner._run(product, OneTouchAnalyticalEngine(), path)

        # spot on day 2 close = 100 * 1.02^3 ≈ 106.12 >= 105
        touch_day = 2
        events = results.day_results[touch_day].lifecycle_events
        assert [e.event_type for e in events] == ["KO"]
        # rebate=1.0, payment_at_hit=True → cashflow = 1.0 * 1.0 (quantity) = 1.0
        assert almost_equal(results.day_results[touch_day].realized_cash, 1.0)
        assert len(results.day_results[-1].positions) == 0
