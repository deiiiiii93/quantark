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
