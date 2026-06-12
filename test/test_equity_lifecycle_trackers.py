"""Tests for the shared equity lifecycle core (state, events, trackers)."""

from datetime import datetime

import pandas as pd
import pytest

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

    def test_backtest_state_is_reexport(self):
        from quantark.asset.equity.lifecycle import AutocallableLifecycleState as Shared
        from quantark.backtest.otc.state import AutocallableLifecycleState as Legacy

        assert Shared is Legacy
