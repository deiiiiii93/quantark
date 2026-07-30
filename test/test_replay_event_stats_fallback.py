"""Opt-in event-stats fallback with provenance (plan Task 11)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.asset.equity.engine.base_engine import BaseEngine  # noqa: E402
from quantark.asset.equity.param import EngineParams  # noqa: E402
from quantark.backtest.replay.single import AutocallableBacktestEngine  # noqa: E402
from quantark.util.exceptions import PricingError  # noqa: E402


class NoneStatsEngine(BaseEngine):
    def __init__(self) -> None:
        super().__init__(EngineParams())

    def price(self, product, pricing_env) -> float:
        return 0.0

    def calculate_event_stats(self, product, pricing_env):
        return None


class RaisingStatsEngine(NoneStatsEngine):
    def calculate_event_stats(self, product, pricing_env):
        raise PricingError("synthetic event-stats failure")


def _engine_with_stats_stub(stub, fallback: str) -> AutocallableBacktestEngine:
    config = fixtures.make_scalar_bsm_config()
    config.engine_config.event_stats_fallback = fallback
    engine = AutocallableBacktestEngine(config)
    engine._inner._replays[0].event_stats_engine = stub
    return engine


@pytest.mark.parametrize("stub", [NoneStatsEngine(), RaisingStatsEngine()],
                         ids=["returns_none", "raises"])
def test_default_none_mode_fails_closed(stub):
    engine = _engine_with_stats_stub(stub, "none")
    with pytest.raises(PricingError):
        engine.run()


@pytest.mark.parametrize("stub_cls", [NoneStatsEngine, RaisingStatsEngine],
                         ids=["returns_none", "raises"])
def test_mc_mode_falls_back_with_warning_and_provenance(stub_cls, caplog):
    engine = _engine_with_stats_stub(stub_cls(), "mc")
    with caplog.at_level(logging.WARNING):
        results = engine.run()
    assert any("MC fallback" in rec.message for rec in caplog.records)
    daily = results.daily_event_summary_df
    assert (daily["event_stats_engine"] == "mc_fallback").all()
    probs = results.event_probability_df
    assert (probs["event_stats_engine"] == "mc_fallback").all()


def test_primary_provenance_recorded():
    config = fixtures.make_scalar_bsm_config()
    # QUAD engines implement calculate_event_stats natively for snowballs.
    from quantark.util.enum.engine_enums import EngineType

    config.engine_config.event_stats_fallback = "none"
    config.engine_config.pricing_engine_type = EngineType.QUADRATURE
    config.engine_config.event_stats_engine_type = EngineType.QUADRATURE
    results = AutocallableBacktestEngine(config).run()
    daily = results.daily_event_summary_df
    assert (daily["event_stats_engine"] == "primary").all()


def test_invalid_fallback_rejected():
    from quantark.backtest.replay.config import AutocallableEngineConfig
    from quantark.util.exceptions import ValidationError

    with pytest.raises(ValidationError, match="event_stats_fallback"):
        AutocallableEngineConfig(event_stats_fallback="quietly_guess")
