"""``TradingClockConfig`` and the config-time refusals of the replay business-day clock (run 7, D1 / D7, AC6).

``AutocallableEngineConfig.trading_clock`` (default ``None``: today's calendar-day replay, unchanged) asks the replay
to price every day on a BUSINESS_DAYS clock. The features whose day times are still ACT/365 (vol models, trade
value, PnL explain, the simulated-path ensembles) refuse it at construction instead of mixing clocks.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from golden.replay_golden import fixtures  # noqa: E402

from quantark.backtest.replay import (  # noqa: E402
    AutocallableBacktestConfig,
    AutocallableEngineConfig,
    HedgeSpec,
    ReplayBacktestConfig,
    ReplayProduct,
)
from quantark.backtest.transaction_costs import ZeroCostModel  # noqa: E402
from quantark.pnlexplain.config import PnLExplainConfig  # noqa: E402
from quantark.priceenv import CurveQuotingBasis  # noqa: E402
from quantark.util.calendar import BusinessDayConvention, CalendarType, create_calendar  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2023, 2026))


def _clock(**kw):
    from quantark.backtest.replay import TradingClockConfig

    return TradingClockConfig(calendar=CAL, **kw)


# --------------------------------------------------------------------------- D1: the config object


def test_trading_clock_config_is_exported_from_both_packages():
    import quantark.backtest as bt
    import quantark.backtest.replay as rp

    assert bt.TradingClockConfig is rp.TradingClockConfig
    assert "TradingClockConfig" in bt.__all__ and "TradingClockConfig" in rp.__all__


def test_trading_clock_config_defaults_and_is_frozen():
    clock = _clock()
    assert clock.calendar is CAL
    assert clock.bus_days_in_year == 252
    assert clock.holiday_time_adjustment is BusinessDayConvention.PRECEDING
    assert clock.vol_quoting is CurveQuotingBasis.CALENDAR_YEAR
    with pytest.raises(dataclasses.FrozenInstanceError):
        clock.bus_days_in_year = 244


def test_trading_clock_config_requires_a_calendar():
    from quantark.backtest.replay import TradingClockConfig

    with pytest.raises(TypeError):
        TradingClockConfig()  # calendar has no default
    with pytest.raises(ValidationError, match="calendar"):
        TradingClockConfig(calendar=None)
    with pytest.raises(ValidationError, match="calendar"):
        TradingClockConfig(calendar="CHINA_SSE")


@pytest.mark.parametrize("days", [0, -1, 244.5, True])
def test_trading_clock_config_refuses_a_bad_denominator(days):
    with pytest.raises(ValidationError, match="bus_days_in_year"):
        _clock(bus_days_in_year=days)


def test_trading_clock_config_refuses_unadjusted_and_non_conventions():
    with pytest.raises(ValidationError, match="holiday_time_adjustment"):
        _clock(holiday_time_adjustment=BusinessDayConvention.UNADJUSTED)
    with pytest.raises(ValidationError, match="holiday_time_adjustment"):
        _clock(holiday_time_adjustment="PRECEDING")
    assert _clock(holiday_time_adjustment=BusinessDayConvention.FOLLOWING).holiday_time_adjustment is (
        BusinessDayConvention.FOLLOWING)


def test_trading_clock_config_vol_quoting_is_a_quoting_basis():
    assert _clock(vol_quoting=CurveQuotingBasis.TRADING_YEAR).vol_quoting is CurveQuotingBasis.TRADING_YEAR
    with pytest.raises(ValidationError, match="vol_quoting"):
        _clock(vol_quoting="calendar")


def test_engine_config_trading_clock_defaults_to_none_and_takes_the_config():
    assert AutocallableEngineConfig().trading_clock is None
    clock = _clock(bus_days_in_year=244)
    assert AutocallableEngineConfig(trading_clock=clock).trading_clock is clock
    with pytest.raises(ValidationError, match="trading_clock"):
        AutocallableEngineConfig(trading_clock=CAL)


# --------------------------------------------------------------------------- D7: refusals on the owning configs


@pytest.mark.parametrize("model", ["localvol", "heston", "heston_slv"])
def test_engine_config_refuses_a_trading_clock_with_a_vol_model(model):
    with pytest.raises(ValidationError, match="trading_clock"):
        AutocallableEngineConfig(vol_source="surface", vol_model=model, trading_clock=_clock())


def _clock_engine_config():
    return AutocallableEngineConfig(trading_clock=_clock())


def _single(**kw):
    return AutocallableBacktestConfig(
        product=fixtures._snowball_product(), market_data=fixtures._market_data(),
        engine_config=_clock_engine_config(), transaction_cost_model=ZeroCostModel(),
        calculate_surfaces=False, **kw)


def _book(**kw):
    return ReplayBacktestConfig(
        products=[ReplayProduct(product=fixtures._snowball_product(), quantity=-1.0, position_id=1,
                                has_lifecycle=True)],
        market_data=fixtures._market_data(), hedge=HedgeSpec(kind="futures"),
        engine_config=_clock_engine_config(), transaction_cost_model=ZeroCostModel(), **kw)


def test_the_backtest_configs_accept_a_trading_clock_otherwise():
    assert _single().engine_config.trading_clock is not None
    assert _book().engine_config.trading_clock is not None


def test_single_config_refuses_pnl_explain_on_a_trading_clock():
    with pytest.raises(ValidationError, match="pnl_explain"):
        _single(pnl_explain=PnLExplainConfig())


def test_book_config_refuses_pnl_explain_on_a_trading_clock():
    with pytest.raises(ValidationError, match="pnl_explain"):
        _book(pnl_explain=PnLExplainConfig())


def test_book_config_refuses_trade_value_on_a_trading_clock():
    with pytest.raises(ValidationError, match="trade_value"):
        _book(trade_value=True)


def test_the_single_engine_forwards_the_trading_clock_to_its_book_config():
    from quantark.backtest.replay import AutocallableBacktestEngine

    config = _single()
    engine = AutocallableBacktestEngine.__new__(AutocallableBacktestEngine)
    # build only the forwarded book config: the numeric fixture product itself is refused at ProductReplay
    try:
        AutocallableBacktestEngine.__init__(engine, config)
    except ValidationError:
        pass
    assert engine._book_config.engine_config.trading_clock is config.engine_config.trading_clock


def test_ensemble_config_refuses_an_engine_config_on_a_trading_clock():
    from quantark.backtest.futures_ledger import FuturesRollPolicy
    from quantark.backtest.simulation.config import (
        CacheConfig,
        EnsembleConfig,
        GateConfig,
        PricingProviderConfig,
    )
    from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy

    with pytest.raises(ValidationError, match="trading_clock"):
        EnsembleConfig(
            products=[ReplayProduct(product=fixtures._snowball_product(), quantity=-1.0, position_id=1,
                                    has_lifecycle=True)],
            engine_config=_clock_engine_config(),
            hedge=HedgeSpec(kind="futures", multiplier=200.0, roll_policy=FuturesRollPolicy()),
            strategy=AutocallableDeltaHedgeStrategy(delta_threshold=0.0, hedge_ratio=1.0, target_delta=0.0),
            transaction_cost_model=ZeroCostModel(),
            pricing=PricingProviderConfig(
                provider="repricing", cache=CacheConfig(memory_bytes=8_000_000),
                gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0)),
        )
