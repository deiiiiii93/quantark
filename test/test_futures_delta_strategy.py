"""Tests for AutocallableDeltaHedgeStrategy on the BaseStrategy hierarchy."""
from __future__ import annotations

import pytest

from quantark.backtest.strategy import AutocallableDeltaHedgeStrategy
from quantark.backtest.strategy.base_strategy import (
    AssetClass,
    BaseStrategy,
    HedgingTarget,
)
from quantark.util.exceptions import ValidationError


def test_is_a_base_strategy():
    strategy = AutocallableDeltaHedgeStrategy()
    assert isinstance(strategy, BaseStrategy)
    assert strategy.asset_class is AssetClass.EQUITY
    assert strategy.hedging_target is HedgingTarget.DELTA
    assert strategy.hedge_instrument == "futures"


def test_constructor_kwargs_unchanged():
    strategy = AutocallableDeltaHedgeStrategy(
        delta_threshold=0.5, hedge_ratio=0.8, target_delta=10.0, round_contracts=False
    )
    assert strategy.get_parameters() == {
        "delta_threshold": 0.5,
        "hedge_ratio": 0.8,
        "target_delta": 10.0,
        "round_contracts": False,
    }


def test_target_contracts_math_preserved():
    strategy = AutocallableDeltaHedgeStrategy()
    # net delta = 40 * -1 = -40; target = -(-40 - 0)/300 * 1.0 = 0.1333 -> 0
    assert strategy.target_contracts(
        product_delta=40.0, product_quantity=-1.0, futures_multiplier=300.0
    ) == 0.0
    # bigger position: net delta = -350*40 = -14000 -> 46.67 -> 47 contracts
    assert strategy.target_contracts(
        product_delta=40.0, product_quantity=-350.0, futures_multiplier=300.0
    ) == 47.0
    unrounded = AutocallableDeltaHedgeStrategy(round_contracts=False)
    assert unrounded.target_contracts(
        product_delta=40.0, product_quantity=-350.0, futures_multiplier=300.0
    ) == pytest.approx(14000.0 / 300.0)


def test_should_rebalance_threshold():
    strategy = AutocallableDeltaHedgeStrategy(delta_threshold=2.0)
    assert not strategy.should_rebalance(10.0, 11.0)
    assert strategy.should_rebalance(10.0, 13.0)


def test_protocol_adapters_agree_with_native_api():
    strategy = AutocallableDeltaHedgeStrategy()
    greeks = {"delta": -14000.0}
    market = {"futures_multiplier": 300.0}
    expected = strategy.target_contracts(
        product_delta=-14000.0, product_quantity=1.0, futures_multiplier=300.0
    )
    size = strategy.calculate_hedge_size(
        current_time=None, portfolio_greeks=greeks, market_data=market
    )
    assert size == expected
    assert strategy.should_hedge(
        current_time=None, portfolio_greeks=greeks, market_data=market,
        current_contracts=0.0,
    ) == strategy.should_rebalance(0.0, expected)


def test_protocol_adapters_require_multiplier():
    strategy = AutocallableDeltaHedgeStrategy()
    with pytest.raises(ValidationError):
        strategy.calculate_hedge_size(
            current_time=None, portfolio_greeks={"delta": 1.0}, market_data={}
        )


def test_validation_preserved():
    with pytest.raises(ValidationError):
        AutocallableDeltaHedgeStrategy(delta_threshold=-1.0)
    with pytest.raises(ValidationError):
        AutocallableDeltaHedgeStrategy(hedge_ratio=1.5)
