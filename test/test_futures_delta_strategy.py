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


# ---------------------------------------------------------------------------
# The separate proportional S/F control
# ---------------------------------------------------------------------------


class TestProportionalControl:
    """``h_j = -D S / (m_j F_j)``: spot neutral, with a known carry residual.

    The legacy sizing above is deliberately untouched; this is a distinct
    policy that isolates the ``S/F`` scaling correction from the effect of
    holding a calendar spread.
    """

    def test_it_is_spot_neutral_where_the_legacy_sizing_is_not(self):
        from quantark.backtest.strategy import (
            ProportionalFuturesDeltaHedgeStrategy,
        )

        spot, futures_price, multiplier = 100.0, 96.0, 200.0
        delta, quantity = 4_000.0, -1.0
        legacy = AutocallableDeltaHedgeStrategy(round_contracts=False)
        scaled = ProportionalFuturesDeltaHedgeStrategy(round_contracts=False)

        legacy_hands = legacy.target_contracts(
            product_delta=delta,
            product_quantity=quantity,
            futures_multiplier=multiplier,
        )
        scaled_hands = scaled.target_contracts(
            product_delta=delta,
            product_quantity=quantity,
            futures_multiplier=multiplier,
            spot=spot,
            futures_price=futures_price,
        )
        book_delta = delta * quantity
        assert legacy_hands == pytest.approx(-book_delta / multiplier)
        assert scaled_hands == pytest.approx(
            -book_delta * spot / (multiplier * futures_price)
        )
        # Residual spot delta: D(1 - F/S) for the legacy sizing, zero here.
        legacy_residual = book_delta + legacy_hands * multiplier * futures_price / spot
        scaled_residual = book_delta + scaled_hands * multiplier * futures_price / spot
        assert legacy_residual == pytest.approx(
            book_delta * (1.0 - futures_price / spot)
        )
        assert legacy_residual != pytest.approx(0.0, abs=1.0)
        assert scaled_residual == pytest.approx(0.0, abs=1e-9)

    def test_the_legacy_sizing_and_signature_are_untouched(self):
        import inspect

        signature = inspect.signature(
            AutocallableDeltaHedgeStrategy().target_contracts
        )
        assert set(signature.parameters) == {
            "product_delta",
            "product_quantity",
            "futures_multiplier",
        }
        assert AutocallableDeltaHedgeStrategy().target_contracts(
            product_delta=40.0, product_quantity=-1.0, futures_multiplier=300.0
        ) == 0.0

    def test_the_inherited_scalar_adapter_requires_both_prices(self):
        from quantark.backtest.strategy import (
            ProportionalFuturesDeltaHedgeStrategy,
        )

        strategy = ProportionalFuturesDeltaHedgeStrategy()
        greeks = {"delta": 1_000.0}
        with pytest.raises(ValidationError):
            strategy.calculate_hedge_size(None, greeks, {"futures_multiplier": 200.0})
        with pytest.raises(ValidationError):
            strategy.calculate_hedge_size(
                None, greeks, {"futures_multiplier": 200.0, "spot": 100.0}
            )
        with pytest.raises(ValidationError):
            strategy.target_contracts(
                product_delta=1.0,
                product_quantity=1.0,
                futures_multiplier=200.0,
                spot=100.0,
                futures_price=0.0,
            )
        assert strategy.calculate_hedge_size(
            None,
            greeks,
            {"futures_multiplier": 200.0, "spot": 100.0, "futures_price": 96.0},
        ) == pytest.approx(round(-1_000.0 * 100.0 / (200.0 * 96.0)))

    def test_it_keeps_the_bands_ratio_and_rounding_of_the_original(self):
        from quantark.backtest.strategy import (
            ProportionalFuturesDeltaHedgeStrategy,
        )

        strategy = ProportionalFuturesDeltaHedgeStrategy(
            delta_threshold=0.5, hedge_ratio=0.5, round_contracts=True
        )
        assert not strategy.should_rebalance(1.0, 1.4)
        assert strategy.should_rebalance(1.0, 1.6)
        assert strategy.target_contracts(
            product_delta=4_000.0,
            product_quantity=-1.0,
            futures_multiplier=200.0,
            spot=100.0,
            futures_price=96.0,
        ) == pytest.approx(round(0.5 * 4_000.0 * 100.0 / (200.0 * 96.0)))
        assert strategy.name == "ProportionalFuturesDeltaHedge"
        assert strategy.get_parameters()["hedge_ratio"] == 0.5
        # The inherited ratio bound is part of the legacy conventions it reuses.
        with pytest.raises(ValidationError):
            ProportionalFuturesDeltaHedgeStrategy(hedge_ratio=1.2)
