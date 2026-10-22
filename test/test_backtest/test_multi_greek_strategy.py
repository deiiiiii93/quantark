"""
Unit tests for multi-Greek hedging strategies and the shared frequency gate.
"""

from datetime import datetime

import pytest

from quantark.backtest.strategy.base_strategy import passes_frequency_gate
from quantark.backtest.strategy.hedge_instruments import (
    OptionHedgeInstrument,
    SpotHedgeInstrument,
)
from quantark.backtest.strategy.hedge_optimizer import HedgeTarget
from quantark.backtest.strategy.multi_greek_strategy import (
    DeltaGammaNeutralStrategy,
    DeltaGammaVegaNeutralStrategy,
    DeltaVegaNeutralStrategy,
    MultiGreekHedgeStrategy,
)
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 1, 5, 10, 0)
SAME_DAY = datetime(2026, 1, 5, 15, 0)
NEXT_DAY = datetime(2026, 1, 6, 10, 0)


class TestFrequencyGate:
    def test_no_breach_never_hedges(self):
        assert not passes_frequency_gate("continuous", False, T0, None)
        assert not passes_frequency_gate("daily", False, T0, None)

    def test_continuous_and_threshold_hedge_on_breach(self):
        assert passes_frequency_gate("continuous", True, T0, T0)
        assert passes_frequency_gate("on_threshold", True, T0, T0)

    def test_daily_once_per_day(self):
        assert passes_frequency_gate("daily", True, T0, None)
        assert not passes_frequency_gate("daily", True, SAME_DAY, T0)
        assert passes_frequency_gate("daily", True, NEXT_DAY, T0)

    def test_hourly_once_per_hour(self):
        in_same_hour = datetime(2026, 1, 5, 10, 45)
        next_hour = datetime(2026, 1, 5, 11, 0)
        assert not passes_frequency_gate("hourly", True, in_same_hour, T0)
        assert passes_frequency_gate("hourly", True, next_hour, T0)


def _make_strategy(**kwargs):
    defaults = dict(
        name="Test",
        targets=[
            HedgeTarget("delta", threshold=100.0),
            HedgeTarget("gamma", threshold=5.0),
        ],
        hedge_instruments=[
            OptionHedgeInstrument(tenor=0.25, name="gamma_option"),
            SpotHedgeInstrument(),
        ],
        rebalance_frequency="continuous",
    )
    defaults.update(kwargs)
    return MultiGreekHedgeStrategy(**defaults)


class TestMultiGreekHedgeStrategy:
    def test_should_hedge_any_target_breach(self):
        strategy = _make_strategy()
        # Neither breached
        assert not strategy.should_hedge(T0, {"delta": 50.0, "gamma": 2.0}, {})
        # Only gamma breached
        assert strategy.should_hedge(T0, {"delta": 50.0, "gamma": 8.0}, {})
        # Only delta breached
        assert strategy.should_hedge(T0, {"delta": 150.0, "gamma": 2.0}, {})

    def test_quantities_neutralize_greeks(self):
        strategy = _make_strategy()
        portfolio = {"delta": -40.0, "gamma": -3.0}
        instrument_greeks = {
            "gamma_option": {"delta": 0.5, "gamma": 0.1},
            "spot": {"delta": 1.0, "gamma": 0.0},
        }
        quantities = strategy.calculate_hedge_quantities(
            T0, portfolio, {}, instrument_greeks
        )
        assert quantities["gamma_option"] == pytest.approx(30.0)
        assert quantities["spot"] == pytest.approx(25.0)

    def test_calculate_hedge_size_unsupported(self):
        with pytest.raises(NotImplementedError):
            _make_strategy().calculate_hedge_size(T0, {}, {})

    def test_delta_threshold_property(self):
        assert _make_strategy().delta_threshold == 100.0

    def test_daily_gate_after_execution(self):
        strategy = _make_strategy(rebalance_frequency="daily")
        breached = {"delta": 200.0, "gamma": 0.0}
        assert strategy.should_hedge(T0, breached, {})
        strategy.on_hedges_executed(T0, {"spot": -200.0})
        assert not strategy.should_hedge(SAME_DAY, breached, {})
        assert strategy.should_hedge(NEXT_DAY, breached, {})

    def test_reset_clears_state(self):
        strategy = _make_strategy(rebalance_frequency="daily")
        strategy.on_hedges_executed(T0, {"spot": -1.0})
        strategy.reset()
        assert strategy.get_statistics()["hedge_count"] == 0
        assert strategy.should_hedge(SAME_DAY, {"delta": 200.0, "gamma": 0.0}, {})

    def test_validation(self):
        with pytest.raises(ValidationError):
            _make_strategy(targets=[])
        with pytest.raises(ValidationError):
            _make_strategy(hedge_instruments=[])
        with pytest.raises(ValidationError):
            _make_strategy(
                targets=[HedgeTarget("delta"), HedgeTarget("delta")]
            )
        with pytest.raises(ValidationError):
            _make_strategy(
                hedge_instruments=[SpotHedgeInstrument(), SpotHedgeInstrument()]
            )
        with pytest.raises(ValidationError):
            _make_strategy(rebalance_frequency="weekly")


class TestConcreteStrategies:
    def test_delta_gamma_defaults(self):
        strategy = DeltaGammaNeutralStrategy()
        assert [t.greek for t in strategy.targets] == ["delta", "gamma"]
        assert [i.name for i in strategy.hedge_instruments] == [
            "gamma_option",
            "spot",
        ]

    def test_delta_vega_defaults(self):
        strategy = DeltaVegaNeutralStrategy()
        assert [t.greek for t in strategy.targets] == ["delta", "vega"]
        assert strategy.hedge_instruments[0].tenor == 1.0

    def test_delta_gamma_vega_defaults(self):
        strategy = DeltaGammaVegaNeutralStrategy()
        assert [t.greek for t in strategy.targets] == ["delta", "gamma", "vega"]
        assert [i.name for i in strategy.hedge_instruments] == [
            "gamma_option",
            "vega_option",
            "spot",
        ]
        # Tenor separation keeps the 3x3 system well conditioned
        assert strategy.hedge_instruments[0].tenor < strategy.hedge_instruments[1].tenor

    def test_delta_gamma_vega_duplicate_instrument_names_raise(self):
        same_name = OptionHedgeInstrument(tenor=0.25, name="opt")
        other = OptionHedgeInstrument(tenor=1.0, name="opt")
        with pytest.raises(ValidationError):
            DeltaGammaVegaNeutralStrategy(
                gamma_instrument=same_name, vega_instrument=other
            )

    def test_get_parameters_serializable(self):
        params = DeltaGammaVegaNeutralStrategy().get_parameters()
        assert len(params["targets"]) == 3
        assert len(params["hedge_instruments"]) == 3
