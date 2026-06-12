"""
Scenario-based hedging strategy.

Instead of hedging local Greeks, this strategy hedges full-revaluation P&L
under user-defined market scenarios (spot crash, vol spike, rate shock,
joint moves). Greeks are local derivatives; real market moves are large
and joint, so for structured-product books scenario hedging can be more
informative than pure delta/vega hedging.

Here the scenario is the hedging *target*: hedges are sized today so the
book is protected if the move happens (pre-emptive stress hedging, hedge
costs paid up front). For the contingent variant — a realized move acting
as a *trigger* that switches on ordinary Greek hedging — see
TriggeredHedgeStrategy in triggered_hedge_strategy.py.

Structurally, scenario hedging is the same linear problem as multi-Greek
hedging: the sensitivity matrix columns hold per-unit scenario P&L of each
hedge instrument instead of per-unit Greeks, and the targets are zero P&L
in each scenario. The same HedgeOptimizer solves it, with the documented
regimes: with more scenarios than instruments the result is the weighted
least-squares hedge (scenario weights set the priorities); with as many
instruments as scenarios the hedge is exact.
"""

from datetime import timedelta
from typing import Dict, List, Optional

from quantark.backtest.strategy.hedge_instruments import (
    BaseHedgeInstrument,
    OptionHedgeInstrument,
    SpotHedgeInstrument,
)
from quantark.backtest.strategy.hedge_optimizer import HedgeTarget
from quantark.backtest.strategy.multi_greek_strategy import MultiGreekHedgeStrategy
from quantark.backtest.strategy.scenarios import (
    MarketScenario,
    instrument_scenario_pnl,
    portfolio_scenario_pnl,
)
from quantark.util.exceptions import ValidationError


class ScenarioHedgeStrategy(MultiGreekHedgeStrategy):
    """
    Hedges portfolio P&L under specified market scenarios.

    Each scenario becomes a hedge target keyed by the scenario's name; the
    portfolio-level measure is full-revaluation P&L under that scenario,
    and each instrument's per-unit measure is its own scenario P&L.

    Attributes:
        scenarios: Scenarios under control
        pnl_threshold: Default absolute scenario loss that triggers a hedge
    """

    def __init__(
        self,
        scenarios: List[MarketScenario],
        name: str = "ScenarioHedge",
        pnl_threshold: float = 0.0,
        thresholds: Optional[Dict[str, float]] = None,
        hedge_instruments: Optional[List[BaseHedgeInstrument]] = None,
        rebalance_frequency: str = "daily",
        min_time_between_hedges: Optional[timedelta] = None,
        instrument_costs: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize scenario-based hedging strategy.

        Args:
            scenarios: Market scenarios to hedge (unique names)
            name: Strategy name
            pnl_threshold: Absolute scenario P&L deviation that triggers a
                hedge (applies to every scenario unless overridden)
            thresholds: Optional per-scenario threshold overrides keyed by
                scenario name
            hedge_instruments: Hedge instrument specs; defaults to a 3M ATM
                option, a 1Y ATM option and spot
            rebalance_frequency: 'daily', 'hourly', 'on_threshold', 'continuous'
            min_time_between_hedges: Minimum time between hedges
            instrument_costs: Optional cost weight per instrument name

        Raises:
            ValidationError: If parameters are invalid
        """
        if not scenarios:
            raise ValidationError("At least one MarketScenario is required")
        scenario_names = [s.name for s in scenarios]
        if len(set(scenario_names)) != len(scenario_names):
            raise ValidationError(f"Duplicate scenario names: {scenario_names}")
        if pnl_threshold < 0:
            raise ValidationError(
                f"pnl_threshold must be non-negative, got {pnl_threshold}"
            )
        thresholds = thresholds or {}
        unknown = set(thresholds) - set(scenario_names)
        if unknown:
            raise ValidationError(
                f"Thresholds reference unknown scenarios: {sorted(unknown)}"
            )

        if hedge_instruments is None:
            hedge_instruments = [
                OptionHedgeInstrument(name="gamma_option", tenor=0.25),
                OptionHedgeInstrument(name="vega_option", tenor=1.0),
                SpotHedgeInstrument(),
            ]

        targets = [
            HedgeTarget(
                greek=scenario.name,
                target=0.0,
                threshold=thresholds.get(scenario.name, pnl_threshold),
                weight=scenario.weight,
            )
            for scenario in scenarios
        ]

        super().__init__(
            name=name,
            targets=targets,
            hedge_instruments=hedge_instruments,
            rebalance_frequency=rebalance_frequency,
            min_time_between_hedges=min_time_between_hedges,
            instrument_costs=instrument_costs,
        )
        self.scenarios = list(scenarios)

    def compute_portfolio_measures(
        self, portfolio, underlying: str, pricing_env
    ) -> Dict[str, float]:
        """Portfolio P&L under each scenario (full revaluation)."""
        return {
            scenario.name: portfolio_scenario_pnl(
                portfolio, underlying, pricing_env, scenario
            )
            for scenario in self.scenarios
        }

    def instrument_measures(
        self,
        instrument: BaseHedgeInstrument,
        product,
        engine,
        pricing_env,
        greeks_calculator,
    ) -> Dict[str, float]:
        """Per-unit P&L of one contract under each scenario."""
        base_price = instrument.unit_price(product, engine, pricing_env)
        return {
            scenario.name: instrument_scenario_pnl(
                product, engine, pricing_env, scenario, base_price=base_price
            )
            for scenario in self.scenarios
        }

    def get_parameters(self) -> Dict[str, object]:
        """Get strategy parameters."""
        params = super().get_parameters()
        params["scenarios"] = [
            {
                "name": s.name,
                "spot_shift": s.spot_shift,
                "vol_shift": s.vol_shift,
                "rate_shift": s.rate_shift,
                "weight": s.weight,
            }
            for s in self.scenarios
        ]
        return params
