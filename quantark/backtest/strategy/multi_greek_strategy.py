"""
Multi-Greek hedging strategies for equity derivatives.

These strategies control several Greeks simultaneously by trading several
hedge instruments per rebalance. Hedge quantities come from a joint linear
solve (see hedge_optimizer): each hedge option carries its own delta along
with its gamma/vega, so per-Greek sequential sizing would be wrong.

Standard OTC desk configurations are provided as convenience subclasses:

- DeltaGammaNeutralStrategy: short-dated ATM option (gamma) + spot
- DeltaVegaNeutralStrategy: longer-dated ATM option (vega) + spot
- DeltaGammaVegaNeutralStrategy: short-dated option (gamma) +
  long-dated option (vega) + spot, a 3x3 solve
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from quantark.backtest.strategy.base_strategy import (
    VALID_REBALANCE_FREQUENCIES,
    AssetClass,
    BaseStrategy,
    HedgingTarget,
    passes_frequency_gate,
)
from quantark.backtest.strategy.hedge_instruments import (
    BaseHedgeInstrument,
    OptionHedgeInstrument,
    SpotHedgeInstrument,
)
from quantark.backtest.strategy.hedge_optimizer import HedgeOptimizer, HedgeTarget
from quantark.util.exceptions import ValidationError


class MultiGreekHedgeStrategy(BaseStrategy):
    """
    Generic multi-target, multi-instrument hedging strategy.

    Monitors a list of HedgeTargets and, when any controlled Greek deviates
    from its target by more than its threshold (subject to frequency
    gating), solves for the instrument quantities that restore all targets
    simultaneously.

    Unlike single-instrument strategies, hedge sizing returns a quantity
    per instrument via calculate_hedge_quantities(); the inherited
    calculate_hedge_size() is not supported.

    Attributes:
        targets: Greeks under control with their target levels/thresholds
        hedge_instruments: Instrument specifications available for hedging
        rebalance_frequency: Frequency gating mode
        min_time_between_hedges: Minimum time between hedges
        instrument_costs: Optional cost weights per instrument name, used
            when there are more instruments than targets
    """

    def __init__(
        self,
        name: str,
        targets: List[HedgeTarget],
        hedge_instruments: List[BaseHedgeInstrument],
        rebalance_frequency: str = "daily",
        min_time_between_hedges: Optional[timedelta] = None,
        instrument_costs: Optional[Dict[str, float]] = None,
        max_condition_number: float = 1e12,
    ):
        """
        Initialize multi-Greek hedging strategy.

        Args:
            name: Strategy name
            targets: Greeks to control (unique greek names)
            hedge_instruments: Hedge instrument specs (unique names)
            rebalance_frequency: 'daily', 'hourly', 'on_threshold', 'continuous'
            min_time_between_hedges: Minimum time between hedges
            instrument_costs: Optional cost weight per instrument name
            max_condition_number: Conditioning limit for the hedge solve

        Raises:
            ValidationError: If parameters are invalid
        """
        if not targets:
            raise ValidationError("At least one HedgeTarget is required")
        if not hedge_instruments:
            raise ValidationError("At least one hedge instrument is required")

        greek_names = [t.greek for t in targets]
        if len(set(greek_names)) != len(greek_names):
            raise ValidationError(f"Duplicate greeks in targets: {greek_names}")

        instrument_names = [inst.name for inst in hedge_instruments]
        if len(set(instrument_names)) != len(instrument_names):
            raise ValidationError(
                f"Duplicate hedge instrument names: {instrument_names}"
            )

        if rebalance_frequency not in VALID_REBALANCE_FREQUENCIES:
            raise ValidationError(
                f"Invalid rebalance_frequency '{rebalance_frequency}'. "
                f"Must be one of {VALID_REBALANCE_FREQUENCIES}"
            )

        # Targets keyed by non-Greek measures (e.g. scenario names) fall
        # back to DELTA as the nominal primary for BaseStrategy metadata
        try:
            primary_target = HedgingTarget(targets[0].greek)
        except ValueError:
            primary_target = HedgingTarget.DELTA

        super().__init__(
            name=name,
            asset_class=AssetClass.EQUITY,
            hedging_target=primary_target,
            hedge_instrument="multi",
        )

        self.targets = list(targets)
        self.hedge_instruments = list(hedge_instruments)
        self.rebalance_frequency = rebalance_frequency
        self.min_time_between_hedges = min_time_between_hedges
        self.instrument_costs = instrument_costs
        self._optimizer = HedgeOptimizer(max_condition_number=max_condition_number)

        # Internal state
        self._hedge_count = 0
        self._last_rebalance_date: Optional[datetime] = None

    @property
    def delta_threshold(self) -> float:
        """Threshold of the delta target, if delta is controlled (else 0)."""
        for target in self.targets:
            if target.greek == "delta":
                return target.threshold
        return 0.0

    def get_target_deviations(
        self, portfolio_greeks: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Deviation of each controlled Greek from its target.

        Args:
            portfolio_greeks: Current portfolio Greeks

        Returns:
            Dictionary mapping greek name to (current - target)
        """
        return {
            t.greek: portfolio_greeks.get(t.greek, 0.0) - t.target
            for t in self.targets
        }

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> bool:
        """
        Hedge when any controlled Greek breaches its threshold, subject to
        frequency gating and the minimum time between hedges.
        """
        if self.min_time_between_hedges is not None:
            time_since_hedge = self.time_since_last_hedge(current_time)
            if (
                time_since_hedge is not None
                and time_since_hedge < self.min_time_between_hedges
            ):
                return False

        deviations = self.get_target_deviations(portfolio_greeks)
        breach = any(
            abs(deviations[t.greek]) > t.threshold for t in self.targets
        )

        return passes_frequency_gate(
            frequency=self.rebalance_frequency,
            breach=breach,
            current_time=current_time,
            last_rebalance_time=self._last_rebalance_date,
        )

    def instrument_measures(
        self,
        instrument: BaseHedgeInstrument,
        product,
        engine,
        pricing_env,
        greeks_calculator,
    ) -> Dict[str, float]:
        """
        Per-unit sensitivity measures of one tradeable contract.

        The hedge solve is agnostic to what the measures are: this base
        implementation returns Greeks; subclasses may return other
        full-revaluation measures (e.g. scenario P&L) under the keys their
        targets reference.

        Args:
            instrument: Hedge instrument spec
            product: Concrete tradeable contract
            engine: Pricing engine for the contract
            pricing_env: Current pricing environment
            greeks_calculator: Calculator shared with the backtest engine

        Returns:
            Dictionary mapping measure name to per-unit value
        """
        return instrument.unit_greeks(product, engine, pricing_env, greeks_calculator)

    def compute_portfolio_measures(
        self, portfolio, underlying: str, pricing_env
    ) -> Dict[str, float]:
        """
        Extra portfolio-level measures needed by this strategy's targets.

        Returned values are merged into the portfolio Greeks dict before
        should_hedge / calculate_hedge_quantities are called. The base
        strategy needs nothing beyond the engine-computed Greeks.

        Args:
            portfolio: Portfolio being hedged
            underlying: Underlying asset identifier
            pricing_env: Current pricing environment

        Returns:
            Dictionary mapping measure name to portfolio-level value
        """
        return {}

    def calculate_hedge_quantities(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        instrument_greeks: Dict[str, Dict[str, float]],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Solve for the quantity of each hedge instrument to trade.

        Args:
            current_time: Current timestamp
            portfolio_greeks: Current portfolio Greeks
            market_data: Current market data
            instrument_greeks: Per-unit Greeks of each tradeable contract,
                keyed by instrument name (supplied by the hedge executor)

        Returns:
            Dictionary mapping instrument name to quantity to trade
        """
        return self._optimizer.solve(
            portfolio_greeks=portfolio_greeks,
            instrument_greeks=instrument_greeks,
            targets=self.targets,
            instrument_costs=self.instrument_costs,
        )

    def calculate_hedge_size(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> float:
        """Not supported: multi-instrument strategies size per instrument."""
        raise NotImplementedError(
            f"{self.__class__.__name__} trades multiple instruments; "
            "use calculate_hedge_quantities() (requires a backtest engine "
            "with multi-instrument hedge execution support)"
        )

    def on_hedges_executed(
        self,
        current_time: datetime,
        quantities: Dict[str, float],
        **kwargs,
    ):
        """
        Update strategy state after a multi-instrument rebalance.

        Args:
            current_time: Execution timestamp
            quantities: Executed quantity per instrument name
        """
        self._last_hedge_time = current_time
        self._last_rebalance_date = current_time
        self._hedge_count += 1

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            "name": self.name,
            "targets": [
                {
                    "greek": t.greek,
                    "target": t.target,
                    "threshold": t.threshold,
                    "weight": t.weight,
                }
                for t in self.targets
            ],
            "hedge_instruments": [repr(inst) for inst in self.hedge_instruments],
            "rebalance_frequency": self.rebalance_frequency,
            "min_time_between_hedges": (
                str(self.min_time_between_hedges)
                if self.min_time_between_hedges
                else None
            ),
            "instrument_costs": self.instrument_costs,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "hedge_count": self._hedge_count,
            "last_hedge_time": self._last_hedge_time,
            "last_rebalance_date": self._last_rebalance_date,
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._hedge_count = 0
        self._last_rebalance_date = None

    def __repr__(self) -> str:
        greeks = ", ".join(t.greek for t in self.targets)
        instruments = ", ".join(inst.name for inst in self.hedge_instruments)
        return (
            f"{self.__class__.__name__}("
            f"targets=[{greeks}], instruments=[{instruments}])"
        )


class DeltaGammaNeutralStrategy(MultiGreekHedgeStrategy):
    """
    Delta-neutral hedging with gamma control.

    Trades an option (default: 3M ATM call, the gamma workhorse on OTC
    desks) to flatten gamma and spot to absorb the residual delta, sized
    jointly via a 2x2 solve.
    """

    def __init__(
        self,
        name: str = "DeltaGammaNeutral",
        delta_threshold: float = 100.0,
        gamma_threshold: float = 10.0,
        target_delta: float = 0.0,
        target_gamma: float = 0.0,
        gamma_instrument: Optional[OptionHedgeInstrument] = None,
        rebalance_frequency: str = "daily",
        min_time_between_hedges: Optional[timedelta] = None,
    ):
        if gamma_instrument is None:
            gamma_instrument = OptionHedgeInstrument(
                name="gamma_option", tenor=0.25, moneyness=1.0
            )
        super().__init__(
            name=name,
            targets=[
                HedgeTarget("delta", target=target_delta, threshold=delta_threshold),
                HedgeTarget("gamma", target=target_gamma, threshold=gamma_threshold),
            ],
            hedge_instruments=[gamma_instrument, SpotHedgeInstrument()],
            rebalance_frequency=rebalance_frequency,
            min_time_between_hedges=min_time_between_hedges,
        )


class DeltaVegaNeutralStrategy(MultiGreekHedgeStrategy):
    """
    Delta-neutral hedging with vega control.

    Trades an option (default: 1Y ATM call — longer tenors carry more vega
    per unit of gamma) to flatten vega and spot for the residual delta.
    """

    def __init__(
        self,
        name: str = "DeltaVegaNeutral",
        delta_threshold: float = 100.0,
        vega_threshold: float = 10.0,
        target_delta: float = 0.0,
        target_vega: float = 0.0,
        vega_instrument: Optional[OptionHedgeInstrument] = None,
        rebalance_frequency: str = "daily",
        min_time_between_hedges: Optional[timedelta] = None,
    ):
        if vega_instrument is None:
            vega_instrument = OptionHedgeInstrument(
                name="vega_option", tenor=1.0, moneyness=1.0
            )
        super().__init__(
            name=name,
            targets=[
                HedgeTarget("delta", target=target_delta, threshold=delta_threshold),
                HedgeTarget("vega", target=target_vega, threshold=vega_threshold),
            ],
            hedge_instruments=[vega_instrument, SpotHedgeInstrument()],
            rebalance_frequency=rebalance_frequency,
            min_time_between_hedges=min_time_between_hedges,
        )


class DeltaGammaVegaNeutralStrategy(MultiGreekHedgeStrategy):
    """
    Joint delta, gamma and vega control.

    Uses two options of different tenors (default: 3M and 1Y ATM calls)
    plus spot. The tenor separation is what makes the 3x3 system well
    conditioned: short-dated options are gamma-rich and vega-poor, long-
    dated options the opposite, and spot carries pure delta.
    """

    def __init__(
        self,
        name: str = "DeltaGammaVegaNeutral",
        delta_threshold: float = 100.0,
        gamma_threshold: float = 10.0,
        vega_threshold: float = 10.0,
        target_delta: float = 0.0,
        target_gamma: float = 0.0,
        target_vega: float = 0.0,
        gamma_instrument: Optional[OptionHedgeInstrument] = None,
        vega_instrument: Optional[OptionHedgeInstrument] = None,
        rebalance_frequency: str = "daily",
        min_time_between_hedges: Optional[timedelta] = None,
    ):
        if gamma_instrument is None:
            gamma_instrument = OptionHedgeInstrument(
                name="gamma_option", tenor=0.25, moneyness=1.0
            )
        if vega_instrument is None:
            vega_instrument = OptionHedgeInstrument(
                name="vega_option", tenor=1.0, moneyness=1.0
            )
        if gamma_instrument.name == vega_instrument.name:
            raise ValidationError(
                "gamma_instrument and vega_instrument must have distinct names"
            )
        super().__init__(
            name=name,
            targets=[
                HedgeTarget("delta", target=target_delta, threshold=delta_threshold),
                HedgeTarget("gamma", target=target_gamma, threshold=gamma_threshold),
                HedgeTarget("vega", target=target_vega, threshold=vega_threshold),
            ],
            hedge_instruments=[
                gamma_instrument,
                vega_instrument,
                SpotHedgeInstrument(),
            ],
            rebalance_frequency=rebalance_frequency,
            min_time_between_hedges=min_time_between_hedges,
        )
