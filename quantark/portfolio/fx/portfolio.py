"""
FX portfolio class for managing multiple FX derivative positions.

Mirrors :class:`~quantark.portfolio.equity.portfolio.EquityPortfolio`: positions
are keyed by ``position_id`` and each currency pair carries its own
:class:`FxPricingEnvironment`, keyed by the pair symbol. Risk aggregation returns
the two-rate FX greeks.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.riskmeasures.fx_greeks_calculator import FxGreeksCalculator
from quantark.priceenv.fx_pricing_environment import FxPricingEnvironment
from quantark.util.exceptions import ValidationError

from .position import _CORE_GREEKS, FXPosition


@dataclass
class FXPortfolio:
    """
    Portfolio container for multiple FX derivative positions.

    Attributes:
        portfolio_name: Name identifier for the portfolio.
        pricing_environments: FX pricing environments keyed by currency pair
            (e.g. ``{"EURUSD": env}``).
        creation_date: When the portfolio was created.
        positions: Positions keyed by ``position_id``.
    """

    portfolio_name: str
    pricing_environments: Dict[str, FxPricingEnvironment]
    creation_date: datetime = field(default_factory=datetime.now)
    positions: Dict[str, FXPosition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.portfolio_name:
            raise ValidationError("Portfolio name is required")
        if not isinstance(self.pricing_environments, dict):
            raise ValidationError("pricing_environments must be a dictionary")

    # ------------------------------------------------------------------ #
    # Position management
    # ------------------------------------------------------------------ #
    def add_position(
        self,
        product: BaseFxProduct,
        quantity: float,
        entry_price: float,
        underlying: str,
        engine: BaseFxEngine,
        entry_timestamp: Optional[datetime] = None,
    ) -> FXPosition:
        """Add a position; the pair must have a registered pricing environment."""
        if underlying not in self.pricing_environments:
            raise ValidationError(
                f"Currency pair '{underlying}' not found in pricing environments. "
                f"Available: {list(self.pricing_environments.keys())}"
            )
        position = FXPosition(
            product=product,
            quantity=quantity,
            entry_price=entry_price,
            underlying=underlying,
            engine=engine,
            entry_timestamp=entry_timestamp or datetime.now(),
        )
        self.positions[position.position_id] = position
        return position

    def remove_position(self, position_id: str) -> Optional[FXPosition]:
        return self.positions.pop(position_id, None)

    def update_position(self, position_id: str, **kwargs: Any) -> FXPosition:
        """Update mutable fields (e.g. ``quantity``, ``entry_price``) of a position."""
        if position_id not in self.positions:
            raise ValidationError(f"Position '{position_id}' not found")
        position = self.positions[position_id]
        for field_name, value in kwargs.items():
            if not hasattr(position, field_name):
                raise ValidationError(f"Unknown position field '{field_name}'")
            setattr(position, field_name, value)
        position.validate()
        return position

    def get_position(self, position_id: str) -> Optional[FXPosition]:
        return self.positions.get(position_id)

    def get_positions_by_underlying(self, underlying: str) -> List[FXPosition]:
        return [p for p in self.positions.values() if p.underlying == underlying]

    # ------------------------------------------------------------------ #
    # Valuation & risk
    # ------------------------------------------------------------------ #
    def get_portfolio_value(self, as_of_date: Optional[datetime] = None) -> float:
        return sum(
            p.get_market_value(self.pricing_environments[p.underlying])
            for p in self.positions.values()
        )

    def get_portfolio_pnl(self) -> float:
        return sum(
            p.get_pnl(self.pricing_environments[p.underlying])
            for p in self.positions.values()
        )

    def get_portfolio_greeks(
        self, greeks_calculator: Optional[FxGreeksCalculator] = None
    ) -> Dict[str, float]:
        calc = greeks_calculator or FxGreeksCalculator()
        aggregated: Dict[str, float] = {key: 0.0 for key in _CORE_GREEKS}
        aggregated["market_value"] = 0.0
        for position in self.positions.values():
            env = self.pricing_environments[position.underlying]
            aggregated["market_value"] += position.get_market_value(env)
            for key, value in position.get_greeks(env, calc).items():
                aggregated[key] += value
        return aggregated

    def get_portfolio_risk_measures(
        self,
        greeks_calculator: Optional[FxGreeksCalculator] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Asset-agnostic risk aggregation entry point (returns FX greeks)."""
        return self.get_portfolio_greeks(greeks_calculator)

    def get_greeks_by_underlying(
        self,
        underlying: str,
        greeks_calculator: Optional[FxGreeksCalculator] = None,
    ) -> Dict[str, float]:
        calc = greeks_calculator or FxGreeksCalculator()
        aggregated: Dict[str, float] = {key: 0.0 for key in _CORE_GREEKS}
        aggregated["market_value"] = 0.0
        env = self.pricing_environments.get(underlying)
        if env is None:
            return aggregated
        for position in self.get_positions_by_underlying(underlying):
            aggregated["market_value"] += position.get_market_value(env)
            for key, value in position.get_greeks(env, calc).items():
                aggregated[key] += value
        return aggregated

    # ------------------------------------------------------------------ #
    # Summary / dunder
    # ------------------------------------------------------------------ #
    def get_summary(self) -> Dict[str, Any]:
        return {
            "portfolio_name": self.portfolio_name,
            "creation_date": self.creation_date.isoformat(),
            "num_positions": len(self.positions),
            "pairs": sorted({p.underlying for p in self.positions.values()}),
            "total_value": self.get_portfolio_value(),
            "total_pnl": self.get_portfolio_pnl(),
        }

    def __len__(self) -> int:
        return len(self.positions)
