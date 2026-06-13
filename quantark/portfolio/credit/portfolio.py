"""
Credit portfolio class for managing multiple CDS-type positions.

Mirrors :class:`~quantark.portfolio.fx.portfolio.FXPortfolio`: positions are
keyed by ``position_id`` and each reference entity carries its own
:class:`CreditPricingEnvironment`, keyed by the entity identifier.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from quantark.asset.credit.engine.base_credit_engine import BaseCreditEngine
from quantark.asset.credit.product.base_credit_product import BaseCreditProduct
from quantark.asset.credit.riskmeasures import CreditGreeksCalculator
from quantark.priceenv import CreditPricingEnvironment
from quantark.util.exceptions import ValidationError

from .position import _CORE_GREEKS, CreditPosition


@dataclass
class CreditPortfolio:
    """
    Portfolio container for multiple credit positions.

    Attributes:
        portfolio_name: Name identifier for the portfolio.
        pricing_environments: Credit pricing environments keyed by reference
            entity (e.g. ``{"ACME": env}``).
        creation_date: When the portfolio was created.
        positions: Positions keyed by ``position_id``.
    """

    portfolio_name: str
    pricing_environments: Dict[str, CreditPricingEnvironment]
    creation_date: datetime = field(default_factory=datetime.now)
    positions: Dict[str, CreditPosition] = field(default_factory=dict)

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
        product: BaseCreditProduct,
        quantity: float,
        entry_price: float,
        reference_entity: str,
        engine: BaseCreditEngine,
        is_qualifying: bool = True,
        payment_currency: str = "USD",
        simm_bucket: Optional[int] = None,
        entry_timestamp: Optional[datetime] = None,
    ) -> CreditPosition:
        """Add a position; the entity must have a registered pricing environment."""
        if reference_entity not in self.pricing_environments:
            raise ValidationError(
                f"Reference entity '{reference_entity}' not found in pricing "
                f"environments. Available: {list(self.pricing_environments.keys())}"
            )
        position = CreditPosition(
            product=product,
            quantity=quantity,
            engine=engine,
            reference_entity=reference_entity,
            entry_price=entry_price,
            is_qualifying=is_qualifying,
            payment_currency=payment_currency,
            simm_bucket=simm_bucket,
            entry_timestamp=entry_timestamp or datetime.now(),
        )
        self.positions[position.position_id] = position
        return position

    def remove_position(self, position_id: str) -> Optional[CreditPosition]:
        return self.positions.pop(position_id, None)

    def update_position(self, position_id: str, **kwargs: Any) -> CreditPosition:
        """Update mutable fields of a position."""
        if position_id not in self.positions:
            raise ValidationError(f"Position '{position_id}' not found")
        position = self.positions[position_id]
        for field_name, value in kwargs.items():
            if not hasattr(position, field_name):
                raise ValidationError(f"Unknown position field '{field_name}'")
            setattr(position, field_name, value)
        position.validate()
        return position

    def get_position(self, position_id: str) -> Optional[CreditPosition]:
        return self.positions.get(position_id)

    def get_positions_by_underlying(self, reference_entity: str) -> List[CreditPosition]:
        return [
            p for p in self.positions.values() if p.reference_entity == reference_entity
        ]

    # ------------------------------------------------------------------ #
    # Valuation & risk
    # ------------------------------------------------------------------ #
    def get_portfolio_value(self, as_of_date: Optional[datetime] = None) -> float:
        return sum(
            p.get_market_value(self.pricing_environments[p.reference_entity])
            for p in self.positions.values()
        )

    def get_portfolio_pnl(self) -> float:
        return sum(
            p.get_pnl(self.pricing_environments[p.reference_entity])
            for p in self.positions.values()
        )

    def get_portfolio_greeks(
        self, greeks_calculator: Optional[CreditGreeksCalculator] = None
    ) -> Dict[str, float]:
        calc = greeks_calculator or CreditGreeksCalculator()
        aggregated: Dict[str, float] = {key: 0.0 for key in _CORE_GREEKS}
        aggregated["market_value"] = 0.0
        for position in self.positions.values():
            env = self.pricing_environments[position.reference_entity]
            aggregated["market_value"] += position.get_market_value(env)
            for key, value in position.get_greeks(env, calc).items():
                aggregated[key] += value
        return aggregated

    def get_portfolio_risk_measures(
        self,
        greeks_calculator: Optional[CreditGreeksCalculator] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        return self.get_portfolio_greeks(greeks_calculator)

    def get_greeks_by_underlying(
        self,
        reference_entity: str,
        greeks_calculator: Optional[CreditGreeksCalculator] = None,
    ) -> Dict[str, float]:
        calc = greeks_calculator or CreditGreeksCalculator()
        aggregated: Dict[str, float] = {key: 0.0 for key in _CORE_GREEKS}
        aggregated["market_value"] = 0.0
        env = self.pricing_environments.get(reference_entity)
        if env is None:
            return aggregated
        for position in self.get_positions_by_underlying(reference_entity):
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
            "entities": sorted({p.reference_entity for p in self.positions.values()}),
            "total_value": self.get_portfolio_value(),
            "total_pnl": self.get_portfolio_pnl(),
        }

    def __len__(self) -> int:
        return len(self.positions)
