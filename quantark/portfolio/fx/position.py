"""
FX position class for tracking individual FX derivative positions.

Mirrors :class:`~quantark.portfolio.equity.position.EquityPosition` but is
typed for FX: it prices through an :class:`FxPricingEnvironment` (spot plus two
rate curves and a vol surface) and reports the two-rate FX greeks
(``delta, gamma, vega, theta, rho_dom, rho_for``).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.riskmeasures.fx_greeks_calculator import FxGreeksCalculator
from quantark.priceenv.fx_pricing_environment import FxPricingEnvironment
from quantark.util.exceptions import ValidationError

#: Core FX greeks aggregated across positions (FX-only fields such as
#: ``delta_premium`` / ``fwd_delta`` are intentionally excluded from aggregation).
_CORE_GREEKS = ("delta", "gamma", "vega", "theta", "rho_dom", "rho_for")


@dataclass
class FXPosition:
    """
    A single FX derivative position in a portfolio.

    Attributes:
        product: The FX product (spot, forward, swap, vanilla/digital/quanto option).
        quantity: Number of contracts (positive=long, negative=short). FX products
            carry their own notional, so quantity is a whole-contract multiplier.
        entry_price: Domestic-currency price/NPV at which the position was entered.
        underlying: Currency-pair identifier, e.g. ``"EURUSD"``.
        engine: FX pricing engine for this position.
        entry_timestamp: When the position was opened.
        position_id: Unique identifier.
    """

    product: BaseFxProduct
    quantity: float
    entry_price: float
    underlying: str
    engine: BaseFxEngine
    entry_timestamp: datetime
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate position parameters."""
        if self.quantity == 0:
            raise ValidationError("Position quantity cannot be zero")
        if not self.underlying:
            raise ValidationError("Underlying (currency pair) identifier is required")
        if self.product is None:
            raise ValidationError("Product is required")
        if self.engine is None:
            raise ValidationError("Engine is required")

    def get_current_price(self, fx_env: FxPricingEnvironment) -> float:
        """Price one contract of the product in the domestic currency."""
        return self.engine.price(self.product, fx_env)

    def get_market_value(self, fx_env: FxPricingEnvironment) -> float:
        """Market value = current price x quantity (domestic currency)."""
        return self.get_current_price(fx_env) * self.quantity

    def get_pnl(self, fx_env: FxPricingEnvironment) -> float:
        """Unrealized P&L versus the entry price (domestic currency)."""
        return (self.get_current_price(fx_env) - self.entry_price) * self.quantity

    def get_greeks(
        self,
        fx_env: FxPricingEnvironment,
        greeks_calculator: Optional[FxGreeksCalculator] = None,
    ) -> Dict[str, float]:
        """Quantity-scaled core FX greeks for this position."""
        calc = greeks_calculator or FxGreeksCalculator()
        raw = calc.calculate(self.product, fx_env, self.engine)
        return {key: raw.get(key, 0.0) * self.quantity for key in _CORE_GREEKS}

    def get_risk_measures(
        self, fx_env: FxPricingEnvironment, **kwargs: Any
    ) -> Dict[str, float]:
        """Asset-agnostic risk-measure entry point (returns FX greeks)."""
        return self.get_greeks(fx_env, kwargs.get("greeks_calculator"))

    def is_long(self) -> bool:
        return self.quantity > 0

    def is_short(self) -> bool:
        return self.quantity < 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "underlying": self.underlying,
            "product_type": type(self.product).__name__,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp.isoformat(),
        }
