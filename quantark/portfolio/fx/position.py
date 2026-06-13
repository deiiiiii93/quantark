"""
FX position class for tracking individual FX derivative positions.

Unifies two roles:

* **Risk-module portfolio layer** — prices through an :class:`FxPricingEnvironment`
  (spot plus two rate curves and a vol surface) and reports the two-rate FX greeks
  (``delta, gamma, vega, theta, rho_dom, rho_for``), consumed by the VaR, stress,
  dynamic-scenario and backtest modules.
* **SIMM sensitivity provider** — implements ``get_simm_sensitivities`` so the same
  position feeds the ISDA SIMM v2.6 initial-margin engine via
  :class:`~quantark.simm.engines.portfolio_adapter.SIMMPortfolioAdapter`.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.riskmeasures.fx_greeks_calculator import FxGreeksCalculator
from quantark.priceenv.fx_pricing_environment import FxPricingEnvironment
from quantark.simm.sensitivity import (
    FXDeltaSensitivity,
    FXVegaSensitivity,
    SensitivityCollection,
    vol_weighted_vega_fx,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

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
        engine: FX pricing engine for this position.
        entry_price: Quote-currency price/NPV at which the position was entered.
        underlying: Currency-pair identifier, e.g. ``"EURUSD"``. Derived from the
            product's currency pair when not supplied.
        entry_timestamp: When the position was opened.
        position_id: Unique identifier.
    """

    product: BaseFxProduct
    quantity: float
    engine: BaseFxEngine
    entry_price: float = 0.0
    underlying: Optional[str] = None
    entry_timestamp: datetime = field(default_factory=datetime.now)
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if self.underlying is None:
            self.underlying = self._derive_underlying()
        self.validate()

    def _derive_underlying(self) -> str:
        pair = self.product.currency_pair
        return f"{pair.base_ccy}{pair.quote_ccy}"

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

    # ------------------------------------------------------------------ #
    # Risk-module interface
    # ------------------------------------------------------------------ #
    def get_current_price(self, fx_env: FxPricingEnvironment) -> float:
        """Price one contract of the product in the quote currency."""
        return self.engine.price(self.product, fx_env)

    def get_market_value(self, fx_env: FxPricingEnvironment) -> float:
        """Market value = current price x quantity (quote currency)."""
        return self.get_current_price(fx_env) * self.quantity

    def get_pnl(self, fx_env: FxPricingEnvironment) -> float:
        """Unrealized P&L versus the entry price (quote currency)."""
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

    # ------------------------------------------------------------------ #
    # SIMM interface (SIMMSensitivityProvider)
    # ------------------------------------------------------------------ #
    def get_simm_sensitivities(self, config: Any, market_data: Any) -> SensitivityCollection:
        """Produce ISDA SIMM FX delta / vega sensitivities for this position.

        ``market_data`` is the per-pair pricing-environment map (the
        ``pricing_environments`` dict of an :class:`FXPortfolio`).
        """
        if not isinstance(market_data, dict):
            raise ValidationError(
                "FX SIMM sensitivity generation requires pricing environments "
                "mapped by currency pair"
            )
        env = market_data.get(self.underlying) or market_data.get(
            str(self.product.currency_pair)
        )
        if env is None:
            raise ValidationError(f"Missing FX pricing environment for {self.underlying}")

        pair = self.product.currency_pair
        calculation_currency = config.calculation_currency.upper()
        if pair.quote_ccy != calculation_currency:
            raise ValidationError(
                f"Built-in FX sensitivity generation requires quote currency "
                f"{pair.quote_ccy} to equal calculation currency {calculation_currency}; "
                "otherwise supply explicit translated sensitivities"
            )

        greeks = self.engine.calculate_greeks(self.product, env)
        result = SensitivityCollection()
        delta_amount = 0.01 * env.effective_spot() * greeks["delta"] * self.quantity
        if not is_zero(delta_amount):
            result.add(
                FXDeltaSensitivity(
                    trade_id=self.position_id,
                    amount=delta_amount,
                    amount_currency=calculation_currency,
                    currency=pair.base_ccy,
                )
            )

        vega = greeks.get("vega", 0.0) * self.quantity
        if not is_zero(vega):
            result.add(
                FXVegaSensitivity(
                    trade_id=self.position_id,
                    amount=vol_weighted_vega_fx(vega, self.underlying),
                    amount_currency=calculation_currency,
                    currency_pair=self.underlying,
                    option_tenor=self.product.get_maturity(env),
                )
            )
        return result
