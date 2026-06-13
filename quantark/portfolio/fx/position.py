"""FX position with built-in compliant SIMM delta and vega generation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.priceenv import FxPricingEnvironment
from quantark.simm.sensitivity import (
    FXDeltaSensitivity,
    FXVegaSensitivity,
    SensitivityCollection,
    vol_weighted_vega_fx,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero


@dataclass
class FxPosition:
    """FX position whose value is expressed in the pair's quote currency."""

    product: BaseFxProduct
    quantity: float
    engine: BaseFxEngine
    entry_price: float = 0.0
    entry_timestamp: datetime = field(default_factory=datetime.now)
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def underlying(self) -> str:
        pair = self.product.currency_pair
        return f"{pair.base_ccy}{pair.quote_ccy}"

    def get_current_price(self, pricing_env: FxPricingEnvironment) -> float:
        return self.engine.price(self.product, pricing_env)

    def get_market_value(self, pricing_env: FxPricingEnvironment) -> float:
        return self.quantity * self.get_current_price(pricing_env)

    def get_simm_sensitivities(self, config: Any, market_data: Any) -> SensitivityCollection:
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
