"""
Analytical pricing engine for FX quanto digital options.
"""

from typing import Optional

from scipy.stats import norm

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_quanto_digital_option import (
    FxQuantoDigitalOption,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero
from .fx_quanto_vanilla_engine import _d1_d2, _quanto_forward, _require_quanto


class FxQuantoDigitalAnalyticalEngine(BaseFxEngine):
    """
    Analytical engine for FX quanto digital (cash-or-nothing) options.

    The in-the-money probability is evaluated under the quanto-adjusted
    forward, the fixed payout converted at the contractual quanto rate, and
    discounting done on the settlement currency curve:

        Call: payout * X_q * df_settle(T_del) * N(d2)
        Put:  payout * X_q * df_settle(T_del) * N(-d2)

    Greeks use the base finite-difference implementation.
    """

    engine_type = EngineType.ANALYTICAL

    def __init__(self, params: Optional[FxEngineParams] = None):
        super().__init__(params)

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        """
        Price an FX quanto digital option in the settlement currency.

        Raises:
            PricingError: If the product is not an FxQuantoDigitalOption
            MarketDataError: If the environment carries no quanto market data
        """
        option = self._check_product(product)
        quanto = _require_quanto(fx_env)

        tau = option.get_maturity(fx_env)
        if tau < 0:
            raise ValidationError(f"Time to expiry must be non-negative, got {tau}")
        spot = fx_env.effective_spot()
        if is_zero(tau):
            return (
                option.get_payoff(spot)
                * quanto.settlement_curve.get_discount_factor(option.get_delivery(fx_env))
            )

        tau_delivery = option.get_delivery(fx_env)
        sigma = fx_env.get_vol(option.strike, tau)
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")

        fwd_q = _quanto_forward(fx_env, quanto, sigma, tau)
        _, d2 = _d1_d2(fwd_q, option.strike, sigma, tau)
        df_settle = quanto.settlement_curve.get_discount_factor(tau_delivery)

        prob = norm.cdf(d2) if option.is_call() else norm.cdf(-d2)
        return (
            option.payout
            * option.quanto_fx_rate
            * df_settle
            * prob
            * option.participation_rate
        )

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxQuantoDigitalOption:
        if not isinstance(product, FxQuantoDigitalOption):
            raise PricingError(
                f"FxQuantoDigitalAnalyticalEngine only supports "
                f"FxQuantoDigitalOption, got {type(product).__name__}"
            )
        return product

    def __repr__(self):
        return "FxQuantoDigitalAnalyticalEngine(analytical)"
