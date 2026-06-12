"""
Analytical Garman-Kohlhagen quanto pricing engine for FX vanilla options.
"""

from typing import Optional, Tuple

from scipy.stats import norm

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_quanto_vanilla_option import (
    FxQuantoVanillaOption,
)
from quantark.priceenv import FxPricingEnvironment, FxQuantoMarketData
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import MarketDataError, PricingError, ValidationError
from quantark.util.numerical import is_zero, safe_exp, safe_log, safe_sqrt


class GarmanKohlhagenQuantoEngine(BaseFxEngine):
    """
    Garman-Kohlhagen engine with quanto adjustment.

    The risk-neutral drift of the underlying pair is shifted by the quanto
    adjustment mu_q = -rho * sigma * sigma_quanto, giving the quanto forward

        F_q = S * exp((r_d - r_f + mu_q) * T)

    (or market_forward * exp(mu_q * T) when a market forward is quoted). The
    payoff is converted at the product's fixed quanto rate and discounted on
    the settlement currency curve:

        Call: N * X_q * df_settle(T_del) * [F_q*N(d1) - K*N(d2)]
        Put:  N * X_q * df_settle(T_del) * [K*N(-d2) - F_q*N(-d1)]

    Greeks use the base finite-difference implementation (the legacy module
    provided no closed-form quanto Greeks).
    """

    engine_type = EngineType.ANALYTICAL

    def __init__(self, params: Optional[FxEngineParams] = None):
        super().__init__(params)

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        """
        Price an FX quanto vanilla option in the settlement currency.

        Raises:
            PricingError: If the product is not an FxQuantoVanillaOption
            MarketDataError: If the environment carries no quanto market data
        """
        option = self._check_product(product)
        quanto = _require_quanto(fx_env)

        tau = option.get_maturity(fx_env)
        if tau < 0:
            raise ValidationError(f"Time to expiry must be non-negative, got {tau}")
        spot = fx_env.effective_spot()
        if is_zero(tau):
            return option.get_payoff(spot) * option.annualization_factor(fx_env)

        tau_delivery = option.get_delivery(fx_env)
        sigma = fx_env.get_vol(option.strike, tau)
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")

        fwd_q = _quanto_forward(fx_env, quanto, sigma, tau)
        d1, d2 = _d1_d2(fwd_q, option.strike, sigma, tau)
        df_settle = quanto.settlement_curve.get_discount_factor(tau_delivery)

        if option.is_call():
            unit_value = fwd_q * norm.cdf(d1) - option.strike * norm.cdf(d2)
        else:
            unit_value = option.strike * norm.cdf(-d2) - fwd_q * norm.cdf(-d1)

        return (
            option.notional
            * option.quanto_fx_rate
            * df_settle
            * unit_value
            * option.participation_rate
            * option.annualization_factor(fx_env)
        )

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxQuantoVanillaOption:
        if not isinstance(product, FxQuantoVanillaOption):
            raise PricingError(
                f"GarmanKohlhagenQuantoEngine only supports FxQuantoVanillaOption, "
                f"got {type(product).__name__}"
            )
        return product

    def __repr__(self):
        return "GarmanKohlhagenQuantoEngine(analytical)"


def _require_quanto(fx_env: FxPricingEnvironment) -> FxQuantoMarketData:
    """Quanto market data accessor shared by the quanto engines."""
    if fx_env.quanto is None:
        raise MarketDataError(
            "Quanto market data (settlement curve, quanto vol, correlation) "
            "is required for quanto pricing"
        )
    return fx_env.quanto


def _quanto_forward(
    fx_env: FxPricingEnvironment,
    quanto: FxQuantoMarketData,
    sigma: float,
    tau: float,
) -> float:
    """
    Quanto-adjusted forward at expiry.

    Applies mu_q = -rho * sigma * sigma_quanto to the market forward when
    quoted, otherwise to the interest-rate-parity drift.
    """
    quanto_adj = -quanto.correlation * sigma * quanto.quanto_vol
    return fx_env.get_forward(tau) * float(safe_exp(quanto_adj * tau))


def _d1_d2(fwd: float, strike: float, sigma: float, tau: float) -> Tuple[float, float]:
    sqrt_tau = float(safe_sqrt(tau))
    denominator = sigma * sqrt_tau
    if is_zero(denominator):
        raise PricingError(f"Vanishing volatility-time term: σ√τ = {denominator}")
    d1 = (float(safe_log(fwd / strike)) + sigma**2 / 2 * tau) / denominator
    d2 = d1 - denominator
    return d1, d2
