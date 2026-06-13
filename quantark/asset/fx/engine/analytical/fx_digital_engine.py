"""
Analytical pricing engine for FX digital options.
"""

from typing import Dict, Optional, Tuple

from scipy.stats import norm

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_digital_option import FxDigitalOption
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxPayoutCurrency
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero, safe_exp, safe_log, safe_sqrt


class FxDigitalOptionAnalyticalEngine(BaseFxEngine):
    """
    Analytical engine for FX digital options under Garman-Kohlhagen dynamics.

    Cash-or-nothing (domestic payout):
        Call: payout * df_dom(T_del) * N(d2)
        Put:  payout * df_dom(T_del) * N(-d2)

    Asset-or-nothing (foreign payout):
        Call: payout * S * df_for(T_del) * N(d1)
        Put:  payout * S * df_for(T_del) * N(-d1)

    Closed-form Greeks are provided for the cash-or-nothing variant
    (legacy formulas); the asset-or-nothing variant uses the base
    finite-difference Greeks.
    """

    engine_type = EngineType.ANALYTICAL

    def __init__(self, params: Optional[FxEngineParams] = None):
        super().__init__(params)

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        """
        Price an FX digital option.

        Raises:
            PricingError: If the product is not an FxDigitalOption
        """
        option = self._check_product(product)

        tau = option.get_maturity(fx_env)
        if tau < 0:
            raise ValidationError(f"Time to expiry must be non-negative, got {tau}")
        spot = fx_env.effective_spot()
        if is_zero(tau):
            return option.get_payoff(spot)

        tau_delivery = option.get_delivery(fx_env)
        d1, d2 = self._d1_d2(option, fx_env, tau)

        if option.payout_currency == FxPayoutCurrency.DOMESTIC:
            df_dom = fx_env.get_domestic_df(tau_delivery)
            prob = norm.cdf(d2) if option.is_call() else norm.cdf(-d2)
            value = option.payout * df_dom * prob
        else:
            df_for = fx_env.get_foreign_df(tau_delivery)
            prob = norm.cdf(d1) if option.is_call() else norm.cdf(-d1)
            value = option.payout * spot * df_for * prob

        return value * option.participation_rate

    def calculate_greeks(
        self, product: BaseFxProduct, fx_env: FxPricingEnvironment
    ) -> Dict[str, float]:
        """
        Closed-form Greeks for cash-or-nothing digitals; finite-difference
        Greeks for asset-or-nothing digitals.

        Conventions match the legacy implementation: vega per 1% vol, theta
        daily decay, rho = dV/dr / 100.
        """
        option = self._check_product(product)
        if option.payout_currency != FxPayoutCurrency.DOMESTIC:
            return super().calculate_greeks(option, fx_env)

        tau = option.get_maturity(fx_env)
        tau_delivery = option.get_delivery(fx_env)
        spot = fx_env.effective_spot()
        sigma = fx_env.get_vol(option.strike, tau)
        value = self.price(option, fx_env)

        d1, d2 = self._d1_d2(option, fx_env, tau)
        df_dom = fx_env.get_domestic_df(tau_delivery)
        r_dom = fx_env.get_domestic_rate(tau_delivery)
        r_for = fx_env.get_foreign_rate(tau_delivery)

        p = option.payout * option.participation_rate
        nd2, nnd2 = norm.cdf(d2), norm.cdf(-d2)
        npd2 = norm.pdf(d2)
        sqrt_tau = float(safe_sqrt(tau))

        sign = 1.0 if option.is_call() else -1.0

        delta = sign * p * df_dom * npd2 / (spot * sigma * sqrt_tau)
        gamma = -sign * p * df_dom * npd2 * d1 / (spot**2 * sigma**2 * tau)
        vega = -sign * p * df_dom * npd2 * d1 / sigma * 0.01

        # Theta: dV/dtau then flipped to calendar decay (daily)
        prob = nd2 if option.is_call() else nnd2
        dvalue_dtau = (
            -p * r_dom * df_dom * prob
            + sign
            * p
            * df_dom
            * npd2
            * ((r_dom - r_for) / (sigma * sqrt_tau) - d1 / (2 * tau))
        )
        theta = -dvalue_dtau / 365.0

        rho_dom = (
            p
            * df_dom
            * (prob * (-tau_delivery) + sign * npd2 * sqrt_tau / sigma)
            / 100.0
        )
        rho_for = -sign * p * df_dom * npd2 * sqrt_tau / sigma / 100.0

        fwd_delta = delta * float(safe_exp(r_for * tau_delivery))
        delta_premium, fwd_delta_premium = self._premium_deltas(
            option, fx_env, delta, fwd_delta
        )

        delta_percentage = delta * spot / value if not is_zero(value) else 0.0
        gamma_percentage = gamma * spot**2 / value if not is_zero(value) else 0.0

        return {
            "price": value,
            "delta": delta,
            "delta_percentage": delta_percentage,
            "delta_premium": delta_premium,
            "fwd_delta": fwd_delta,
            "fwd_delta_premium": fwd_delta_premium,
            "gamma": gamma,
            "gamma_percentage": gamma_percentage,
            "vega": vega,
            "theta": theta,
            "rho_dom": rho_dom,
            "rho_for": rho_for,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxDigitalOption:
        if not isinstance(product, FxDigitalOption):
            raise PricingError(
                f"FxDigitalOptionAnalyticalEngine only supports FxDigitalOption, "
                f"got {type(product).__name__}"
            )
        return product

    @staticmethod
    def _d1_d2(
        option: FxDigitalOption, fx_env: FxPricingEnvironment, tau: float
    ) -> Tuple[float, float]:
        sigma = fx_env.get_vol(option.strike, tau)
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        fwd = fx_env.get_forward(tau)
        sqrt_tau = float(safe_sqrt(tau))
        denominator = sigma * sqrt_tau
        if is_zero(denominator):
            raise PricingError(f"Vanishing volatility-time term: σ√τ = {denominator}")
        d1 = (float(safe_log(fwd / option.strike)) + sigma**2 / 2 * tau) / denominator
        d2 = d1 - denominator
        return d1, d2

    def __repr__(self):
        return "FxDigitalOptionAnalyticalEngine(analytical)"
