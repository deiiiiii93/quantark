"""
Analytical Garman-Kohlhagen pricing engine for FX European vanilla options.
"""

from typing import Dict, Optional, Tuple

from scipy.stats import norm

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_close, is_zero, safe_exp, safe_log, safe_sqrt


class GarmanKohlhagenEngine(BaseFxEngine):
    """
    Garman-Kohlhagen analytical engine for FX European vanilla options.

    Extends Black-Scholes to two interest rates. Per unit of foreign (base)
    currency, the option value in domestic (quote) currency is:

        Call: [F*N(d1) - K*N(d2)] * df_dom(T_del)
        Put:  [K*N(-d2) - F*N(-d1)] * df_dom(T_del)

    where F is the outright forward at expiry (interest rate parity from the
    effective spot, or a market-quoted forward). Delivery after expiry changes
    only the payment discount factor; the payoff remains fixed at expiry.

    Greeks are computed with closed forms following the legacy conventions
    (vega per 1% vol, theta daily, rho = dV/dr / 100).
    """

    engine_type = EngineType.ANALYTICAL

    def __init__(self, params: Optional[FxEngineParams] = None):
        super().__init__(params)

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        """
        Price an FX vanilla option.

        Raises:
            PricingError: If the product is not an FxVanillaOption
        """
        option = self._check_product(product)

        tau = option.get_maturity(fx_env)
        if tau < 0:
            raise ValidationError(f"Time to expiry must be non-negative, got {tau}")
        spot = fx_env.effective_spot()
        if is_zero(tau):
            return (
                option.get_payoff(spot)
                * fx_env.get_domestic_df(option.get_delivery(fx_env))
                * option.annualization_factor(fx_env)
            )

        fwd, strike, tau_delivery = self._forward_and_strike(option, fx_env, tau)
        sigma = fx_env.get_vol(option.strike, tau)
        self._validate_market(spot, sigma)

        d1, d2 = self._d1_d2(fwd, strike, sigma, tau)
        df_dom = fx_env.get_domestic_df(tau_delivery)

        if option.is_call():
            unit_value = fwd * norm.cdf(d1) - strike * norm.cdf(d2)
        else:
            unit_value = strike * norm.cdf(-d2) - fwd * norm.cdf(-d1)

        value = (
            option.notional
            * unit_value
            * df_dom
            * option.participation_rate
            * option.annualization_factor(fx_env)
        )

        if value < 0 and not is_zero(value / option.notional):
            raise PricingError(f"Negative option value computed: {value}")
        return value

    def calculate_greeks(
        self, product: BaseFxProduct, fx_env: FxPricingEnvironment
    ) -> Dict[str, float]:
        """
        Closed-form Garman-Kohlhagen Greeks.

        Conventions (legacy-compatible):
            - delta/gamma: spot sensitivities of the (possibly annualized) value
            - fwd_delta = delta * exp(r_f * T_del)
            - delta_premium / fwd_delta_premium: foreign-premium adjustment,
              applied after annualization
            - vega: per 1% vol change
            - theta: daily decay (dV/dt / 365)
            - rho_dom / rho_for: dV/dr / 100
        """
        option = self._check_product(product)

        tau = option.get_maturity(fx_env)
        tau_delivery = option.get_delivery(fx_env)
        if not is_close(tau, tau_delivery) or fx_env.market_forward is not None:
            return super().calculate_greeks(option, fx_env)

        spot = fx_env.effective_spot()
        value = self.price(option, fx_env)

        fwd, strike, tau_delivery = self._forward_and_strike(option, fx_env, tau)
        sigma = fx_env.get_vol(option.strike, tau)
        self._validate_market(spot, sigma)

        d1, d2 = self._d1_d2(fwd, strike, sigma, tau)
        df_dom = fx_env.get_domestic_df(tau_delivery)
        df_for = fx_env.get_foreign_df(tau_delivery)
        r_dom = fx_env.get_domestic_rate(tau_delivery)
        r_for = fx_env.get_foreign_rate(tau_delivery)

        n = option.notional * option.participation_rate
        nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
        nnd1, nnd2 = norm.cdf(-d1), norm.cdf(-d2)
        npd1 = norm.pdf(d1)
        sqrt_tau = float(safe_sqrt(tau))

        if option.is_call():
            delta = n * df_for * nd1
        else:
            delta = -n * df_for * nnd1

        gamma = n * df_for * npd1 / (spot * sigma * sqrt_tau)
        vega = n * spot * df_for * npd1 * sqrt_tau * 0.01

        theta_diffusion = -n * spot * df_for * npd1 * sigma / (2 * sqrt_tau)
        if option.is_call():
            theta_carry = (
                n * r_for * spot * df_for * nd1
                - n * r_dom * strike * df_dom * nd2
            )
        else:
            theta_carry = (
                -n * r_for * spot * df_for * nnd1
                + n * r_dom * strike * df_dom * nnd2
            )
        theta = (theta_diffusion + theta_carry) / 365.0

        if option.is_call():
            rho_dom = n * strike * df_dom * tau_delivery * nd2 / 100.0
            rho_for = -n * spot * df_for * tau_delivery * nd1 / 100.0
        else:
            rho_dom = -n * strike * df_dom * tau_delivery * nnd2 / 100.0
            rho_for = n * spot * df_for * tau_delivery * nnd1 / 100.0

        fwd_delta = delta * float(safe_exp(r_for * tau_delivery))

        # Annualization scales all sensitivities; premium adjustment is a
        # fixed amount added afterwards (legacy behavior).
        factor = option.annualization_factor(fx_env)
        delta *= factor
        fwd_delta *= factor
        gamma *= factor
        vega *= factor
        theta *= factor
        rho_dom *= factor
        rho_for *= factor

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
    def _check_product(product: BaseFxProduct) -> FxVanillaOption:
        if not isinstance(product, FxVanillaOption):
            raise PricingError(
                f"GarmanKohlhagenEngine only supports FxVanillaOption, "
                f"got {type(product).__name__}"
            )
        return product

    @staticmethod
    def _validate_market(spot: float, sigma: float) -> None:
        if spot <= 0:
            raise ValidationError(f"Spot must be positive, got {spot}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        if sigma > 5.0:
            raise ValidationError(
                f"Volatility too high for numerical stability: {sigma}"
            )

    @staticmethod
    def _forward_and_strike(
        option: FxVanillaOption, fx_env: FxPricingEnvironment, tau: float
    ) -> Tuple[float, float, float]:
        """
        Forward at expiry, contractual strike, and delivery time.
        """
        tau_delivery = option.get_delivery(fx_env)
        fwd = fx_env.get_forward(tau)

        return fwd, option.strike, tau_delivery

    @staticmethod
    def _d1_d2(fwd: float, strike: float, sigma: float, tau: float) -> Tuple[float, float]:
        sqrt_tau = float(safe_sqrt(tau))
        denominator = sigma * sqrt_tau
        if is_zero(denominator):
            raise PricingError(f"Vanishing volatility-time term: σ√τ = {denominator}")
        d1 = (float(safe_log(fwd / strike)) + sigma**2 / 2 * tau) / denominator
        d2 = d1 - denominator
        return d1, d2

    def __repr__(self):
        return "GarmanKohlhagenEngine(analytical)"
