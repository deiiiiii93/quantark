"""
Analytical pricing engine for FX digital options.
"""

from typing import Dict, Optional, Tuple

from scipy.stats import norm

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.analytical.garman_kohlhagen_engine import GarmanKohlhagenEngine
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_digital_option import FxDigitalOption
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.param.vol.vannavolga import VannaVolgaVolSurface
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxPayoutCurrency, OptionType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_close, is_zero, safe_exp, safe_log, safe_sqrt


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

    # Strike bump for the call-spread replication of the digital (-dC/dK).
    _H_REL = 1e-4
    _H_FLOOR = 1e-6

    def __init__(self, params: Optional[FxEngineParams] = None):
        super().__init__(params)
        self._vanilla_engine = GarmanKohlhagenEngine()

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
            return option.get_payoff(spot) * fx_env.get_domestic_df(
                option.get_delivery(fx_env)
            )

        # Under a Vanna-Volga smile the level-only N(d2) digital misses the
        # skew term (-vega * d-sigma/d-K). Price by static replication off the
        # smile-consistent vanilla instead.
        if isinstance(fx_env.vol_surface, VannaVolgaVolSurface):
            return self._replicated_digital(option, fx_env)

        tau_delivery = option.get_delivery(fx_env)
        d1, d2 = self._d1_d2(option, fx_env, tau)

        if option.payout_currency == FxPayoutCurrency.DOMESTIC:
            df_dom = fx_env.get_domestic_df(tau_delivery)
            prob = norm.cdf(d2) if option.is_call() else norm.cdf(-d2)
            value = option.payout * df_dom * prob
        else:
            df_dom = fx_env.get_domestic_df(tau_delivery)
            fwd = fx_env.get_forward(tau)
            prob = norm.cdf(d1) if option.is_call() else norm.cdf(-d1)
            value = option.payout * fwd * df_dom * prob

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

        # Under a VV smile, route Greeks through bump-and-reprice (price() then
        # uses replication, capturing skew); the closed forms below are
        # sticky-strike at sigma(K) and miss the smile dynamics.
        if isinstance(fx_env.vol_surface, VannaVolgaVolSurface):
            return super().calculate_greeks(option, fx_env)

        if option.payout_currency != FxPayoutCurrency.DOMESTIC:
            return super().calculate_greeks(option, fx_env)

        tau = option.get_maturity(fx_env)
        tau_delivery = option.get_delivery(fx_env)
        if not is_close(tau, tau_delivery) or fx_env.market_forward is not None:
            return super().calculate_greeks(option, fx_env)

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

    def _vanilla_call_price(
        self, option: FxDigitalOption, fx_env: FxPricingEnvironment, strike: float
    ) -> float:
        """Smile-consistent unit-notional vanilla CALL price at `strike`."""
        vanilla = FxVanillaOption(
            strike=strike,
            option_type=OptionType.CALL,
            currency_pair=option.currency_pair,
            maturity=option.maturity,
            expiry_date=option.expiry_date,
            delivery=option.delivery,
            delivery_date=option.delivery_date,
            notional_foreign=1.0,
        )
        return self._vanilla_engine.price(vanilla, fx_env)

    def _replicated_digital(
        self, option: FxDigitalOption, fx_env: FxPricingEnvironment
    ) -> float:
        """Smile-consistent digital via static replication (-dC/dK call spread).

        Captures the smile skew because the two vanilla legs are priced at
        sigma(K +/- h) through the surface. Derives all four digital variants
        from the cash-or-nothing call atom by no-arbitrage identities.
        """
        K = option.strike
        h = max(self._H_FLOOR, self._H_REL * K)
        if K - h <= 0.0:
            raise PricingError(
                f"Replication strike bump h={h} too large for strike {K}"
            )
        c_up = self._vanilla_call_price(option, fx_env, K + h)
        c_dn = self._vanilla_call_price(option, fx_env, K - h)
        cash_call = (c_dn - c_up) / (2.0 * h)  # = -dC/dK

        tau_delivery = option.get_delivery(fx_env)
        df_dom = fx_env.get_domestic_df(tau_delivery)

        if option.payout_currency == FxPayoutCurrency.DOMESTIC:
            base = cash_call if option.is_call() else (df_dom - cash_call)
        else:
            c_k = self._vanilla_call_price(option, fx_env, K)
            asset_call = c_k + K * cash_call
            if option.is_call():
                base = asset_call
            else:
                # Total asset-or-nothing PV (call + put = PV[S_T]) must use the
                # SAME forward basis as the vanilla legs and the closed-form
                # path (fwd * df_dom), not s_eff * df_for, so the two agree under
                # a market_forward override or non-standard delivery.
                tau_m = option.get_maturity(fx_env)
                fwd = fx_env.get_forward(tau_m)
                base = fwd * df_dom - asset_call

        return option.payout * base * option.participation_rate

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
