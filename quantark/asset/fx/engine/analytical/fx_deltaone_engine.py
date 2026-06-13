"""
Pricing engine for FX delta-one products (spot, forward, swap).
"""

from typing import Dict, Optional, Union

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.deltaone import FxForward, FxSpot, FxSwap
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError


class FxDeltaOneEngine(BaseFxEngine):
    """
    Discounting engine for linear FX products.

    Spot / forward (net cash settlement in quote currency):
        NPV_quote = N * (F_mkt - K) * df_dom(T)
        NPV_base  = N * (F_mkt - K) / S * df_for(T)

    where F_mkt is the outright market forward to the settlement date
    (interest rate parity from spot, or a quoted market forward) and the
    quote-currency NPV is the complete economic value.

    Swap (physical exchange on both legs):
        quote leg:  N * K_near * df_dom(t_near) - N * K_far * df_dom(t_far)
        base leg:   N * df_for(t_far) - N * df_for(t_near)
        NPV (quote) = quote leg + base leg * S

    A near leg that settled before the valuation date contributes nothing.

    price() returns the total NPV in domestic (quote) currency;
    price_details() returns the full breakdown.
    """

    engine_type = EngineType.ANALYTICAL

    def __init__(self, params: Optional[FxEngineParams] = None):
        super().__init__(params)

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        """
        Total NPV in domestic (quote) currency.

        Raises:
            PricingError: If the product is not an FX delta-one product
        """
        details = self.price_details(product, fx_env)
        return float(details["npv"])

    def price_details(
        self, product: BaseFxProduct, fx_env: FxPricingEnvironment
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Full pricing breakdown.

        Common keys: npv (quote currency), npv_quote_currency,
        npv_base_currency, currency_pair. Spot/forward additionally report
        market_forward_rate, forward_points, contract_forward_points and the
        discount factors; swaps report per-leg discount factors and
        near_leg_expired.
        """
        if isinstance(product, FxSwap):
            return self._swap_details(product, fx_env)
        if isinstance(product, (FxForward, FxSpot)):
            return self._outright_details(product, fx_env)
        raise PricingError(
            f"FxDeltaOneEngine only supports FxSpot, FxForward and FxSwap, "
            f"got {type(product).__name__}"
        )

    # ------------------------------------------------------------------
    # Outright (spot / forward)
    # ------------------------------------------------------------------

    def _outright_details(
        self,
        product: Union[FxForward, FxSpot],
        fx_env: FxPricingEnvironment,
    ) -> Dict[str, Union[float, bool, str]]:
        t = product.get_maturity(fx_env)
        spot = fx_env.spot
        forward = fx_env.get_forward(t)
        df_dom = fx_env.get_domestic_df(t)
        df_for = fx_env.get_foreign_df(t)

        n = product.notional_base
        k = product.contract_rate
        npv_quote = n * (forward - k) * df_dom
        npv_base = n * (forward - k) / spot * df_for

        return {
            "currency_pair": str(product.currency_pair),
            "npv": npv_quote,
            "npv_quote_currency": npv_quote,
            "npv_base_currency": npv_base,
            "market_spot_rate": spot,
            "market_forward_rate": forward,
            "contract_rate": k,
            "forward_points": forward - spot,
            "contract_forward_points": k - spot,
            "domestic_discount_factor": df_dom,
            "foreign_discount_factor": df_for,
            "years_to_settlement": t,
        }

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------

    def _swap_details(
        self, swap: FxSwap, fx_env: FxPricingEnvironment
    ) -> Dict[str, Union[float, bool, str]]:
        t_far = swap.get_maturity(fx_env)
        df_dom_far = fx_env.get_domestic_df(t_far)
        df_for_far = fx_env.get_foreign_df(t_far)

        near_expired = swap.is_near_leg_expired(fx_env)
        if near_expired:
            df_dom_near = 0.0
            df_for_near = 0.0
        else:
            t_near = swap.get_near_time(fx_env)
            df_dom_near = fx_env.get_domestic_df(t_near)
            df_for_near = fx_env.get_foreign_df(t_near)

        n = swap.notional_base
        # Near leg: sell base / receive quote; far leg: buy base / pay quote
        npv_quote = n * swap.near_rate * df_dom_near - n * swap.far_rate * df_dom_far
        npv_base = n * df_for_far - n * df_for_near

        spot = fx_env.spot
        return {
            "currency_pair": str(swap.currency_pair),
            "npv": npv_quote + npv_base * spot,
            "npv_quote_currency": npv_quote,
            "npv_base_currency": npv_base,
            "market_spot_rate": spot,
            "swap_points": swap.swap_points,
            "near_leg_expired": near_expired,
            "domestic_discount_factor_near": df_dom_near,
            "foreign_discount_factor_near": df_for_near,
            "domestic_discount_factor_far": df_dom_far,
            "foreign_discount_factor_far": df_for_far,
            "years_to_far_settlement": t_far,
        }

    def __repr__(self):
        return "FxDeltaOneEngine(analytical)"
