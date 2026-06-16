"""
Vanna-Volga FX barrier engine adapting the functional VV pricers to the
standard BaseFxEngine contract (price + bump-and-reprice Greeks).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option import FxOneTouchOption, FxBarrierOption
from quantark.param.vol.vannavolga import FXEnv, VannaVolgaVolSurface
from quantark.priceenv import FxPricingEnvironment
from quantark.util.exceptions import MarketDataError, PricingError

from .vv_barrier import VVBarrierResult, price_vv_one_touch
from .vv_vanilla_barrier import price_vv_barrier


class VannaVolgaBarrierEngine(BaseFxEngine):
    """Prices FxOneTouchOption and FxBarrierOption via Vanna-Volga.

    Greeks are inherited from BaseFxEngine (bump-and-reprice); they work
    because the VannaVolgaVolSurface re-anchors sticky-delta under spot/rate
    bumps and shifts all quotes under vega bumps (see the chain wiring).
    """

    def __init__(self, params: Optional[FxEngineParams] = None):
        super().__init__(params=params)

    def _build_fx_env(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> FXEnv:
        tau = product.get_maturity(fx_env)
        return FXEnv(
            spot=fx_env.spot,
            rd=fx_env.get_domestic_rate(tau),
            rf=fx_env.get_foreign_rate(tau),
            tau=tau,
        )

    def _surface(self, fx_env: FxPricingEnvironment) -> VannaVolgaVolSurface:
        surface = fx_env.vol_surface
        if not isinstance(surface, VannaVolgaVolSurface):
            raise MarketDataError(
                "VannaVolgaBarrierEngine requires a VannaVolgaVolSurface "
                f"vol surface; got {type(surface).__name__}."
            )
        return surface

    def price_details(
        self, product: BaseFxProduct, fx_env: FxPricingEnvironment
    ) -> VVBarrierResult:
        env = self._build_fx_env(product, fx_env)
        surface = self._surface(fx_env)
        if isinstance(product, FxOneTouchOption):
            result = price_vv_one_touch(
                env, surface.quotes, product.barrier, product.is_up,
                conv=surface.conv, premium_included_atm=surface.premium_included_atm,
            )
            # price_vv_one_touch returns a UNIT one-touch; scale by the
            # product payout so price and bump-and-reprice Greeks are correct.
            if product.payout != 1.0:
                result = dataclasses.replace(
                    result,
                    bstv=result.bstv * product.payout,
                    vv=result.vv * product.payout,
                )
            return result
        if isinstance(product, FxBarrierOption):
            return price_vv_barrier(
                env, surface.quotes, product.strike, product.barrier,
                product.is_up, product.is_call(),
                knock_in=(product.knock_type.value == "knock_in"),
                rebate=product.rebate, rebate_at_hit=product.rebate_at_hit,
                conv=surface.conv, premium_included_atm=surface.premium_included_atm,
            )
        raise PricingError(
            f"VannaVolgaBarrierEngine cannot price {type(product).__name__}; "
            "expected FxOneTouchOption or FxBarrierOption."
        )

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        result = self.price_details(product, fx_env)
        return result.vv
