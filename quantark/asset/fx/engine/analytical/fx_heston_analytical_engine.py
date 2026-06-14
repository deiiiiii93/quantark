"""FX Heston semi-analytical engine for European vanillas."""

from __future__ import annotations

from typing import Dict, Optional, Union

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.localvol_common import check_fx_v1_restrictions, fx_contract_value
from quantark.asset.fx.engine.localvol_greeks import fx_local_vol_model_greeks
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType, HestonAnalyticalMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.analytical_kernel import heston_call_price, heston_put_price


def _resolve_method(method) -> str:
    if method is None:
        return "gatheral"
    if isinstance(method, tuple):
        if len(method) != 2 or method[0] != EngineType.ANALYTICAL:
            raise ValidationError(f"method tuple must be EngineType.ANALYTICAL(method), got {method!r}")
        method = method[1]
    if isinstance(method, HestonAnalyticalMethod):
        return method.name.lower()
    if isinstance(method, str):
        m = method.lower()
        if m not in ("lewis", "gatheral", "weber"):
            raise ValidationError(f"unknown Heston analytical method: {method}")
        return m
    raise ValidationError(f"invalid method specifier: {method!r}")


class FxHestonAnalyticalEngine(BaseFxEngine):
    """Semi-analytical Heston FX vanilla pricing (carry = foreign rate).

    GK-consistent sizing (notional*participation*annualization). v1 restrictions
    enforced (expiry==delivery, no market_forward, spot_days==0). Greeks expose
    delta/gamma/theta/rho_dom/rho_for (no IV-bump vega).
    """

    engine_type = EngineType.ANALYTICAL

    def __init__(self, model_params: HestonParams,
                 method: Union[str, HestonAnalyticalMethod, tuple, None] = None,
                 params: Optional[FxEngineParams] = None):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams instance")
        super().__init__(params)
        self.model_params = model_params
        self.method = _resolve_method(method)

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
        if not isinstance(product, FxVanillaOption):
            raise PricingError("FxHestonAnalyticalEngine supports FxVanillaOption only")
        check_fx_v1_restrictions(product, fx_env)
        T = float(product.get_maturity(fx_env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        s0 = float(fx_env.effective_spot())
        r = float(fx_env.get_domestic_rate(T))
        carry = float(fx_env.get_foreign_rate(T))
        k = float(product.strike)
        if product.is_call():
            unit = heston_call_price(s0, k, T, self.model_params, r, carry, method=self.method)
        else:
            unit = heston_put_price(s0, k, T, self.model_params, r, carry, method=self.method)
        return fx_contract_value(product, fx_env, unit)

    def calculate_greeks(self, product: BaseFxProduct,
                         fx_env: FxPricingEnvironment) -> Dict[str, float]:
        return fx_local_vol_model_greeks(
            lambda p, e, _surface: self.price(p, e), product, fx_env, None,
            spot_bump=self.params.spot_bump, rate_bump=self.params.rate_bump,
            theta_days=int(self.params.theta_days),
        )
