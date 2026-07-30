"""Equity Heston semi-analytical engine for European vanillas."""

from __future__ import annotations

from typing import Dict, Optional, Union

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.localvol_greeks import local_vol_model_greeks
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType, HestonAnalyticalMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.analytical_kernel import heston_call_price, heston_put_price


def _resolve_method(method) -> str:
    if method is None:
        return "gatheral"
    if isinstance(method, tuple):  # EngineType.ANALYTICAL(HestonAnalyticalMethod.X)
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


class HestonAnalyticalEngine(BaseEngine):
    """Semi-analytical Heston European vanilla pricing (Lewis/Gatheral/Weber).

    Model parameters (HestonParams) are supplied at construction; market data (spot,
    rate curve, dividend yield) comes from the pricing environment. Greeks expose
    delta/gamma/theta/rho (no scalar vega; structured volatility risk is separate).
    """

    engine_type = EngineType.ANALYTICAL
    settlement_support = SettlementSupport.TERMINAL_ONLY
    supports_lifecycle_state = True

    def __init__(self, model_params: HestonParams,
                 method: Union[str, HestonAnalyticalMethod, tuple, None] = None,
                 engine_params: Optional[EngineParams] = None):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams instance")
        super().__init__(engine_params if engine_params is not None else EngineParams())
        self.model_params = model_params
        self.method = _resolve_method(method)

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError("HestonAnalyticalEngine supports EuropeanVanillaOption only")
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv
        timing = resolve_terminal_timing(product, pricing_env)
        T = float(product.get_maturity(pricing_env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        s0 = float(pricing_env.spot)
        r = float(pricing_env.get_rate(T))
        carry = float(pricing_env.get_div_yield(T))
        k = float(product.strike)
        if product.option_type == OptionType.CALL:
            unit = heston_call_price(s0, k, T, self.model_params, r, carry, method=self.method)
        else:
            unit = heston_put_price(s0, k, T, self.model_params, r, carry, method=self.method)
        return (
            unit
            * float(getattr(product, "contract_multiplier", 1.0))
            * timing.delay_df
            + pending_receivable_pv(lifecycle_state, pricing_env)
        )

    def calculate_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> Dict[str, float]:
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return {"price": fixed_pv, "delta": 0.0, "gamma": 0.0}
        bump = self.params.get_effective_bump_config()
        return local_vol_model_greeks(
            lambda p, e, _surface: self.price(
                p,
                e,
                lifecycle_state=lifecycle_state,
            ),
            product,
            pricing_env,
            None,
            spot_bump=bump.spot_bump, rate_bump=bump.rate_bump,
            theta_days=bump.time_bump_days,
        )
