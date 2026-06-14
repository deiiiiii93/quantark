"""Equity Heston ADI PDE solver for European vanillas."""

from __future__ import annotations

from typing import Dict, Optional, Union

from copy import deepcopy

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.param.rrf import ParallelShiftRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import ADIScheme, EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.pde_kernel import (
    price_european_heston_pde,
    price_delta_gamma_heston_pde,
)


class HestonPDESolver(BaseEngine):
    """Heston ADI PDE pricing (Douglas / Craig-Sneyd) for European vanillas.

    For European options the Heston price depends only on the terminal forward and
    discount, so constant curve-consistent (r, carry) is curve-exact. Greeks
    delta/gamma/theta/rho (no vega) hold HestonParams fixed and pin the spatial grid.
    """

    engine_type = EngineType.PDE

    def __init__(self, model_params: HestonParams,
                 scheme: Union[ADIScheme, str] = ADIScheme.CRAIG_SNEYD,
                 n_x: int = 200, n_v: int = 100, n_t: int = 100, use_sparse: bool = False,
                 engine_params: Optional[EngineParams] = None):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams instance")
        super().__init__(engine_params if engine_params is not None else EngineParams())
        self.model_params = model_params
        try:
            self.scheme = (ADIScheme[scheme.upper()] if isinstance(scheme, str) else scheme)
        except KeyError:
            raise ValidationError(f"unknown ADI scheme: {scheme}")
        self.n_x, self.n_v, self.n_t, self.use_sparse = n_x, n_v, n_t, use_sparse
        self._greeks_grid_spot = 0.0

    def _price(self, product: BaseEquityProduct, env: PricingEnvironment) -> float:
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError("HestonPDESolver supports EuropeanVanillaOption only")
        T = float(product.get_maturity(env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        unit = price_european_heston_pde(
            s0=float(env.spot), strike=float(product.strike),
            is_call=product.option_type == OptionType.CALL, T=T, params=self.model_params,
            r=float(env.get_rate(T)), carry=float(env.get_div_yield(T)),
            n_x=self.n_x, n_v=self.n_v, n_t=self.n_t, scheme=self.scheme,
            use_sparse=self.use_sparse, grid_spot=self._greeks_grid_spot,
        )
        return unit * float(getattr(product, "contract_multiplier", 1.0))

    def price(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> float:
        return self._price(product, pricing_env)

    def calculate_greeks(self, product: BaseEquityProduct,
                         pricing_env: PricingEnvironment) -> Dict[str, float]:
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError("HestonPDESolver supports EuropeanVanillaOption only")
        T = float(product.get_maturity(pricing_env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        bump = self.params.get_effective_bump_config()
        mult = float(getattr(product, "contract_multiplier", 1.0))

        # delta/gamma from a SINGLE PDE solve (spatial derivatives of the solved surface).
        price, delta, gamma = price_delta_gamma_heston_pde(
            s0=float(pricing_env.spot), strike=float(product.strike),
            is_call=product.option_type == OptionType.CALL, T=T, params=self.model_params,
            r=float(pricing_env.get_rate(T)), carry=float(pricing_env.get_div_yield(T)),
            n_x=self.n_x, n_v=self.n_v, n_t=self.n_t, scheme=self.scheme, use_sparse=self.use_sparse,
        )
        greeks: Dict[str, float] = {
            "price": price * mult, "delta": delta * mult, "gamma": gamma * mult,
        }

        # rho via bumped rate curve (reprice); theta via shrunk maturity (reprice).
        base = price * mult
        env_ru = deepcopy(pricing_env)
        env_ru.rate_curve = ParallelShiftRateCurve(pricing_env.rate_curve, bump.rate_bump)
        env_rd = deepcopy(pricing_env)
        env_rd.rate_curve = ParallelShiftRateCurve(pricing_env.rate_curve, -bump.rate_bump)
        greeks["rho"] = (self._price(product, env_ru) - self._price(product, env_rd)) / (2.0 * bump.rate_bump) / 100.0

        # Theta: shrink a float maturity; for date-based products advance the valuation date.
        maturity_attr = getattr(product, "maturity", None)
        if maturity_attr is not None and maturity_attr > 0:
            eff = min(bump.time_bump_days / 365.0, 0.5 * float(maturity_attr))
            shifted = deepcopy(product)
            shifted.maturity = float(maturity_attr) - eff
            greeks["theta"] = (self._price(shifted, pricing_env) - base) / (eff * 365.0)
        else:
            from datetime import timedelta
            env_fut = deepcopy(pricing_env)
            env_fut.valuation_date = pricing_env.valuation_date + timedelta(days=bump.time_bump_days)
            greeks["theta"] = (self._price(product, env_fut) - base) / bump.time_bump_days
        return greeks
