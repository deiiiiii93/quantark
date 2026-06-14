"""FX Heston ADI PDE solver for European vanillas."""

from __future__ import annotations

from typing import Dict, Optional, Union

from copy import deepcopy

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.localvol_common import check_fx_v1_restrictions, fx_contract_value
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.param.rrf import ParallelShiftRateCurve
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import ADIScheme, EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.pde_kernel import (
    price_european_heston_pde,
    price_delta_gamma_heston_pde,
)


class FxHestonPDESolver(BaseFxEngine):
    """Heston ADI PDE FX pricing (carry = foreign rate). GK sizing; v1 restrictions;
    greeks delta/gamma/theta/rho_dom/rho_for (no vega), grid pinned across spot bumps."""

    engine_type = EngineType.PDE

    def __init__(self, model_params: HestonParams,
                 scheme: Union[ADIScheme, str] = ADIScheme.CRAIG_SNEYD,
                 n_x: int = 200, n_v: int = 100, n_t: int = 100, use_sparse: bool = False,
                 params: Optional[FxEngineParams] = None):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams instance")
        super().__init__(params)
        self.model_params = model_params
        try:
            self.scheme = (ADIScheme[scheme.upper()] if isinstance(scheme, str) else scheme)
        except KeyError:
            raise ValidationError(f"unknown ADI scheme: {scheme}")
        self.n_x, self.n_v, self.n_t, self.use_sparse = n_x, n_v, n_t, use_sparse
        self._greeks_grid_spot = 0.0

    def _price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
        if not isinstance(product, FxVanillaOption):
            raise PricingError("FxHestonPDESolver supports FxVanillaOption only")
        check_fx_v1_restrictions(product, fx_env)
        T = float(product.get_maturity(fx_env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        unit = price_european_heston_pde(
            s0=float(fx_env.effective_spot()), strike=float(product.strike),
            is_call=product.is_call(), T=T, params=self.model_params,
            r=float(fx_env.get_domestic_rate(T)), carry=float(fx_env.get_foreign_rate(T)),
            n_x=self.n_x, n_v=self.n_v, n_t=self.n_t, scheme=self.scheme,
            use_sparse=self.use_sparse, grid_spot=self._greeks_grid_spot,
        )
        return fx_contract_value(product, fx_env, unit)

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        return self._price(product, fx_env)

    def calculate_greeks(self, product: BaseFxProduct,
                         fx_env: FxPricingEnvironment) -> Dict[str, float]:
        from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
        if not isinstance(product, FxVanillaOption):
            raise PricingError("FxHestonPDESolver supports FxVanillaOption only")
        check_fx_v1_restrictions(product, fx_env)
        T = float(product.get_maturity(fx_env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        size = (float(product.notional) * float(product.participation_rate)
                * float(product.annualization_factor(fx_env)))

        price_u, delta_u, gamma_u = price_delta_gamma_heston_pde(
            s0=float(fx_env.effective_spot()), strike=float(product.strike),
            is_call=product.is_call(), T=T, params=self.model_params,
            r=float(fx_env.get_domestic_rate(T)), carry=float(fx_env.get_foreign_rate(T)),
            n_x=self.n_x, n_v=self.n_v, n_t=self.n_t, scheme=self.scheme, use_sparse=self.use_sparse,
        )
        greeks: Dict[str, float] = {
            "price": price_u * size, "delta": delta_u * size, "gamma": gamma_u * size,
        }
        base = price_u * size
        rb = self.params.rate_bump
        env_du = deepcopy(fx_env)
        env_du.domestic_curve = ParallelShiftRateCurve(fx_env.domestic_curve, rb)
        env_dd = deepcopy(fx_env)
        env_dd.domestic_curve = ParallelShiftRateCurve(fx_env.domestic_curve, -rb)
        greeks["rho_dom"] = (self._price(product, env_du) - self._price(product, env_dd)) / (2.0 * rb) / 100.0
        env_fu = deepcopy(fx_env)
        env_fu.foreign_curve = ParallelShiftRateCurve(fx_env.foreign_curve, rb)
        env_fd = deepcopy(fx_env)
        env_fd.foreign_curve = ParallelShiftRateCurve(fx_env.foreign_curve, -rb)
        greeks["rho_for"] = (self._price(product, env_fu) - self._price(product, env_fd)) / (2.0 * rb) / 100.0

        maturity_attr = getattr(product, "maturity", None)
        if maturity_attr is not None and maturity_attr > 0:
            eff = min(self.params.theta_days / 365.0, 0.5 * float(maturity_attr))
            shifted = deepcopy(product)
            shifted.maturity = float(maturity_attr) - eff
            if getattr(shifted, "delivery", None) is not None:
                shifted.delivery = shifted.delivery - eff
            greeks["theta"] = (self._price(shifted, fx_env) - base) / (eff * 365.0)
        else:
            from datetime import timedelta
            env_fut = deepcopy(fx_env)
            env_fut.valuation_date = fx_env.valuation_date + timedelta(days=self.params.theta_days)
            greeks["theta"] = (self._price(product, env_fut) - base) / self.params.theta_days
        return greeks
