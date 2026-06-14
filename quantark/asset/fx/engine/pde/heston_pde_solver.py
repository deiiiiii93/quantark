"""FX Heston ADI PDE solver for European vanillas."""

from __future__ import annotations

from typing import Dict, Optional, Union

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.localvol_common import check_fx_v1_restrictions, fx_contract_value
from quantark.asset.fx.engine.localvol_greeks import fx_local_vol_model_greeks
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import ADIScheme, EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde


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
        self._greeks_grid_spot = float(fx_env.effective_spot())
        try:
            return fx_local_vol_model_greeks(
                lambda p, e, _s: self._price(p, e), product, fx_env, None,
                spot_bump=self.params.spot_bump, rate_bump=self.params.rate_bump,
                theta_days=int(self.params.theta_days),
            )
        finally:
            self._greeks_grid_spot = 0.0
