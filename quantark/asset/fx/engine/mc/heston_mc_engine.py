"""FX Heston Monte Carlo engine for European vanillas."""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.localvol_common import check_fx_v1_restrictions, fx_contract_value
from quantark.asset.fx.engine.localvol_greeks import fx_local_vol_model_greeks
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType, HestonMCScheme
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.curves import forward_rates_on_grid
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.mc_kernel import price_european_heston_mc


class FxHestonMCEngine(BaseFxEngine):
    """Monte Carlo Heston FX pricing (carry = foreign rate). GK-consistent sizing;
    v1 restrictions enforced; greeks delta/gamma/theta/rho_dom/rho_for, no vega."""

    engine_type = EngineType.MONTE_CARLO

    def __init__(self, model_params: HestonParams,
                 scheme: Union[HestonMCScheme, str] = HestonMCScheme.QUADEXP,
                 params: Optional[FxEngineParams] = None, num_paths: int = 100_000,
                 time_steps: int = 100, seed: int = 42, use_antithetic: bool = True):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams instance")
        super().__init__(params)
        self.model_params = model_params
        try:
            self.scheme = (HestonMCScheme[scheme.upper()] if isinstance(scheme, str) else scheme)
        except KeyError:
            raise ValidationError(f"unknown Heston MC scheme: {scheme}")
        if not isinstance(self.scheme, HestonMCScheme):
            raise ValidationError("scheme must be a HestonMCScheme")
        self.num_paths, self.time_steps, self.seed = num_paths, time_steps, seed
        self.use_antithetic = use_antithetic

    def _price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
        if not isinstance(product, FxVanillaOption):
            raise PricingError("FxHestonMCEngine supports FxVanillaOption only")
        check_fx_v1_restrictions(product, fx_env)
        T = float(product.get_maturity(fx_env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        n = int(self.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(fx_env.domestic_curve, t_grid)
        carry_fwd = forward_rates_on_grid(fx_env.foreign_curve, t_grid)
        unit = price_european_heston_mc(
            s0=float(fx_env.effective_spot()), strike=float(product.strike),
            is_call=product.is_call(), params=self.model_params, step_dt=np.diff(t_grid),
            r_fwd=r_fwd, carry_fwd=carry_fwd,
            disc_factor=float(fx_env.domestic_curve.get_discount_factor(T)),
            scheme=self.scheme, num_paths=self.num_paths, seed=self.seed,
            use_antithetic=self.use_antithetic,
        )
        return fx_contract_value(product, fx_env, unit)

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        return self._price(product, fx_env)

    def calculate_greeks(self, product: BaseFxProduct,
                         fx_env: FxPricingEnvironment) -> Dict[str, float]:
        return fx_local_vol_model_greeks(
            lambda p, e, _s: self._price(p, e), product, fx_env, None,
            spot_bump=self.params.spot_bump, rate_bump=self.params.rate_bump,
            theta_days=int(self.params.theta_days),
        )
