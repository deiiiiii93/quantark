"""FX Heston Stochastic-Local-Volatility Monte Carlo engine (European vanillas)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.localvol_common import (
    build_fx_local_vol, check_fx_v1_restrictions, fx_contract_value,
)
from quantark.asset.fx.engine.localvol_greeks import fx_local_vol_model_greeks
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.curves import forward_rates_on_grid
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.slv import BinMethod, price_european_slv_mc


class FxHestonSLVMCEngine(BaseFxEngine):
    """FX Heston SLV MC pricing (carry = foreign rate). GK sizing; v1 restrictions;
    greeks delta/gamma/theta/rho_dom/rho_for (no vega), local-vol surface held fixed."""

    engine_type = EngineType.MONTE_CARLO

    def __init__(self, model_params: HestonParams, eta: float = 1.0,
                 num_bins: int = 20, bin_method: BinMethod = BinMethod.EQUAL_WEIGHTED,
                 params: Optional[FxEngineParams] = None, num_paths: int = 60_000,
                 time_steps: int = 100, seed: int = 42,
                 local_vol_surface: Optional[LocalVolSurface] = None):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams instance")
        if eta < 0:
            raise ValidationError("eta must be non-negative")
        super().__init__(params)
        self.model_params, self.eta = model_params, eta
        self.num_bins, self.bin_method = num_bins, bin_method
        self.num_paths, self.time_steps, self.seed = num_paths, time_steps, seed
        self._prebuilt = local_vol_surface

    def _price_with_surface(self, product: BaseFxProduct, fx_env: FxPricingEnvironment,
                            lv: LocalVolSurface) -> float:
        from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
        if not isinstance(product, FxVanillaOption):
            raise PricingError("FxHestonSLVMCEngine supports FxVanillaOption only")
        check_fx_v1_restrictions(product, fx_env)
        T = float(product.get_maturity(fx_env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        n = int(self.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(fx_env.domestic_curve, t_grid)
        carry_fwd = forward_rates_on_grid(fx_env.foreign_curve, t_grid)
        unit = price_european_slv_mc(
            s0=float(fx_env.effective_spot()), strike=float(product.strike),
            is_call=product.is_call(), params=self.model_params, lv_surface=lv,
            step_dt=np.diff(t_grid), r_fwd=r_fwd, carry_fwd=carry_fwd,
            disc_factor=float(fx_env.domestic_curve.get_discount_factor(T)), eta=self.eta,
            num_paths=self.num_paths, num_bins=self.num_bins, bin_method=self.bin_method,
            seed=self.seed,
        )
        return fx_contract_value(product, fx_env, unit)

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        return self._price_with_surface(product, fx_env, build_fx_local_vol(fx_env, self._prebuilt))

    def calculate_greeks(self, product: BaseFxProduct,
                         fx_env: FxPricingEnvironment) -> Dict[str, float]:
        lv = build_fx_local_vol(fx_env, self._prebuilt)
        return fx_local_vol_model_greeks(
            self._price_with_surface, product, fx_env, lv,
            spot_bump=self.params.spot_bump, rate_bump=self.params.rate_bump,
            theta_days=int(self.params.theta_days),
        )
