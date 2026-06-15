"""FX Local-Volatility Monte Carlo engine (European vanillas)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.localvol_common import (
    build_fx_local_vol,
    check_fx_v1_restrictions,
    fx_contract_value,
)
from quantark.asset.fx.engine.localvol_greeks import fx_local_vol_model_greeks
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.curves import forward_rates_on_grid
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc


class FxLocalVolMCEngine(BaseFxEngine):
    """Monte Carlo FX pricing under a Dupire local-volatility surface (carry = r_f).

    Greeks expose delta/gamma/theta/rho_dom/rho_for (no IV-bump vega), holding the
    calibrated surface fixed.
    """

    engine_type = EngineType.MONTE_CARLO

    def __init__(self, params: Optional[FxEngineParams] = None, num_paths: int = 100_000,
                 time_steps: int = 200, seed: int = 42, use_antithetic: bool = True,
                 local_vol_surface: Optional[LocalVolSurface] = None):
        super().__init__(params)
        self.num_paths, self.time_steps, self.seed = num_paths, time_steps, seed
        self.use_antithetic, self._prebuilt = use_antithetic, local_vol_surface

    def _price_with_surface(self, product: BaseFxProduct, fx_env: FxPricingEnvironment,
                            lv: LocalVolSurface) -> float:
        if not isinstance(product, _supported_products()):
            raise PricingError("FxLocalVolMCEngine supports FxVanillaOption only")
        check_fx_v1_restrictions(product, fx_env)
        T = float(product.get_maturity(fx_env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        n = int(self.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(fx_env.domestic_curve, t_grid)
        carry_fwd = forward_rates_on_grid(fx_env.foreign_curve, t_grid)
        disc = float(fx_env.domestic_curve.get_discount_factor(T))
        unit = price_european_lv_mc(
            s0=float(fx_env.effective_spot()), strike=float(product.strike),
            is_call=product.is_call(), lv_surface=lv, step_dt=np.diff(t_grid),
            r_fwd=r_fwd, carry_fwd=carry_fwd, disc_factor=disc, num_paths=self.num_paths,
            seed=self.seed, use_antithetic=self.use_antithetic,
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


def _supported_products():
    from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
    return (FxVanillaOption,)
