"""Equity Heston Stochastic-Local-Volatility Monte Carlo engine (European vanillas)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.localvol_greeks import local_vol_model_greeks
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.curves import forward_carry_on_grid, forward_rates_on_grid
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.slv import BinMethod, LeverageSurface, price_european_slv_mc


class HestonSLVMCEngine(BaseEngine):
    """Heston Stochastic-Local-Volatility MC pricing for European vanillas.

    The Dupire local-vol leg is built from the environment's market GridVolSurface (or a
    prebuilt LocalVolSurface). Leverage is calibrated on-the-fly by binning. Greeks
    (delta/gamma/theta/rho, no vega) hold HestonParams + the local-vol surface fixed.
    """

    engine_type = EngineType.MONTE_CARLO
    settlement_support = SettlementSupport.TERMINAL_ONLY
    supports_lifecycle_state = True

    def __init__(self, model_params: HestonParams, eta: float = 1.0,
                 num_bins: int = 20, bin_method: BinMethod = BinMethod.EQUAL_WEIGHTED,
                 params: Optional[MCParams] = None,
                 local_vol_surface: Optional[LocalVolSurface] = None,
                 leverage_surface: Optional[LeverageSurface] = None):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams instance")
        if eta < 0:
            raise ValidationError("eta must be non-negative")
        if leverage_surface is not None and not isinstance(leverage_surface, LeverageSurface):
            raise ValidationError("leverage_surface must be a LeverageSurface when provided")
        super().__init__(params if params is not None else MCParams())
        self.model_params = model_params
        self.eta, self.num_bins, self.bin_method = eta, num_bins, bin_method
        self._prebuilt = local_vol_surface
        self._prebuilt_leverage = leverage_surface

    def _build_surface(self, env: PricingEnvironment) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError("HestonSLVMCEngine requires a GridVolSurface or prebuilt LocalVolSurface")
        return build_dupire_local_vol(env.vol_surface, spot=env.spot,
                                      rate_curve=env.rate_curve, div_yield=env.get_div_yield)

    def _price_with_surface(self, product: BaseEquityProduct, env: PricingEnvironment,
                            lv: LocalVolSurface) -> float:
        return self._price_with_artifacts(
            product,
            env,
            lv,
            self._prebuilt_leverage,
        )

    def _price_with_artifacts(self, product: BaseEquityProduct, env: PricingEnvironment,
                              lv: LocalVolSurface,
                              leverage: Optional[LeverageSurface],
                              *,
                              payment_df: Optional[float] = None) -> float:
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError("HestonSLVMCEngine supports EuropeanVanillaOption only")
        T = float(product.get_maturity(env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        unit = price_european_slv_mc(
            s0=float(env.spot), strike=float(product.strike),
            is_call=product.option_type == OptionType.CALL, params=self.model_params,
            lv_surface=lv, step_dt=np.diff(t_grid), r_fwd=r_fwd, carry_fwd=carry_fwd,
            disc_factor=(
                float(env.get_discount_factor(T))
                if payment_df is None
                else float(payment_df)
            ),
            eta=self.eta,
            num_paths=int(self.params.num_paths), num_bins=self.num_bins,
            bin_method=self.bin_method, seed=int(self.params.seed),
            leverage_surface=leverage,
            use_antithetic=bool(getattr(self.params, "use_antithetic", False)),
        )
        return unit * float(getattr(product, "contract_multiplier", 1.0))

    def _price_with_settlement(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        lv: LocalVolSurface,
        lifecycle_state=None,
    ) -> float:
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv
        timing = resolve_terminal_timing(product, pricing_env)
        return (
            self._price_with_artifacts(
                product,
                pricing_env,
                lv,
                self._prebuilt_leverage,
                payment_df=timing.payment_df,
            )
            + pending_receivable_pv(lifecycle_state, pricing_env)
        )

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        return self._price_with_settlement(
            product,
            pricing_env,
            self._build_surface(pricing_env),
            lifecycle_state,
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
        lv = self._build_surface(pricing_env)
        bump = self.params.get_effective_bump_config()
        return local_vol_model_greeks(
            lambda p, e, surface: self._price_with_settlement(
                p,
                e,
                surface,
                lifecycle_state,
            ),
            product,
            pricing_env,
            lv,
            spot_bump=bump.spot_bump, rate_bump=bump.rate_bump, theta_days=bump.time_bump_days,
        )
