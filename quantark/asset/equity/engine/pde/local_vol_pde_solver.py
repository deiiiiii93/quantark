"""Equity Local-Volatility Crank-Nicolson PDE solver (European vanillas)."""

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
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.curves import forward_carry_on_grid, forward_rates_on_grid
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.localvol.pde_kernel import price_european_lv_pde


class LocalVolPDESolver(BaseEngine):
    """Crank-Nicolson PDE pricing under a Dupire local-volatility surface.

    Deterministic (never invokes MC). Per-step forward rates/carry from the curves are
    threaded into the kernel. Greeks (delta/gamma/theta/rho; no vega) hold the
    calibrated surface fixed.
    """

    engine_type = EngineType.PDE
    settlement_support = SettlementSupport.TERMINAL_ONLY
    supports_lifecycle_state = True

    def __init__(self, params: Optional[PDEParams] = None,
                 local_vol_surface: Optional[LocalVolSurface] = None):
        super().__init__(params if params is not None else PDEParams())
        self._prebuilt = local_vol_surface
        self._greeks_smax = 0.0  # >0 pins the spatial grid across Greek bumps

    def _build_surface(self, env: PricingEnvironment) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError(
                "LocalVolPDESolver requires a GridVolSurface (market IV grid) or a "
                "prebuilt LocalVolSurface"
            )
        return build_dupire_local_vol(
            env.vol_surface, spot=env.spot, rate_curve=env.rate_curve,
            div_yield=env.get_div_yield,
        )

    def _price_with_surface(self, product: BaseEquityProduct, env: PricingEnvironment,
                            lv: LocalVolSurface) -> float:
        if not isinstance(product, _supported_products()):
            raise PricingError("LocalVolPDESolver supports EuropeanVanillaOption only")
        T = float(product.get_maturity(env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        is_call = product.option_type == OptionType.CALL
        s_max = self._greeks_smax if self._greeks_smax > 0 else (
            float(self.params.s_max) if self.params.s_max > 0 else 0.0
        )
        unit = price_european_lv_pde(
            s0=float(env.spot), strike=float(product.strike), is_call=is_call, T=T,
            lv_surface=lv, step_dt=np.diff(t_grid), r_fwd=r_fwd, carry_fwd=carry_fwd,
            n_s=int(self.params.grid_size), s_max=s_max, theta=float(self.params.theta),
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
            self._price_with_surface(product, pricing_env, lv)
            * timing.delay_df
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
        # Pin the spatial grid (s_max) across spot bumps so finite differences are not
        # contaminated by grid movement.
        base_smax = float(self.params.s_max) if self.params.s_max > 0 else (
            4.0 * max(float(pricing_env.spot), float(getattr(product, "strike", pricing_env.spot)))
        )
        self._greeks_smax = base_smax
        try:
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
                spot_bump=bump.spot_bump, rate_bump=bump.rate_bump,
                theta_days=bump.time_bump_days,
            )
        finally:
            self._greeks_smax = 0.0


def _supported_products():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    return (EuropeanVanillaOption,)
