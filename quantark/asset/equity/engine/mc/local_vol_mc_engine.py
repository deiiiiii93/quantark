"""Equity Local-Volatility Monte Carlo engine (European vanillas)."""

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
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc


class LocalVolMCEngine(BaseEngine):
    """Monte Carlo pricing under a Dupire local-volatility surface.

    The local-vol surface is built from the environment's market GridVolSurface unless
    a prebuilt LocalVolSurface is supplied at construction. Greeks (delta/gamma/theta/rho;
    no vega) hold the calibrated surface fixed.
    """

    engine_type = EngineType.MONTE_CARLO
    settlement_support = SettlementSupport.TERMINAL_ONLY
    supports_lifecycle_state = True

    def __init__(self, params: Optional[MCParams] = None,
                 local_vol_surface: Optional[LocalVolSurface] = None):
        super().__init__(params if params is not None else MCParams())
        self._prebuilt = local_vol_surface

    def _build_surface(self, env: PricingEnvironment) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError(
                "LocalVolMCEngine requires a GridVolSurface (market IV grid) or a "
                "prebuilt LocalVolSurface"
            )
        return build_dupire_local_vol(
            env.vol_surface, spot=env.spot, rate_curve=env.rate_curve,
            div_yield=env.get_div_yield,
        )

    def _price_with_surface(
        self,
        product: BaseEquityProduct,
        env: PricingEnvironment,
        lv: LocalVolSurface,
        *,
        payment_df: Optional[float] = None,
    ) -> float:
        if not isinstance(product, _supported_products()):
            raise PricingError("LocalVolMCEngine supports EuropeanVanillaOption only")
        T = float(product.get_maturity(env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        disc = (
            float(env.get_discount_factor(T))
            if payment_df is None
            else float(payment_df)
        )
        is_call = product.option_type == OptionType.CALL
        unit = price_european_lv_mc(
            s0=float(env.spot), strike=float(product.strike), is_call=is_call,
            lv_surface=lv, step_dt=np.diff(t_grid), r_fwd=r_fwd, carry_fwd=carry_fwd,
            disc_factor=disc, num_paths=int(self.params.num_paths),
            seed=int(self.params.seed), use_antithetic=bool(self.params.use_antithetic),
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
            self._price_with_surface(
                product,
                pricing_env,
                lv,
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
            spot_bump=bump.spot_bump, rate_bump=bump.rate_bump,
            theta_days=bump.time_bump_days,
        )


def _supported_products():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    return (EuropeanVanillaOption,)
