"""Equity Heston Monte Carlo engine for European vanillas."""

from __future__ import annotations

from typing import Dict, Optional, Union

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
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType, HestonMCScheme
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.curves import forward_carry_on_grid, forward_rates_on_grid
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.mc_kernel import price_european_heston_mc


class HestonMCEngine(BaseEngine):
    """Monte Carlo Heston pricing (Euler / EulerLog / QuadExp) for European vanillas.

    Model params at construction; market data from the environment. Greeks expose
    delta/gamma/theta/rho (no vega), holding HestonParams fixed.
    """

    engine_type = EngineType.MONTE_CARLO
    settlement_support = SettlementSupport.TERMINAL_ONLY
    supports_lifecycle_state = True

    def __init__(self, model_params: HestonParams,
                 scheme: Union[HestonMCScheme, str] = HestonMCScheme.QUADEXP,
                 params: Optional[MCParams] = None):
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams instance")
        super().__init__(params if params is not None else MCParams())
        self.model_params = model_params
        try:
            self.scheme = (HestonMCScheme[scheme.upper()] if isinstance(scheme, str) else scheme)
        except KeyError:
            raise ValidationError(f"unknown Heston MC scheme: {scheme}")
        if not isinstance(self.scheme, HestonMCScheme):
            raise ValidationError("scheme must be a HestonMCScheme")

    def _price(
        self,
        product: BaseEquityProduct,
        env: PricingEnvironment,
        *,
        payment_df: Optional[float] = None,
    ) -> float:
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError("HestonMCEngine supports EuropeanVanillaOption only")
        T = float(product.get_maturity(env))
        if T <= 0:
            raise ValidationError("maturity must be positive")
        n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        unit = price_european_heston_mc(
            s0=float(env.spot), strike=float(product.strike),
            is_call=product.option_type == OptionType.CALL, params=self.model_params,
            step_dt=np.diff(t_grid), r_fwd=r_fwd, carry_fwd=carry_fwd,
            disc_factor=(
                float(env.get_discount_factor(T))
                if payment_df is None
                else float(payment_df)
            ),
            scheme=self.scheme,
            num_paths=int(self.params.num_paths), seed=int(self.params.seed),
            use_antithetic=bool(self.params.use_antithetic),
        )
        return unit * float(getattr(product, "contract_multiplier", 1.0))

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv
        timing = resolve_terminal_timing(product, pricing_env)
        return (
            self._price(product, pricing_env, payment_df=timing.payment_df)
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
            lambda p, e, _s: self.price(
                p,
                e,
                lifecycle_state=lifecycle_state,
            ),
            product,
            pricing_env,
            None,
            spot_bump=bump.spot_bump, rate_bump=bump.rate_bump, theta_days=bump.time_bump_days,
        )
