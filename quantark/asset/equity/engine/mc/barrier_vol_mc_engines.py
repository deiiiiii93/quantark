"""Monte-Carlo barrier engines under the vol-model processes (Local Vol / Heston / SLV).

Thin wrappers over the ``quantark.volmodels`` barrier MC kernels: they resolve the per-step
forward rate/carry from the curves, resolve continuous vs discrete monitoring, price the
``BarrierOption``, and apply ``participation_rate`` once. Per-observation ``observation_schedule``
is out of scope for v1 and rejected. Mirrors the European ``LocalVolMCEngine`` / ``HestonMCEngine``
/ ``HestonSLVMCEngine`` wrappers.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType, ObservationType
from quantark.util.enum.engine_enums import EngineType, HestonMCScheme
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.curves import forward_carry_on_grid, forward_rates_on_grid
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.localvol.mc_kernel import price_barrier_lv_mc
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.mc_kernel import price_barrier_heston_mc
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.volmodels.slv.slv_mc_kernel import price_barrier_slv_mc


def _check_barrier(engine_name: str, product) -> None:
    if not isinstance(product, BarrierOption):
        raise PricingError(f"{engine_name} supports BarrierOption only")
    if product.observation_schedule is not None:
        raise ValidationError("observation_schedule (per-date barriers) is out of scope for v1")


def _monitoring(product: BarrierOption, t_grid: np.ndarray):
    """Return (continuous: bool, observe_idx: Optional[np.ndarray]) snapping discrete dates to grid."""
    if product.observation_type == ObservationType.CONTINUOUS:
        return True, None
    dates = product.observation_dates or []
    if not dates:
        raise ValidationError("discrete monitoring requires observation_dates")
    idx = []
    T = float(t_grid[-1])
    for d in dates:
        if d > T + 1e-12:
            raise ValidationError(f"observation date {d} exceeds maturity {T}")
        idx.append(int(np.argmin(np.abs(t_grid - d))))
    return False, np.unique(np.asarray(idx, dtype=int))


def _barrier_kwargs(product: BarrierOption, continuous, observe_idx):
    return dict(barrier=float(product.barrier), is_up=product.is_up_barrier, is_out=product.is_knock_out,
                rebate=float(product.rebate), pay_at_hit=bool(product.pay_at_hit),
                continuous=continuous, observe_idx=observe_idx)


class LocalVolBarrierMCEngine(BaseEngine):
    """Barrier MC under a Dupire local-volatility surface."""

    engine_type = EngineType.MONTE_CARLO

    def __init__(self, params: Optional[MCParams] = None, local_vol_surface: Optional[LocalVolSurface] = None):
        super().__init__(params or MCParams())
        self._prebuilt = local_vol_surface

    def price(self, product: BaseEquityProduct, env: PricingEnvironment) -> float:
        _check_barrier("LocalVolBarrierMCEngine", product)
        T = float(product.get_maturity(env)); n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        lv = self._prebuilt
        if lv is None:
            if not isinstance(env.vol_surface, GridVolSurface):
                raise PricingError("LocalVolBarrierMCEngine needs a GridVolSurface or a prebuilt LocalVolSurface")
            lv = build_dupire_local_vol(env.vol_surface, spot=env.spot, rate_curve=env.rate_curve,
                                        div_yield=env.get_div_yield)
        continuous, observe_idx = _monitoring(product, t_grid)
        unit = price_barrier_lv_mc(float(env.spot), float(product.strike),
                                   product.option_type == OptionType.CALL, lv,
                                   np.diff(t_grid), r_fwd, carry_fwd, float(env.get_discount_factor(T)),
                                   num_paths=int(self.params.num_paths), seed=self.params.seed,
                                   use_antithetic=bool(self.params.use_antithetic),
                                   **_barrier_kwargs(product, continuous, observe_idx))
        return unit * float(product.participation_rate) * float(getattr(product, "contract_multiplier", 1.0))


class HestonBarrierMCEngine(BaseEngine):
    """Barrier MC under the Heston stochastic-volatility model."""

    engine_type = EngineType.MONTE_CARLO

    def __init__(self, model_params: HestonParams, params: Optional[MCParams] = None):
        super().__init__(params or MCParams())
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        self.model_params = model_params

    def price(self, product: BaseEquityProduct, env: PricingEnvironment) -> float:
        _check_barrier("HestonBarrierMCEngine", product)
        T = float(product.get_maturity(env)); n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        continuous, observe_idx = _monitoring(product, t_grid)
        unit = price_barrier_heston_mc(float(env.spot), float(product.strike),
                                       product.option_type == OptionType.CALL, self.model_params,
                                       np.diff(t_grid), r_fwd, carry_fwd, float(env.get_discount_factor(T)),
                                       num_paths=int(self.params.num_paths), seed=self.params.seed,
                                       **_barrier_kwargs(product, continuous, observe_idx))
        return unit * float(product.participation_rate) * float(getattr(product, "contract_multiplier", 1.0))


class HestonSLVBarrierMCEngine(BaseEngine):
    """Barrier MC under Heston-SLV using a precomputed leverage surface."""

    engine_type = EngineType.MONTE_CARLO

    def __init__(self, model_params: HestonParams, leverage_surface: LeverageSurface,
                 params: Optional[MCParams] = None, eta: float = 1.0):
        super().__init__(params or MCParams())
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        if not isinstance(leverage_surface, LeverageSurface):
            raise ValidationError("leverage_surface must be a calibrated LeverageSurface")
        self.model_params = model_params
        self.leverage_surface = leverage_surface
        self.eta = float(eta)

    def price(self, product: BaseEquityProduct, env: PricingEnvironment) -> float:
        _check_barrier("HestonSLVBarrierMCEngine", product)
        T = float(product.get_maturity(env)); n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        continuous, observe_idx = _monitoring(product, t_grid)
        unit = price_barrier_slv_mc(float(env.spot), float(product.strike),
                                    product.option_type == OptionType.CALL, self.model_params,
                                    self.leverage_surface, np.diff(t_grid), r_fwd, carry_fwd,
                                    float(env.get_discount_factor(T)), eta=self.eta,
                                    num_paths=int(self.params.num_paths), seed=self.params.seed,
                                    **_barrier_kwargs(product, continuous, observe_idx))
        return unit * float(product.participation_rate) * float(getattr(product, "contract_multiplier", 1.0))
