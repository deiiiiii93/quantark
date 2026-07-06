"""PDE barrier solvers under the vol-model processes (Local Vol / Heston / SLV).

Thin wrappers over the ``quantark.volmodels`` barrier PDE kernels — the 1D Crank-Nicolson LV
kernel and the 2D ADI Heston/SLV kernels — pricing the existing ``BarrierOption``. Continuous
monitoring is exact (barrier on the domain boundary for LV; per-step injection for the 2D ADI);
discrete monitoring enforces the knock-out at the observation nodes. ``participation_rate`` is
applied once; ``observation_schedule`` is rejected (v1). Deterministic — no MC inside PDE.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.param import GridVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType, ObservationType
from quantark.util.enum.engine_enums import ADIScheme, EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.curves import forward_carry_on_grid, forward_rates_on_grid
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.localvol.pde_kernel import price_barrier_lv_pde
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.pde_kernel import price_barrier_heston_pde
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.volmodels.slv.slv_pde_kernel import price_barrier_slv_pde


def _check(name, product):
    if not isinstance(product, BarrierOption):
        raise PricingError(f"{name} supports BarrierOption only")
    sched = product.observation_schedule
    recs = getattr(sched, "records", None) if sched is not None else None
    if recs:
        barriers = {(r.barrier, r.upper_barrier, r.lower_barrier) for r in recs}
        if len(barriers) > 1 or len({r.payoff for r in recs}) > 1:
            raise ValidationError("per-observation ObservationSchedule (varying barriers/payoffs) is out of scope for v1")


def _bkw(product):
    """Barrier kwargs shared by both leg pricings (rebate passed separately via the closure)."""
    return dict(barrier=float(product.barrier), is_up=product.is_up_barrier, is_out=product.is_knock_out,
                pay_at_hit=bool(product.pay_at_hit))


def _obs_dates(product, T):
    """(continuous, dates): CONTINUOUS -> (True, None); EXPIRY -> (False, []) (terminal only);
    DISCRETE -> (False, observation_dates)."""
    if product.observation_type == ObservationType.CONTINUOUS:
        return True, None
    if product.observation_type == ObservationType.EXPIRY:
        return False, [float(T)]  # single observation AT maturity (the terminal knock-out)
    dates = product.observation_dates or []
    if not dates:
        raise ValidationError("discrete monitoring requires observation_dates")
    for d in dates:
        if d > T + 1e-12:
            raise ValidationError(f"observation date {d} exceeds maturity {T}")
    return False, list(dates)


def _apply_participation(product, price_of_rebate):
    """Apply participation to the OPTION leg only (never the rebate leg), by superposition:
    V = V(full rebate) + (participation - 1) * V(rebate=0). Reduces to participation*V when
    rebate == 0, and to V when participation == 1 -- one PDE solve in the common cases."""
    p = float(product.participation_rate)
    reb = float(product.rebate)
    if p == 1.0 or reb == 0.0:
        return p * price_of_rebate(reb)
    return price_of_rebate(reb) + (p - 1.0) * price_of_rebate(0.0)


class LocalVolBarrierPDESolver(BaseEngine):
    """1D Crank-Nicolson barrier PDE under a Dupire local-volatility surface."""

    engine_type = EngineType.PDE

    def __init__(self, params: Optional[PDEParams] = None, local_vol_surface: Optional[LocalVolSurface] = None):
        super().__init__(params or PDEParams())
        self._prebuilt = local_vol_surface

    def price(self, product: BaseEquityProduct, env: PricingEnvironment) -> float:
        _check("LocalVolBarrierPDESolver", product)
        T = float(product.get_maturity(env)); n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        lv = self._prebuilt
        if lv is None:
            if not isinstance(env.vol_surface, GridVolSurface):
                raise PricingError("LocalVolBarrierPDESolver needs a GridVolSurface or a prebuilt LocalVolSurface")
            lv = build_dupire_local_vol(env.vol_surface, spot=env.spot, rate_curve=env.rate_curve,
                                        div_yield=env.get_div_yield)
        continuous, dates = _obs_dates(product, T)
        observe_steps = None if continuous else [int(np.argmin(np.abs(t_grid - d))) for d in dates]
        is_call = product.option_type == OptionType.CALL

        def price_of_rebate(reb):
            return price_barrier_lv_pde(float(env.spot), float(product.strike), is_call, T, lv,
                                        np.diff(t_grid), r_fwd, carry_fwd, rebate=reb,
                                        continuous=continuous, observe_steps=observe_steps,
                                        n_s=int(self.params.grid_size), **_bkw(product))

        unit = _apply_participation(product, price_of_rebate)
        return unit * float(getattr(product, "contract_multiplier", 1.0))


class _Heston2DBarrierBase(BaseEngine):
    engine_type = EngineType.PDE

    def __init__(self, n_x=200, n_v=100, n_t=100, scheme=ADIScheme.CRAIG_SNEYD):
        super().__init__(PDEParams())
        self.n_x, self.n_v, self.n_t, self.scheme = int(n_x), int(n_v), int(n_t), scheme

    def _observe_taus(self, product, T):
        continuous, dates = _obs_dates(product, T)
        return continuous, (None if continuous else [T - d for d in dates])


class HestonBarrierPDESolver(_Heston2DBarrierBase):
    """2D ADI barrier PDE under the Heston model."""

    def __init__(self, model_params: HestonParams, n_x=200, n_v=100, n_t=100, scheme=ADIScheme.CRAIG_SNEYD):
        super().__init__(n_x, n_v, n_t, scheme)
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        self.model_params = model_params

    def price(self, product: BaseEquityProduct, env: PricingEnvironment) -> float:
        _check("HestonBarrierPDESolver", product)
        T = float(product.get_maturity(env))
        continuous, observe_taus = self._observe_taus(product, T)
        is_call = product.option_type == OptionType.CALL

        def price_of_rebate(reb):
            return price_barrier_heston_pde(float(env.spot), float(product.strike), is_call, T,
                                            self.model_params, float(env.get_rate(T)), float(env.get_div_yield(T)),
                                            rebate=reb, continuous=continuous, observe_taus=observe_taus,
                                            n_x=self.n_x, n_v=self.n_v, n_t=self.n_t, scheme=self.scheme,
                                            **_bkw(product))

        unit = _apply_participation(product, price_of_rebate)
        return unit * float(getattr(product, "contract_multiplier", 1.0))


class HestonSLVBarrierPDESolver(_Heston2DBarrierBase):
    """2D ADI barrier PDE under Heston-SLV (given a calibrated leverage surface)."""

    def __init__(self, model_params: HestonParams, leverage_surface: LeverageSurface,
                 eta: float = 1.0, n_x=200, n_v=100, n_t=100, scheme=ADIScheme.CRAIG_SNEYD):
        super().__init__(n_x, n_v, n_t, scheme)
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        if not isinstance(leverage_surface, LeverageSurface):
            raise ValidationError("leverage_surface must be a calibrated LeverageSurface")
        self.model_params = model_params
        self.leverage_surface = leverage_surface
        self.eta = float(eta)

    def price(self, product: BaseEquityProduct, env: PricingEnvironment) -> float:
        _check("HestonSLVBarrierPDESolver", product)
        T = float(product.get_maturity(env))
        continuous, observe_taus = self._observe_taus(product, T)
        is_call = product.option_type == OptionType.CALL

        def price_of_rebate(reb):
            return price_barrier_slv_pde(float(env.spot), float(product.strike), is_call, T,
                                         self.model_params, self.leverage_surface,
                                         float(env.get_rate(T)), float(env.get_div_yield(T)),
                                         eta=self.eta, rebate=reb, continuous=continuous, observe_taus=observe_taus,
                                         n_x=self.n_x, n_v=self.n_v, n_t=self.n_t, scheme=self.scheme,
                                         **_bkw(product))

        unit = _apply_participation(product, price_of_rebate)
        return unit * float(getattr(product, "contract_multiplier", 1.0))
