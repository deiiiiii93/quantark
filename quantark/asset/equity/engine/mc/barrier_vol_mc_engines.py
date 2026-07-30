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
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.execution.errors import CapabilityError
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


def _reject_nonuniform_schedule(product) -> None:
    """A discrete-dates product carries a uniform ObservationSchedule (built via from_legacy);
    that is fine. Only a schedule with per-observation VARYING barriers/payoffs is out of scope."""
    sched = product.observation_schedule
    recs = getattr(sched, "records", None) if sched is not None else None
    if not recs:
        return
    # A legacy (uniform) schedule repeats one barrier level and no per-date payoff -> fine.
    # Reject only if the barrier VARIES across observations or a per-date payoff is set.
    barriers = {(r.barrier, r.upper_barrier, r.lower_barrier) for r in recs}
    payoffs = {r.payoff for r in recs}
    if len(barriers) > 1 or len(payoffs) > 1:
        raise ValidationError("per-observation ObservationSchedule (varying barriers/payoffs) is out of scope for v1")


def _check_barrier(engine_name: str, product) -> None:
    if not isinstance(product, BarrierOption):
        raise PricingError(f"{engine_name} supports BarrierOption only")
    _reject_nonuniform_schedule(product)


def _monitoring(product: BarrierOption, t_grid: np.ndarray):
    """Return (continuous: bool, observe_idx: Optional[np.ndarray]) snapping discrete dates to grid.

    ObservationType.EXPIRY (barrier checked only at maturity) is discrete monitoring at the
    single terminal node; CONTINUOUS uses the Brownian bridge; DISCRETE snaps observation_dates.
    """
    if product.observation_type == ObservationType.CONTINUOUS:
        return True, None
    if product.observation_type == ObservationType.EXPIRY:
        return False, np.asarray([t_grid.size - 1], dtype=int)  # terminal node only
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
                continuous=continuous, observe_idx=observe_idx,
                participation=float(product.participation_rate))


def _settlement_scale(engine, product, env, lifecycle_state):
    """Return fixed lifecycle PV or the safe whole-kernel terminal scale."""
    validate_settlement_capability(engine, product, lifecycle_state)
    fixed_pv = terminal_lifecycle_pv(lifecycle_state, env)
    if fixed_pv is not None:
        return fixed_pv, None

    terminal_timing = resolve_terminal_timing(product, env)
    convention = getattr(product, "settlement_convention", None)
    delayed_hit = bool(
        product.pay_at_hit
        and convention is not None
        and float(convention.lag) != 0.0
    )

    schedule = getattr(product, "observation_schedule", None)
    if product.pay_at_hit and schedule is not None and schedule.records:
        resolved = schedule.resolve(
            env,
            default_barrier=product.barrier,
            default_payoff=product.rebate,
            require_single=True,
            product=product,
        )
        delayed_hit = delayed_hit or any(
            not np.isclose(rec.settlement_time, rec.observation_time)
            for rec in resolved
        )

    if product.pay_at_hit and (
        delayed_hit or not np.isclose(terminal_timing.delay_df, 1.0)
    ):
        raise CapabilityError(
            f"{type(engine).__name__} cannot separate delayed first-hit "
            "cashflows from terminal cashflows in the vol-model kernel"
        )
    return None, terminal_timing.delay_df


def _finish_price(
    engine,
    unit,
    se,
    product,
    env,
    lifecycle_state,
    settlement_scale,
):
    mult = float(getattr(product, "contract_multiplier", 1.0))
    scale = float(settlement_scale)
    engine._last_std_error = float(se) * mult * scale
    return (
        float(unit) * mult * scale
        + pending_receivable_pv(lifecycle_state, env)
    )


class _StdErrMixin:
    """Records the MC standard error of the last pricing run.

    Mirrors ``BarrierOptionMCEngine.get_last_std_error`` (house style): ``price`` returns the
    point estimate and stores the standard error on ``self._last_std_error`` for callers that
    want to quantify Monte-Carlo noise (e.g. an MC-vs-PDE cross-check gate).
    """

    def get_last_std_error(self):
        return getattr(self, "_last_std_error", None)


class LocalVolBarrierMCEngine(_StdErrMixin, BaseEngine):
    """Barrier MC under a Dupire local-volatility surface."""

    engine_type = EngineType.MONTE_CARLO
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

    def __init__(self, params: Optional[MCParams] = None, local_vol_surface: Optional[LocalVolSurface] = None):
        super().__init__(params or MCParams())
        self._prebuilt = local_vol_surface

    def price(
        self,
        product: BaseEquityProduct,
        env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        _check_barrier("LocalVolBarrierMCEngine", product)
        fixed_pv, settlement_scale = _settlement_scale(
            self, product, env, lifecycle_state
        )
        if fixed_pv is not None:
            self._last_std_error = 0.0
            return fixed_pv
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
        unit, se = price_barrier_lv_mc(float(env.spot), float(product.strike),
                                       product.option_type == OptionType.CALL, lv,
                                       np.diff(t_grid), r_fwd, carry_fwd, float(env.get_discount_factor(T)),
                                       num_paths=int(self.params.num_paths), seed=self.params.seed,
                                       use_antithetic=bool(self.params.use_antithetic), return_stderr=True,
                                       **_barrier_kwargs(product, continuous, observe_idx))
        return _finish_price(
            self, unit, se, product, env, lifecycle_state, settlement_scale
        )


class HestonBarrierMCEngine(_StdErrMixin, BaseEngine):
    """Barrier MC under the Heston stochastic-volatility model."""

    engine_type = EngineType.MONTE_CARLO
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

    def __init__(self, model_params: HestonParams, params: Optional[MCParams] = None):
        super().__init__(params or MCParams())
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        self.model_params = model_params

    def price(
        self,
        product: BaseEquityProduct,
        env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        _check_barrier("HestonBarrierMCEngine", product)
        fixed_pv, settlement_scale = _settlement_scale(
            self, product, env, lifecycle_state
        )
        if fixed_pv is not None:
            self._last_std_error = 0.0
            return fixed_pv
        T = float(product.get_maturity(env)); n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        continuous, observe_idx = _monitoring(product, t_grid)
        unit, se = price_barrier_heston_mc(float(env.spot), float(product.strike),
                                           product.option_type == OptionType.CALL, self.model_params,
                                           np.diff(t_grid), r_fwd, carry_fwd, float(env.get_discount_factor(T)),
                                           num_paths=int(self.params.num_paths), seed=self.params.seed,
                                           return_stderr=True,
                                           **_barrier_kwargs(product, continuous, observe_idx))
        return _finish_price(
            self, unit, se, product, env, lifecycle_state, settlement_scale
        )


class HestonSLVBarrierMCEngine(_StdErrMixin, BaseEngine):
    """Barrier MC under Heston-SLV using a precomputed leverage surface."""

    engine_type = EngineType.MONTE_CARLO
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

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

    def price(
        self,
        product: BaseEquityProduct,
        env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        _check_barrier("HestonSLVBarrierMCEngine", product)
        fixed_pv, settlement_scale = _settlement_scale(
            self, product, env, lifecycle_state
        )
        if fixed_pv is not None:
            self._last_std_error = 0.0
            return fixed_pv
        T = float(product.get_maturity(env)); n = int(self.params.time_steps)
        t_grid = np.linspace(0.0, T, n + 1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        continuous, observe_idx = _monitoring(product, t_grid)
        unit, se = price_barrier_slv_mc(float(env.spot), float(product.strike),
                                        product.option_type == OptionType.CALL, self.model_params,
                                        self.leverage_surface, np.diff(t_grid), r_fwd, carry_fwd,
                                        float(env.get_discount_factor(T)), eta=self.eta,
                                        num_paths=int(self.params.num_paths), seed=self.params.seed,
                                        return_stderr=True,
                                        **_barrier_kwargs(product, continuous, observe_idx))
        return _finish_price(
            self, unit, se, product, env, lifecycle_state, settlement_scale
        )
