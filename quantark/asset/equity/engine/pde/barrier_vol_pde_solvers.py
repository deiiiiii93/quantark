"""PDE barrier solvers under the vol-model processes (Local Vol / Heston / SLV).

``LocalVolBarrierPDESolver`` marches on the shared 1D core (spec 3C §5.3). The Heston and SLV
solvers are thin wrappers over the ``quantark.volmodels`` 2D ADI barrier kernels, pricing the
existing ``BarrierOption``: continuous monitoring by per-step injection, discrete monitoring at the
observation nodes; ``participation_rate`` is applied once; a per-observation
``observation_schedule`` is rejected (v1). Below the kernels' vol-of-vol floor the variance is deterministic and both
solvers march the 1D local-vol limit sigma_loc(S, t) = L(S, t) sqrt(v(t)) on the core instead. Deterministic — no MC
inside PDE.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver, barrier_expiry_outcome
from quantark.asset.equity.engine.pde.core.adapters.local_vol import LocalVolBarrierAdapter
from quantark.asset.equity.engine.pde.local_vol_surface import LocalVolSurfaceMixin
from quantark.asset.equity.engine.settlement_support import (
    at_exercise_instant,
    resolve_terminal_timing,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.param.pde_removed_names import reject_removed_kwargs
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.execution.errors import CapabilityError
from quantark.priceenv import PricingEnvironment, TermMarketContext
from quantark.util.enum import OptionType, ObservationType
from quantark.util.enum.engine_enums import ADIScheme, EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.deterministic_limit import DeterministicVarianceLocalVol
from quantark.volmodels.heston.pde_kernel import _SIGMA_MIN, price_barrier_heston_pde
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


def _settlement_scale(engine, product, env) -> float:
    """Return the safe whole-kernel terminal scale or fail closed."""
    terminal_timing = resolve_terminal_timing(product, env)
    delayed_hit = False
    schedule = getattr(product, "observation_schedule", None)
    if product.pay_at_hit and schedule is not None and schedule.records:
        resolved = schedule.resolve(
            env,
            default_barrier=product.barrier,
            default_payoff=product.rebate,
            require_single=True,
            product=product,
        )
        delayed_hit = any(
            not np.isclose(rec.settlement_time, rec.observation_time)
            for rec in resolved
        )
    convention = getattr(product, "settlement_convention", None)
    delayed_hit = delayed_hit or bool(
        product.pay_at_hit
        and convention is not None
        and float(convention.lag) != 0.0
    )
    if product.pay_at_hit and (
        delayed_hit or not np.isclose(terminal_timing.delay_df, 1.0)
    ):
        raise CapabilityError(
            f"{type(engine).__name__} cannot separate delayed first-hit "
            "cashflows from terminal cashflows in the vol-model kernel"
        )
    return float(terminal_timing.delay_df)


def _refuse_ended(engine_name: str, product, lifecycle_state=None) -> None:
    """The 2D model's remaining European has no path here: an ended schedule is refused (spec 2026-10-07 theta
    boundary 3.3), never priced under another model."""
    from quantark.asset.equity.engine.monitoring_ended import knock_in_carried, monitoring_ended
    if monitoring_ended(product) and not knock_in_carried(product, lifecycle_state):
        raise CapabilityError(
            f"{engine_name}: the product's monitoring has ended; price the remaining European with a European "
            "engine of the same model")


class LocalVolBarrierPDESolver(LocalVolSurfaceMixin, BarrierPDESolver):
    """Single barriers under a Dupire local-volatility surface, marched on the core (spec 3C §5.3): EXPIRY, knock-in
    rebates, per-observation schedules, lagged pay-at-hit rebates, bump contexts and session grids are the base
    family's. Greeks: price, delta, gamma from the grid (decision 21)."""

    _adapter_class = LocalVolBarrierAdapter

    def __init__(self, params: Optional[PDEParams] = None, local_vol_surface: Optional[LocalVolSurface] = None,
                 **removed):
        reject_removed_kwargs(type(self).__name__, removed)
        super().__init__(params)
        self._init_local_vol(local_vol_surface)

    def price(self, product, pricing_env):
        return self._with_surface(pricing_env, lambda: BarrierPDESolver.price(self, product, pricing_env))

    def calculate_greeks(self, product, pricing_env):
        return self._with_surface(pricing_env, lambda: BarrierPDESolver.calculate_greeks(self, product, pricing_env))

    def _session_outputs(self, product, pricing_env, **kwargs):
        return self._with_surface(pricing_env, lambda: BarrierPDESolver._session_outputs(
            self, product, pricing_env, **kwargs))

    def calculate_spot_greeks_curve(self, product, pricing_env, spot_levels):
        return self._with_surface(pricing_env, lambda: BarrierPDESolver.calculate_spot_greeks_curve(
            self, product, pricing_env, spot_levels))

    def _vanilla_solver(self):
        """The knock-in's vanilla part on the same surface object (spec 3C §5.3)."""
        from .local_vol_pde_solver import LocalVolPDESolver
        surface = self._active_lv_surface if self._active_lv_surface is not None else self._prebuilt
        return LocalVolPDESolver(self.params, local_vol_surface=surface)

    def create_bump_context(self, product, pricing_env):
        # the parts' layouts are bound with this solve's surface in scope (bump_parts reads _vanilla_solver)
        return self._with_surface(pricing_env, lambda: BarrierPDESolver.create_bump_context(self, product, pricing_env))


class _Heston2DBarrierBase(BaseEngine):
    engine_type = EngineType.PDE
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL

    def __init__(self, n_x=200, n_v=100, n_t=100, scheme=ADIScheme.CRAIG_SNEYD):
        super().__init__(PDEParams())
        self.n_x, self.n_v, self.n_t, self.scheme = int(n_x), int(n_v), int(n_t), scheme

    def _admit(self, product, env) -> float:
        """The route's refusals (product, ended schedule, delayed first-hit cash), then the terminal settlement
        scale."""
        name = type(self).__name__
        _check(name, product)
        _refuse_ended(name, product)
        return _settlement_scale(self, product, env)

    def calculate_greeks(self, product, pricing_env, *, lifecycle_state=None):
        """At the exercise instant, the Black-Scholes barrier PDE route's greeks of the settled value (price, the
        payoff slope, gamma 0; W9, ruling 2026-10-10); otherwise bump-and-reprice."""
        if lifecycle_state is None:
            self._admit(product, pricing_env)
            if at_exercise_instant(float(product.get_maturity(pricing_env))):
                return barrier_expiry_outcome(product, pricing_env, greeks=True)
        return super().calculate_greeks(product, pricing_env, lifecycle_state=lifecycle_state)

    def _observe_taus(self, product, T):
        continuous, dates = _obs_dates(product, T)
        return continuous, (None if continuous else [T - d for d in dates])

    def _deterministic_limit(self, product, env, leverage=None) -> float:
        """Vol-of-vol below ``_SIGMA_MIN``: the variance is deterministic and the 2D ADI grid ill-conditioned, so the
        barrier is the 1D local-vol problem sigma_loc(S, t) = L(S, t) sqrt(v(t)), marched on the core at its
        standard preset (the core prices participation, rebates, the multiplier and terminal settlement itself)."""
        surface = DeterministicVarianceLocalVol(self.model_params, leverage)
        return LocalVolBarrierPDESolver(local_vol_surface=surface).price(product, env)


class HestonBarrierPDESolver(_Heston2DBarrierBase):
    """2D ADI barrier PDE under the Heston model."""

    def __init__(self, model_params: HestonParams, n_x=200, n_v=100, n_t=100, scheme=ADIScheme.CRAIG_SNEYD):
        super().__init__(n_x, n_v, n_t, scheme)
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        self.model_params = model_params

    def price(self, product: BaseEquityProduct, env: PricingEnvironment) -> float:
        from quantark.asset.equity.engine.monitoring_ended import knock_in_carried

        settlement_scale = self._admit(product, env)
        if self.model_params.sigma < _SIGMA_MIN:
            return self._deterministic_limit(product, env)
        T = float(product.get_maturity(env))
        if at_exercise_instant(T):
            return barrier_expiry_outcome(product, env)     # settled before the model runs (W9, ruling 2026-10-10)
        t_grid = np.linspace(0.0, T, self.n_t + 1)
        market = TermMarketContext.from_env(
            env, t_grid, ref_strike=None
        )
        continuous, observe_taus = self._observe_taus(product, T)
        is_call = product.option_type == OptionType.CALL

        def price_of_rebate(reb):
            return price_barrier_heston_pde(float(env.spot), float(product.strike), is_call, T,
                                            self.model_params, float(env.get_rate(T)), float(env.get_div_yield(T)),
                                            rebate=reb, continuous=continuous, observe_taus=observe_taus,
                                            n_x=self.n_x, n_v=self.n_v, n_t=self.n_t, scheme=self.scheme,
                                            market_context=market, knocked_in=knock_in_carried(product),
                                            **_bkw(product))

        unit = _apply_participation(product, price_of_rebate)
        return (
            unit
            * float(getattr(product, "contract_multiplier", 1.0))
            * settlement_scale
        )


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
        from quantark.asset.equity.engine.monitoring_ended import knock_in_carried

        settlement_scale = self._admit(product, env)
        if self.model_params.sigma < _SIGMA_MIN:
            return self._deterministic_limit(product, env, self.leverage_surface)
        T = float(product.get_maturity(env))
        if at_exercise_instant(T):
            return barrier_expiry_outcome(product, env)     # settled before the model runs (W9, ruling 2026-10-10)
        t_grid = np.linspace(0.0, T, self.n_t + 1)
        market = TermMarketContext.from_env(
            env, t_grid, ref_strike=None
        )
        continuous, observe_taus = self._observe_taus(product, T)
        is_call = product.option_type == OptionType.CALL

        def price_of_rebate(reb):
            return price_barrier_slv_pde(float(env.spot), float(product.strike), is_call, T,
                                         self.model_params, self.leverage_surface,
                                         float(env.get_rate(T)), float(env.get_div_yield(T)),
                                         eta=self.eta, rebate=reb, continuous=continuous, observe_taus=observe_taus,
                                         n_x=self.n_x, n_v=self.n_v, n_t=self.n_t, scheme=self.scheme,
                                         market_context=market, knocked_in=knock_in_carried(product),
                                         **_bkw(product))

        unit = _apply_participation(product, price_of_rebate)
        return (
            unit
            * float(getattr(product, "contract_multiplier", 1.0))
            * settlement_scale
        )
