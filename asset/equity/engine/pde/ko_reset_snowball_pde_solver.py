"""
PDE solver for KO-reset Snowball options using the unified PDESystemState.

Applies the pre-KI KO schedule to the V0 surface and the post-KI KO schedule
to the V1 surface (ABSOLUTE post-KO mode only).
"""

from collections import OrderedDict, defaultdict
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from asset.equity.engine.pde.base_pde_solver import PDESolutionResult
from asset.equity.engine.pde.core import (
    PDESystemState,
    KnockOutEvent,
    KnockInEvent,
    MaturityEvent,
)
from asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from priceenv import PricingEnvironment
from util.enum import CouponPayType, ObservationType, PostKOScheduleMode
from util.exceptions import PricingError, ValidationError
from util.numerical import is_close, is_zero


class KOResetSnowballPDESolver(SnowballPDESolver):
    """
    PDE solver for KO-reset Snowball options using the generic PDESystemState.
    """

    _supported_product_type: type = KnockOutResetSnowballOption
    _solver_name: str = "KOResetSnowballPDESolver"

    def __init__(self, params=None):
        super().__init__(params=params)
        self._pre_ko_records_cache: "OrderedDict[Tuple, List[ResolvedObservationRecord]]" = OrderedDict()
        self._post_ko_records_cache: "OrderedDict[Tuple, List[ResolvedObservationRecord]]" = OrderedDict()

    def _solve(self, product: KnockOutResetSnowballOption, pricing_env: PricingEnvironment) -> PDESolutionResult:
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(product, spot, pricing_env, ki_continuous)

        strike = product.strike
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # Clear per-solve caches
        self._matrix_cache.clear()
        self._banded_cache.clear()

        if self._profile_enabled: self._reset_profile_stats()

        # 1. Build Grids
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(product, pricing_env, spot, sigma, tau, r, q)
        num_x, num_t = len(x_vec), len(t_vec)

        # 2. Initialize State (2 surfaces: 0=No-KI, 1=KI)
        state = PDESystemState(num_x, num_t, num_states=2)
        self._state = state

        # 3. Set Terminal Conditions
        self._set_terminal_condition_v0(state.get_slice(num_t - 1)[:, 0], x_vec, s_vec, product, pricing_env)
        self._set_terminal_condition_v1(state.get_slice(num_t - 1)[:, 1], x_vec, s_vec, product, pricing_env)

        # 4. Build Events
        events_by_step = self._build_ko_reset_events(t_vec, product, pricing_env, tau, ki_continuous)
        
        if (num_t - 1) in events_by_step:
            for event in events_by_step[num_t - 1]:
                event.apply(state, num_t - 1, t_vec[num_t - 1], s_vec, pricing_env)

        # 5. Matrices
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)

        # 6. Time Stepping
        params = self.params
        smooth_js = set()
        # ... (Rannacher setup inherited behavior if possible, but let's replicate for clarity) ...
        if params.use_rannacher and params.auto_grid and params.rannacher_at_events:
            for et in self._get_event_times(product, tau) or []:
                idx = int(np.argmin(np.abs(t_vec - et)))
                if 0 < idx < num_t - 1 and is_close(float(t_vec[idx]), float(et)):
                    smooth_js.update([idx - 1 - k for k in range(params.rannacher_steps) if idx - 1 - k >= 0])

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            tau_remaining = tau - t_vec[j]
            theta = params.theta
            if params.use_rannacher and (num_t - 1 - j) < params.rannacher_steps: theta = 1.0
            elif j in smooth_js: theta = params.event_theta

            # Set Boundary Conditions
            self._set_boundary_conditions_v0(state.get_slice(j)[:, 0], x_vec, s_vec, j, tau_remaining, product, pricing_env)
            self._set_boundary_conditions_v1(state.get_slice(j)[:, 1], x_vec, s_vec, j, tau_remaining, product, pricing_env)

            # Solve
            banded, lower1, main1, upper1 = self._get_banded_system(l, c, u, dt, theta, full_coeffs=(l,c,u))
            def injector(rhs_buffer, t_idx):
                self._inject_boundary_contributions(rhs_buffer, state.grids[:, :, :], l, u, t_idx, dt, theta)
            state.solve_step_banded(j, j + 1, banded, (lower1, main1, upper1), injector)
            
            # Apply Events
            if j in events_by_step:
                for event in events_by_step[j]:
                    event.apply(state, j, t_vec[j], s_vec, pricing_env)
                    
            if ki_continuous and product.has_ki_barrier:
                ki_barrier = product.barrier_config.ki_barrier
                if isinstance(ki_barrier, list): ki_barrier = ki_barrier[0]
                event = KnockInEvent(float(ki_barrier), product.is_reverse, source_idx=1, target_idx=0)
                event.apply(state, j, t_vec[j], s_vec, pricing_env)

        # 7. Result
        spot_log = np.log(spot)
        solution_idx = 1 if knocked_in_at_valuation else 0
        solution_vec = state.get_slice(0)[:, solution_idx]
        return PDESolutionResult(solution_vec, x_vec, s_vec, spot_log)

    def _build_ko_reset_events(self, t_vec, product, pricing_env, tau, ki_continuous):
        events = defaultdict(list)
        
        # Pre-KI KO Events (Target V0 only)
        pre_records = self._get_cached_pre_ko_records(pricing_env, product)
        for rec in pre_records:
            if not 0.0 <= rec.observation_time <= tau: continue
            idx = self._aligned_time_index(t_vec, rec.observation_time, "Pre-KO")
            # Apply only to State 0
            events[idx].append(KnockOutEvent(rec, product.is_reverse, state_indices=[0]))
            
        # Post-KI KO Events (Target V1 only)
        if not product.barrier_config.disable_ko_after_ki:
            post_records = self._get_cached_post_ko_records(pricing_env, product)
            for rec in post_records:
                if not 0.0 <= rec.observation_time <= tau: continue
                idx = self._aligned_time_index(t_vec, rec.observation_time, "Post-KO")
                # Apply only to State 1
                events[idx].append(KnockOutEvent(rec, product.is_reverse, state_indices=[1]))

        # V0 matures at the end of the pre-KO schedule when no KI happens.
        pre_maturity = product.get_pre_maturity_time(pricing_env)
        if 0.0 < pre_maturity < tau and not is_close(pre_maturity, tau):
            idx = self._aligned_time_index(t_vec, pre_maturity, "Pre-Maturity")

            def v0_payoff(s, env):
                return product.get_maturity_payoff_v0(
                    s, accumulated_coupons=0.0, pricing_env=env
                )

            # Insert before KO/KI at the same time so KO can overwrite if triggered.
            events[idx].insert(0, MaturityEvent(v0_payoff, state_indices=[0]))

        # KI Events (Discrete)
        if product.has_ki_barrier and not ki_continuous:
            ki_profile = self._get_cached_ki_profile(pricing_env, product)
            for obs_time in ki_profile.get("observation_times", []):
                if not 0.0 <= obs_time <= tau: continue
                idx = self._aligned_time_index(t_vec, obs_time, "KI")
                ki_barrier = product.barrier_config.ki_barrier
                if isinstance(ki_barrier, list): ki_barrier = ki_barrier[0]
                events[idx].append(KnockInEvent(float(ki_barrier), product.is_reverse, source_idx=1, target_idx=0))
                
        return events

    def _validate_product(self, product: KnockOutResetSnowballOption) -> None:
        super()._validate_product(product)
        if product.post_ko_mode != PostKOScheduleMode.ABSOLUTE:
            raise ValidationError("KOResetSnowballPDESolver only supports PostKOScheduleMode.ABSOLUTE")
        if product.post_barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise ValidationError("Post-KO monitoring must be discrete for PDE solver.")

    def _resolve_ko_records(self, product, pricing_env, config) -> List[ResolvedObservationRecord]:
        resolved_schedule, rates, schedule_records = product._resolve_ko_schedule(config, pricing_env)
        principal = product.initial_price * product.contract_multiplier if product.payoff_config.include_principal else 0.0
        ko_records = []
        for idx, rec in enumerate(resolved_schedule):
            rate = rates[idx]
            accrual = product.compute_ko_accrual_factor(rec.observation_time, schedule_records[idx], pricing_env)
            payoff = principal + product.initial_price * product.contract_multiplier * float(rate) * float(accrual)
            settle = rec.settlement_time
            if product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY:
                settle = product.get_maturity(pricing_env)
            ko_records.append(
                ResolvedObservationRecord(
                    observation_time=rec.observation_time,
                    barrier=rec.barrier,
                    payoff=payoff,
                    settlement_time=settle,
                )
            )
        return ko_records

    def _get_cached_pre_ko_records(self, pricing_env, product):
        if not self._is_cache_enabled(): return self._resolve_ko_records(product, pricing_env, product.barrier_config)
        key = self._observation_cache_key(pricing_env, product, "pre_ko")
        if key in self._pre_ko_records_cache: return self._pre_ko_records_cache[key]
        records = self._resolve_ko_records(product, pricing_env, product.barrier_config)
        self._pre_ko_records_cache[key] = records
        return records

    def _get_cached_post_ko_records(self, pricing_env, product):
        if not self._is_cache_enabled(): return self._resolve_ko_records(product, pricing_env, product.post_barrier_config)
        key = self._observation_cache_key(pricing_env, product, "post_ko")
        if key in self._post_ko_records_cache: return self._post_ko_records_cache[key]
        records = self._resolve_ko_records(product, pricing_env, product.post_barrier_config)
        self._post_ko_records_cache[key] = records
        return records

    def _get_cached_ko_records(self, pricing_env, product):
        return self._get_cached_pre_ko_records(pricing_env, product)

    def _get_event_times(self, product: KnockOutResetSnowballOption, tau: float) -> Optional[List[float]]:
        times = super()._get_event_times(product, tau) or []
        # Post-KI times from raw config
        post_config = product.post_barrier_config
        if post_config.ko_observation_dates:
            for t in post_config.ko_observation_dates:
                if 0 < t < tau: times.append(t)
        elif post_config.ko_observation_schedule:
            for r in post_config.ko_observation_schedule.records:
                if r.observation_time and 0 < r.observation_time < tau:
                    times.append(r.observation_time)
        return sorted(set(times)) if times else None

    def _get_barriers(self, product: KnockOutResetSnowballOption) -> List[float]:
        barriers = super()._get_barriers(product)
        post_barrier = product.post_barrier_config.ko_barrier
        if isinstance(post_barrier, list): barriers.extend(post_barrier)
        elif post_barrier > 0: barriers.append(post_barrier)
        return barriers

    def __repr__(self) -> str: return "KOResetSnowballPDESolver()"
