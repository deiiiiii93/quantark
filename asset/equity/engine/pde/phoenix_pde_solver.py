"""
PDE solver for Phoenix options using the unified PDESystemState.

Handles memory coupons via N-state vectorization.
"""

from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from asset.equity.engine.pde.base_pde_solver import PDESolutionResult
from asset.equity.engine.pde.core import (
    PDESystemState,
    PDEEvent,
    KnockOutEvent,
    KnockInEvent,
    PhoenixCouponEvent,
)
from asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option.phoenix_option import PhoenixOption
from priceenv import PricingEnvironment
from util.enum import ObservationType, CouponPayType
from util.exceptions import ValidationError
from util.numerical import is_close

class PhoenixPDESolver(SnowballPDESolver):
    """
    PDE solver for Phoenix options using generic PDESystemState.
    """
    _supported_product_type: type = PhoenixOption
    _solver_name: str = "PhoenixPDESolver"

    def _solve(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> PDESolutionResult:
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product, spot, pricing_env, ki_continuous
        )

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

        # 2. Determine State Size
        ko_records = self._get_cached_ko_records(pricing_env, product)
        num_obs = len(ko_records)
        use_memory = product.has_memory_coupon
        if use_memory and num_obs > 50:
             raise ValidationError(f"Too many observations ({num_obs}) for Memory Phoenix PDE.")
        
        max_k = num_obs if use_memory else 0
        num_memory_states = max_k + 1
        num_states = 2 * num_memory_states
        
        v0_indices = list(range(num_memory_states))
        v1_indices = list(range(num_memory_states, num_states))

        state = PDESystemState(num_x, num_t, num_states=num_states)
        self._state = state

        # 3. Set Terminal Conditions
        for k in range(num_memory_states):
            payoff_v0 = np.array([product.get_maturity_payoff_v0(s, accumulated_coupons=0.0, pricing_env=pricing_env) for s in s_vec])
            state.get_slice(num_t-1)[:, v0_indices[k]] = payoff_v0
            
        payoff_v1 = np.array([product.get_maturity_payoff_v1(s, pricing_env) for s in s_vec])
        for k in range(num_memory_states):
            state.get_slice(num_t-1)[:, v1_indices[k]] = payoff_v1

        # 4. Build Events
        events_by_step = self._build_phoenix_events(
            t_vec, product, pricing_env, tau, ki_continuous, v0_indices, v1_indices, ko_records
        )
        
        for event in events_by_step[num_t - 1]:
            event.apply(state, num_t - 1, t_vec[num_t - 1], s_vec, pricing_env)

        # 5. Matrices
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)

        # 6. Time Stepping
        params = self.params
        smooth_js = set()
        if params.use_rannacher and params.auto_grid and params.rannacher_at_events:
            event_times = self._get_event_times(product, tau)
            if event_times:
                for et in event_times:
                    idx = int(np.argmin(np.abs(t_vec - et)))
                    if 0 < idx < num_t - 1 and is_close(float(t_vec[idx]), float(et)):
                        for k in range(params.rannacher_steps):
                            smooth_idx = idx - 1 - k
                            if smooth_idx >= 0: smooth_js.add(smooth_idx)

        ki_mask = None
        if ki_continuous and product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                ki_barrier = ki_barrier[0]
            if product.is_reverse:
                ki_mask = s_vec >= float(ki_barrier)
            else:
                ki_mask = s_vec <= float(ki_barrier)

        rhs_buffer = np.empty((num_x - 2, state.num_states), dtype=float)

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            tau_remaining = tau - t_vec[j]
            theta = params.theta
            if params.use_rannacher and (num_t - 1 - j) < params.rannacher_steps: theta = 1.0
            elif j in smooth_js: theta = params.event_theta

            # Set Boundary Conditions
            for k_idx in v0_indices:
                self._set_boundary_conditions_v0(state.get_slice(j)[:, k_idx], x_vec, s_vec, j, tau_remaining, product, pricing_env)
            for k_idx in v1_indices:
                self._set_boundary_conditions_v1(state.get_slice(j)[:, k_idx], x_vec, s_vec, j, tau_remaining, product, pricing_env)

            # Solve
            banded, lower1, main1, upper1 = self._get_banded_system(l, c, u, dt, theta, full_coeffs=(l,c,u))
            def injector(rhs_buffer, t_idx):
                self._inject_boundary_contributions(rhs_buffer, state.grids[:, :, :], l, u, t_idx, dt, theta)
            state.solve_step_banded(
                j,
                j + 1,
                banded,
                (lower1, main1, upper1),
                injector,
                rhs_buffer=rhs_buffer,
            )
            
            # Apply Events
            for event in events_by_step[j]:
                event.apply(state, j, t_vec[j], s_vec, pricing_env)
                    
            if ki_mask is not None:
                for k in range(num_memory_states):
                    state.grids[ki_mask, j, v0_indices[k]] = state.grids[
                        ki_mask, j, v1_indices[k]
                    ]

        # 7. Result
        spot_log = np.log(spot)
        solution_idx = v1_indices[0] if knocked_in_at_valuation else v0_indices[0]
        solution_vec = state.get_slice(0)[:, solution_idx]
        return PDESolutionResult(solution_vec, x_vec, s_vec, spot_log)

    def _build_phoenix_events(self, t_vec, product, pricing_env, tau, ki_continuous, v0_idx, v1_idx, ko_records):
        events: List[List[PDEEvent]] = [[] for _ in range(len(t_vec))]
        ko_times = [rec.observation_time for rec in ko_records]
        num_obs = len(ko_times)
        
        coupon_barrier = product.coupon_config.coupon_barrier
        if isinstance(coupon_barrier, list):
            coupon_barriers = np.array(coupon_barrier, dtype=float)
        else:
            coupon_barriers = np.full(num_obs, float(coupon_barrier))
            
        coupon_amounts = []
        for i in range(num_obs):
            yf = ko_times[i] if i == 0 else ko_times[i] - ko_times[i-1]
            amt = product.get_coupon_payoff(i, year_fraction=yf)
            coupon_amounts.append(amt)
        
        accumulated_vector = np.concatenate(([0.0], np.cumsum(coupon_amounts)))
        
        for i, rec in enumerate(ko_records):
            if not 0.0 <= rec.observation_time <= tau: continue
            idx = self._aligned_time_index(t_vec, rec.observation_time, "Obs")
            
            # Order: KO then Coupon (Coupon is added, KO is fixed payoff)
            events[idx].append(KnockOutEvent(rec, product.is_reverse))
            
            c_event = PhoenixCouponEvent(
                barrier=coupon_barriers[i],
                base_coupon=coupon_amounts[i],
                accumulated_vector=accumulated_vector,
                settlement_time=rec.settlement_time,
                is_reverse=product.is_reverse,
                is_memory=product.has_memory_coupon,
                v0_indices=v0_idx,
                v1_indices=v1_idx
            )
            if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY:
                c_event.settlement_time = tau
            events[idx].append(c_event)
            
        if product.has_ki_barrier and not ki_continuous:
            ki_profile = self._get_cached_ki_profile(pricing_env, product)
            for obs_time in ki_profile.get("observation_times", []):
                if not 0.0 <= obs_time <= tau: continue
                idx = self._aligned_time_index(t_vec, obs_time, "KI")
                ki_barrier = product.barrier_config.ki_barrier
                if isinstance(ki_barrier, list): ki_barrier = ki_barrier[0]
                for k in range(len(v0_idx)):
                    events[idx].append(KnockInEvent(float(ki_barrier), product.is_reverse, source_idx=v1_idx[k], target_idx=v0_idx[k]))
        return events

    def _get_event_times(self, product: PhoenixOption, tau: float) -> Optional[List[float]]:
        times = super()._get_event_times(product, tau) or []
        # Coupon times are usually same as KO, but double check
        return sorted(set(times))
