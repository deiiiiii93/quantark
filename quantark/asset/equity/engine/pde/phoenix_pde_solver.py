"""
PDE solver for Phoenix options using the Two-Surface method.

Adds coupon jumps at observation times on top of the Snowball PDE framework.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from time import perf_counter

from quantark.asset.equity.engine.pde.base_pde_solver import PDESolutionResult
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_close, is_zero


class PhoenixPDESolver(SnowballPDESolver):
    """
    Two-Surface PDE solver for Phoenix options with coupon jumps.

    KO and KI behavior follows SnowballPDESolver. Coupon payoffs are added at
    observation times based on the coupon barrier.
    """

    # Override class attributes for product type checking
    _supported_product_type: type = PhoenixOption
    _solver_name: str = "PhoenixPDESolver"

    def __init__(self, params=None):
        super().__init__(params=params)
        self._coupon_observation_indices: Dict[int, int] = {}
        self._coupon_barriers: np.ndarray = np.array([])
        self._coupon_amounts: np.ndarray = np.array([])
        self._coupon_cumulative: np.ndarray = np.array([])

    # --- Native event stats (no MC): reuse the Snowball KO/KI machinery ---

    def _event_stats_product_type(self) -> type:
        return PhoenixOption

    def _make_event_stats(self, **fields):
        from quantark.asset.equity.engine.event_stats import PhoenixEventStats

        return PhoenixEventStats(**fields)

    def _n_extra_event_cols(self, n_ko: int) -> int:
        # One coupon-trigger indicator column per observation.
        return n_ko

    def _set_extra_event_indicators(
        self, v0, v1, s_vec, n_ko, ko_idx, rec, product, pricing_env, t_vec, t_idx
    ) -> None:
        # Set the coupon-trigger column on the coupon-pay mask AFTER the KO jump,
        # so a coupon at a simultaneous KO is still counted (matches the Phoenix MC
        # reference: coupon_hit gated on "alive entering obs i", incl. first_ko_idx==i).
        if ko_idx is None or ko_idx >= self._coupon_barriers.shape[0]:
            return
        coupon_barrier = float(self._coupon_barriers[ko_idx])
        pay_mask = self._get_barrier_mask(
            s_vec, coupon_barrier, product.is_reverse, is_up_barrier=True
        )
        df_delay = self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=1.0,
            current_time=float(t_vec[t_idx]),
            settlement_time=rec.settlement_time,
        )
        coup_col = n_ko + ko_idx
        v0[pay_mask, coup_col] = df_delay
        v1[pay_mask, coup_col] = df_delay

    def _extract_extra_event_stats(
        self, initial_grid, x_vec, spot_log, n_ko, ko_records, pricing_env, product
    ) -> dict:
        ed_coup = np.array(
            [
                float(np.interp(spot_log, x_vec, initial_grid[:, n_ko + i]))
                for i in range(n_ko)
            ],
            dtype=float,
        )
        coupon_probability = np.zeros(n_ko, dtype=float)
        for i, rec in enumerate(ko_records):
            obs_time = float(rec.observation_time)
            settle = float(
                rec.settlement_time if rec.settlement_time is not None else obs_time
            )
            df0 = pricing_env.get_discount_factor(settle)
            if df0 > 0.0:
                coupon_probability[i] = float(ed_coup[i] / df0)
        result = {"coupon_probability": coupon_probability}
        ecc = self._coupon_cashflow_from_probability(
            coupon_probability, n_ko, ko_records, pricing_env, product
        )
        if ecc is not None:
            result["expected_discounted_coupon_cashflow"] = ecc
        return result

    def _coupon_cashflow_from_probability(
        self, coupon_probability, n_ko, ko_records, pricing_env, product
    ):
        """Expected discounted coupon cashflow for non-memory coupons, else None.

        With a deterministic per-period coupon amount and a deterministic
        settlement time, E[DF(0->settle) * amount * 1{coupon}] factors exactly as
        DF(0->settle) * amount * P(coupon). For memory coupons the paid amount is
        path-dependent and cannot be recovered from the trigger indicator, so we
        omit the field (probability stays correct) rather than report a wrong value.
        """
        if product.has_memory_coupon:
            return None
        expiry = product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
        maturity = float(product.get_maturity(pricing_env))
        ecc = np.zeros(n_ko, dtype=float)
        for i, rec in enumerate(ko_records):
            obs_time = float(rec.observation_time)
            settle = maturity if expiry else obs_time
            amt = (
                float(self._coupon_amounts[i])
                if i < self._coupon_amounts.shape[0]
                else 0.0
            )
            ecc[i] = float(
                pricing_env.get_discount_factor(settle) * amt * coupon_probability[i]
            )
        return ecc

    # price() and calculate_greeks() are inherited from SnowballPDESolver
    # The _check_product_type() method uses _supported_product_type to validate

    # _validate_product is identical to parent, so we inherit it

    def get_critical_points(
        self, product: PhoenixOption, pricing_env: PricingEnvironment
    ) -> List[float]:
        points = super().get_critical_points(product, pricing_env)

        coupon_barrier = product.coupon_config.coupon_barrier
        if isinstance(coupon_barrier, list):
            points.extend([b for b in coupon_barrier if b > 0])
        elif coupon_barrier > 0:
            points.append(coupon_barrier)

        return sorted(set([p for p in points if p > 0]))

    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        barriers = super()._get_barriers(product)
        if not isinstance(product, PhoenixOption):
            return barriers

        coupon_barrier = product.coupon_config.coupon_barrier
        if isinstance(coupon_barrier, list):
            barriers.extend([b for b in coupon_barrier if b > 0])
        elif coupon_barrier is not None and coupon_barrier > 0:
            barriers.append(coupon_barrier)

        return barriers

    def _get_immediate_ko_payoff(
        self, product: PhoenixOption, pricing_env: PricingEnvironment
    ) -> float:
        ko_records = product.resolve_ko_observations(pricing_env)
        ko_record_0 = self._find_record_at_time(ko_records, 0.0)
        if ko_record_0 is None:
            raise ValidationError(
                "Immediate KO payoff requested but no KO observation exists at valuation date."
            )

        spot = pricing_env.spot
        coupon_payoff = 0.0
        if product.is_coupon_triggered(spot, 0):
            coupon_payoff = product.get_coupon_payoff(0)

        payoff = float(ko_record_0.payoff or 0.0) + float(coupon_payoff)
        settlement_time = ko_record_0.settlement_time
        if settlement_time is not None and settlement_time > 0.0:
            df = pricing_env.get_discount_factor(settlement_time)
            return float(payoff) * float(df)
        return float(payoff)

    def _calculate_terminal_value(
        self, product: PhoenixOption, spot: float, pricing_env: PricingEnvironment
    ) -> float:
        """Calculate terminal payoff when already expired."""
        knocked_in = self._is_already_knocked_in(product, spot)
        return product.get_payoff(
            spot,
            knocked_in=knocked_in,
            accumulated_coupons=0.0,
            pricing_env=pricing_env,
        )

    def _solve(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> PDESolutionResult:
        """
        Core Two-Surface PDE solving logic for Phoenix options.
        Overrides Snowball logic to handle vector states for memory coupons.
        """
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        # Determine knocked-in state at valuation
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product, spot, pricing_env, ki_continuous=ki_continuous
        )
        self._knocked_in_at_valuation = knocked_in_at_valuation

        # Extract market data
        strike = product.strike
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # Store product properties
        self._is_reverse = product.is_reverse
        self._ki_continuous = ki_continuous
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                self._ki_barrier = ki_barrier[0]
            else:
                self._ki_barrier = ki_barrier

        if self._profile_enabled:
            self._reset_profile_stats()

        # Build grids
        if self._profile_enabled:
            t0 = perf_counter()
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )
        if self._profile_enabled:
            self._profile_stats["grid_build"] += perf_counter() - t0

        # Memory coupon setup
        use_memory = product.has_memory_coupon
        num_obs = len(self._coupon_barriers)
        if use_memory and num_obs > 50:
             raise ValidationError(
                f"Too many observations ({num_obs}) for Memory Phoenix PDE engine. "
                "Limit is 50 to prevent performance degradation. Use MC engine instead."
            )
        
        # Determine number of memory states to track at maturity
        max_k = num_obs if use_memory else 0
        
        # Initialize lists of grids
        num_x, num_t = len(x_vec), len(t_vec)
        
        # grid_v0_list[k] is the V0 surface for k missed coupons
        grid_v0_list = [np.zeros((num_x, num_t)) for _ in range(max_k + 1)]
        grid_v1_list = [np.zeros((num_x, num_t)) for _ in range(max_k + 1)]

        # Set terminal conditions for all memory states
        self._set_terminal_condition_vector(
            grid_v0_list, grid_v1_list, x_vec, s_vec, product, pricing_env
        )

        # Apply terminal coupon/KO/KI if maturity is an observation time.
        terminal_tidx = len(t_vec) - 1
        coupon_obs_idx = self._coupon_observation_indices.get(terminal_tidx)
        if coupon_obs_idx is not None:
            self._apply_coupon_jump_vector(
                grid_v0_list,
                grid_v1_list,
                s_vec,
                terminal_tidx,
                current_time=tau,
                product=product,
                pricing_env=pricing_env,
                obs_idx=coupon_obs_idx,
            )

        if product.has_ki_barrier:
            should_apply_ki = self._ki_continuous or terminal_tidx in self._ki_observation_indices
            if should_apply_ki:
                for k in range(len(grid_v0_list)):
                    self._apply_ki_jump(grid_v0_list[k], grid_v1_list[k], s_vec, terminal_tidx, product)

        # Maturity KO is stored in `_ko_terminal_record` by the inherited
        # `_build_grids` (it is intentionally kept out of
        # `_ko_observation_indices`). Apply it here, after the terminal
        # coupon/KI jumps, mirroring SnowballPDESolver._solve(). Using
        # `_apply_ko_jump_vector` preserves the same-date coupon-at-KO semantics,
        # since the terminal coupon index is registered in
        # `_coupon_observation_indices`.
        if self._has_terminal_ko and self._ko_terminal_record is not None:
            self._apply_ko_jump_vector(
                grid_v0_list,
                grid_v1_list,
                s_vec,
                terminal_tidx,
                current_time=tau,
                product=product,
                pricing_env=pricing_env,
                ko_record=self._ko_terminal_record,
            )

        # Build operator matrices
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        A = self._build_operator_matrix(l, c, u, num_x)

        # Time stepping with vector state
        self._time_stepping_vector_surface(
            grid_v0_list,
            grid_v1_list,
            A,
            l,
            c,
            u,
            x_vec,
            s_vec,
            t_vec,
            dt_vec,
            product,
            pricing_env,
            r,
            q,
            sigma,
            tau,
        )

        # Result is from state 0 (no accumulated memory at valuation)
        spot_log = np.log(spot)
        if knocked_in_at_valuation:
            solution_vec = grid_v1_list[0][:, 0]
        else:
            solution_vec = grid_v0_list[0][:, 0]

        return PDESolutionResult(
            solution_vec=solution_vec,
            x_vec=x_vec,
            s_vec=s_vec,
            spot_log=spot_log,
        )

    def _set_terminal_condition_vector(
        self,
        grid_v0_list: List[np.ndarray],
        grid_v1_list: List[np.ndarray],
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
    ) -> None:
        """Set terminal conditions for all memory states."""
        # V1 (Knocked-In): Payoff usually doesn't depend on memory (coupon lost?)
        payoff_v1 = np.array(
            [product.get_maturity_payoff_v1(s, pricing_env) for s in s_vec]
        )
        for grid in grid_v1_list:
            grid[:, -1] = payoff_v1

        # V0 (Not Knocked-In): Base payoff (coupons added via jumps)
        for k, grid in enumerate(grid_v0_list):
            payoff_v0 = np.array(
                [
                    product.get_maturity_payoff_v0(
                        s, accumulated_coupons=0.0, pricing_env=pricing_env
                    )
                    for s in s_vec
                ]
            )
            grid[:, -1] = payoff_v0

    def _build_grids(
        self,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ):
        result = super()._build_grids(product, pricing_env, spot, sigma, tau, r, q)
        _, _, _, t_vec, _ = result

        self._coupon_observation_indices.clear()
        ko_records = self._get_cached_ko_records(pricing_env, product)
        if not ko_records:
            return result

        ko_times = [rec.observation_time for rec in ko_records]
        num_obs = len(ko_times)

        coupon_barrier = product.coupon_config.coupon_barrier
        if isinstance(coupon_barrier, list):
            if len(coupon_barrier) != num_obs:
                raise ValidationError(
                    "Coupon barrier schedule length does not match KO observations."
                )
            self._coupon_barriers = np.array(coupon_barrier, dtype=float)
        else:
            self._coupon_barriers = np.full(num_obs, float(coupon_barrier))

        period_year_fractions = np.array(
            product.get_coupon_period_year_fractions(ko_times),
            dtype=float,
        )
        self._coupon_amounts = np.array(
            [
                product.get_coupon_payoff(i, year_fraction=period_year_fractions[i])
                for i in range(num_obs)
            ],
            dtype=float,
        )
        self._coupon_cumulative = np.concatenate(
            ([0.0], np.cumsum(self._coupon_amounts))
        )
        for obs_idx, obs_time in enumerate(ko_times):
            if is_close(obs_time, 0.0):
                self._coupon_observation_indices[0] = obs_idx
            elif is_close(obs_time, tau):
                self._coupon_observation_indices[len(t_vec) - 1] = obs_idx
            elif 0.0 < obs_time < tau:
                idx = self._aligned_time_index(t_vec, obs_time, "Coupon observation")
                self._coupon_observation_indices[idx] = obs_idx

        return result

    def _accumulated_coupon_amount(self, obs_idx: int, missed_count: int) -> float:
        if missed_count <= 0 or obs_idx <= 0:
            return 0.0
        start = max(obs_idx - missed_count, 0)
        return float(self._coupon_cumulative[obs_idx] - self._coupon_cumulative[start])

    def _time_stepping_vector_surface(
        self,
        grid_v0_list: List[np.ndarray],
        grid_v1_list: List[np.ndarray],
        A: sp.csc_matrix,
        l: np.ndarray,
        c: np.ndarray,
        u: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_vec: np.ndarray,
        dt_vec: np.ndarray,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        r: float,
        q: float,
        sigma: float,
        tau: float,
    ) -> None:
        """Backward time stepping for vector surfaces."""
        params = self.params
        num_t, num_x = len(t_vec), len(x_vec)
        I_int = sp.eye(num_x - 2, format="csc")
        use_banded = params.use_banded_solver
        n_int = num_x - 2
        
        # Reuse caches
        self._matrix_cache.clear()
        self._banded_cache.clear()

        # Temporary buffers for RHS/Sol
        rhs = None
        if use_banded and n_int > 2:
            rhs = np.empty(n_int, dtype=float)

        smooth_js = set()
        event_theta = params.event_theta
        event_steps = params.event_rannacher_steps
        if params.use_rannacher and params.auto_grid and params.rannacher_at_events:
            event_times = self._get_event_times(product, tau)
            if event_times and event_steps > 0:
                for et in event_times:
                    idx = int(np.argmin(np.abs(t_vec - et)))
                    if 0 < idx < num_t - 1 and is_close(float(t_vec[idx]), float(et)):
                        for k in range(event_steps):
                            smooth_idx = idx - 1 - k
                            if smooth_idx >= 0:
                                smooth_js.add(smooth_idx)

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            theta = params.theta
            
            steps_from_end = num_t - 1 - j
            if params.use_rannacher and steps_from_end < params.rannacher_steps:
                theta = 1.0
            elif j in smooth_js:
                theta = event_theta

            banded, lower1, main1, upper1 = (None, None, None, None)
            M1, M2_lu = (None, None)
            
            if use_banded and n_int > 2:
                banded, lower1, main1, upper1 = self._get_banded_system(l, c, u, dt, theta)
            else:
                M1, M2_lu = self._get_matrices(I_int, A, dt, theta)

            tau_remaining = tau - t_vec[j]
            
            # Step V0 grids
            self._step_grids(grid_v0_list, j, dt, theta, x_vec, s_vec, tau_remaining, product, pricing_env, 
                             use_banded, banded, lower1, main1, upper1, M1, M2_lu, rhs, l, u, is_v1=False)
            
            # Step V1 grids
            self._step_grids(grid_v1_list, j, dt, theta, x_vec, s_vec, tau_remaining, product, pricing_env,
                             use_banded, banded, lower1, main1, upper1, M1, M2_lu, rhs, l, u, is_v1=True)

            # Apply modifications (Coupons, KO, KI)
            self._apply_step_modifications_vector_surface(
                grid_v0_list, grid_v1_list, x_vec, s_vec, j, tau_remaining, product, pricing_env
            )

    def _step_grids(self, grid_list, j, dt, theta, x_vec, s_vec, tau_remaining, product, pricing_env,
                    use_banded, banded, lower1, main1, upper1, M1, M2_lu, rhs, l, u, is_v1):
        """Helper to diffuse a list of grids."""
        for grid in grid_list:
            # Set boundary
            if is_v1:
                self._set_boundary_conditions_v1(grid, x_vec, s_vec, j, tau_remaining, product, pricing_env)
            else:
                self._set_boundary_conditions_v0(grid, x_vec, s_vec, j, tau_remaining, product, pricing_env)
            
            v_next = grid[1:-1, j + 1]
            
            if use_banded and banded is not None:
                np.multiply(main1, v_next, out=rhs)
                rhs[1:] += lower1 * v_next[:-1]
                rhs[:-1] += upper1 * v_next[1:]
                
                if len(grid) > 2:
                    rhs[0] += dt * ((1.0 - theta) * l[1] * grid[0, j + 1] + theta * l[1] * grid[0, j])
                    rhs[-1] += dt * ((1.0 - theta) * u[-2] * grid[-1, j + 1] + theta * u[-2] * grid[-1, j])
                
                sol = solve_banded((1, 1), banded, rhs, overwrite_b=True, check_finite=False)
                grid[1:-1, j] = sol
            else:
                rhs_val = M1 @ v_next
                if len(grid) > 2:
                    rhs_val[0] += dt * ((1.0 - theta) * l[1] * grid[0, j + 1] + theta * l[1] * grid[0, j])
                    rhs_val[-1] += dt * ((1.0 - theta) * u[-2] * grid[-1, j + 1] + theta * u[-2] * grid[-1, j])
                
                grid[1:-1, j] = M2_lu.solve(rhs_val)

    def _apply_step_modifications_vector_surface(
        self,
        grid_v0_list: List[np.ndarray],
        grid_v1_list: List[np.ndarray],
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
    ) -> None:
        current_time = self._total_tau - tau

        # 1. Coupon Jump (Fan-In)
        coupon_obs_idx = self._coupon_observation_indices.get(t_idx)
        if coupon_obs_idx is not None:
            self._apply_coupon_jump_vector(
                grid_v0_list,
                grid_v1_list,
                s_vec,
                t_idx,
                current_time,
                product,
                pricing_env,
                coupon_obs_idx,
            )

        # 2. KI Jump
        if product.has_ki_barrier:
            should_apply_ki = self._ki_continuous or t_idx in self._ki_observation_indices
            if should_apply_ki:
                # Apply to all states
                for k in range(len(grid_v0_list)):
                    self._apply_ki_jump(grid_v0_list[k], grid_v1_list[k], s_vec, t_idx, product)

        # 3. KO Jump
        ko_record = self._ko_observation_indices.get(t_idx)
        if ko_record is not None:
            self._apply_ko_jump_vector(
                grid_v0_list,
                grid_v1_list,
                s_vec,
                t_idx,
                current_time,
                product,
                pricing_env,
                ko_record,
            )

    def _apply_coupon_jump_vector(
        self,
        grid_v0_list: List[np.ndarray],
        grid_v1_list: List[np.ndarray],
        s_vec: np.ndarray,
        t_idx: int,
        current_time: float,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        obs_idx: int,
    ) -> None:
        if obs_idx < 0 or obs_idx >= self._coupon_barriers.shape[0]:
            return

        barrier = float(self._coupon_barriers[obs_idx])
        coupon_amt = float(self._coupon_amounts[obs_idx])
        use_memory = product.has_memory_coupon
        
        settlement_time = (
            self._total_tau
            if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
            else current_time
        )
        coupon_discount = self._df_between_times(pricing_env, current_time, settlement_time)

        # Coupon barrier behaves like KO (UP barrier) - pay when above
        pay_mask = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)
        
        max_k = obs_idx if use_memory else 0
        diffused_v0_0 = grid_v0_list[0][:, t_idx].copy()
        diffused_v1_0 = grid_v1_list[0][:, t_idx].copy()
        
        for k in range(max_k + 1):
            accumulated_pay = (
                self._accumulated_coupon_amount(obs_idx, k) if use_memory else 0.0
            )
            total_pay = (coupon_amt + accumulated_pay) * coupon_discount
            
            # V0
            val_pay_0 = diffused_v0_0 + total_pay
            next_k = k + 1 if use_memory else 0
            val_miss_0 = grid_v0_list[next_k][:, t_idx]
            
            grid_v0_list[k][pay_mask, t_idx] = val_pay_0[pay_mask]
            grid_v0_list[k][~pay_mask, t_idx] = val_miss_0[~pay_mask]
            
            # V1
            val_pay_1 = diffused_v1_0 + total_pay
            val_miss_1 = grid_v1_list[next_k][:, t_idx]
            
            grid_v1_list[k][pay_mask, t_idx] = val_pay_1[pay_mask]
            grid_v1_list[k][~pay_mask, t_idx] = val_miss_1[~pay_mask]

    def _apply_ko_jump_vector(
        self,
        grid_v0_list: List[np.ndarray],
        grid_v1_list: List[np.ndarray],
        s_vec: np.ndarray,
        t_idx: int,
        current_time: float,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        ko_record: ResolvedObservationRecord,
    ) -> None:
        barrier = ko_record.barrier
        base_payoff = float(ko_record.payoff or 0.0)
        
        ko_mask = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)
            
        coupon_amt = 0.0
        obs_idx = self._coupon_observation_indices.get(t_idx)
        if obs_idx is not None:
            coupon_amt = float(self._coupon_amounts[obs_idx])
            
        use_memory = product.has_memory_coupon
        max_k = obs_idx if (obs_idx is not None and use_memory) else 0

        if obs_idx is not None:
            coupon_barrier = float(self._coupon_barriers[obs_idx])
            pay_mask = self._get_barrier_mask(s_vec, coupon_barrier, product.is_reverse, is_up_barrier=True)
        else:
            pay_mask = np.zeros_like(s_vec, dtype=bool)
        
        df = 1.0
        if ko_record.settlement_time is not None and ko_record.settlement_time > current_time:
            df = self._df_between_times(pricing_env, current_time, ko_record.settlement_time)
            
        for k in range(len(grid_v0_list)):
            effective_k = k if k <= max_k else max_k 
            accumulated_pay = (
                self._accumulated_coupon_amount(obs_idx, effective_k)
                if (use_memory and obs_idx is not None)
                else 0.0
            )

            total_payoff = np.full_like(
                s_vec, (base_payoff + accumulated_pay) * df, dtype=float
            )
            if coupon_amt > 0.0:
                total_payoff = np.where(
                    pay_mask, total_payoff + coupon_amt * df, total_payoff
                )
            
            if ko_mask.any():
                grid_v0_list[k][ko_mask, t_idx] = total_payoff[ko_mask]
                grid_v1_list[k][ko_mask, t_idx] = total_payoff[ko_mask]
