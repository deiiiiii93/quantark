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

from quantark.asset.equity.engine.pde.backward_operator import BackwardOperator
from quantark.asset.equity.engine.pde.base_pde_solver import PDESolutionResult
from quantark.asset.equity.engine.pde.event_projection import (
    project_piecewise_event,
)
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
        df_delay = self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=1.0,
            current_time=float(t_vec[t_idx]),
            settlement_time=rec.settlement_time,
        )
        coup_col = n_ko + ko_idx
        if self._event_uses_projection(t_idx):
            for v in (v0, v1):
                v[:, coup_col] = self._project_event_values(
                    s_vec, coupon_barrier, product.is_reverse, True,
                    v[:, coup_col], df_delay,
                )
            return
        pay_mask = self._event_nodal_mask(
            s_vec, coupon_barrier, product.is_reverse, True,
            at_valuation=(t_idx == 0),
        )
        v0[pay_mask, coup_col] = df_delay
        v1[pay_mask, coup_col] = df_delay

    def _t0_extra_indicator_overrides(
        self,
        product,
        pricing_env,
        spot,
        n_ko,
        rec0_pos,
        rec0,
        t_vec,
        ko_triggered,
        df_delay0,
    ) -> dict:
        """Deterministic valuation-date coupon indicator at the known spot.

        Mirrors _set_extra_event_indicators (a coupon at a simultaneous KO is
        still counted); a triggered t=0 KO kills every later coupon stream."""
        overrides: dict = {}
        if rec0_pos is None or rec0_pos >= self._coupon_barriers.shape[0]:
            return overrides
        coupon_barrier = float(self._coupon_barriers[rec0_pos])
        pay = bool(
            self._event_nodal_mask(
                np.asarray([float(spot)], dtype=float),
                coupon_barrier,
                product.is_reverse,
                True,
                at_valuation=True,
            )[0]
        )
        overrides[n_ko + rec0_pos] = df_delay0 if pay else 0.0
        if ko_triggered:
            for i in range(n_ko):
                if i != rec0_pos:
                    overrides[n_ko + i] = 0.0
        return overrides

    def _extract_extra_event_stats(
        self,
        initial_grid,
        x_vec,
        spot_log,
        n_ko,
        ko_records,
        pricing_env,
        product,
        col_overrides=None,
    ) -> dict:
        overrides = col_overrides or {}

        def _read(col: int) -> float:
            if col in overrides:
                return float(overrides[col])
            return float(np.interp(spot_log, x_vec, initial_grid[:, col]))

        ed_coup = np.array([_read(n_ko + i) for i in range(n_ko)], dtype=float)
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

        # State preamble shared with session preparation (see
        # SnowballPDESolver._prepare_solve_state).
        knocked_in_at_valuation = self._prepare_solve_state(product, pricing_env)
        self._reset_t0_readout_state()

        # Extract market data
        strike = product.strike
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # BGK state is resolved at the top of _build_grids (see
        # SnowballPDESolver._build_grids) so subclasses cannot skip it.

        if self._profile_enabled:
            self._reset_profile_stats()

        # Build grids: declarative layer on the migrated path (grid geometry
        # + damping data); the certified vector-surface event application
        # below stays inline — it already flows through the moved projection
        # primitives, and its coupon-memory dispatch is solver-owned state.
        if self._profile_enabled:
            t0 = perf_counter()
        self._active_layout = None
        self._active_schedule = None
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
            should_apply_ki = (
                self._ki_continuous
                or self._bgk_active
                or terminal_tidx in self._ki_observation_indices
            )
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

        # Term-structure step coefficients (one set for flat inputs)
        sc = self._build_step_coefficients(pricing_env, product.strike, t_vec, dx_vec, num_x)
        sc = self._flat_exact_step_coefficients(sc, r, q, sigma, dx_vec, num_x)
        step_coeffs = None if sc.n_unique == 1 else sc

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
            step_coeffs=step_coeffs,
        )

        # Result is from state 0 (no accumulated memory at valuation)
        spot_log = np.log(spot)
        if knocked_in_at_valuation:
            solution_vec = grid_v1_list[0][:, 0]
        else:
            solution_vec = grid_v0_list[0][:, 0]

        readout_vec, readout_override = self._compose_t0_readout(
            1 if knocked_in_at_valuation else 0
        )
        return PDESolutionResult(
            solution_vec=solution_vec,
            x_vec=x_vec,
            s_vec=s_vec,
            spot_log=spot_log,
            readout_vec=readout_vec,
            readout_override=readout_override,
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
        if self._active_layout is None:
            # Legacy path only: the layer path registers coupons inside
            # _populate_observation_maps (single site).
            _, _, _, t_vec, _ = result
            self._register_coupon_observations(product, pricing_env, t_vec, tau)
        return result

    def _register_coupon_observations(
        self,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        t_vec: np.ndarray,
        tau: float,
        step_of=None,
    ) -> None:
        """Coupon barrier/amount arrays + observation-index map.

        ``step_of`` (the layout's exact-float map) resolves indices on the
        migrated path; the legacy path keeps ``_aligned_time_index``.
        """
        self._coupon_observation_indices.clear()
        ko_records = self._get_cached_ko_records(pricing_env, product)
        if not ko_records:
            return

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
                if step_of is not None:
                    idx = step_of(obs_time)
                else:
                    idx = self._aligned_time_index(
                        t_vec, obs_time, "Coupon observation"
                    )
                self._coupon_observation_indices[idx] = obs_idx

    def _uses_grid_layer(self) -> bool:
        return True

    def _populate_observation_maps(self, product, pricing_env, layout, tau):
        # Coupon registration rides along wherever the layer populates the
        # KO/KI maps (_solve AND the event-stats sweep) — single site.
        super()._populate_observation_maps(product, pricing_env, layout, tau)
        self._register_coupon_observations(
            product, pricing_env, layout.time.t, tau, step_of=layout.time.step_at
        )

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
        step_coeffs=None,
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
        self._term_A_cache = {}

        # Temporary buffers for RHS/Sol
        rhs = None
        if use_banded and n_int > 2:
            rhs = np.empty(n_int, dtype=float)

        # Canonical damping schedule (terminal Rannacher + event smoothing):
        # from the layout's frozensets on the migrated path, else legacy.
        if self._active_layout is not None:
            theta_schedule = self._theta_schedule_from_layout(self._active_layout)
        else:
            theta_schedule = BackwardOperator.theta_by_step(
                np.asarray(t_vec),
                np.asarray(dt_vec),
                params,
                self._get_event_times(product, tau),
            )

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            theta = float(theta_schedule[j])

            if step_coeffs is not None:
                coeff_key = int(step_coeffs.set_index[j])
                l, c, u = step_coeffs.lcu_sets[coeff_key]
            else:
                coeff_key = 0

            banded, lower1, main1, upper1 = (None, None, None, None)
            M1, M2_lu = (None, None)
            
            if use_banded and n_int > 2:
                banded, lower1, main1, upper1 = self._get_banded_system(
                    l, c, u, dt, theta, coeff_key=coeff_key
                )
            else:
                if step_coeffs is not None:
                    A = self._operator_matrix_for_set(step_coeffs, coeff_key, num_x)
                M1, M2_lu = self._get_matrices(I_int, A, dt, theta, coeff_key=coeff_key)

            tau_remaining = tau - t_vec[j]
            
            # Step V0 grids
            self._step_grids(grid_v0_list, j, dt, theta, x_vec, s_vec, tau_remaining, product, pricing_env, 
                             use_banded, banded, lower1, main1, upper1, M1, M2_lu, rhs, l, u, is_v1=False)
            
            # Step V1 grids
            self._step_grids(grid_v1_list, j, dt, theta, x_vec, s_vec, tau_remaining, product, pricing_env,
                             use_banded, banded, lower1, main1, upper1, M1, M2_lu, rhs, l, u, is_v1=True)

            # Apply modifications (Coupons, KO, KI). The valuation-date
            # level-0 columns are captured BEFORE their events: t=0 events
            # are deterministic at the known spot, so the readout uses this
            # smooth 0+ branch plus pointwise transitions instead of
            # interpolating across the nodal t=0 jump.
            if j == 0 and self._t0_has_events(product):
                self._t0_pre_event_cols = (
                    grid_v0_list[0][:, 0].copy(),
                    grid_v1_list[0][:, 0].copy(),
                )
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

        # 1. Coupon Jump (Fan-In). When a KO observation shares the date and
        # events project, the KO site applies the coupon fan-in and the KO
        # transition as ONE piecewise cell average — sequential projection
        # double-averages a shared dual cell [2026-07-24 review, finding 3].
        coupon_obs_idx = self._coupon_observation_indices.get(t_idx)
        if coupon_obs_idx is not None and not self._joint_coupon_ko_projection_active(
            t_idx
        ):
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
            should_apply_ki = (
                self._ki_continuous
                or self._bgk_active
                or t_idx in self._ki_observation_indices
            )
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

        max_k = obs_idx if use_memory else 0
        diffused_v0_0 = grid_v0_list[0][:, t_idx].copy()
        diffused_v1_0 = grid_v1_list[0][:, t_idx].copy()

        if self._event_uses_projection(t_idx):
            for k in range(max_k + 1):
                accumulated_pay = (
                    self._accumulated_coupon_amount(obs_idx, k) if use_memory else 0.0
                )
                total_pay = (coupon_amt + accumulated_pay) * coupon_discount
                next_k = k + 1 if use_memory else 0

                val_miss_0 = grid_v0_list[next_k][:, t_idx]
                grid_v0_list[k][:, t_idx] = self._project_event_values(
                    s_vec, barrier, product.is_reverse, True,
                    val_miss_0, diffused_v0_0 + total_pay,
                )
                val_miss_1 = grid_v1_list[next_k][:, t_idx]
                grid_v1_list[k][:, t_idx] = self._project_event_values(
                    s_vec, barrier, product.is_reverse, True,
                    val_miss_1, diffused_v1_0 + total_pay,
                )
            return

        # Coupon barrier behaves like KO (UP barrier) - pay when above
        pay_mask = self._event_nodal_mask(
            s_vec, barrier, product.is_reverse, True, at_valuation=(t_idx == 0)
        )

        if t_idx == 0 and self._use_cell_average_events():
            self._capture_t0_coupon_readout(
                s_vec,
                barrier,
                product,
                pricing_env,
                diffused_v0_0,
                diffused_v1_0,
                grid_v0_list,
                grid_v1_list,
                use_memory,
                obs_idx,
                coupon_amt,
                coupon_discount,
            )

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

    def _t0_has_events(self, product) -> bool:
        if super()._t0_has_events(product):
            return True
        return (
            self._use_cell_average_events()
            and 0 in self._coupon_observation_indices
        )

    def _joint_coupon_ko_projection_active(self, t_idx: int) -> bool:
        """Coincident coupon + KO date whose projection runs as ONE pass.

        _apply_step_modifications_vector_surface skips the standalone coupon
        exactly when this holds, and _apply_ko_jump_vector performs the joint
        piecewise projection instead [2026-07-24 review, finding 3]."""
        if not self._event_uses_projection(t_idx):
            return False
        obs_idx = self._coupon_observation_indices.get(t_idx)
        rec = self._ko_observation_indices.get(t_idx)
        if obs_idx is None or rec is None:
            return False
        if obs_idx < 0 or obs_idx >= self._coupon_barriers.shape[0]:
            return False
        b_ko = float(rec.barrier) if rec.barrier is not None else 0.0
        return b_ko > 0.0 and float(self._coupon_barriers[obs_idx]) > 0.0

    def _apply_joint_coupon_ko_projection(
        self,
        grid_v0_list: List[np.ndarray],
        grid_v1_list: List[np.ndarray],
        s_vec: np.ndarray,
        t_idx: int,
        current_time: float,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        obs_idx: int,
        ko_barrier: float,
        base_payoff: float,
        df: float,
        coupon_amt: float,
        use_memory: bool,
        max_k: int,
    ) -> None:
        """One-pass cell average of the coincident coupon + KO event.

        The contractual post-event value is piecewise in spot with thresholds
        at the coupon and KO barriers:

            KO region:      (base + coupon_at_ko * 1_pay) * df
            survive & pay:  level-0 diffused value + coupon
            survive & miss: next memory level's value

        Sequential projection double-averages any dual cell the two
        thresholds share (equal/nearby barriers); for well-separated barriers
        this one-pass average coincides with the sequential result
        [2026-07-24 review, finding 3].
        """
        n = len(s_vec)
        x_vec = np.log(np.asarray(s_vec, dtype=float))
        coupon_barrier = float(self._coupon_barriers[obs_idx])
        settlement_time = (
            self._total_tau
            if product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
            else current_time
        )
        coupon_discount = self._df_between_times(
            pricing_env, current_time, settlement_time
        )
        trig_up = not bool(product.is_reverse)
        breaks = sorted((np.log(coupon_barrier), np.log(float(ko_barrier))))
        lows = [-np.inf] + breaks
        his = breaks + [np.inf]
        b_c_x, b_ko_x = np.log(coupon_barrier), np.log(float(ko_barrier))

        diffused_cols = (
            grid_v0_list[0][:, t_idx].copy(),
            grid_v1_list[0][:, t_idx].copy(),
        )
        for k in range(len(grid_v0_list)):
            effective_k = k if k <= max_k else max_k
            accumulated_ko = (
                self._accumulated_coupon_amount(obs_idx, effective_k)
                if use_memory
                else 0.0
            )
            coupon_at_ko = coupon_amt + accumulated_ko
            in_coupon_fan = k <= max_k
            if in_coupon_fan:
                accumulated_k = (
                    self._accumulated_coupon_amount(obs_idx, k)
                    if use_memory
                    else 0.0
                )
                total_pay = (coupon_amt + accumulated_k) * coupon_discount
                next_k = k + 1 if use_memory else 0
            for grids, diffused0 in (
                (grid_v0_list, diffused_cols[0]),
                (grid_v1_list, diffused_cols[1]),
            ):
                branches = []
                for j in range(len(breaks) + 1):
                    if trig_up:
                        m_ko = b_ko_x <= lows[j]
                        m_pay = b_c_x <= lows[j]
                    else:
                        m_ko = his[j] <= b_ko_x
                        m_pay = his[j] <= b_c_x
                    if m_ko:
                        branches.append(
                            np.full(
                                n,
                                (base_payoff + (coupon_at_ko if m_pay else 0.0))
                                * df,
                            )
                        )
                    elif in_coupon_fan and m_pay:
                        branches.append(diffused0 + total_pay)
                    elif in_coupon_fan:
                        branches.append(grids[next_k][:, t_idx])
                    else:
                        branches.append(grids[k][:, t_idx])
                grids[k][:, t_idx] = project_piecewise_event(
                    x_vec, breaks, branches
                )

    def _capture_t0_coupon_readout(
        self,
        s_vec: np.ndarray,
        barrier: float,
        product: PhoenixOption,
        pricing_env: PricingEnvironment,
        diffused_v0_0: np.ndarray,
        diffused_v1_0: np.ndarray,
        grid_v0_list: List[np.ndarray],
        grid_v1_list: List[np.ndarray],
        use_memory: bool,
        obs_idx: int,
        coupon_amt: float,
        coupon_discount: float,
    ) -> None:
        """Pointwise-exact valuation-date coupon readout at the actual spot.

        The nodal t=0 application leaves a value jump at the coupon barrier;
        interpolating the post-event column across it blends the pay/miss
        branches (a grid-dependent fraction of the coupon). Instead resolve
        today's trigger at the known spot with the same inclusive ownership
        the nodal mask uses, interpolate the smooth branch the spot actually
        follows, and add the coupon cash [2026-07-24 review, finding 1].
        """
        spot = float(pricing_env.spot)
        pay = bool(
            self._event_nodal_mask(
                np.asarray([spot], dtype=float),
                barrier,
                product.is_reverse,
                True,
                at_valuation=True,
            )[0]
        )
        x_vec = np.log(np.asarray(s_vec, dtype=float))
        spot_log = float(np.log(spot))
        if pay:
            accumulated = (
                self._accumulated_coupon_amount(obs_idx, 0) if use_memory else 0.0
            )
            total_pay = (coupon_amt + accumulated) * coupon_discount
            cols = (diffused_v0_0.copy(), diffused_v1_0.copy())
            values = tuple(
                float(np.interp(spot_log, x_vec, col)) + total_pay for col in cols
            )
        else:
            next_k = 1 if use_memory else 0
            cols = (
                grid_v0_list[next_k][:, 0].copy(),
                grid_v1_list[next_k][:, 0].copy(),
            )
            values = tuple(float(np.interp(spot_log, x_vec, col)) for col in cols)
        self._t0_readout_cols = cols
        self._t0_readout_values = values

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

        coupon_amt = 0.0
        obs_idx = self._coupon_observation_indices.get(t_idx)
        if obs_idx is not None:
            coupon_amt = float(self._coupon_amounts[obs_idx])

        use_memory = product.has_memory_coupon
        max_k = obs_idx if (obs_idx is not None and use_memory) else 0

        if not self._event_uses_projection(t_idx):
            at_val = t_idx == 0
            ko_mask = self._event_nodal_mask(
                s_vec, barrier, product.is_reverse, True, at_valuation=at_val
            )
            if obs_idx is not None:
                coupon_barrier = float(self._coupon_barriers[obs_idx])
                pay_mask = self._event_nodal_mask(
                    s_vec, coupon_barrier, product.is_reverse, True,
                    at_valuation=at_val,
                )
            else:
                pay_mask = np.zeros_like(s_vec, dtype=bool)

        df = 1.0
        if ko_record.settlement_time is not None and ko_record.settlement_time > current_time:
            df = self._df_between_times(pricing_env, current_time, ko_record.settlement_time)

        if self._event_uses_projection(t_idx):
            if self._joint_coupon_ko_projection_active(t_idx):
                # Coincident coupon + KO: fan-in and KO transition are ONE
                # piecewise contractual function of spot — project its exact
                # cell average in a single pass (the standalone coupon
                # application was skipped) [2026-07-24 review, finding 3].
                self._apply_joint_coupon_ko_projection(
                    grid_v0_list,
                    grid_v1_list,
                    s_vec,
                    t_idx,
                    current_time,
                    product,
                    pricing_env,
                    obs_idx,
                    float(barrier),
                    base_payoff,
                    df,
                    coupon_amt,
                    use_memory,
                    max_k,
                )
                return
            for k in range(len(grid_v0_list)):
                effective_k = k if k <= max_k else max_k
                accumulated_pay = (
                    self._accumulated_coupon_amount(obs_idx, effective_k)
                    if (use_memory and obs_idx is not None)
                    else 0.0
                )
                # Degenerate fallback (no coupon obs, or non-positive
                # barriers where the log-space breaks are undefined):
                # sequential inner projection as before.
                coupon_at_ko = coupon_amt + accumulated_pay
                if coupon_at_ko > 0.0 and obs_idx is not None:
                    coupon_barrier = float(self._coupon_barriers[obs_idx])
                    total_payoff = self._project_event_values(
                        s_vec, coupon_barrier, product.is_reverse, True,
                        base_payoff * df, (base_payoff + coupon_at_ko) * df,
                    )
                else:
                    total_payoff = np.full(len(s_vec), base_payoff * df)
                grid_v0_list[k][:, t_idx] = self._project_event_values(
                    s_vec, barrier, product.is_reverse, True,
                    grid_v0_list[k][:, t_idx], total_payoff,
                )
                grid_v1_list[k][:, t_idx] = self._project_event_values(
                    s_vec, barrier, product.is_reverse, True,
                    grid_v1_list[k][:, t_idx], total_payoff,
                )
            return

        for k in range(len(grid_v0_list)):
            effective_k = k if k <= max_k else max_k
            accumulated_pay = (
                self._accumulated_coupon_amount(obs_idx, effective_k)
                if (use_memory and obs_idx is not None)
                else 0.0
            )

            # Memory semantics (matching the Phoenix Monte-Carlo engine's
            # convention): accrued coupons are released ONLY when the current
            # observation's coupon condition is met — a KO below the coupon
            # barrier forfeits them along with the current coupon.
            total_payoff = np.full_like(s_vec, base_payoff * df, dtype=float)
            coupon_at_ko = coupon_amt + accumulated_pay
            if coupon_at_ko > 0.0:
                total_payoff = np.where(
                    pay_mask, total_payoff + coupon_at_ko * df, total_payoff
                )

            if ko_mask.any():
                grid_v0_list[k][ko_mask, t_idx] = total_payoff[ko_mask]
                grid_v1_list[k][ko_mask, t_idx] = total_payoff[ko_mask]
