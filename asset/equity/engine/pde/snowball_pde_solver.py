"""
PDE solver for Snowball (autocallable) options using the unified PDESystemState.

This solver maintains two value surfaces via PDESystemState:
- State 0: Value when knock-in (KI) has NOT occurred
- State 1: Value when knock-in (KI) HAS occurred

The surfaces interact at barrier observation times via PDEEvents:
- KO Event: Sets value to payoff (on both surfaces)
- KI Event: Copies value from State 1 to State 0
"""

from collections import OrderedDict, defaultdict
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from asset.equity.engine.pde.base_pde_solver import BasePDESolver, PDESolutionResult
from asset.equity.engine.pde.core import PDESystemState, KnockOutEvent, KnockInEvent
from asset.equity.engine.event_stats import AutocallableEventStats
from asset.equity.param import PDEParams
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from asset.equity.product.option.snowball_option import SnowballOption
from priceenv import PricingEnvironment
from util.enum import ObservationType, ProtectionType
from util.exceptions import PricingError, ValidationError
from util.numerical import Tolerance, is_close, is_zero, safe_divide


class SnowballPDESolver(BasePDESolver):
    """
    PDE solver for Snowball (autocallable) options using the generic PDESystemState.
    """

    _supported_product_type: type = SnowballOption
    _solver_name: str = "SnowballPDESolver"

    def __init__(
        self, params: Optional[PDEParams] = None, enable_profiling: bool = False
    ):
        super().__init__(params)
        self._profile_enabled = enable_profiling
        self._profile_stats: Dict[str, float] = {}
        
        # Caches
        self._ko_records_cache: "OrderedDict[Tuple, List[ResolvedObservationRecord]]" = OrderedDict()
        self._ki_profile_cache: "OrderedDict[Tuple, Dict[str, List[Optional[float]]]]" = OrderedDict()
        
        # State tracking (for observation alignment)
        self._total_tau: float = 0.0
        self._state: Optional[PDESystemState] = None

    def enable_profiling(self, enabled: bool = True) -> None:
        self._profile_enabled = enabled

    def get_profile_stats(self) -> Dict[str, float]:
        return dict(self._profile_stats)

    def _reset_profile_stats(self) -> None:
        self._profile_stats = {
            "grid_build": 0.0,
            "boundary": 0.0,
            "matrix_build": 0.0,
            "rhs": 0.0,
            "solve": 0.0,
            "barrier": 0.0,
        }

    def _solve(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> PDESolutionResult:
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        # Check KI status at valuation
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product, spot, pricing_env, ki_continuous=ki_continuous
        )

        strike = product.strike
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # Clear per-solve caches
        self._matrix_cache.clear()
        self._banded_cache.clear()

        if self._profile_enabled:
            self._reset_profile_stats()

        # 1. Build Grids
        if self._profile_enabled: t0 = perf_counter()
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )
        if self._profile_enabled: self._profile_stats["grid_build"] += perf_counter() - t0

        num_x, num_t = len(x_vec), len(t_vec)
        
        # 2. Initialize State (2 surfaces: 0=No-KI, 1=KI)
        state = PDESystemState(num_x, num_t, num_states=2)
        self._state = state
        
        # 3. Set Terminal Conditions
        self._set_terminal_condition_v0(
            state.get_slice(num_t - 1)[:, 0], x_vec, s_vec, product, pricing_env
        )
        self._set_terminal_condition_v1(
            state.get_slice(num_t - 1)[:, 1], x_vec, s_vec, product, pricing_env
        )

        # 4. Build Events
        events_by_step = self._build_events_map(
            t_vec, product, pricing_env, tau, ki_continuous
        )
        
        # Apply terminal events (if any at t=T)
        if (num_t - 1) in events_by_step:
            for event in events_by_step[num_t - 1]:
                event.apply(state, num_t - 1, t_vec[num_t - 1], s_vec, pricing_env)

        # 5. Build Operator Matrices
        if self._profile_enabled: t0 = perf_counter()
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        if self._profile_enabled: self._profile_stats["matrix_build"] += perf_counter() - t0

        # 6. Time Stepping
        params = self.params
        smooth_js = set()
        if params.use_rannacher and params.auto_grid and params.rannacher_at_events:
            for et in self._get_event_times(product, tau) or []:
                idx = int(np.argmin(np.abs(t_vec - et)))
                if 0 < idx < num_t - 1 and is_close(float(t_vec[idx]), float(et)):
                    smooth_js.update([idx - 1 - k for k in range(params.rannacher_steps) if idx - 1 - k >= 0])

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            steps_from_end = num_t - 1 - j
            current_time = t_vec[j]
            tau_remaining = tau - current_time
            
            theta = params.theta
            if params.use_rannacher and steps_from_end < params.rannacher_steps:
                theta = 1.0
            elif j in smooth_js:
                theta = params.event_theta

            # Set Boundary Conditions (for V0 and V1)
            if self._profile_enabled: t0 = perf_counter()
            self._set_boundary_conditions_v0(
                state.get_slice(j)[:, 0], x_vec, s_vec, j, tau_remaining, product, pricing_env
            )
            self._set_boundary_conditions_v1(
                state.get_slice(j)[:, 1], x_vec, s_vec, j, tau_remaining, product, pricing_env
            )
            if self._profile_enabled: self._profile_stats["boundary"] += perf_counter() - t0

            # Solve Step
            if self._profile_enabled: t0 = perf_counter()
            
            banded, lower1, main1, upper1 = self._get_banded_system(
                l, c, u, dt, theta, full_coeffs=(l, c, u)
            )
            
            def injector(rhs_buffer, t_idx):
                self._inject_boundary_contributions(rhs_buffer, state.grids[:, :, :], l, u, t_idx, dt, theta)
            
            state.solve_step_banded(j, j + 1, banded, (lower1, main1, upper1), injector)
            
            if self._profile_enabled: self._profile_stats["solve"] += perf_counter() - t0

            # Apply Events
            if j in events_by_step:
                if self._profile_enabled: t0 = perf_counter()
                for event in events_by_step[j]:
                    event.apply(state, j, current_time, s_vec, pricing_env)
                if self._profile_enabled: self._profile_stats["barrier"] += perf_counter() - t0
                
            # Apply Continuous KI if needed
            if ki_continuous and product.has_ki_barrier:
                ki_barrier = product.barrier_config.ki_barrier
                if isinstance(ki_barrier, list): ki_barrier = ki_barrier[0]
                event = KnockInEvent(float(ki_barrier), product.is_reverse, source_idx=1, target_idx=0)
                event.apply(state, j, current_time, s_vec, pricing_env)

        # 7. Result
        spot_log = np.log(spot)
        solution_idx = 1 if knocked_in_at_valuation else 0
        solution_vec = state.get_slice(0)[:, solution_idx]

        return PDESolutionResult(
            solution_vec=solution_vec,
            x_vec=x_vec,
            s_vec=s_vec,
            spot_log=spot_log,
        )

    @property
    def _grid_v0(self):
        if self._state is not None:
            return self._state.grids[:, :, 0]
        return None

    @property
    def _grid_v1(self):
        if self._state is not None:
            return self._state.grids[:, :, 1]
        return None

    def _build_events_map(self, t_vec, product, pricing_env, tau, ki_continuous):
        events = defaultdict(list)
        
        # KO Events
        ko_records = self._get_cached_ko_records(pricing_env, product)
        for rec in ko_records:
            if not 0.0 <= rec.observation_time <= tau:
                continue
            idx = self._aligned_time_index(t_vec, rec.observation_time, "KO")
            events[idx].append(KnockOutEvent(rec, product.is_reverse))
            
        # KI Events (Discrete)
        if product.has_ki_barrier and not ki_continuous:
            ki_profile = self._get_cached_ki_profile(pricing_env, product)
            for obs_time in ki_profile.get("observation_times", []):
                if not 0.0 <= obs_time <= tau:
                    continue
                idx = self._aligned_time_index(t_vec, obs_time, "KI")
                ki_barrier = product.barrier_config.ki_barrier
                if isinstance(ki_barrier, list): ki_barrier = ki_barrier[0]
                events[idx].append(KnockInEvent(float(ki_barrier), product.is_reverse, source_idx=1, target_idx=0))
                
        return events

    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        """Collect all barrier levels for spatial grid construction."""
        barriers = []
        if hasattr(product, "barrier_config"):
            ko_barrier = product.barrier_config.ko_barrier
            if isinstance(ko_barrier, list): barriers.extend(ko_barrier)
            elif ko_barrier > 0: barriers.append(ko_barrier)
            if product.barrier_config.ki_barrier is not None:
                ki_barrier = product.barrier_config.ki_barrier
                if isinstance(ki_barrier, list): barriers.extend(ki_barrier)
                elif ki_barrier > 0: barriers.append(ki_barrier)
        return barriers

    def _get_event_times(
        self, product: BaseEquityProduct, tau: float
    ) -> Optional[List[float]]:
        """Collect all observation times for time grid alignment."""
        event_times = []

        if hasattr(product, "barrier_config"):
            # KO observation times
            ko_schedule = product.barrier_config.ko_observation_schedule
            if ko_schedule is not None:
                for rec in ko_schedule.records:
                    if rec.observation_time is not None:
                        t = rec.observation_time
                        if 0 < t < tau:
                            event_times.append(t)
            elif product.barrier_config.ko_observation_dates is not None:
                for t in product.barrier_config.ko_observation_dates:
                    if 0 < t < tau:
                        event_times.append(t)

            # KI observation times (if discrete)
            ki_continuous = (
                product.barrier_config.ki_continuous
                or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            )
            if not ki_continuous:
                ki_schedule = product.barrier_config.ki_observation_schedule
                if ki_schedule is not None:
                    for rec in ki_schedule.records:
                        if rec.observation_time is not None:
                            t = rec.observation_time
                            if 0 < t < tau:
                                event_times.append(t)
                elif product.barrier_config.ki_observation_dates is not None:
                    for t in product.barrier_config.ki_observation_dates:
                        if 0 < t < tau:
                            event_times.append(t)

        return sorted(set(event_times)) if event_times else None

    def price(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> float:
        self._check_product_type(product)
        if pricing_env is None:
            raise ValidationError(f"PricingEnvironment is required for {self._solver_name}")
        self._validate_product(product)
        
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)
        
        if tau <= 0 or is_zero(tau):
            return self._calculate_terminal_value(product, spot, pricing_env)
            
        if self._is_knocked_out_at_valuation(product, spot, pricing_env):
            return self._get_immediate_ko_payoff(product, pricing_env)
            
        result = self._solve(product, pricing_env)
        return self._interpolate_price(result.solution_vec, result.x_vec, result.spot_log)

    def calculate_greeks(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> Dict[str, float]:
        self._check_product_type(product)
        if pricing_env is None:
            raise ValidationError(f"PricingEnvironment is required for {self._solver_name}")
        self._validate_product(product)
        
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)
        
        if tau <= 0 or is_zero(tau):
            return {"price": self._calculate_terminal_value(product, spot, pricing_env), "delta": 0.0, "gamma": 0.0}
            
        if self._is_knocked_out_at_valuation(product, spot, pricing_env):
            return {"price": self._get_immediate_ko_payoff(product, pricing_env), "delta": 0.0, "gamma": 0.0}
            
        result = self._solve(product, pricing_env)
        price = self._interpolate_price(result.solution_vec, result.x_vec, result.spot_log)
        delta, gamma = self._calculate_delta_gamma(result.solution_vec, result.x_vec, result.spot_log, spot)
        return {"price": price, "delta": delta, "gamma": gamma}

    def _set_terminal_condition_v0(self, grid_slice, x_vec, s_vec, product, pricing_env):
        payoffs = np.array(
            [product.get_maturity_payoff_v0(s, pricing_env=pricing_env) for s in s_vec]
        )
        grid_slice[:] = payoffs

    def _set_terminal_condition_v1(self, grid_slice, x_vec, s_vec, product, pricing_env):
        payoffs = np.array([product.get_maturity_payoff_v1(s, pricing_env) for s in s_vec])
        grid_slice[:] = payoffs

    def _set_boundary_conditions_v0(self, grid_slice, x_vec, s_vec, t_idx, tau, product, pricing_env):
        current_time = self._total_tau - tau
        df_to_maturity = self._df_between_times(pricing_env, current_time, self._total_tau)
        principal = product.initial_price * product.contract_multiplier if product.payoff_config.include_principal else 0.0
        rebate = product.payoff_config.rebate_rate * product.initial_price * product.contract_multiplier
        grid_slice[0] = (principal + rebate) * df_to_maturity
        max_ko = self._get_max_ko_barrier(product)
        if s_vec[-1] >= max_ko:
            grid_slice[-1] = self._get_ko_payoff_at_time(product, pricing_env, current_time, t_idx)
        else:
            grid_slice[-1] = (principal + rebate) * df_to_maturity

    def _set_boundary_conditions_v1(self, grid_slice, x_vec, s_vec, t_idx, tau, product, pricing_env):
        current_time = self._total_tau - tau
        df_to_maturity = self._df_between_times(pricing_env, current_time, self._total_tau)
        principal = product.initial_price * product.contract_multiplier if product.payoff_config.include_principal else 0.0
        strike = product.strike
        initial = product.initial_price
        participation = product.payoff_config.participation_rate
        if product.is_reverse:
            grid_slice[0] = principal * df_to_maturity
        else:
            if self.params.boundary_mode == "asymptotic" and product.payoff_config.protection_type == ProtectionType.NONE:
                df, df_div = self._get_asymptotic_discount_factors(pricing_env, tau)
                effective_strike = strike
                effective_participation = participation
                airbag = product.airbag_config
                if airbag.airbag_barrier is not None and s_vec[0] < airbag.airbag_barrier:
                    effective_participation = airbag.airbag_participation_rate
                    if airbag.airbag_strike is not None: effective_strike = airbag.airbag_strike
                slope = effective_participation * product.contract_multiplier
                grid_slice[0] = principal * df + slope * (s_vec[0] * df_div - effective_strike * df)
            else:
                max_loss = participation * (-strike / initial) * (initial * product.contract_multiplier)
                if product.payoff_config.protection_type.name == "FULL": max_loss = 0.0
                elif product.payoff_config.protection_type.name == "PARTIAL":
                    floor = -product.payoff_config.protection_rate * (initial * product.contract_multiplier)
                    max_loss = max(max_loss, floor)
                grid_slice[0] = (principal + max_loss) * df_to_maturity
        if product.is_reverse:
            protection = product.payoff_config.protection_type
            if protection == ProtectionType.NONE:
                df, df_div = self._get_asymptotic_discount_factors(pricing_env, tau)
                participation = product.payoff_config.participation_rate
                effective_strike = strike
                airbag = product.airbag_config
                if airbag.airbag_barrier is not None and s_vec[-1] > airbag.airbag_barrier:
                    participation = airbag.airbag_participation_rate
                    if airbag.airbag_strike is not None: effective_strike = airbag.airbag_strike
                slope = participation * product.contract_multiplier
                grid_slice[-1] = (principal + slope * effective_strike) * df - slope * s_vec[-1] * df_div
            else:
                if protection == ProtectionType.PARTIAL:
                    floor = product.payoff_config.protection_rate * product.initial_price * product.contract_multiplier
                    grid_slice[-1] = (principal - floor) * df_to_maturity
                else:
                    grid_slice[-1] = principal * df_to_maturity
        else:
            grid_slice[-1] = principal * df_to_maturity

    def _check_product_type(self, product: BaseEquityProduct) -> None:
        if not isinstance(product, self._supported_product_type):
            raise PricingError(f"{self._solver_name} only supports {self._supported_product_type.__name__}")

    def _validate_product(self, product: SnowballOption) -> None:
        ki_continuous = product.barrier_config.ki_continuous or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        if ki_continuous and product.has_ki_barrier:
            if isinstance(product.barrier_config.ki_barrier, list):
                raise ValidationError("Continuous KI monitoring requires scalar ki_barrier.")

    def _aligned_time_index(self, t_vec: np.ndarray, obs_time: float, label: str) -> int:
        for idx, t_val in enumerate(t_vec):
            if is_close(float(t_val), float(obs_time), abs_tol=Tolerance.PRECISION):
                return int(idx)
        nearest = int(np.argmin(np.abs(t_vec - obs_time)))
        raise ValidationError(f"{label} time {obs_time} does not align with grid (nearest {t_vec[nearest]}).")

    def _build_grids(self, product, pricing_env, spot, sigma, tau, r, q):
        self._total_tau = tau
        return super()._build_grids(product, pricing_env, spot, sigma, tau, r, q)

    def _grid_cache_key(self, product, pricing_env, spot, sigma, tau, r, q):
        base_key = super()._grid_cache_key(product, pricing_env, spot, sigma, tau, r, q)
        if not hasattr(product, "barrier_config"): return base_key
        ko_records = self._get_cached_ko_records(pricing_env, product)
        ko_key = tuple(sorted((round(rec.observation_time, 12), round(rec.barrier if rec.barrier else 0.0, 12)) for rec in ko_records))
        return base_key + (ko_key,)

    def _get_cached_ko_records(self, pricing_env, product):
        if not self._is_cache_enabled(): return product.resolve_ko_observations(pricing_env)
        key = self._observation_cache_key(pricing_env, product, "ko")
        if key in self._ko_records_cache: return self._ko_records_cache[key]
        records = product.resolve_ko_observations(pricing_env)
        self._ko_records_cache[key] = records
        return records

    def _get_cached_ki_profile(self, pricing_env, product):
        if not self._is_cache_enabled(): return product.get_ki_observation_profile(pricing_env)
        key = self._observation_cache_key(pricing_env, product, "ki")
        if key in self._ki_profile_cache: return self._ki_profile_cache[key]
        profile = product.get_ki_observation_profile(pricing_env)
        self._ki_profile_cache[key] = profile
        return profile

    def _observation_cache_key(self, pricing_env, product, kind):
        strategy = self._resolve_cache_strategy()
        return (kind, strategy, id(product), pricing_env.valuation_date)

    def _is_knocked_in_at_valuation(self, product, spot, pricing_env, ki_continuous):
        if not product.has_ki_barrier: return False
        if ki_continuous: return self._is_already_knocked_in(product, spot)
        ki_records = product.resolve_ki_observations(pricing_env)
        rec0 = self._find_record_at_time(ki_records, 0.0)
        if rec0: return (spot >= rec0.barrier) if product.is_reverse else (spot <= rec0.barrier)
        return False

    def _is_already_knocked_in(self, product, spot):
        if not product.has_ki_barrier: return False
        b = product.barrier_config.ki_barrier
        if isinstance(b, list): b = b[0]
        return (spot >= b) if product.is_reverse else (spot <= b)

    def _is_knocked_out_at_valuation(self, product, spot, pricing_env):
        if product.barrier_config.ko_observation_type != ObservationType.DISCRETE: return False
        ko_records = product.resolve_ko_observations(pricing_env)
        rec0 = self._find_record_at_time(ko_records, 0.0)
        if rec0: return (spot <= rec0.barrier) if product.is_reverse else (spot >= rec0.barrier)
        return False

    def _get_immediate_ko_payoff(self, product, pricing_env):
        ko_records = product.resolve_ko_observations(pricing_env)
        rec0 = self._find_record_at_time(ko_records, 0.0)
        if not rec0: raise ValidationError("No KO observation at valuation.")
        payoff = rec0.payoff if rec0.payoff is not None else 0.0
        if rec0.settlement_time is not None and rec0.settlement_time > 0:
            df = pricing_env.get_discount_factor(rec0.settlement_time)
            return float(payoff) * float(df)
        return float(payoff)

    def _calculate_terminal_value(self, product, spot, pricing_env):
        ki = self._is_already_knocked_in(product, spot)
        return product.get_payoff(spot, pricing_env, knocked_in=ki)

    @staticmethod
    def _find_record_at_time(records, time):
        for r in records: 
            if is_close(r.observation_time, time): return r
        return None

    @staticmethod
    def _df_between_times(pricing_env, t1, t2):
        if t2 <= t1: return 1.0
        return float(safe_divide(pricing_env.get_discount_factor(t2), pricing_env.get_discount_factor(t1), 1.0))

    def _get_max_ko_barrier(self, product):
        b = product.barrier_config.ko_barrier
        if isinstance(b, list): return max(b)
        return b

    def _get_ko_payoff_at_time(self, product, pricing_env, time, t_idx):
        ko_records = product.resolve_ko_observations(pricing_env)
        for r in ko_records:
            if is_close(r.observation_time, time):
                return self._cashflow_value_at_time(pricing_env, r.payoff, time, r.settlement_time)
        return 0.0

    def _cashflow_value_at_time(self, pricing_env, cf, t, t_settle):
        return float(cf) * self._df_between_times(pricing_env, t, t_settle)

    def set_terminal_condition(self, *args): pass
    def set_boundary_conditions(self, *args): pass
    def calculate_event_stats(self, product, pricing_env): return None
