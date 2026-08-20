"""
PDE solver for KO-reset Snowball options using the Two-Surface method.

Applies the pre-KI KO schedule to the V0 surface and the post-KI KO schedule
to the V1 surface (ABSOLUTE post-KO mode only).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.pde.base_pde_solver import PDESolutionResult
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType, PostKOScheduleMode
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import Tolerance, is_close, is_zero


class KOResetSnowballPDESolver(SnowballPDESolver):
    """
    PDE solver for KO-reset Snowball options using the two-surface method.

    - V0 surface uses pre-KI KO schedule
    - V1 surface uses post-KI KO schedule
    """

    # Override class attributes for product type checking
    _supported_product_type: type = KnockOutResetSnowballOption
    _solver_name: str = "KOResetSnowballPDESolver"

    def __init__(self, params=None):
        super().__init__(params=params)
        self._pre_ko_records_cache: "OrderedDict[Tuple, List[ResolvedObservationRecord]]" = (
            OrderedDict()
        )
        self._post_ko_records_cache: "OrderedDict[Tuple, List[ResolvedObservationRecord]]" = (
            OrderedDict()
        )
        self._pre_ko_observation_indices: Dict[int, ResolvedObservationRecord] = {}
        self._post_ko_observation_indices: Dict[int, ResolvedObservationRecord] = {}
        self._pre_ko_terminal_record: Optional[ResolvedObservationRecord] = None
        #: Time index at which the not-yet-KI surface is seeded, and the
        #: payoff to seed it with. Both stay None when the pre-KI schedule
        #: runs to the full maturity, which is the un-staged case.
        self._v0_seed_idx: Optional[int] = None
        self._v0_seed_values = None
        self._post_ko_terminal_record: Optional[ResolvedObservationRecord] = None
        self._has_pre_terminal_ko: bool = False
        self._has_post_terminal_ko: bool = False

    # price() is inherited from SnowballPDESolver using _check_product_type()

    def calculate_event_stats(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        *,
        npv: Optional[float] = None,
        streams: Optional[frozenset] = None,
    ) -> Optional[object]:
        # streams is ignored: the KO-reset stats are QUAD-delegated (full
        # distribution), not a prunable PDE indicator sweep. npv, when supplied
        # by the single-pass price_with_events, is the PDE value used for the
        # reconciliation residual (avoids a redundant self.price() solve).
        if not isinstance(product, KnockOutResetSnowballOption):
            return None
        if pricing_env is None:
            return None

        from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
            KOResetSnowballQuadEngine,
        )

        grid_points = max(501, int(getattr(self.params, "grid_size", 0) or 0))
        quad_stats = KOResetSnowballQuadEngine(
            QuadParams(grid_points=grid_points)
        ).calculate_event_stats(product, pricing_env)
        if quad_stats is None:
            return None

        pde_pv = float(npv) if npv is not None else float(self.price(product, pricing_env))
        pv_delta = pde_pv - float(quad_stats.pv)
        return replace(
            quad_stats,
            pv=pde_pv,
            expected_discounted_maturity_cashflow=(
                float(quad_stats.expected_discounted_maturity_cashflow) + pv_delta
            ),
        )

    def _validate_product(self, product: KnockOutResetSnowballOption) -> None:
        super()._validate_product(product)
        if product.post_ko_mode != PostKOScheduleMode.ABSOLUTE:
            raise ValidationError(
                "KOResetSnowballPDESolver only supports PostKOScheduleMode.ABSOLUTE"
            )
        if product.post_barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise ValidationError("Post-KO monitoring must be discrete for PDE solver.")
        if product.post_barrier_config.ko_observation_schedule is None:
            raise ValidationError("Post-KO observation schedule is required.")

    def _resolve_ko_records(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        config,
    ) -> List[ResolvedObservationRecord]:
        resolved_schedule, rates, schedule_records = product._resolve_ko_schedule(
            config, pricing_env
        )
        principal_component = (
            product.initial_price * product.contract_multiplier
            if product.payoff_config.include_principal
            else 0.0
        )
        maturity_time: Optional[float] = None
        ko_records: List[ResolvedObservationRecord] = []
        for idx, rec in enumerate(resolved_schedule):
            rate = rates[idx]
            schedule_record = schedule_records[idx]
            accrual_factor = product.compute_ko_accrual_factor(
                rec.observation_time, schedule_record, pricing_env
            )
            coupon_payoff = (
                product.initial_price
                * product.contract_multiplier
                * float(rate)
                * float(accrual_factor)
            )
            payoff = principal_component + coupon_payoff

            settlement_time = rec.settlement_time
            if product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY:
                maturity_time = (
                    maturity_time
                    if maturity_time is not None
                    else product.get_maturity(pricing_env)
                )
                settlement_time = maturity_time

            ko_records.append(
                ResolvedObservationRecord(
                    observation_time=rec.observation_time,
                    barrier=rec.barrier,
                    payoff=payoff,
                    settlement_time=settlement_time,
                )
            )
        return ko_records

    def _get_cached_pre_ko_records(
        self, pricing_env: PricingEnvironment, product: KnockOutResetSnowballOption
    ) -> List[ResolvedObservationRecord]:
        if not self._is_cache_enabled():
            return self._resolve_ko_records(
                product, pricing_env, product.barrier_config
            )
        key = self._observation_cache_key(pricing_env, product, "pre_ko")
        cached = self._pre_ko_records_cache.get(key)
        if cached is not None:
            self._pre_ko_records_cache.move_to_end(key)
            return cached
        records = self._resolve_ko_records(
            product, pricing_env, product.barrier_config
        )
        self._pre_ko_records_cache[key] = records
        self._pre_ko_records_cache.move_to_end(key)
        if len(self._pre_ko_records_cache) > self.params.grid_cache_max_entries:
            self._pre_ko_records_cache.popitem(last=False)
        return records

    def _get_cached_post_ko_records(
        self, pricing_env: PricingEnvironment, product: KnockOutResetSnowballOption
    ) -> List[ResolvedObservationRecord]:
        if not self._is_cache_enabled():
            return self._resolve_ko_records(
                product, pricing_env, product.post_barrier_config
            )
        key = self._observation_cache_key(pricing_env, product, "post_ko")
        cached = self._post_ko_records_cache.get(key)
        if cached is not None:
            self._post_ko_records_cache.move_to_end(key)
            return cached
        records = self._resolve_ko_records(
            product, pricing_env, product.post_barrier_config
        )
        self._post_ko_records_cache[key] = records
        self._post_ko_records_cache.move_to_end(key)
        if len(self._post_ko_records_cache) > self.params.grid_cache_max_entries:
            self._post_ko_records_cache.popitem(last=False)
        return records

    def _get_cached_ko_records(
        self, pricing_env: PricingEnvironment, product: KnockOutResetSnowballOption
    ) -> List[ResolvedObservationRecord]:
        return self._get_cached_pre_ko_records(pricing_env, product)

    def _uses_grid_layer(self) -> bool:
        return True

    def _populate_observation_maps(self, product, pricing_env, layout, tau):
        super()._populate_observation_maps(product, pricing_env, layout, tau)
        self._register_post_ko_observations(
            product, pricing_env, tau, step_of=layout.time.step_at
        )

    def _register_post_ko_observations(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        tau: float,
        step_of=None,
        t_vec=None,
    ) -> None:
        """Post-KI (reset) KO index map — exact lookup on the migrated path.

        Also snapshots the pre-KO maps the parent just built (the V0 surface
        fires the pre schedule; V1 fires the post/reset schedule [§11.7]).
        """
        self._pre_ko_observation_indices = dict(self._ko_observation_indices)
        self._pre_ko_terminal_record = self._ko_terminal_record
        self._has_pre_terminal_ko = self._has_terminal_ko

        self._post_ko_observation_indices.clear()
        self._post_ko_terminal_record = None
        self._has_post_terminal_ko = False

        post_records = self._filter_observations_by_tau(
            self._get_cached_post_ko_records(pricing_env, product), tau
        )
        for rec in post_records:
            obs_time = rec.observation_time
            if is_close(obs_time, 0.0):
                self._post_ko_observation_indices[0] = rec
            elif is_close(obs_time, tau):
                self._post_ko_terminal_record = rec
                self._has_post_terminal_ko = True
            elif 0.0 < obs_time < tau:
                if step_of is not None:
                    idx = step_of(obs_time)
                else:
                    idx = self._aligned_time_index(
                        t_vec, obs_time, "Post-KO observation"
                    )
                self._post_ko_observation_indices[idx] = rec

    def _build_grids(
        self,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        result = super()._build_grids(product, pricing_env, spot, sigma, tau, r, q)
        if self._active_layout is None:
            _, _, _, t_vec, _ = result
            self._register_post_ko_observations(
                product, pricing_env, tau, t_vec=t_vec
            )
        return result

    def _post_ki_ko_times(
        self, product: KnockOutResetSnowballOption, tau: float
    ) -> List[float]:
        """Post-KI (reset) KO observation times, interior to (0, tau).

        In the two-surface KO-reset PDE the post-KI KO schedule fires on the V1
        (knocked-in) surface; those dates must be grid nodes exactly [§11.7].
        """
        out = []
        post_schedule = product.post_barrier_config.ko_observation_schedule
        if post_schedule is not None:
            out += [
                rec.observation_time
                for rec in post_schedule.records
                if rec.observation_time is not None
            ]
        elif product.post_barrier_config.ko_observation_dates is not None:
            out += list(product.post_barrier_config.ko_observation_dates)
        return sorted({float(t) for t in out if t is not None and 0.0 < float(t) < tau})

    def _ko_coupon_align_times(
        self, product: KnockOutResetSnowballOption, tau: float
    ) -> List[float]:
        """Pre-KI KO ∪ post-KI (reset) KO — both must align exactly [§11.7]."""
        pre = super()._ko_coupon_align_times(product, tau)
        post = self._post_ki_ko_times(product, tau)
        return sorted(set(pre) | set(post))

    def get_critical_points(
        self, product: KnockOutResetSnowballOption, pricing_env: PricingEnvironment
    ) -> List[float]:
        points = super().get_critical_points(product, pricing_env)
        post_barrier = product.post_barrier_config.ko_barrier
        if isinstance(post_barrier, list):
            points.extend([b for b in post_barrier if b > 0])
        elif post_barrier > 0:
            points.append(post_barrier)
        return sorted(set([p for p in points if p > 0]))

    def _get_barriers(self, product: KnockOutResetSnowballOption) -> List[float]:
        barriers = super()._get_barriers(product)
        post_barrier = product.post_barrier_config.ko_barrier
        if isinstance(post_barrier, list):
            barriers.extend(post_barrier)
        elif post_barrier > 0:
            barriers.append(post_barrier)
        return barriers

    def _solve(
        self, product: KnockOutResetSnowballOption, pricing_env: PricingEnvironment
    ) -> PDESolutionResult:
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        # State preamble shared with session preparation (see
        # SnowballPDESolver._prepare_solve_state).
        knocked_in_at_valuation = self._prepare_solve_state(product, pricing_env)

        strike = product.strike
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        if self._profile_enabled:
            self._reset_profile_stats()

        if self._profile_enabled:
            t0 = perf_counter()
        self._active_layout = None
        self._active_schedule = None  # certified pre/post dispatch stays inline
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )
        if self._profile_enabled:
            self._profile_stats["grid_build"] += perf_counter() - t0

        num_x, num_t = len(x_vec), len(t_vec)
        self._grid_v0 = np.zeros((num_x, num_t))
        self._grid_v1 = np.zeros((num_x, num_t))

        self._set_terminal_condition_v1(
            self._grid_v1, x_vec, s_vec, product, pricing_env
        )

        # A not-yet-knocked-in contract matures on the PRE-KI schedule; only a
        # knocked-in one runs on to `tau`. Seeding V0 at the end of the grid
        # would march it back through a period it can never reach, so when the
        # two horizons differ V0 stays zero above the pre-KI maturity -- its
        # true value there -- and is seeded at that index during stepping.
        pre_maturity = float(product.get_pre_maturity_time(pricing_env))
        self._v0_seed_idx = None
        self._v0_seed_values = None
        staged_v0 = (
            self._pre_ko_observation_indices
            and pre_maturity < tau
            and not is_close(pre_maturity, tau, abs_tol=Tolerance.PRECISION)
        )
        if staged_v0:
            seed = np.zeros((num_x, 1))
            self._set_terminal_condition_v0(seed, x_vec, s_vec, product, pricing_env)
            self._v0_seed_values = seed[:, -1].copy()
            self._v0_seed_idx = max(self._pre_ko_observation_indices)
        else:
            self._set_terminal_condition_v0(
                self._grid_v0, x_vec, s_vec, product, pricing_env
            )

        if self._has_pre_terminal_ko and self._pre_ko_terminal_record is not None:
            self._apply_terminal_ko_single(
                self._grid_v0,
                s_vec,
                product,
                pricing_env,
                self._pre_ko_terminal_record,
            )

        if (
            self._has_post_terminal_ko
            and self._post_ko_terminal_record is not None
            and not product.barrier_config.disable_ko_after_ki
        ):
            self._apply_terminal_ko_single(
                self._grid_v1,
                s_vec,
                product,
                pricing_env,
                self._post_ko_terminal_record,
            )

        if product.has_ki_barrier and not staged_v0:
            is_terminal_ki = self._ki_continuous or self._bgk_active
            if not is_terminal_ki:
                if (num_t - 1) in self._ki_observation_indices:
                    is_terminal_ki = True
            if is_terminal_ki:
                self._apply_ki_jump(
                    self._grid_v0, self._grid_v1, s_vec, num_t - 1, product
                )

        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        A = self._build_operator_matrix(l, c, u, num_x)

        # Term-structure step coefficients (one set for flat inputs)
        sc = self._build_step_coefficients(pricing_env, product.strike, t_vec, dx_vec, num_x)
        sc = self._flat_exact_step_coefficients(sc, r, q, sigma, dx_vec, num_x)
        step_coeffs = None if sc.n_unique == 1 else sc

        self._time_stepping_two_surface(
            self._grid_v0,
            self._grid_v1,
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

        spot_log = np.log(spot)
        if knocked_in_at_valuation:
            solution_vec = self._grid_v1[:, 0]
        else:
            solution_vec = self._grid_v0[:, 0]

        return PDESolutionResult(
            solution_vec=solution_vec,
            x_vec=x_vec,
            s_vec=s_vec,
            spot_log=spot_log,
        )

    def _apply_step_modifications_two_surface(
        self,
        grid_v0: np.ndarray,
        grid_v1: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
    ) -> None:
        current_time = self._total_tau - tau

        if self._v0_seed_idx is not None and t_idx == self._v0_seed_idx:
            grid_v0[:, t_idx] = self._v0_seed_values

        pre_record = self._pre_ko_observation_indices.get(t_idx)
        if pre_record is not None:
            self._apply_ko_jump_single(
                grid_v0,
                s_vec,
                t_idx,
                current_time,
                product,
                pricing_env,
                pre_record,
            )

        post_record = self._post_ko_observation_indices.get(t_idx)
        if (
            post_record is not None
            and not product.barrier_config.disable_ko_after_ki
        ):
            self._apply_ko_jump_single(
                grid_v1,
                s_vec,
                t_idx,
                current_time,
                product,
                pricing_env,
                post_record,
            )

        v0_live = self._v0_seed_idx is None or t_idx <= self._v0_seed_idx
        if product.has_ki_barrier and v0_live:
            should_apply_ki = (
                self._ki_continuous or t_idx in self._ki_observation_indices
            )
            if should_apply_ki:
                self._apply_ki_jump(grid_v0, grid_v1, s_vec, t_idx, product)

    def _apply_ko_jump_single(
        self,
        grid: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        current_time: float,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        ko_record: ResolvedObservationRecord,
    ) -> None:
        barrier = ko_record.barrier
        payoff = ko_record.payoff if ko_record.payoff is not None else 0.0
        cashflow_value = self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=payoff,
            current_time=current_time,
            settlement_time=ko_record.settlement_time,
        )

        if self._event_uses_projection(t_idx):
            grid[:, t_idx] = self._project_event_values(
                s_vec, barrier, product.is_reverse, True,
                grid[:, t_idx], cashflow_value,
            )
            return
        mask = self._event_nodal_mask(
            s_vec, barrier, product.is_reverse, True, at_valuation=(t_idx == 0)
        )
        grid[mask, t_idx] = cashflow_value

    def _apply_terminal_ko_single(
        self,
        grid: np.ndarray,
        s_vec: np.ndarray,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        ko_record: ResolvedObservationRecord,
    ) -> None:
        barrier = ko_record.barrier
        payoff = ko_record.payoff if ko_record.payoff is not None else 0.0
        cashflow_value = self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=payoff,
            current_time=self._total_tau,
            settlement_time=ko_record.settlement_time,
        )

        if self._use_cell_average_events():
            grid[:, -1] = self._project_event_values(
                s_vec, barrier, product.is_reverse, True,
                grid[:, -1], cashflow_value,
            )
            return
        mask = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)
        grid[mask, -1] = cashflow_value

    def __repr__(self) -> str:
        return "KOResetSnowballPDESolver()"
