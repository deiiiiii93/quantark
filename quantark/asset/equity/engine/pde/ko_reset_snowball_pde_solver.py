"""
PDE solver for KO-reset Snowball options using the Two-Surface method.

Applies the pre-KI KO schedule to the V0 surface and the post-KI KO schedule
to the V1 surface (ABSOLUTE post-KO mode only).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.pde.base_pde_solver import GRID_CACHE_MAX_ENTRIES
from quantark.asset.equity.engine.pde.core.adapters.ko_reset import KOResetAdapter
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType, PostKOScheduleMode
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close


class KOResetSnowballPDESolver(SnowballPDESolver):
    """
    PDE solver for KO-reset Snowball options using the two-surface method.

    - V0 surface uses pre-KI KO schedule
    - V1 surface uses post-KI KO schedule
    """

    # Override class attributes for product type checking
    _supported_product_type: type = KnockOutResetSnowballOption
    _solver_name: str = "KOResetSnowballPDESolver"
    _adapter_class = KOResetAdapter

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

    def _constructor_kwargs(self) -> dict:
        return {"params": self.params}   # no enable_profiling argument

    def _event_stats_product_type(self) -> type:
        return KnockOutResetSnowballOption

    def _make_event_stats(self, **fields):
        from quantark.asset.equity.engine.event_stats import KOResetEventStats

        fields.setdefault("provenance", self._native_provenance())
        return KOResetEventStats(**fields)

    # price() / calculate_greeks() / calculate_event_stats() are SnowballPDESolver's; KOResetAdapter
    # marches them (the statistics as the value's PDE group, spec 3B §5.4, decision 6).

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
                settlement_time = self._terminal_payment_time(
                    product,
                    pricing_env,
                )

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
        if len(self._pre_ko_records_cache) > GRID_CACHE_MAX_ENTRIES:
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
        if len(self._post_ko_records_cache) > GRID_CACHE_MAX_ENTRIES:
            self._post_ko_records_cache.popitem(last=False)
        return records

    def _get_cached_ko_records(
        self, pricing_env: PricingEnvironment, product: KnockOutResetSnowballOption
    ) -> List[ResolvedObservationRecord]:
        return self._get_cached_pre_ko_records(pricing_env, product)

    def _decide_variant_at_valuation(self, product, spot, pricing_env, carried):
        """Zero time to maturity: the pending observations of the schedule in force are decided on the spot.

        ``_apply_step_modifications_two_surface`` applies this instant's pre-KI knock-out to the
        not-knocked-in surface and its post-KI knock-out to the knocked-in one before the knock-in moves
        value across, so a fresh knock-in sees the post-KI observation, as the decision applies it. The
        inherited shortcut went straight to the carried state's terminal payoff.
        """
        product.require_alive(pricing_env, carried)
        decision = product.decide_observations_at_valuation(
            spot, pricing_env, knocked_in=carried
        )
        if decision.knocked_out:
            return self._knocked_out_value(product, pricing_env, decision), decision.knocked_in
        return None, decision.knocked_in

    def _valuation_decision(self, product, pricing_env):
        """The decision of the schedule in force (spec 3B decision 15), on a contract still alive: one
        past its first schedule and not knocked in has matured (``require_alive`` raises)."""
        carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        product.require_alive(pricing_env, carried)
        return product.decide_observations_at_valuation(pricing_env.spot, pricing_env, knocked_in=carried)

    def _ko_settlement_time(self, product, pricing_env, record):
        """A knock-out's payment time: its settlement, or the terminal payment under EXPIRY accrual."""
        if product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY:
            return self._terminal_payment_time(product, pricing_env)
        return record.settlement_time

    def _knocked_out_value(self, product, pricing_env, decision) -> float:
        """The knock-out the decision fired, pre or post schedule: its contractual cash at its payment."""
        record = decision.ko_record
        settlement_time = self._ko_settlement_time(product, pricing_env, record)
        if settlement_time is not None and settlement_time > 0.0:
            return float(record.payoff) * float(pricing_env.get_discount_factor(settlement_time))
        return float(record.payoff)

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

    def _post_ki_ko_times(
        self, product: KnockOutResetSnowballOption, tau: float
    ) -> List[float]:
        """Post-KI (reset) KO observation times, interior to (0, tau).

        In the two-surface KO-reset PDE the post-KI KO schedule fires on the V1
        (knocked-in) surface; those dates must be grid nodes exactly [§11.7].
        """
        out = []
        post_schedule = product.post_barrier_config.ko_observation_schedule
        if post_schedule is not None and post_schedule.uses_dates():
            # a schedule stated in dates resolves through the prepared environment, as the pre schedule's
            # does (_ko_coupon_align_times): its raw records carry no year fraction (spec 3B F11)
            out += [
                rec.observation_time
                for rec in self._get_cached_post_ko_records(
                    self._grid_request_environment("post knock-out"), product)
            ]
        elif post_schedule is not None:
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
        self._two_surface_events(grid_v0[:, t_idx], grid_v1[:, t_idx], s_vec, t_idx, grid_v0.shape[1],
                                 self._total_tau - tau, product, pricing_env)

    def _two_surface_events(self, v0, v1, s_vec, t_idx, num_t, current_time, product, pricing_env,
                            breach=None) -> None:
        """The column form of ``_apply_step_modifications_two_surface`` (spec 3B §5.1): the staged seed,
        the pre schedule on the not-knocked-in surface, the post schedule on the knocked-in one, the
        knock-in, in today's order; ``breach`` as ``_ko_columns``, and its ``seed`` fills the staged
        pre-maturity index."""
        if self._v0_seed_idx is not None and t_idx == self._v0_seed_idx:
            v0[...] = self._v0_seed_values if breach is None else breach.seed(v0)

        pre_record = self._pre_ko_observation_indices.get(t_idx)
        # the first-schedule knock-out wins a tie with a discrete knock-in (see _pre_ko_goes_last)
        pre_ko_last = pre_record is not None and self._pre_ko_goes_last()
        if pre_record is not None and not pre_ko_last:
            self._ko_columns([v0], s_vec, t_idx, pre_record, product, pricing_env, current_time,
                             breach=breach, surfaces=("alive",))

        post_record = self._post_ko_observation_indices.get(t_idx)
        if (
            post_record is not None
            and not product.barrier_config.disable_ko_after_ki
        ):
            self._ko_columns([v1], s_vec, t_idx, post_record, product, pricing_env, current_time,
                             breach=breach, surfaces=("ki",))

        v0_live = self._v0_seed_idx is None or t_idx <= self._v0_seed_idx
        if product.has_ki_barrier and v0_live:
            should_apply_ki = (
                self._ki_continuous or t_idx in self._ki_observation_indices
            )
            if should_apply_ki:
                self._ki_columns(v0, v1, s_vec, t_idx, num_t, product)

        if pre_ko_last:
            self._ko_columns([v0], s_vec, t_idx, pre_record, product, pricing_env, current_time,
                             breach=breach, surfaces=("alive",))

    def _pre_ko_goes_last(self) -> bool:
        """Whether a first-schedule knock-out is applied after the knock-in sharing its observation.

        For a DISCRETE knock-in, always: the knocked-in surface does not carry that knock-out, so
        copying it into the not-knocked-in surface would let the knock-in win the tie. A continuously
        monitored knock-in breached at a node was touched before the observation, so it did come
        first and the second schedule is already in force there.
        """
        return not self._ki_continuous

    def _apply_ko_jump_single(
        self,
        grid: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        current_time: float,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        ko_record: ResolvedObservationRecord,
        surface: str = "alive",
    ) -> None:
        """One surface's knock-out: ``"alive"`` for the pre schedule, ``"ki"`` for the post schedule."""
        self._ko_columns([grid[:, t_idx]], s_vec, t_idx, ko_record, product, pricing_env, current_time,
                         surfaces=(surface,))

    def _apply_terminal_ko_single(
        self,
        grid: np.ndarray,
        s_vec: np.ndarray,
        product: KnockOutResetSnowballOption,
        pricing_env: PricingEnvironment,
        ko_record: ResolvedObservationRecord,
        surface: str = "alive",
    ) -> None:
        """One surface's terminal knock-out: ``"alive"`` for the pre schedule, ``"ki"`` for the post."""
        self._terminal_ko_columns([grid[:, -1]], s_vec, ko_record, product, pricing_env,
                                  surfaces=(surface,))

    def __repr__(self) -> str:
        return "KOResetSnowballPDESolver()"
