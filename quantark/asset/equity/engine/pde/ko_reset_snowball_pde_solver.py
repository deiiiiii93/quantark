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
from quantark.asset.equity.engine.settlement_support import (
    first_schedule_payment_time,
    first_schedule_settlement,
    knocked_in_settlement,
)
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType, PostKOScheduleMode
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close, same_instant


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
    #: the not-knocked-in surface fires the pre-KI schedule, the knocked-in one the post-KI schedule
    _fold_alive_schedule = "pre"
    _fold_ki_schedule = "post"

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
        #: The knocked-in surface likewise (spec D31, run 5 T7): a knocked-in note ends at schedule 2's last fixing,
        #: seeded at that index when it falls before the grid's last node (``knocked_in_settlement``); ``_v1_end`` is
        #: ``(schedule 2's end, the seed's delay factor)`` for the knocked-in boundary rows. None when unstaged.
        self._v1_seed_idx: Optional[int] = None
        self._v1_seed_values = None
        self._v1_end: Optional[Tuple[float, float]] = None
        self._post_ko_terminal_record: Optional[ResolvedObservationRecord] = None
        self._has_pre_terminal_ko: bool = False
        self._has_post_terminal_ko: bool = False
        # same-instant knock-out groups of >= 2 effective records per schedule (0.16.0 D14): step -> the group in
        # contract order (the step's slot holds its first record); the maturity instant's group
        self._pre_ko_groups: Dict[int, Tuple[ResolvedObservationRecord, ...]] = {}
        self._post_ko_groups: Dict[int, Tuple[ResolvedObservationRecord, ...]] = {}
        self._pre_ko_terminal_group: Optional[Tuple[ResolvedObservationRecord, ...]] = None
        self._post_ko_terminal_group: Optional[Tuple[ResolvedObservationRecord, ...]] = None

    def _constructor_kwargs(self) -> dict:
        return {"params": self.params}   # no enable_profiling argument

    def _event_stats_product_type(self) -> type:
        return KnockOutResetSnowballOption

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
        first_schedule = config is product.barrier_config
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
                # EXPIRY pay: with the note's end -- schedule 1's maturity for a first-schedule knock-out (a path
                # never knocked in ends there, spec D30), the knocked-in note's end (schedule 2's last fixing, D31)
                # for a second-schedule one
                settlement_time = (
                    self._first_schedule_payment_time(product, pricing_env)
                    if first_schedule
                    else self._knocked_in_payment_time(
                        product, pricing_env, [float(r.observation_time) for r in resolved_schedule])
                )

            ko_records.append(
                ResolvedObservationRecord(
                    observation_time=rec.observation_time,
                    barrier=rec.barrier,
                    payoff=payoff,
                    settlement_time=settlement_time,
                    # contract order and the statistics' per-record keys of a same-instant pair (0.16.0 D14)
                    observation_date=rec.observation_date,
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

    def _first_schedule_payment_time(self, product, pricing_env) -> float:
        """When the never-knocked-in note is paid, and under EXPIRY pay a first-schedule knock-out (spec D30): the
        explicit settlement date when booked (spec D33), else schedule 1's maturity plus the terminal delay
        (``first_schedule_payment_time``)."""
        return first_schedule_payment_time(product, pricing_env, self._terminal_timing(product, pricing_env))

    def _first_schedule_settlement(self, product, pricing_env):
        """``first_schedule_settlement``: the payment (and its delay factor) of a note that ends with schedule 1."""
        return first_schedule_settlement(product, pricing_env, self._terminal_timing(product, pricing_env))

    def _knocked_in_settlement(self, product, pricing_env, post_schedule_times=None):
        """``knocked_in_settlement``: the payment (and its delay factor) of a knocked-in note that ends at schedule 2's
        last fixing before the grid's last node (spec D31, run 5 T7); None when schedule 2 ends with the product.
        ``post_schedule_times``: schedule 2's active observation times (default: the cached records')."""
        if post_schedule_times is None:
            post_schedule_times = [float(rec.observation_time)
                                   for rec in self._get_cached_post_ko_records(pricing_env, product)]
        return knocked_in_settlement(product, pricing_env, self._terminal_timing(product, pricing_env),
                                     post_schedule_times=post_schedule_times)

    def _knocked_in_payment_time(self, product, pricing_env, post_schedule_times=None) -> float:
        """When the knocked-in note is paid, and under EXPIRY pay a second-schedule knock-out: schedule 2's end's
        settlement (``knocked_in_settlement``), the product terminal's when schedule 2 ends with the product."""
        settlement = self._knocked_in_settlement(product, pricing_env, post_schedule_times)
        if settlement is None:
            return self._terminal_payment_time(product, pricing_env)
        return float(settlement.payment_time)

    def _ko_settlement_time(self, product, pricing_env, record, regime):
        """A knock-out's payment time: its settlement, or under EXPIRY accrual the end of the note it ends --
        schedule 1's maturity for a ``"pre"`` knock-out (spec D30), the knocked-in note's payment for a ``"post"``
        one (schedule 2's end, D31)."""
        if product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY:
            if regime == "pre":
                return self._first_schedule_payment_time(product, pricing_env)
            return self._knocked_in_payment_time(product, pricing_env)
        return record.settlement_time

    def _get_immediate_ko_payoff(self, product, pricing_env) -> float:
        """The 2D (ADI) routes read the knock-out decided at valuation here. Under EXPIRY pay a first-schedule one is
        paid at schedule 1's maturity (``_knocked_out_value``, spec D30); everything else as the snowball reads it."""
        if product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY:
            decision = self._valuation_decision(product, pricing_env)
            if decision.knocked_out and decision.ko_regime == "pre":
                return self._knocked_out_value(product, pricing_env, decision)
        if self._shared_valuation_instant(product, pricing_env):
            # several records of one schedule at the valuation instant (0.16.0 D14): the decision's record pays -- the
            # first in contract order the spot breaches, not the instant's first record
            decision = self._valuation_decision(product, pricing_env)
            if decision.knocked_out:
                return self._knocked_out_value(product, pricing_env, decision)
        return super()._get_immediate_ko_payoff(product, pricing_env)

    def _shared_valuation_instant(self, product, pricing_env) -> bool:
        """Whether a schedule holds two or more knock-out records at the valuation instant."""
        return any(
            sum(1 for rec in records if rec.observation_time is not None and same_instant(rec.observation_time, 0.0))
            >= 2
            for records in (self._get_cached_pre_ko_records(pricing_env, product),
                            self._get_cached_post_ko_records(pricing_env, product)))

    def _knocked_out_value(self, product, pricing_env, decision) -> float:
        """The knock-out the decision fired, pre or post schedule: its contractual cash at its payment."""
        record = decision.ko_record
        settlement_time = self._ko_settlement_time(product, pricing_env, record, decision.ko_regime)
        if settlement_time is not None and settlement_time > 0.0:
            return float(record.payoff) * float(pricing_env.get_discount_factor(settlement_time))
        return float(record.payoff)

    # --- cash-leg folding --------------------------------------------------------------------

    def _fold_ko_records(self, product, pricing_env, tau: float):
        """Both schedules' in-life records: the pre schedule's, then the post schedule's."""
        pre = self._filter_observations_by_tau(self._get_cached_pre_ko_records(pricing_env, product), tau)
        post = self._filter_observations_by_tau(self._get_cached_post_ko_records(pricing_env, product), tau)
        return [("pre", rec) for rec in pre] + [("post", rec) for rec in post]

    def _fold_shared_instants(self, product, tagged):
        """Each schedule's knock-out records sharing an env instant (0.16.0 D10/D14): a group of two or more effective
        records is refused by the binder (each pays its own cash there), a duplicate binds its effective record."""
        out = []
        for schedule in ("pre", "post"):
            records = [rec for tag, rec in tagged if tag == schedule]
            if len(records) >= 2:
                out.extend(SnowballPDESolver._instant_groups(product, records))
        return tuple(out)

    def _fold_no_ki_timing(self, product, pricing_env):
        """The alive branch matures on the first schedule: determined at its maturity, paid on the explicit
        settlement date when booked (spec D33), else the terminal settlement lag later; discounted from there to the
        pre-maturity by its own ratio."""
        settlement = self._first_schedule_settlement(product, pricing_env)
        pre_maturity, payment = settlement.determination_time, settlement.payment_time
        return pre_maturity, payment, self._df_between_times(pricing_env, pre_maturity, payment)

    def _fold_ki_timing(self, product, pricing_env):
        """The knocked-in branch: the product terminal's when schedule 2 ends with the product, else determined at
        schedule 2's last fixing and paid there by ``knocked_in_settlement`` (D31), discounted from its payment to
        schedule 2's end by its own ratio (as the alive branch at schedule 1's end)."""
        settlement = self._knocked_in_settlement(product, pricing_env)
        if settlement is None:
            return super()._fold_ki_timing(product, pricing_env)
        post_end, payment = settlement.determination_time, settlement.payment_time
        return post_end, payment, self._df_between_times(pricing_env, post_end, payment)

    def _folded_knock_out_cash(self, product, pricing_env, decision, fold) -> float:
        """The folded cash of the record the valuation-date decision fired, on its own schedule, at
        its payment (as ``_knocked_out_value`` discounts its payoff)."""
        bound = self._bind_fold(product, pricing_env, fold)
        record = decision.ko_record
        schedule = "pre" if decision.ko_regime == "pre" else "post"
        amount = bound.ko_amount(schedule, record.observation_time)
        settlement_time = self._ko_settlement_time(product, pricing_env, record, decision.ko_regime)
        if settlement_time is not None and settlement_time > 0.0:
            return float(amount) * float(pricing_env.get_discount_factor(settlement_time))
        return float(amount)

    def _uses_grid_layer(self) -> bool:
        return True

    def _populate_observation_maps(self, product, pricing_env, layout, tau):
        super()._populate_observation_maps(product, pricing_env, layout, tau)
        self._register_post_ko_observations(
            product, pricing_env, tau, step_of=layout.time.step_at
        )
        self._register_same_instant_groups(product, pricing_env, layout, tau)

    def _register_same_instant_groups(self, product, pricing_env, layout, tau) -> None:
        """The records of one env instant (0.16.0 D14): each schedule's knock-out records sharing an instant are one
        group in contract order (``same_instant_ko_groups``: a record whose region an earlier one's contains is inert),
        the step's slot holding the group's first record (the discrete knock-in records sharing an instant are the
        base solver's: ``_register_ki_instants``, one knock-in at the widest level). Instants of one record keep
        today's maps."""
        self._pre_ko_groups, self._post_ko_groups = {}, {}
        self._pre_ko_terminal_group = self._post_ko_terminal_group = None
        step_of = layout.time.step_at
        for records, slots, groups, schedule in (
                (self._get_cached_pre_ko_records(pricing_env, product), self._pre_ko_observation_indices,
                 self._pre_ko_groups, "pre"),
                (self._get_cached_post_ko_records(pricing_env, product), self._post_ko_observation_indices,
                 self._post_ko_groups, "post")):
            records = self._filter_observations_by_tau(records, tau)
            if len(records) < 2:
                continue
            for _, effective in SnowballPDESolver._instant_groups(product, records):
                if not effective:
                    continue
                group = tuple(effective) if len(effective) >= 2 else None
                t = float(effective[0].observation_time)
                if same_instant(t, tau):
                    if schedule == "pre":
                        self._pre_ko_terminal_record, self._pre_ko_terminal_group = effective[0], group
                        self._ko_terminal_record = effective[0]
                    else:
                        self._post_ko_terminal_record, self._post_ko_terminal_group = effective[0], group
                    continue
                step = 0 if is_close(t, 0.0) else step_of(t)
                slots[step] = effective[0]
                if schedule == "pre":       # the base map the boundary rows read names the same record
                    self._ko_observation_indices[step] = effective[0]
                if group is not None:
                    groups[step] = group

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
            elif same_instant(obs_time, tau):
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

    def _set_terminal_condition_v0(self, grid, x_vec, s_vec, product, pricing_env) -> None:
        # the pre-KI contract tenor does not depend on spot: resolve its schedule once, not per node
        tenor = product._resolve_pre_contract_tenor(pricing_env)
        payoffs = np.array(
            [product.get_maturity_payoff_v0(s, pricing_env, contract_tenor=tenor) for s in s_vec]
        )
        # seeded at schedule 1's maturity: its payment's delay factor, the terminal's unless an explicit settlement
        # date governs (spec D33)
        settlement = self._first_schedule_settlement(product, pricing_env)
        delay_df = settlement.delay_df if settlement.settlement_date_governs else self._structured_terminal_delay_df
        grid[:, -1] = payoffs * delay_df
        if self._fold is not None:
            grid[:, -1] += self._fold.no_ki * self._fold.no_ki_df

    def _set_terminal_condition_v1(self, grid, x_vec, s_vec, product, pricing_env) -> None:
        """The knocked-in redemption; seeded at schedule 2's last fixing when the note ends there before the grid's
        last node (D31), with that payment's delay factor (``knocked_in_settlement``: an explicit settlement date's
        own ratio when it governs, else the terminal's)."""
        settlement = self._knocked_in_settlement(product, pricing_env)
        if settlement is None or not settlement.settlement_date_governs:
            super()._set_terminal_condition_v1(grid, x_vec, s_vec, product, pricing_env)
            return
        payoffs = np.array([product.get_maturity_payoff_v1(s, pricing_env) for s in s_vec])
        grid[:, -1] = payoffs * settlement.delay_df
        if self._fold is not None:
            grid[:, -1] += self._fold.ki * self._fold.ki_df

    def _boundary_rows_v1(self, t_idx, tau, product, pricing_env, s_vec):
        """The knocked-in rows of a note that ends at schedule 2's last fixing (``_v1_end``, D31): the snowball's
        rows for a contract maturing there (discounted to that end, with its seed's delay factor). Past it the
        knocked-in claim is dead and the rows are never read (the seed overwrites the column at that index)."""
        end = self._v1_end
        if end is None:
            return super()._boundary_rows_v1(t_idx, tau, product, pricing_env, s_vec)
        post_end, delay_df = end
        tau_v1 = tau - (self._total_tau - post_end)
        if tau_v1 < 0.0 and not same_instant(tau_v1, 0.0):
            return super()._boundary_rows_v1(t_idx, tau, product, pricing_env, s_vec)
        saved = self._total_tau, self._structured_terminal_delay_df
        self._total_tau, self._structured_terminal_delay_df = post_end, delay_df
        try:
            return super()._boundary_rows_v1(t_idx, max(tau_v1, 0.0), product, pricing_env, s_vec)
        finally:
            self._total_tau, self._structured_terminal_delay_df = saved

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
        if self._v1_seed_idx is not None and t_idx == self._v1_seed_idx:
            # the knocked-in note's end (D31): seeded before schedule 2's last knock-out applies on it
            v1[...] = self._v1_seed_values if breach is None else breach.seed_ki(v1)

        pre_record = self._pre_ko_observation_indices.get(t_idx)
        # the first-schedule knock-out wins a tie with a discrete knock-in (see _pre_ko_goes_last)
        pre_ko_last = pre_record is not None and self._pre_ko_goes_last()
        pre_group = self._pre_ko_groups.get(t_idx) if self._pre_ko_groups else None

        def pre_knock_out():
            if pre_group is not None:   # a same-instant group: one stage (0.16.0 D14)
                self._ko_group_columns([v0], s_vec, t_idx, pre_group, product, pricing_env, current_time,
                                       breach=breach, surfaces=("alive",))
            else:
                self._ko_columns([v0], s_vec, t_idx, pre_record, product, pricing_env, current_time,
                                 breach=breach, surfaces=("alive",))

        if pre_record is not None and not pre_ko_last:
            pre_knock_out()

        post_record = self._post_ko_observation_indices.get(t_idx)
        if (
            post_record is not None
            and not product.barrier_config.disable_ko_after_ki
        ):
            post_group = self._post_ko_groups.get(t_idx) if self._post_ko_groups else None
            if post_group is not None:
                self._ko_group_columns([v1], s_vec, t_idx, post_group, product, pricing_env, current_time,
                                       breach=breach, surfaces=("ki",))
            else:
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
            pre_knock_out()

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
        self._terminal_ko_single_columns([grid[:, -1]], s_vec, product, pricing_env, ko_record, surface)

    def _terminal_ko_single_columns(self, cols, s_vec, product, pricing_env, ko_record, surface, breach=None):
        """``_terminal_ko_columns`` of one schedule's maturity knock-out on ``cols``: its same-instant group when the
        maturity instant holds one (0.16.0 D14), else its record."""
        group = self._pre_ko_terminal_group if surface == "alive" else self._post_ko_terminal_group
        if group is not None:
            self._terminal_ko_group_columns(cols, s_vec, group, product, pricing_env, breach=breach,
                                            surfaces=(surface,))
        else:
            self._terminal_ko_columns(cols, s_vec, ko_record, product, pricing_env, breach=breach,
                                      surfaces=(surface,))

    def __repr__(self) -> str:
        return "KOResetSnowballPDESolver()"
