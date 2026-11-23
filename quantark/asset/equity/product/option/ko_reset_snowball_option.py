"""
Knock-out reset snowball option implementation.

This product behaves like a standard snowball before knock-in (KI). Once KI
occurs, the knock-out (KO) schedule switches to a second, independent schedule.
The post-KI schedule can be applied in two modes:
- ABSOLUTE: post-KI schedule uses absolute times/dates.
- REBASED: post-KI schedule uses offsets from the KI time.
"""

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from quantark.asset.equity.settlement import SettlementConvention
from quantark.util.calendar import add_months
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import (
    BarrierType,
    ObservationAggregation,
    ObservationType,
    PostKOScheduleMode,
    ProtectionType,
    TenorEnd,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, is_close, is_valid_number

from .observation_schedule import (
    at_valuation_instant,
    ObservationRecord,
    ObservationsAtValuation,
    knock_in_decided_at_valuation,
    ObservationSchedule,
    PricingEnv,
    ResolvedObservationRecord,
)
from .snowball_config import AccrualConfig, AirbagConfig, BarrierConfig, PayoffConfig
from .accrual import (AccrualKind, AccrualPeriod, KnockInAnchor, contract_tenor as _contract_tenor, ko_accrual_factor,
                      leg_annualized)
from .accrual_build import snapshot_terms
from .contract_clock import canonical_clock_after, canonicalize_contract_clock, contract_instants, snap_to_instants
from .snowball_option import SnowballOption


@dataclass(frozen=True)
class KOResetObservationsAtValuation(ObservationsAtValuation):
    """``ObservationsAtValuation`` plus the schedule a knock-out came from.

    Attributes:
        ko_regime: ``"pre"`` (the pre-KI schedule) or ``"post"`` (the post-KI one) when ``knocked_out``.
            ``ko_index`` is the position in that schedule's resolved records, and ``ko_record.payoff``
            carries the contractual knock-out cash (the schedule's own resolution leaves it at zero).
    """

    ko_regime: Optional[str] = None


@dataclass
class KnockOutResetSnowballOption(SnowballOption):
    """
    Snowball option with KO schedule reset after KI.

    The pre-KI KO schedule is defined by `barrier_config` (same as SnowballOption).
    The post-KI KO schedule is defined by `post_barrier_config` (KO-only).

    Two rules fix the contract's life:

    1. A knock-in REPLACES the first knock-out schedule by the second (usually more dates at lower
       rates), and the knocked-in state stays to the end. A knocked-in contract is a knocked-in
       snowball on the second schedule, alive until that schedule ends.
    2. The second schedule only becomes effective through a knock-in DURING the first one. A
       contract not knocked in when the first schedule ends matures there, so a knock-in is never
       tested after that (``first_schedule_ended``, ``require_alive``).
    3. Schedule 2 ends last (spec D35): its last date is on or after schedule 1's last date, so every
       knock-in meets a schedule 2 in force until the note ends. ABSOLUTE checks the booked dates; REBASED
       checks the earliest possible knock-in (the first knock-in observation) plus schedule 2's last offset.
       A contract that breaks it is refused at construction (``ValidationError``, on both clocks).

    Two rules fix what one instant decides:

    * A first-schedule knock-out and a knock-in observed at the same instant: the knock-out wins, as
      on every autocallable, so that knock-in does not replace the schedule the knock-out belongs to.
    * Otherwise the second schedule is in force FROM the observation that knocks the contract in,
      that observation included.
    """

    post_barrier_config: BarrierConfig = field(
        default_factory=lambda: BarrierConfig(ko_barrier=1.0, ko_rate=0.0)
    )
    post_ko_mode: PostKOScheduleMode = PostKOScheduleMode.ABSOLUTE
    # Not a dataclass field: an instance attribute only when set (``__init__``), so a contract without it keeps the
    # attributes, fingerprints and cache keys it always had.
    first_schedule_settlement_time = None

    def __init__(
        self,
        initial_price: float,
        strike: float,
        barrier_config: BarrierConfig,
        post_barrier_config: BarrierConfig,
        payoff_config: Optional[PayoffConfig] = None,
        accrual_config: Optional[AccrualConfig] = None,
        airbag_config: Optional[AirbagConfig] = None,
        contract_multiplier: float = 1.0,
        is_reverse: bool = False,
        maturity: Optional[float] = None,
        tenor: Optional[float] = None,
        initial_date: Optional[datetime] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        maturity_date: Optional[datetime] = None,
        tenor_end: TenorEnd = TenorEnd.EXERCISE,
        annualization_day_count: DayCountConvention = DayCountConvention.ACT_365,
        post_ko_mode: PostKOScheduleMode = PostKOScheduleMode.ABSOLUTE,
        settlement_convention: Optional[SettlementConvention] = None,
        first_schedule_settlement_time: Optional[float] = None,
    ):
        """``first_schedule_settlement_time`` (numeric clock only): when a note that ends with its first schedule is
        paid -- the never-knocked-in maturity and, under EXPIRY coupon pay, a first-schedule knock-out (spec D30) --
        when the contract fixes that payment explicitly. It is the numeric counterpart of an explicit
        ``settlement_date`` for that payment (settlement precedence, spec D33): the float-time intraday twin of a
        dated note whose settlement date pays its first schedule carries it (spec D34). ``None`` (the default): an
        explicit ``settlement_date`` when booked, else schedule 1's maturity plus the settlement delay
        (``settlement_support.first_schedule_settlement``). A dated contract books a ``settlement_date`` instead.
        """
        self.post_barrier_config = post_barrier_config
        self.post_ko_mode = post_ko_mode
        if first_schedule_settlement_time is not None:
            self.first_schedule_settlement_time = first_schedule_settlement_time

        super().__init__(
            initial_price=initial_price,
            strike=strike,
            barrier_config=barrier_config,
            payoff_config=payoff_config,
            accrual_config=accrual_config,
            airbag_config=airbag_config,
            contract_multiplier=contract_multiplier,
            is_reverse=is_reverse,
            maturity=maturity,
            tenor=tenor,
            initial_date=initial_date,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
            maturity_date=maturity_date,
            tenor_end=tenor_end,
            annualization_day_count=annualization_day_count,
            settlement_convention=settlement_convention,
        )

    def validate(self, *, allow_zero_maturity: bool = False) -> None:
        self._validate_post_ko_mode()
        self._validate_first_schedule_settlement_time()
        super().validate(allow_zero_maturity=allow_zero_maturity)
        self._validate_dated_rebased_end()
        self._validate_schedule_two_ends_last()
        self._validate_first_schedule_settlement_range()

    def __eq__(self, other):
        """The dataclass fields, and ``first_schedule_settlement_time`` when either side sets it (it is not a field,
        so contracts without it compare exactly as before; CP9 M3)."""
        if other.__class__ is not self.__class__:
            return NotImplemented
        names = [f.name for f in fields(self) if f.compare]
        if tuple(getattr(self, n) for n in names) != tuple(getattr(other, n) for n in names):
            return False
        return self.first_schedule_settlement_time == other.first_schedule_settlement_time

    __hash__ = None

    def _validate_first_schedule_settlement_range(self) -> None:
        """``first_schedule_settlement_time`` is the payment of a note that ends with schedule 1 (spec D34), so it is
        refused (CP9 I1, owner ruling option A):

        - before schedule 1's maturity (its last booked observation): a payment cannot precede its determination;
        - when schedule 1 runs to the product maturity, unless it equals the product terminal's payment time: such a
          note has one maturity, paid by the product terminal (``first_schedule_settlement``), so another time means
          nothing.

        Times compare with ``is_close`` (round-off is equality). Checked on the booked records, as an aged contract
        keeps both its records and this time on one clock; a contract whose schedule 1 has no record left is not
        checked (only a knocked-in note is alive there, and it never pays on schedule 1).
        """
        value = getattr(self, "first_schedule_settlement_time", None)
        schedule = None if self.barrier_config is None else self.barrier_config.ko_observation_schedule
        if value is None or schedule is None or not schedule.records:
            return
        pre_end = max(float(r.observation_time) for r in schedule.records)
        if value < pre_end and not is_close(value, pre_end, abs_tol=Tolerance.ZERO):
            raise ValidationError(
                f"first_schedule_settlement_time {value!r} precedes schedule 1's maturity {pre_end!r}: the payment of a "
                "note that ends with schedule 1 cannot come before it ends"
            )
        from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
        from quantark.param import FlatRateCurve, FlatVolSurface, NoDividend, SpotQuote
        from quantark.priceenv import PricingEnvironment

        # market-free: a numeric contract's times are on its own clock, and only times are read here
        env = PricingEnvironment(rate_curve=FlatRateCurve(0.0), valuation_date=datetime(2000, 1, 3),
                                 spot_quote=SpotQuote(float(self.initial_price) or 1.0),
                                 vol_surface=FlatVolSurface(0.2), div_yield=NoDividend())
        terminal = resolve_terminal_timing(self, env)
        one_maturity = is_close(pre_end, float(terminal.determination_time), abs_tol=Tolerance.PRECISION)
        if one_maturity and not is_close(value, float(terminal.payment_time), abs_tol=Tolerance.ZERO):
            raise ValidationError(
                f"first_schedule_settlement_time {value!r}: schedule 1 runs to the product maturity, so the note has one "
                f"maturity, paid at the product terminal's payment time {float(terminal.payment_time)!r}; another "
                "first-schedule payment means nothing (book it as the terminal settlement)"
            )

    # ------------------------------------------------------------------
    # Validity: schedule 2 ends last (spec D35, owner ruling 2026-11-14)
    # ------------------------------------------------------------------

    def _validate_schedule_two_ends_last(self) -> None:
        """Schedule 2's last date must be on or after schedule 1's last date (equality allowed).

        A knock-in replaces schedule 1 by schedule 2 and the knocked-in note lives until schedule 2 ends; a schedule 2
        that could end before schedule 1 leaves a knock-in with no schedule 2 in force, which no contract means. Checked
        on the booked records (passed ones included), so an aged contract is held to the rule it was booked under:

        - ABSOLUTE: schedule 2's last booked date (numeric: time) against schedule 1's;
        - REBASED: schedule 2 hangs from the knock-in, and any path could knock in on the first knock-in observation,
          so that observation plus schedule 2's last offset (dated: ``rebased_dates_after``, calendar months) must not
          fall before schedule 1's last date. A REBASED contract monitors its knock-in discretely
          (``_validate_post_observation_parameters``), so the earliest possible knock-in is always that observation.
          Every later knock-in ends later, so each knocked-in copy (``with_absolute_schedule_2``) is valid too.

        Numeric times compare with ``is_close`` (round-off is equality). Raises ``ValidationError`` naming the dates.
        """
        pre = self.barrier_config.ko_observation_schedule
        post = self.post_barrier_config.ko_observation_schedule
        if pre is None or post is None or not pre.records or not post.records:
            return
        from quantark.asset.clock import ProductClock, product_clock

        dated = product_clock(self) is ProductClock.DATE

        def last(records):
            if dated:
                return max(r.observation_date for r in records)
            return max(float(r.observation_time) for r in records)

        def show(value) -> str:
            return f"{value:%Y-%m-%d}" if dated else f"{value!r}"

        def before(end, pre_end) -> bool:
            if dated:
                return end < pre_end
            return end < pre_end and not is_close(end, pre_end, abs_tol=Tolerance.ZERO)

        pre_end = last(pre.records)
        if self.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            end = last(post.records)
            if before(end, pre_end):
                raise ValidationError(
                    "KO-reset snowball (ABSOLUTE): schedule 2 must end on or after schedule 1, because a knock-in "
                    f"replaces schedule 1 by schedule 2 for the rest of the note's life: schedule 2's last observation "
                    f"{show(end)} precedes schedule 1's last observation {show(pre_end)}"
                )
            return
        ki = self.barrier_config.ki_observation_schedule
        records = [] if ki is None else ki.records
        if not records:
            return
        if dated:
            first = min(r.observation_date for r in records)
            end = max(self.rebased_dates_after(first))
            offset = f"{max(self._rebased_offset_months())} months"
        else:
            first = min(float(r.observation_time) for r in records)
            last_offset = max(float(r.observation_time) for r in post.records)
            end = first + last_offset
            offset = f"{last_offset!r}"
        if before(end, pre_end):
            raise ValidationError(
                "KO-reset snowball (REBASED): schedule 2 must end on or after schedule 1, because a knock-in "
                f"replaces schedule 1 by schedule 2 for the rest of the note's life: a knock-in on the first knock-in "
                f"observation {show(first)} plus schedule 2's last offset ({offset}) ends {show(end)}, before schedule "
                f"1's last observation {show(pre_end)}"
            )

    def _validate_first_schedule_settlement_time(self) -> None:
        """A finite time on the numeric clock (spec D34): a dated contract states the payment as its
        ``settlement_date`` (clock rule)."""
        from quantark.asset.clock import ProductClock, product_clock

        value = getattr(self, "first_schedule_settlement_time", None)
        if value is None:
            return
        if not is_valid_number(value):
            raise ValidationError(f"first_schedule_settlement_time must be a finite time, got {value!r}")
        if product_clock(self) is ProductClock.DATE:
            raise ValidationError(
                "KnockOutResetSnowballOption is on the date clock (exercise_date set): first_schedule_settlement_time "
                "is a numeric time with no date for it; a dated contract books its settlement_date (clock rule)"
            )
        self.first_schedule_settlement_time = float(value)

    # ------------------------------------------------------------------
    # A REBASED contract on the date clock (ship spec 2026-10-26, Decision 6)
    # ------------------------------------------------------------------

    def _dated_rebased(self) -> bool:
        from quantark.asset.clock import ProductClock, product_clock

        return self.post_ko_mode == PostKOScheduleMode.REBASED and product_clock(self) is ProductClock.DATE

    def _rebased_offset_months(self) -> List[int]:
        """Schedule 2's offsets as whole calendar months: a dated REBASED offset is booked as ``k / 12`` years and
        means ``k`` months (user decision 2026-10-26); any other offset, and a settlement offset (it has no date
        form: the settlement follows the settlement convention), is refused."""
        schedule = self.post_barrier_config.ko_observation_schedule
        months: List[int] = []
        for record in [] if schedule is None else schedule.records:
            if record.settlement_time is not None:
                raise ValidationError(
                    "a dated REBASED KO-reset cannot book a schedule-2 settlement offset (settlement_time): it has no "
                    "date form; the settlement follows the product's settlement convention"
                )
            scaled = float(record.observation_time) * 12.0
            whole = round(scaled)
            if not is_close(scaled, whole):
                raise ValidationError(
                    "a dated REBASED KO-reset's schedule-2 offsets are calendar months: each must be a whole number "
                    f"of months (k / 12 years), got {record.observation_time}"
                )
            months.append(int(whole))
        return months

    def rebased_dates_after(self, knock_in_date: datetime) -> Tuple[datetime, ...]:
        """The schedule-2 dates of a dated REBASED contract knocked in on ``knock_in_date``: ``add_months(date, k)``
        for each offset's ``k`` months, unadjusted (the day of month kept, clamped at the month end)."""
        return tuple(add_months(knock_in_date, k) for k in self._rebased_offset_months())

    @property
    def rebased_extended_schedule(self) -> Mapping[int, Tuple[datetime, ...]]:
        """A dated REBASED contract's extended schedule, read-only: for each knock-in record (by its index in the
        knock-in schedule), the schedule-2 dates a knock-in on that record's date places (``rebased_dates_after``).
        Generated (and checked) at construction; empty for any other contract."""
        if not self._dated_rebased():
            return MappingProxyType({})
        schedule = self.barrier_config.ki_observation_schedule
        records = [] if schedule is None else schedule.records
        return MappingProxyType({
            index: self.rebased_dates_after(record.observation_date) for index, record in enumerate(records)
        })

    def _validate_dated_rebased_end(self) -> None:
        """A dated REBASED contract's latest end (the last generated date) is a floor on its ``exercise_date``: a
        given one (or the one ``maturity_date`` supplies, as on every equity option) must not precede it; one still
        missing takes the latest end."""
        if not self._dated_rebased():
            return
        ends = [dates[-1] for dates in self.rebased_extended_schedule.values() if dates]
        if not ends:
            return
        latest = max(ends)
        if self.exercise_date is None:
            self.exercise_date = latest
        elif self.exercise_date < latest:
            raise ValidationError(
                f"a dated REBASED KO-reset's exercise_date ({self.exercise_date:%Y-%m-%d}) precedes its latest end "
                f"({latest:%Y-%m-%d}), the last schedule-2 date of a knock-in on its last knock-in date"
            )

    def rebased_schedule_two_times(
        self, pricing_env: PricingEnv, ki_times: Sequence[float], ki_source_indices: Sequence[int],
        offsets: np.ndarray,
    ) -> List[np.ndarray]:
        """Schedule 2's observation times for a knock-in on each knock-in record (``ki_times`` with their source
        indices in the knock-in schedule), in years from the valuation date: ``t_ki + offsets`` on the numeric
        clock, the resolved generated dates (``rebased_extended_schedule``) on the date clock."""
        if not self._dated_rebased():
            # a knock-in time plus an offset is a second spelling of any contract instant it lands on: snap it
            instants = contract_instants(self)
            return [snap_to_instants(t_ki + np.asarray(offsets, dtype=float), instants) for t_ki in ki_times]
        extended = self.rebased_extended_schedule
        return [
            np.array([ObservationRecord(observation_date=d).resolve_time(pricing_env) for d in extended[index]],
                     dtype=float)
            for index in ki_source_indices
        ]

    def _validate_barrier_parameters(self) -> None:
        super()._validate_barrier_parameters()
        self._validate_post_barrier_parameters()

    def _validate_observation_parameters(self) -> None:
        super()._validate_observation_parameters()
        self._validate_post_observation_parameters()

    def _validate_post_barrier_parameters(self) -> None:
        """Validate post-KI KO barrier configuration."""
        config = self.post_barrier_config

        self._validate_barrier_array(config.ko_barrier, "Post KO barrier")
        self._validate_rate_array(config.ko_rate, "Post KO rate")

        obs_len = self._get_observation_length_for_config(config)
        if obs_len is not None:
            self._validate_array_length(config.ko_barrier, obs_len, "Post KO barrier")
            self._validate_array_length(config.ko_rate, obs_len, "Post KO rate")

        if config.ki_barrier is not None:
            raise ValidationError("post_barrier_config must not define KI barriers")

    def _validate_post_observation_parameters(self) -> None:
        """Validate post-KI KO observation schedules."""
        config = self.post_barrier_config

        if not isinstance(config.ko_observation_type, ObservationType):
            raise ValidationError(
                f"Invalid post KO observation type: {config.ko_observation_type}"
            )
        if config.ko_observation_type != ObservationType.DISCRETE:
            raise ValidationError("Post KO monitoring must be discrete")

        if (
            config.ko_observation_schedule is None
            and config.ko_observation_dates is None
        ):
            raise ValidationError(
                "Post KO observation dates or schedule required for discrete monitoring"
            )
        self._validate_observation_dates(
            config.ko_observation_dates, "Post KO"
        )

        if self.post_ko_mode == PostKOScheduleMode.REBASED:
            if self.barrier_config.ki_barrier is None:
                raise ValidationError("Rebased post-KO schedule requires KI barrier")
            if (
                self.barrier_config.ki_observation_type != ObservationType.DISCRETE
                or self.barrier_config.ki_continuous
            ):
                raise ValidationError(
                    "Rebased post-KO schedule requires discrete KI monitoring"
                )
            schedule = config.ko_observation_schedule
            if schedule is not None:
                for rec in schedule.records:
                    if rec.observation_date is not None:
                        raise ValidationError(
                            "Rebased post-KO schedule must use observation_time offsets, not dates"
                        )

    def _validate_post_ko_mode(self) -> None:
        if not isinstance(self.post_ko_mode, PostKOScheduleMode):
            raise ValidationError(f"Invalid post_ko_mode: {self.post_ko_mode}")

    def _get_observation_length_for_config(
        self, config: BarrierConfig
    ) -> Optional[int]:
        if config.ko_observation_schedule is not None:
            return len(config.ko_observation_schedule.records)
        if config.ko_observation_dates is not None:
            return len(config.ko_observation_dates)
        return None

    def _build_observation_schedules(self) -> None:
        super()._build_observation_schedules()

        needs_post_schedule = (
            self.post_barrier_config.ko_observation_schedule is None
            and self.post_barrier_config.ko_observation_dates is not None
        )
        if not needs_post_schedule:
            return

        post_barriers = (
            self.post_barrier_config.ko_barrier
            if isinstance(self.post_barrier_config.ko_barrier, list)
            else None
        )
        post_rates = (
            self.post_barrier_config.ko_rate
            if isinstance(self.post_barrier_config.ko_rate, list)
            else None
        )

        records = []
        for i, t in enumerate(self.post_barrier_config.ko_observation_dates):
            barrier_val = (
                post_barriers[i] if post_barriers else self.post_barrier_config.ko_barrier
            )
            rate_val = (
                post_rates[i] if post_rates else self.post_barrier_config.ko_rate
            )
            records.append(
                ObservationRecord(
                    observation_time=t,
                    barrier=barrier_val,
                    return_rate=rate_val,
                    is_rate_annualized=False,
                )
            )
        post_schedule = ObservationSchedule(
            records=records,
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        )

        self.post_barrier_config = replace(
            self.post_barrier_config, ko_observation_schedule=post_schedule
        )

    @canonical_clock_after
    def time_shift(
        self, time_bump: float, bumped_date: datetime, pricing_env, *, keep_landing: bool = False
    ) -> bool:
        """Age the product: the snowball's schedules and an ABSOLUTE schedule 2, the post-knock-in knock-out
        schedule. A REBASED schedule 2 holds offsets from the knock-in time and stays.

        ``keep_landing`` keeps a record landing on the bumped instant at time 0 (theta, spec 2026-10-07) on every
        schedule. Returns True only when every observation has passed: schedule 1 and the knock-ins behind a
        knocked-in contract leave it alive on schedule 2. Schedule 2's records keep their accrual periods, so its
        cash does not move (accrual time spec §1); its explicit factors drop with its records.
        """
        super().time_shift(time_bump, bumped_date, pricing_env, keep_landing=keep_landing)
        if self.first_schedule_settlement_time is not None:
            # a time on the numeric clock, like the records: it moves with the origin
            self.first_schedule_settlement_time = float(self.first_schedule_settlement_time) - float(time_bump)
        config = self.barrier_config
        passed = config is None or not (
            BarrierConfig._has_future(config.ko_observation_schedule, config.ko_observation_dates)
            or BarrierConfig._has_future(config.ki_observation_schedule, config.ki_observation_dates)
        )
        if self.post_ko_mode != PostKOScheduleMode.ABSOLUTE:
            return passed
        before = self._get_observation_length_for_config(self.post_barrier_config) or 0
        self.post_barrier_config, _ = self.post_barrier_config.time_shift(
            time_bump, bumped_date, pricing_env, keep_landing=keep_landing
        )
        dropped = before - (self._get_observation_length_for_config(self.post_barrier_config) or 0)
        # the elapsed time was banked by the snowball's shift above: this one only takes the prefix off
        self.accrual_config = self.accrual_config.shifted(0.0, dropped_post=dropped)
        post = self.post_barrier_config
        return passed and not BarrierConfig._has_future(post.ko_observation_schedule, post.ko_observation_dates)

    def knock_in_anchor_required(self) -> bool:
        """Whether a knocked-in contract needs its ``KnockInAnchor`` (accrual time spec §5.1): a REBASED contract
        whose knock-in records accrue. A non-accruing one (``NONE`` periods: dated, no ``initial_date``, nothing
        annualized) has no anchor to resolve, so none is required."""
        if self.post_ko_mode != PostKOScheduleMode.REBASED:
            return False
        schedule = self.barrier_config.ki_observation_schedule
        records = [] if schedule is None else schedule.records
        return any(r.accrual is not None and r.accrual.kind is not AccrualKind.NONE for r in records)

    def with_absolute_schedule_2(
        self, knock_in_time: Union[float, datetime], *, anchor: Optional[KnockInAnchor]
    ) -> "KnockOutResetSnowballOption":
        """The ABSOLUTE contract a REBASED one becomes once knocked in at ``knock_in_time``.

        On the date clock ``knock_in_time`` is the knock-in DATE: schedule 2 becomes the dated records on the
        generated dates of that knock-in (``rebased_dates_after``) and the contract's ``exercise_date`` the last of
        them (ship spec 2026-10-26, Decision 6). Below, the numeric clock.

        ``knock_in_time`` is TIMING: the knock-in close on this product's own clock (years from its numeric origin;
        negative for a knock-in before it). Every schedule-2 record is observed at ``knock_in_time + offset`` and the
        contract ends at ``knock_in_time + max(offsets)``, the REBASED maturity (``get_post_maturity_time(ki_time)``),
        so V1 is paid and discounted when it ends.

        ``anchor`` is ACCRUAL: the knock-in record that fired, at its contractual fixing (accrual time spec §5.1),
        resolved once at the knock-in. Each converted record keeps its quoted ``relative`` period with the anchor
        embedded, and so does the ``post`` terminal (§5.2): a DATE anchor plus a quoted offset is never rounded into a
        date, and the copy books the cash the un-knocked contract's Monte Carlo route pays on the same path. The
        periods become explicit, so a re-validated copy keeps them. A contract whose knock-in records accrue refuses a
        missing anchor; it is never looked up by date.
        """
        if self.post_ko_mode != PostKOScheduleMode.REBASED:
            raise ValidationError("with_absolute_schedule_2 converts a REBASED contract only")
        schedule = self.post_barrier_config.ko_observation_schedule
        if schedule is None or not schedule.records:
            raise ValidationError("a REBASED contract needs its schedule-2 offsets to be placed")
        if anchor is None and self.knock_in_anchor_required():
            raise ValidationError(
                "a knocked-in REBASED KO-reset needs its knock-in anchor (the knock-in record that fired, at its "
                "contractual fixing; accrual time spec §5.1): schedule 2's accrual hangs from it"
            )

        def embedded(period: Optional[AccrualPeriod]) -> Optional[AccrualPeriod]:
            if period is None:
                return None
            if period.relative and anchor is not None:
                return replace(period, anchor=anchor, terms=None)
            return replace(period, terms=None)

        dated = self._dated_rebased()
        if dated != isinstance(knock_in_time, datetime):
            raise ValidationError(
                "a dated REBASED KO-reset converts on its knock-in date, a numeric one on its knock-in time; got "
                f"{knock_in_time!r}"
            )
        if dated:
            dates = self.rebased_dates_after(knock_in_time)
            product = deepcopy(self)
            product.post_barrier_config = replace(
                self.post_barrier_config,
                ko_observation_schedule=replace(schedule, records=[
                    replace(record, observation_time=None, observation_date=date, accrual=embedded(record.accrual))
                    for record, date in zip(schedule.records, dates)
                ]),
                ko_observation_dates=None,
            )
            product.post_ko_mode = PostKOScheduleMode.ABSOLUTE
            product.exercise_date = max(dates)
            product.maturity = None
            return self._with_converted_terminals(product, embedded)

        shift = float(knock_in_time)
        instants = contract_instants(self)

        def placed(offset: float) -> float:
            # the knock-in time plus an offset, on the contract's instant when it is one (as MC places it)
            return float(snap_to_instants([shift + float(offset)], instants)[0])

        records = [
            replace(
                record,
                observation_time=placed(record.observation_time),
                settlement_time=None if record.settlement_time is None else placed(record.settlement_time),
                accrual=embedded(record.accrual),
            )
            for record in schedule.records
        ]
        product = deepcopy(self)
        product.post_barrier_config = replace(
            self.post_barrier_config,
            ko_observation_schedule=replace(schedule, records=records),
            ko_observation_dates=None,
        )
        product.post_ko_mode = PostKOScheduleMode.ABSOLUTE
        product.maturity = placed(max(float(record.observation_time) for record in schedule.records))
        canonicalize_contract_clock(product)
        return self._with_converted_terminals(product, embedded)

    def _with_converted_terminals(self, product: "KnockOutResetSnowballOption", embedded) -> \
            "KnockOutResetSnowballOption":
        terminals = dict(getattr(self, "_terminal_accrual", None) or {})
        if "post" in terminals:
            terminals["post"] = embedded(terminals["post"])
        product._terminal_accrual = terminals
        # the converted terminals are the contract's, kept as they are when the copy is validated again
        product._terminal_accrual_explicit = True
        snapshot_terms(product)
        return product

    def _resolve_ko_schedule(
        self, config: BarrierConfig, pricing_env: PricingEnv
    ) -> Tuple[List[ResolvedObservationRecord], List[float], List[ObservationRecord]]:
        schedule = config.ko_observation_schedule
        if schedule is None:
            raise ValidationError("KO observation schedule is required to resolve KO observations.")

        default_barrier = (
            None if isinstance(config.ko_barrier, list) else config.ko_barrier
        )
        active_indices = [
            index
            for index, record in enumerate(schedule.records)
            if not (
                (
                    record.observation_time is not None
                    and float(record.observation_time) < 0.0
                )
                or (
                    record.observation_date is not None
                    and pricing_env is not None
                    and record.observation_date < pricing_env.valuation_date
                )
            )
        ]
        if not active_indices:
            return [], [], []
        active_schedule = ObservationSchedule(
            records=[schedule.records[index] for index in active_indices],
            aggregation_mode=schedule.aggregation_mode,
            frequency=schedule.frequency,
        )
        resolved_schedule = active_schedule.resolve(
            pricing_env=pricing_env,
            default_barrier=default_barrier,
            default_payoff=0.0,
            require_single=True,
            product=self,
        )

        rates: List[float] = []
        for source_idx in active_indices:
            rate = schedule.records[source_idx].return_rate
            if rate is None:
                rate = self._get_barrier_at(
                    config.ko_rate, source_idx, "KO rate"
                )
            rates.append(rate)

        return (
            resolved_schedule,
            rates,
            [schedule.records[index] for index in active_indices],
        )

    def get_pre_ko_observation_profile(self, pricing_env: PricingEnv) -> Dict[str, List[float]]:
        resolved, rates, records = self._resolve_ko_schedule(
            self.barrier_config, pricing_env
        )
        return {
            "observation_times": [rec.observation_time for rec in resolved],
            "barriers": [rec.barrier for rec in resolved],
            "rates": rates,
            "records": records,
        }

    def get_post_ko_observation_profile(self, pricing_env: PricingEnv) -> Dict[str, List[float]]:
        resolved, rates, records = self._resolve_ko_schedule(
            self.post_barrier_config, pricing_env
        )
        return {
            "observation_times": [rec.observation_time for rec in resolved],
            "barriers": [rec.barrier for rec in resolved],
            "rates": rates,
            "records": records,
            "mode": self.post_ko_mode,
        }

    def get_pre_maturity_time(self, pricing_env: PricingEnv) -> float:
        profile = self.get_pre_ko_observation_profile(pricing_env)
        times = profile["observation_times"]
        if times:
            return max(times)
        # No pre-KI observation is left, so the first schedule has ended (``first_schedule_ended``) and
        # only a knocked-in contract is still alive; it never reads this time. Callers that price the
        # not-knocked-in state must go through ``require_alive`` first.
        return super().get_maturity(pricing_env)

    def first_schedule_ended(self, pricing_env: PricingEnv) -> bool:
        """Whether every pre-KI knock-out observation is behind the valuation date.

        An aged contract drops the observations it has passed, so an ended first schedule is one with
        no observation left. An observation AT the valuation date is still pending.
        """
        return not self.get_pre_ko_observation_profile(pricing_env)["observation_times"]

    def require_alive(self, pricing_env: PricingEnv, knocked_in: bool) -> None:
        """Refuse the one state the contract cannot be in: past its first schedule and not knocked in.

        Such a contract matured when the first schedule ended (rule 2). Pricing it anyway would extend
        the first schedule to the final maturity and go on testing knock-ins, which every engine did.
        """
        if not knocked_in and self.first_schedule_ended(pricing_env):
            raise ValidationError(
                "KO-reset snowball: the pre-KI schedule has ended and the contract is not knocked in, so it "
                "matured at the end of that schedule; only a knocked-in contract runs on the post-KI schedule."
            )

    def _resolve_pre_contract_tenor(self, pricing_env: PricingEnv) -> float:
        """The not-knocked-in (V0) tenor: origin to schedule 1's last contractual fixing (accrual time spec §4.4)."""
        return _contract_tenor(self, pricing_env, "pre")

    def _resolve_post_contract_tenor(self, pricing_env: PricingEnv, *, knock_in=None) -> float:
        """The knocked-in (V1) tenor: to schedule 2's last contractual fixing; a REBASED contract's hangs from its
        knock-in anchor (``knock_in``)."""
        return _contract_tenor(self, pricing_env, "post", knock_in=knock_in)

    def get_contract_tenor(self, pricing_env: PricingEnv = None) -> float:
        """
        For KO-reset snowball, default contract tenor follows the pre-KI schedule.
        Post-KI tenor is handled explicitly in get_maturity_payoff_v1.
        """
        return _contract_tenor(self, pricing_env, "contract" if pricing_env is None else "pre")

    def get_post_maturity_time(
        self, pricing_env: PricingEnv, ki_time: Optional[float] = None
    ) -> float:
        profile = self.get_post_ko_observation_profile(pricing_env)
        times = profile["observation_times"]
        if not times:
            return self.get_pre_maturity_time(pricing_env)
        if self.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            return max(times)
        if ki_time is None:
            raise ValidationError("ki_time is required for rebased post-KO maturity.")
        return float(snap_to_instants([float(ki_time) + max(times)], contract_instants(self))[0])

    def get_max_maturity_time(self, pricing_env: PricingEnv) -> float:
        base_maturity = super().get_maturity(pricing_env)
        pre_maturity = self.get_pre_maturity_time(pricing_env)
        post_profile = self.get_post_ko_observation_profile(pricing_env)
        post_times = post_profile["observation_times"]
        if not post_times:
            return max(base_maturity, pre_maturity)

        if self.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            post_max = max(post_times)
        elif self._dated_rebased():
            # the latest generated end of a knock-in still ahead
            active = self.active_ki_source_indices(pricing_env)
            extended = self.rebased_extended_schedule
            ends = [extended[index][-1] for index in active if extended[index]]
            post_max = (
                ObservationRecord(observation_date=max(ends)).resolve_time(pricing_env) if ends else pre_maturity
            )
        else:
            ki_profile = self.get_ki_observation_profile(pricing_env)
            ki_times = ki_profile["observation_times"]
            if not ki_times:
                post_max = pre_maturity
            else:
                post_max = max(ki_times) + max(post_times)
        return max(base_maturity, pre_maturity, post_max)

    def get_maturity_payoff_v0(
        self, spot: float, pricing_env: PricingEnv = None, *,
        contract_tenor: Optional[float] = None,
    ) -> float:
        """
        KO-reset V0 payoff uses pre-KI contract tenor when annualized.

        ``contract_tenor`` is ``_resolve_pre_contract_tenor(pricing_env)`` resolved by a caller that
        prices many spots; it does not depend on spot. ``None`` resolves it here.
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        principal = (
            self.initial_price * self.contract_multiplier
            if self.payoff_config.include_principal
            else 0.0
        )

        if (
            self.payoff_config.call_rebate_enabled
            and self.payoff_config.call_strike is not None
        ):
            call_payoff = max(spot - self.payoff_config.call_strike, 0.0)
            rebate = (
                self.payoff_config.call_participation_rate
                * self.contract_multiplier
                * call_payoff
            )
            if leg_annualized(self, "rebate"):
                contract_tenor = contract_tenor or self._resolve_pre_contract_tenor(
                    pricing_env
                )
                rebate *= contract_tenor
        else:
            if leg_annualized(self, "rebate"):
                contract_tenor = contract_tenor or self._resolve_pre_contract_tenor(
                    pricing_env
                )
                rebate = (
                    self.payoff_config.rebate_rate
                    * self.initial_price
                    * self.contract_multiplier
                    * contract_tenor
                )
            else:
                rebate = (
                    self.payoff_config.rebate_rate
                    * self.initial_price
                    * self.contract_multiplier
                )

        return principal + rebate

    def get_maturity_payoff_v1(
        self, spot: float, pricing_env: PricingEnv = None, *,
        contract_tenor: Optional[float] = None,
    ) -> float:
        """
        KO-reset V1 payoff uses post-KI tenor when annualized.

        ``contract_tenor`` is the post-KI tenor resolved by a caller that knows the knock-in (a REBASED
        contract's tenor hangs from it); ``None`` resolves it here, from a carried anchor if the copy has one.
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        principal = (
            self.initial_price * self.contract_multiplier
            if self.payoff_config.include_principal
            else 0.0
        )

        airbag_barrier = self.airbag_config.airbag_barrier
        participation_rate = self.payoff_config.participation_rate
        effective_strike = self.strike

        if airbag_barrier is not None:
            if self.is_reverse:
                is_unsafe = spot > airbag_barrier
            else:
                is_unsafe = spot < airbag_barrier

            if is_unsafe:
                participation_rate = self.airbag_config.airbag_participation_rate
                effective_strike = (
                    self.airbag_config.airbag_strike
                    if self.airbag_config.airbag_strike is not None
                    else self.strike
                )

        if self.is_reverse:
            raw_diff = effective_strike - spot
        else:
            raw_diff = spot - effective_strike

        downside = (
            participation_rate * min(raw_diff, 0.0) * self.contract_multiplier
        )
        if leg_annualized(self, "knock_in"):
            if contract_tenor is None:
                contract_tenor = self._resolve_post_contract_tenor(
                    pricing_env, knock_in=getattr(self, "_otc_lifecycle_knock_in_anchor", None)
                )
            downside *= contract_tenor

        if self.payoff_config.protection_type == ProtectionType.FULL:
            floor = 0.0
            downside = max(downside, -floor)
        elif self.payoff_config.protection_type == ProtectionType.PARTIAL:
            floor = (
                self.payoff_config.protection_rate
                * self.initial_price
                * self.contract_multiplier
            )
            downside = max(downside, -floor)

        return principal + downside

    def decide_observations_at_valuation(
        self,
        spot: float,
        pricing_env: PricingEnv,
        knocked_in: bool = False,
    ) -> KOResetObservationsAtValuation:
        """Apply the observations scheduled AT the valuation instant (time zero) to the known spot.

        The knock-out schedule in force follows the knock-in state: the pre-KI schedule before a
        knock-in, the post-KI schedule after one. A contract not yet knocked in tests its pre-KI
        knock-out, then its knock-in; a knocked-in one -- carried, or decided at this instant -- tests
        the post-KI knock-out, unless the contract disables knock-outs after a knock-in.

        The second schedule is in force from the observation that knocks the contract in: a post-KI
        observation at that same instant is tested. Every engine applies it that way (the lattice engines
        hand a fresh knock-in a knocked-in surface that has already applied it; Monte Carlo counts a
        post-KI observation at or after the knock-in). It matters only when the post-KI level is at or
        below the knock-in level, so that one spot can breach both.
        """
        def breached(config: BarrierConfig, regime: str, state: bool):
            resolved, rates, sources = self._resolve_ko_schedule(config, pricing_env)
            for index, (record, rate, source) in enumerate(zip(resolved, rates, sources)):
                if not at_valuation_instant(record.observation_time) or record.barrier is None:
                    continue
                hit = spot <= record.barrier if self.is_reverse else spot >= record.barrier
                if hit:
                    cash = self._ko_cash(record.observation_time, rate, source, pricing_env)
                    return KOResetObservationsAtValuation(
                        True, state, replace(record, payoff=cash), index, regime
                    )
            return None

        fresh = False
        if not knocked_in:
            decided = breached(self.barrier_config, "pre", False)
            if decided is not None:
                return decided
            # rule 2: a knock-in is only tested while the first schedule is live (its last observation
            # included); past it, a contract not knocked in has matured
            fresh = (
                self.has_ki_barrier
                and not self.first_schedule_ended(pricing_env)
                and knock_in_decided_at_valuation(self, spot, pricing_env)
            )
        state = bool(knocked_in or fresh)
        # TODO: a REBASED post-KI schedule holds offsets from the knock-in time, which a carried state does
        # not record, so nothing places its records at this instant; they are left undecided here.
        if (
            state
            and not self.barrier_config.disable_ko_after_ki
            and self.post_ko_mode == PostKOScheduleMode.ABSOLUTE
        ):
            decided = breached(self.post_barrier_config, "post", True)
            if decided is not None:
                return decided
        return KOResetObservationsAtValuation(False, state)

    def _ko_cash(
        self,
        observation_time: float,
        rate: float,
        schedule_record: ObservationRecord,
        pricing_env: PricingEnv,
    ) -> float:
        """Contractual cash of one knock-out: the principal, if included, plus rate x accrual on the notional."""
        notional = self.initial_price * self.contract_multiplier
        principal = notional if self.payoff_config.include_principal else 0.0
        accrual = self.compute_ko_accrual_factor(observation_time, schedule_record, pricing_env)
        return float(principal + notional * float(rate) * float(accrual))

    def compute_ko_accrual_factor(
        self,
        observation_time: Optional[float],
        schedule_record: ObservationRecord,
        pricing_env: PricingEnv,
        *,
        knock_in=None,
    ) -> float:
        """The knock-out leg of ``schedule_record``'s accrual period (accrual time spec §5). ``observation_time`` is
        unused, kept for callers: accrual is the record's own, not its observation time. A REBASED schedule-2
        record needs its knock-in anchor (``knock_in``)."""
        return ko_accrual_factor(self, schedule_record, pricing_env, knock_in=knock_in)

    def cache_key(self) -> Dict[str, object]:
        key = super().cache_key()

        def _serialize_enum(value) -> Optional[str]:
            return value.name if hasattr(value, "name") else None

        def _serialize_schedule(
            schedule: Optional[ObservationSchedule],
        ) -> Optional[Dict[str, object]]:
            if schedule is None:
                return None
            frequency = schedule.frequency
            if hasattr(frequency, "name"):
                frequency_value = frequency.name
            else:
                frequency_value = frequency
            return {
                "aggregation_mode": _serialize_enum(schedule.aggregation_mode),
                "frequency": frequency_value,
                "records": [_serialize_value(rec) for rec in schedule.records],
            }

        def _serialize_value(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, datetime):
                return value.isoformat()
            if hasattr(value, "name"):
                return value.name
            if isinstance(value, ObservationSchedule):
                return _serialize_schedule(value)
            if hasattr(value, "__dataclass_fields__"):
                return {
                    f.name: _serialize_value(getattr(value, f.name))
                    for f in value.__dataclass_fields__.values()
                }
            if isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_serialize_value(v) for v in value]
            return str(value)

        key["post_barrier_config"] = _serialize_value(self.post_barrier_config)
        key["post_ko_mode"] = _serialize_value(self.post_ko_mode)
        if self.first_schedule_settlement_time is not None:       # absent otherwise: existing keys stay as they were
            key["first_schedule_settlement_time"] = float(self.first_schedule_settlement_time)
        return key
