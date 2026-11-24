"""Fold cash legs into an autocallable's own payoff.

A foldable leg's PV is affine in the parent's event probabilities. Writing
``v_e = leg.value(basis_e)`` for the distribution that puts probability one on
event ``e``, the event route's leg PV is exactly

    v_anchor + Σ_{e ≠ anchor} P_e · (v_e − v_anchor)

where the anchor is MATURITY_NO_KO, the bucket the event route derives as the
residual (and the one ``EventDistribution.normalized`` adjusts). So a leg set
folds into a constant (``Σ v_anchor``, added outside the solve) plus, for every
other event, a cash amount paid at that event's settlement. An engine adds that
cash exactly where it adds its own payoff for the event, and one plain solve
then prices product and legs together.

The basis distributions come from the event route's own conversion code
(``EventDistribution.from_statistics``) fed a solve-free structural
``EventStatistics`` record on the product's event axis (the shared observation
grid; DCN: ``dcn_fold_slots``), so the folded structure is the event route's,
not a re-derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantark.cashleg.accrual_leg import AccrualLeg
from quantark.cashleg.autocallable_leg import AutocallableCashLeg
from quantark.cashleg.base import CashLeg
from quantark.cashleg.deterministic_leg import DeterministicLeg
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, almost_equal, is_close, is_non_positive, same_instant

__all__ = [
    "FoldMode",
    "FoldedEventCash",
    "FoldOverlay",
    "FoldedTradeValue",
    "ensure_foldable",
    "fold_cash_legs",
    "folded_trade_value",
    "is_event_independent",
    "is_foldable",
]

#: Exact types only: a subclass may override ``value()`` with a non-affine rule.
_FOLDABLE_TYPES = (DeterministicLeg, AccrualLeg, FixedPayoffLeg, AutocallableCashLeg)
_FOLDABLE_STREAMS = frozenset(
    {EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI}
)
_BRANCHES = ("no_ki", "ki")


class FoldMode(Enum):
    """What a folded trade valuation reports."""

    TOTAL = "total"            # trade total PV and greeks only
    DRILLDOWN = "drilldown"    # product, total, per-leg PV and per-leg greeks


@dataclass(frozen=True)
class FoldedEventCash:
    """Cash folded onto one event, per product unit, paid at ``payment_time``."""

    schedule: str               # "main" | "pre" | "post"
    determination_time: float
    payment_time: float
    amount: float


@dataclass(frozen=True)
class FoldOverlay:
    """Cash legs folded onto an autocallable's knock-out records and maturity branches.

    ``ko`` follows the event route's event order. ``constant_pv`` is absolute (not per
    unit) and is added outside the solve. Frozen and made of floats and strings, so it
    hashes into an engine's cache keys.
    """

    ko: Tuple[FoldedEventCash, ...]
    maturity_no_ki: Optional[FoldedEventCash]
    maturity_ki: Optional[FoldedEventCash]
    constant_pv: float = 0.0

    @property
    def is_empty(self) -> bool:
        """No cash enters the solve: engines then price exactly as without an overlay."""
        entries = list(self.ko) + [e for e in (self.maturity_no_ki, self.maturity_ki) if e]
        return all(almost_equal(e.amount, 0.0, tol=0.0) for e in entries)

    def bind_ko(self, records: Sequence[Tuple[str, float, float]]) -> np.ndarray:
        """Amounts aligned to an engine's ``(schedule, determination_time, payment_time)`` records.

        Binding is one to one: an engine record without an entry, an entry left without a
        record, or a payment time that differs raises ``ValidationError`` — cash is never
        dropped or misdiscounted silently.
        """
        used = [False] * len(self.ko)
        amounts = np.zeros(len(records), dtype=float)
        for k, (schedule, determination, payment) in enumerate(records):
            match = next(
                (
                    i
                    for i, entry in enumerate(self.ko)
                    if not used[i]
                    and entry.schedule == schedule
                    and almost_equal(
                        entry.determination_time, float(determination), tol=Tolerance.ZERO
                    )
                ),
                None,
            )
            if match is None:
                raise ValidationError(
                    f"engine knock-out record ({schedule}, t={float(determination)}) has no "
                    "overlay entry, or its entry was bound twice"
                )
            entry = self.ko[match]
            if not almost_equal(entry.payment_time, float(payment), tol=Tolerance.ZERO):
                raise ValidationError(
                    f"knock-out record ({schedule}, t={float(determination)}) settles at "
                    f"{float(payment)} but the folded cash assumed payment at {entry.payment_time}"
                )
            used[match] = True
            amounts[k] = entry.amount
        unbound = [entry for entry, u in zip(self.ko, used) if not u]
        if unbound:
            raise ValidationError(
                f"overlay entry ({unbound[0].schedule}, t={unbound[0].determination_time}) "
                "has no engine record"
            )
        return amounts

    def bind_terminal(self, branch: str, determination_time: float, payment_time: float) -> float:
        """The amount for maturity branch ``"no_ki"`` or ``"ki"``, checked against the engine's timing.

        A branch the overlay does not carry (the trivial, knock-out-free structure) binds to 0.
        """
        if branch not in _BRANCHES:
            raise ValidationError(f"unknown maturity branch {branch!r}; expected one of {_BRANCHES}")
        entry = self.maturity_no_ki if branch == "no_ki" else self.maturity_ki
        if entry is None:
            return 0.0
        if not almost_equal(entry.determination_time, float(determination_time), tol=Tolerance.ZERO):
            raise ValidationError(
                f"maturity branch {branch} has determination time {float(determination_time)} in the engine "
                f"but at {entry.determination_time} by the folded cash"
            )
        if not almost_equal(entry.payment_time, float(payment_time), tol=Tolerance.ZERO):
            raise ValidationError(
                f"maturity branch {branch} settles at {float(payment_time)} but the folded cash "
                f"assumed payment at {entry.payment_time}"
            )
        return entry.amount


# --- foldability --------------------------------------------------------------------------

def is_foldable(leg: CashLeg) -> bool:
    """An allowlisted leg type (exactly) reading only knock-out and maturity streams."""
    return type(leg) in _FOLDABLE_TYPES and leg.required_event_types() <= _FOLDABLE_STREAMS


def is_event_independent(leg: CashLeg) -> bool:
    """A foldable leg whose value reads no event probability: valued directly, never solved."""
    return is_foldable(leg) and leg.required_event_types() == frozenset()


def ensure_foldable(legs: Sequence[CashLeg]) -> None:
    """Raise ``CapabilityError`` naming the first leg that cannot fold."""
    from quantark.execution.errors import CapabilityError

    for leg in legs:
        if not is_foldable(leg):
            streams = sorted(e.value for e in leg.required_event_types())
            raise CapabilityError(
                f"{type(leg).__name__} (streams {streams}) cannot be folded into the product "
                "solve; only DeterministicLeg, AccrualLeg, FixedPayoffLeg and AutocallableCashLeg "
                "reading knock-out / maturity streams fold. Price it on the event route."
            )


# --- the solve-free event structure ---------------------------------------------------------

@dataclass(frozen=True)
class _Slot:
    schedule: str
    determination_time: float
    payment_time: float


@dataclass(frozen=True)
class _Structure:
    """Event slots in the event route's order, and the distribution of given event weights."""

    ko: Tuple[_Slot, ...]
    no_ki: _Slot
    ki: Optional[_Slot]
    distribution: Callable[[np.ndarray, float, float], EventDistribution]

    def weights(self) -> List[Tuple[np.ndarray, float, float]]:
        """``(p_ko, p_no_ki, p_ki)`` of every basis distribution, KO records first."""
        n = len(self.ko)
        bases = []
        for i in range(n):
            p = np.zeros(n)
            p[i] = 1.0
            bases.append((p, 0.0, 0.0))
        bases.append((np.zeros(n), 1.0, 0.0))
        if self.ki is not None:
            bases.append((np.zeros(n), 0.0, 1.0))
        return bases


def _trivial_structure(product, pricing_env) -> _Structure:
    tau = max(float(product.get_maturity(pricing_env)), 0.0)
    return _Structure(
        ko=(),
        no_ki=_Slot("main", tau, tau),
        ki=None,
        distribution=lambda p, no_ki, ki: EventDistribution.trivial(tau),
    )


def _in_life(records, tau):
    """The engines' ``_filter_observations_by_tau``: observations in [0, tau], sorted."""
    kept = [
        rec for rec in records
        if rec.observation_time is not None and 0.0 <= rec.observation_time <= tau
    ]
    kept.sort(key=lambda rec: float(rec.observation_time))
    return kept


def _settlement(rec) -> float:
    return float(rec.settlement_time if rec.settlement_time is not None else rec.observation_time)


#: why a fold structure's statistics carry no KI / coupon-probability streams: folding reads KO and maturity only
_FOLD_REASON = "not requested"


def _all_dates(dates) -> Optional[tuple]:
    dates = tuple(dates)
    return None if any(d is None for d in dates) else dates


def _structure_statistics(
    *,
    family: str,
    times,
    payments,
    p_ko,
    survival,
    maturity: dict,
    ki_settlement: dict,
    p_no_ki: float,
    p_ki: float,
    schedule=None,
    dates=None,
    payment_dates=None,
    coupon_payments=None,
    coupon_payment_dates=None,
):
    """Solve-free ``EventStatistics`` of given event weights on a product's event axis (zero cash).

    The event route's distribution of these weights is ``EventDistribution.from_statistics`` of this record,
    so a basis distribution is built by the same conversion the engines' statistics go through.
    """
    from quantark.asset.equity.engine.event_stats import (
        CouponTable,
        EventStatistics,
        KnockInStatistics,
        ObservationTable,
        OutcomeRow,
        ReceivablesTable,
        StatisticsProvenance,
        build_cash_ledger,
    )

    times = np.asarray(times, dtype=float)
    n = times.size
    zeros = np.zeros(n)
    flags = np.zeros(n, dtype=bool)
    survival = np.asarray(survival, dtype=float)
    observations = ObservationTable(
        times=times, payment_times=np.asarray(payments, dtype=float), dates=dates, payment_dates=payment_dates,
        is_ko_obs=np.ones(n, dtype=bool), is_coupon_obs=np.full(n, family in ("phoenix", "dcn")),
        schedule=schedule, probability=np.asarray(p_ko, dtype=float), survival=survival,
        discounted_cash=zeros, cash_to_pay=zeros, payment_is_path_dependent=flags,
    )
    unavailable = {key: _FOLD_REASON for key in
                   ("ki_table", "knock_in.probability", "knock_in.expected_time", "knock_in.expected_loss")}
    coupons = None
    if family in ("phoenix", "dcn"):
        unavailable["coupons.probability"] = _FOLD_REASON
        coupons = CouponTable(
            probability=None, survival=np.r_[1.0, survival[:-1]] if n else zeros,
            payment_times=np.asarray(payments if coupon_payments is None else coupon_payments, dtype=float),
            payment_dates=payment_dates if coupon_payments is None else coupon_payment_dates,
            discounted_cash=zeros, cash_to_pay=zeros, payment_is_path_dependent=flags,
        )
    maturity_row = OutcomeRow(probability=float(p_no_ki), discounted_cash=0.0, cash_to_pay=0.0, **maturity)
    ki_row = OutcomeRow(probability=float(p_ki), discounted_cash=0.0, cash_to_pay=0.0, **ki_settlement)
    receivables = ReceivablesTable.empty()
    final_time = float(times[-1]) if n else maturity_row.time
    return EventStatistics(
        product_family=family, pv=0.0, observations=observations, coupons=coupons, maturity=maturity_row,
        ki_settlement=ki_row,
        knock_in=KnockInStatistics(probability=None, probability_no_ko=ki_row.probability, expected_time=None,
                                   expected_loss=None),
        ki_table=None, receivables=receivables,
        ledger=build_cash_ledger(observations=observations, coupons=coupons, maturity=maturity_row,
                                 ki_settlement=ki_row, receivables=receivables),
        expected_lifetime=float(np.sum(observations.probability * times)) + (float(p_no_ki) + float(p_ki)) * final_time,
        provenance=StatisticsProvenance(statistics_model="fold_structure", price_model="fold_structure"),
        unavailable=unavailable,
    )


def _autocallable_structure(product, pricing_env) -> _Structure:
    """Snowball / Phoenix: the PDE readout's records and terminal timing."""
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption

    tau = float(product.get_maturity(pricing_env))
    if is_non_positive(tau, tol=0.0):                   # the engines' expired outcome: trivial
        return _trivial_structure(product, pricing_env)
    records = _in_life(product.resolve_ko_observations(pricing_env), tau)
    if not records:
        return _trivial_structure(product, pricing_env)
    terminal = resolve_terminal_timing(product, pricing_env)
    ko_times = np.array([float(rec.observation_time) for rec in records], dtype=float)
    payments = np.array([_settlement(rec) for rec in records], dtype=float)
    family = "phoenix" if isinstance(product, PhoenixOption) else "snowball"
    timing = dict(time=float(terminal.determination_time), payment_time=float(terminal.payment_time),
                  date=terminal.determination_date, payment_date=terminal.payment_date)
    # dates only when every record and the terminal row carry one (the legacy ledger's all-or-nothing rule)
    dates = _all_dates([*(rec.observation_date for rec in records), terminal.determination_date])
    payment_dates = _all_dates([*(rec.settlement_date for rec in records), terminal.payment_date])
    dates = None if dates is None else dates[:-1]
    payment_dates = None if payment_dates is None else payment_dates[:-1]

    def distribution(p_ko, p_no_ki, p_ki):
        p_ko = np.asarray(p_ko, dtype=float)
        stats = _structure_statistics(
            family=family, times=ko_times, payments=payments, p_ko=p_ko,
            survival=np.maximum(0.0, 1.0 - np.cumsum(p_ko)), maturity=timing, ki_settlement=timing,
            p_no_ki=p_no_ki, p_ki=p_ki, dates=dates, payment_dates=payment_dates,
        )
        return EventDistribution.from_statistics(stats)

    terminal_slot = _Slot("main", float(terminal.determination_time), float(terminal.payment_time))
    return _Structure(
        ko=tuple(_Slot("main", float(rec.observation_time), _settlement(rec)) for rec in records),
        no_ki=terminal_slot,
        ki=terminal_slot,
        distribution=distribution,
    )


def _ko_reset_structure(product, pricing_env) -> _Structure:
    """KO-reset: both schedules on one axis (pre first at a tie) and the two maturity buckets."""
    from quantark.asset.equity.engine.settlement_support import (
        first_schedule_payment_time, first_schedule_settlement, resolve_terminal_timing)
    from quantark.execution.errors import CapabilityError
    from quantark.util.enum import CouponPayType, PostKOScheduleMode

    if product.post_ko_mode == PostKOScheduleMode.REBASED:
        # DESIGN §4.11: out of scope until a fold binder takes the absolute termination distribution
        raise CapabilityError(
            "a REBASED KO-reset settles at knock-in-dependent times; its cash legs cannot be folded "
            "(no fold binder takes the absolute termination distribution); value them on the event route"
        )
    tau = float(product.get_maturity(pricing_env))
    if is_non_positive(tau, tol=0.0):                   # the engines' expired outcome: trivial
        return _trivial_structure(product, pricing_env)
    terminal = resolve_terminal_timing(product, pricing_env)
    expiry = product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY

    def schedule(config, expiry_payment):
        resolved, _, _ = product._resolve_ko_schedule(config, pricing_env)
        times, pays = [], []
        for rec in _in_life(resolved, tau):
            times.append(float(rec.observation_time))
            pays.append(expiry_payment if expiry else _settlement(rec))
        return np.array(times, dtype=float), np.array(pays, dtype=float)

    # EXPIRY pay: with the note's end -- schedule 1's maturity for a first-schedule knock-out (spec D30)
    pre_t, pre_pay = schedule(product.barrier_config,
                              first_schedule_payment_time(product, pricing_env, terminal) if expiry else None)
    post_t, post_pay = schedule(product.post_barrier_config, float(terminal.payment_time))
    if pre_t.size + post_t.size == 0:
        return _trivial_structure(product, pricing_env)
    delay = float(terminal.payment_time) - float(terminal.determination_time)
    # the note that ends with schedule 1: paid on the explicit settlement date when booked (spec D33), else the
    # terminal delay after schedule 1's maturity -- where every engine (MC included) settles it
    first_schedule = first_schedule_settlement(product, pricing_env, terminal)
    pre_maturity, post_maturity = first_schedule.determination_time, tau
    pre_payment = first_schedule.payment_time
    times = np.r_[pre_t, post_t]
    tag = np.r_[np.zeros(pre_t.size), np.ones(post_t.size)]
    order = np.lexsort((tag, times))                      # the event route's merged order
    pays = np.r_[pre_pay, post_pay]
    merged_times, merged_pays = times[order], pays[order]
    schedule_tags = tuple("pre" if tag[i] == 0 else "post" for i in order)
    # F5: the pre-maturity mass leaves the survival at its own determination when that precedes the last record
    last = float(merged_times[-1])
    drops_pre_maturity = pre_maturity < last and not same_instant(pre_maturity, last)
    after_pre_maturity = merged_times >= pre_maturity

    def distribution(p_ko, p_no_ki, p_ki):
        p_ko = np.asarray(p_ko, dtype=float)
        survival = 1.0 - np.cumsum(p_ko)
        if drops_pre_maturity:
            survival = survival - np.where(after_pre_maturity, float(p_no_ki), 0.0)
        stats = _structure_statistics(
            family="ko_reset", times=merged_times, payments=merged_pays, p_ko=p_ko, survival=survival,
            schedule=schedule_tags, p_no_ki=p_no_ki, p_ki=p_ki,
            maturity=dict(time=pre_maturity, payment_time=pre_payment),
            ki_settlement=dict(time=post_maturity, payment_time=post_maturity + delay),
        )
        return EventDistribution.from_statistics(stats)

    return _Structure(
        ko=tuple(
            _Slot("pre" if tag[i] == 0 else "post", float(times[i]), float(pays[i])) for i in order
        ),
        no_ki=_Slot("pre", pre_maturity, pre_payment),
        ki=_Slot("post", post_maturity, post_maturity + delay),
        distribution=distribution,
    )


def _dcn_structure(product, pricing_env) -> _Structure:
    """DCN: the shared monthly observation grid and the loss-leg settlement (``dcn_fold_slots``, spec D12 axis)."""
    from quantark.asset.equity.engine.dcn_valuation import dcn_fold_slots
    from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context

    ctx = build_dcn_grid_context(product, pricing_env)
    if np.asarray(ctx.obs_cols).size == 0:
        return _trivial_structure(product, pricing_env)
    slots = dcn_fold_slots(ctx)
    times = np.array([t for _, t, _ in slots.ko], dtype=float)
    payments = np.array([p for _, _, p in slots.ko], dtype=float)
    observations = product.schedule.monthly
    pay = product.payment_dates()
    timing = dict(time=slots.determination, payment_time=slots.payment,
                  date=observations[-1].observation_date, payment_date=pay.loss_payment_date)
    dates = _all_dates(m.observation_date for m in observations)
    ko_dates = _all_dates(pay.ko_payment_dates)
    coupon_dates = _all_dates(pay.coupon_payment_dates)

    def distribution(p_ko, p_no_ki, p_ki):
        p_ko = np.asarray(p_ko, dtype=float)
        stats = _structure_statistics(
            family="dcn", times=times, payments=payments, p_ko=p_ko,
            survival=np.maximum(0.0, 1.0 - np.cumsum(p_ko)), maturity=timing, ki_settlement=timing,
            p_no_ki=p_no_ki, p_ki=p_ki, dates=dates, payment_dates=ko_dates,
            coupon_payments=np.asarray(ctx.coupon_pay_times, dtype=float), coupon_payment_dates=coupon_dates,
        )
        return EventDistribution.from_statistics(stats)

    maturity = _Slot("main", slots.determination, slots.payment)
    return _Structure(
        ko=tuple(_Slot(schedule, t, p) for schedule, t, p in slots.ko),
        no_ki=maturity,
        ki=maturity,
        distribution=distribution,
    )


def _event_structure(product, pricing_env) -> _Structure:
    from quantark.asset.equity.product.option.ko_reset_snowball_option import (
        KnockOutResetSnowballOption,
    )
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.execution.errors import CapabilityError

    from quantark.asset.equity.product.option.dcn_option import DCNOption

    if isinstance(product, DCNOption):
        return _dcn_structure(product, pricing_env)
    if isinstance(product, KnockOutResetSnowballOption):
        return _ko_reset_structure(product, pricing_env)
    if isinstance(product, (SnowballOption, PhoenixOption)):
        return _autocallable_structure(product, pricing_env)
    raise CapabilityError(f"cash legs cannot be folded into a {type(product).__name__}")


# --- folding --------------------------------------------------------------------------------

def fold_cash_legs(
    legs: Sequence[CashLeg],
    product,
    pricing_env,
    *,
    quantity: float,
    unit_notional: float,
) -> FoldOverlay:
    """Fold ``legs`` (absolute, buyer's direction) into per-unit cash on ``product``'s events.

    For any engine whose event weights are ``P``:
    ``quantity · (npv + Σ_e P_e·amount_e·DF(pay_e)) + constant_pv`` equals
    ``quantity · npv + Σ leg PV`` of the event route on those weights.
    """
    ensure_foldable(legs)
    quantity = float(quantity)
    if almost_equal(quantity, 0.0, tol=0.0):
        raise ValidationError("a position with quantity 0 has no product unit to fold its cash legs onto")

    structure = _event_structure(product, pricing_env)
    # a DCN's engines sign every cash they add by its direction, so its overlay is buyer-view (D9)
    unit = quantity * float(getattr(product, "direction_sign", 1))
    bases = structure.weights()
    dists = [structure.distribution(*w) for w in bases]
    anchor = len(structure.ko)                           # MATURITY_NO_KO follows the KO records
    diffs = np.zeros(len(bases))
    constant = 0.0
    for leg in legs:
        values = np.array([leg.value(d, pricing_env, unit_notional) for d in dists], dtype=float)
        if not is_event_independent(leg) and len(bases) > 1:
            _check_affine(leg, structure, values, anchor, pricing_env, unit_notional)
        diffs += values - values[anchor]
        constant += float(values[anchor])

    def cash(slot, diff):
        return FoldedEventCash(
            slot.schedule,
            slot.determination_time,
            slot.payment_time,
            float(diff) / pricing_env.get_discount_factor(slot.payment_time) / unit,
        )

    ko = tuple(cash(slot, diffs[i]) for i, slot in enumerate(structure.ko))
    no_ki = FoldedEventCash(
        structure.no_ki.schedule,
        structure.no_ki.determination_time,
        structure.no_ki.payment_time,
        0.0,
    )
    ki = None if structure.ki is None else cash(structure.ki, diffs[anchor + 1])
    return FoldOverlay(ko=ko, maturity_no_ki=no_ki, maturity_ki=ki, constant_pv=constant)


def _check_affine(leg, structure, values, anchor, pricing_env, unit_notional) -> None:
    """One mixed distribution must value as the probed decomposition predicts."""
    n = len(structure.ko)
    m = len(values)
    w = 1.0 / m
    mixed = structure.distribution(np.full(n, w), w, w if structure.ki is not None else 0.0)
    predicted = float(values[anchor] + w * np.sum(values - values[anchor]))
    actual = float(leg.value(mixed, pricing_env, unit_notional))
    scale = max(1.0, float(np.max(np.abs(values))))
    if not is_close(actual, predicted, rel_tol=Tolerance.RELATIVE, abs_tol=Tolerance.RELATIVE * scale):
        raise ValidationError(
            f"{type(leg).__name__} {leg.name or leg.leg_id} is not affine in the event "
            f"probabilities (mixed {actual!r} vs folded {predicted!r}); it cannot be folded"
        )


# --- one trade valuation, folded --------------------------------------------------------------

@dataclass(frozen=True)
class FoldedTradeValue:
    """A trade valued with its cash legs folded into the product solve.

    ``total`` is absolute: ``quantity · npv + Σ leg PV``. ``npv`` (per unit) and ``per_leg``
    (absolute, keyed by ``leg_id``) are filled only in ``FoldMode.DRILLDOWN``.
    """

    total: float
    npv: Optional[float] = None
    per_leg: Dict[str, float] = field(default_factory=dict)


def folded_trade_value(
    engine,
    product,
    pricing_env,
    legs: Sequence[CashLeg],
    *,
    quantity: float,
    unit_notional: float,
    mode: FoldMode,
    lifecycle_state=None,
) -> FoldedTradeValue:
    """Value a trade by folding its cash legs into ``engine``'s solve (ship spec D11).

    TOTAL: one folded solve with every leg. DRILLDOWN: the plain solve, the folded
    total, and one folded solve per event-dependent leg (leg j's PV is
    ``quantity·(folded_j − npv) + constant_j``); event-independent legs are valued
    directly and never solved. The overlay is rebuilt from ``pricing_env`` on every
    call, so a bumped market gets its own discount ratios.
    """
    if not isinstance(mode, FoldMode):
        raise ValidationError(f"mode must be a FoldMode, got {mode!r}")
    from quantark.execution.errors import CapabilityError

    if not getattr(engine, "supports_cash_leg_folding", False):
        raise CapabilityError(
            f"{type(engine).__name__} does not fold cash legs into its solve; "
            "price the legs on the event route"
        )
    if mode is FoldMode.DRILLDOWN:
        # one gate for every per-leg caller: breakdown, trade risk, kernel cells, theta (D11c, CP4)
        reason = engine.fold_drilldown_refusal(product, pricing_env)
        if reason is not None:
            raise CapabilityError(f"FoldMode.DRILLDOWN refused on {type(engine).__name__}: {reason}")
    kwargs = {} if lifecycle_state is None else {"lifecycle_state": lifecycle_state}
    quantity = float(quantity)
    overlay = fold_cash_legs(
        legs, product, pricing_env, quantity=quantity, unit_notional=unit_notional
    )
    folded = float(engine.price_folded(product, pricing_env, overlay, **kwargs))
    total = quantity * folded + overlay.constant_pv
    if mode is FoldMode.TOTAL:
        return FoldedTradeValue(total=total)

    npv = float(engine.price(product, pricing_env, **kwargs))
    per_leg: Dict[str, float] = {}
    for leg in legs:
        own = fold_cash_legs(
            [leg], product, pricing_env, quantity=quantity, unit_notional=unit_notional
        )
        if own.is_empty:
            per_leg[leg.leg_id] = own.constant_pv
        else:
            alone = float(engine.price_folded(product, pricing_env, own, **kwargs))
            per_leg[leg.leg_id] = quantity * (alone - npv) + own.constant_pv
    return FoldedTradeValue(total=total, npv=npv, per_leg=per_leg)
