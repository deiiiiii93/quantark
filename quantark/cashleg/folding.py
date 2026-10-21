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
(``EventDistribution.from_autocallable_stats``) fed a solve-free structural
record, so the folded structure is the event route's, not a re-derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from dataclasses import field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantark.cashleg.accrual_leg import AccrualLeg
from quantark.cashleg.autocallable_leg import AutocallableCashLeg
from quantark.cashleg.base import CashLeg
from quantark.cashleg.deterministic_leg import DeterministicLeg
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, almost_equal, is_close, is_non_positive

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


def _autocallable_structure(product, pricing_env) -> _Structure:
    """Snowball / Phoenix: the PDE readout's records, ledger and terminal timing."""
    from quantark.asset.equity.engine.event_stats import (
        AutocallableEventStats,
        payment_aware_cashflow_fields,
    )
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing

    tau = float(product.get_maturity(pricing_env))
    if is_non_positive(tau, tol=0.0):                   # the engines' expired outcome: trivial
        return _trivial_structure(product, pricing_env)
    records = _in_life(product.resolve_ko_observations(pricing_env), tau)
    if not records:
        return _trivial_structure(product, pricing_env)
    terminal = resolve_terminal_timing(product, pricing_env)
    ko_times = np.array([float(rec.observation_time) for rec in records], dtype=float)
    ledger = payment_aware_cashflow_fields(
        pricing_env,
        determination_times=[*ko_times, float(terminal.determination_time)],
        payment_times=[*(_settlement(rec) for rec in records), float(terminal.payment_time)],
        expected_discounted_cashflows=np.zeros(len(records) + 1),
        determination_dates=[*(rec.observation_date for rec in records), terminal.determination_date],
        payment_dates=[*(rec.settlement_date for rec in records), terminal.payment_date],
    )

    def distribution(p_ko, p_no_ki, p_ki):
        stats = AutocallableEventStats(
            pv=0.0,
            ko_times=ko_times,
            ko_probability=np.asarray(p_ko, dtype=float),
            survival_probability=np.maximum(0.0, 1.0 - np.cumsum(p_ko)),
            expected_discounted_ko_cashflow=np.zeros(len(records)),
            ki_probability=float(p_ki),
            expected_discounted_maturity_cashflow=0.0,
            ki_survive_knocked_in_probability=float(p_ki),
            **ledger,
        )
        return EventDistribution.from_autocallable_stats(stats)

    terminal_slot = _Slot("main", float(terminal.determination_time), float(terminal.payment_time))
    return _Structure(
        ko=tuple(_Slot("main", float(rec.observation_time), _settlement(rec)) for rec in records),
        no_ki=terminal_slot,
        ki=terminal_slot,
        distribution=distribution,
    )


def _ko_reset_structure(product, pricing_env) -> _Structure:
    """KO-reset: both schedules on one axis (pre first at a tie) and the two maturity buckets."""
    from quantark.asset.equity.engine.event_stats import (
        KOResetEventStats,
        payment_aware_cashflow_fields,
    )
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.execution.errors import CapabilityError
    from quantark.util.enum import CouponPayType, PostKOScheduleMode

    if product.post_ko_mode == PostKOScheduleMode.REBASED:
        raise CapabilityError(
            "a REBASED KO-reset schedule settles at knock-in-dependent times; its cash legs "
            "cannot be folded (the event route refuses it too)"
        )
    tau = float(product.get_maturity(pricing_env))
    if is_non_positive(tau, tol=0.0):                   # the engines' expired outcome: trivial
        return _trivial_structure(product, pricing_env)
    terminal = resolve_terminal_timing(product, pricing_env)
    expiry = product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY

    def schedule(config):
        resolved, _, _ = product._resolve_ko_schedule(config, pricing_env)
        times, pays = [], []
        for rec in _in_life(resolved, tau):
            times.append(float(rec.observation_time))
            pays.append(float(terminal.payment_time) if expiry else _settlement(rec))
        return np.array(times, dtype=float), np.array(pays, dtype=float)

    pre_t, pre_pay = schedule(product.barrier_config)
    post_t, post_pay = schedule(product.post_barrier_config)
    if pre_t.size + post_t.size == 0:
        return _trivial_structure(product, pricing_env)
    delay = float(terminal.payment_time) - float(terminal.determination_time)
    pre_maturity, post_maturity = float(product.get_pre_maturity_time(pricing_env)), tau
    ledger = payment_aware_cashflow_fields(
        pricing_env,
        determination_times=[*pre_t, *post_t, pre_maturity, post_maturity],
        payment_times=[*pre_pay, *post_pay, pre_maturity + delay, post_maturity + delay],
        expected_discounted_cashflows=np.zeros(pre_t.size + post_t.size + 2),
    )
    times = np.r_[pre_t, post_t]
    tag = np.r_[np.zeros(pre_t.size), np.ones(post_t.size)]
    order = np.lexsort((tag, times))                      # the event route's merged order
    pays = np.r_[pre_pay, post_pay]
    n_pre = pre_t.size

    def distribution(p_ko, p_no_ki, p_ki):
        p = np.empty(order.size)
        p[order] = np.asarray(p_ko, dtype=float)          # back to [pre..., post...]
        p_pre, p_post = p[:n_pre], p[n_pre:]
        stats = KOResetEventStats(
            pv=0.0,
            ko_times=pre_t,
            ko_probability=p_pre,
            survival_probability=np.maximum(0.0, 1.0 - np.cumsum(p_pre)),
            expected_discounted_ko_cashflow=np.zeros(n_pre),
            ki_probability=float(p_ki),
            expected_discounted_maturity_cashflow=0.0,
            ki_survive_knocked_in_probability=float(p_ki),
            pre_ko_times=pre_t,
            pre_ko_probability=p_pre,
            post_ko_times=post_t,
            post_ko_probability=p_post,
            pre_ko_probability_total=float(np.sum(p_pre)),
            post_ko_probability_total=float(np.sum(p_post)),
            pre_ko_payment_times=pre_pay,
            post_ko_payment_times=post_pay,
            pre_maturity_probability=float(p_no_ki),
            post_maturity_probability=float(p_ki),
            pre_maturity_time=pre_maturity,
            post_maturity_time=post_maturity,
            pre_maturity_payment_time=pre_maturity + delay,
            post_maturity_payment_time=post_maturity + delay,
            **ledger,
        )
        return EventDistribution.from_autocallable_stats(stats)

    return _Structure(
        ko=tuple(
            _Slot("pre" if tag[i] == 0 else "post", float(times[i]), float(pays[i])) for i in order
        ),
        no_ki=_Slot("pre", pre_maturity, pre_maturity + delay),
        ki=_Slot("post", post_maturity, post_maturity + delay),
        distribution=distribution,
    )


def _dcn_structure(product, pricing_env) -> _Structure:
    """DCN: monthly KO observations and the loss-leg settlement, as ``DCNMCEngine``'s ledger lays them out."""
    from quantark.asset.equity.engine.event_stats import DCNEventStats, payment_aware_cashflow_fields
    from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context

    ctx = build_dcn_grid_context(product)
    obs = np.asarray(ctx.times[ctx.obs_cols], dtype=float)
    n = obs.size
    if n == 0:
        return _trivial_structure(product, pricing_env)
    observations = product.schedule.monthly
    pay = product.payment_dates()
    ledger = payment_aware_cashflow_fields(
        pricing_env,
        determination_times=np.concatenate([obs, obs, [obs[-1]]]),
        payment_times=np.concatenate([ctx.coupon_pay_times, ctx.ko_pay_times, [ctx.loss_pay_time]]),
        expected_discounted_cashflows=np.zeros(2 * n + 1),
        determination_dates=[*(m.observation_date for m in observations),
                             *(m.observation_date for m in observations),
                             observations[-1].observation_date],
        payment_dates=[*pay.coupon_payment_dates, *pay.ko_payment_dates, pay.loss_payment_date],
    )

    def distribution(p_ko, p_no_ki, p_ki):
        stats = DCNEventStats(
            ki_probability=float(p_ki),
            ko_probability=float(np.sum(p_ko)),
            ko_timing_distribution=tuple(float(x) for x in p_ko),
            coupon_probability=tuple(0.0 for _ in range(n)),
            expected_life_years=0.0,
            prob_survive_no_ki=float(p_no_ki),
            prob_survive_ki=float(p_ki),
            expected_discounted_loss_leg=0.0,
            **ledger,
        )
        return EventDistribution.from_dcn_stats(stats)

    maturity = _Slot("main", float(obs[-1]), float(ctx.loss_pay_time))
    return _Structure(
        ko=tuple(_Slot("main", float(t), float(p)) for t, p in zip(obs, ctx.ko_pay_times)),
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
