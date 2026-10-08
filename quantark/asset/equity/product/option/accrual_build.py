"""Turning a contract's accrual terms into periods at construction (accrual time spec §4.2–4.7).

Every derived period records the terms it came from (``AccrualPeriod.terms``). A product that is built again
(``dataclasses.replace``, ``validate()`` after a change) keeps a derived period while those terms are unchanged, so
ageing and snapping never move cash, and derives it anew from the current terms when they changed. A period the
caller set explicitly (``terms is None``) is always kept.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Dict, List, Optional

from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.calendar import BusinessDayConvention
from quantark.util.enum import AccrualEndType, CouponPayType, PostKOScheduleMode
from quantark.util.exceptions import ValidationError

from .accrual import BUSINESS_DAYS_TERMS, AccrualBasis, AccrualKind, AccrualPeriod
from .observation_schedule import ObservationRecord, ObservationSchedule

R5 = ("a dated contract whose annualized cash needs an accrual start needs initial_date (accrual time spec R5): "
      "its accrual would otherwise start 'today'")


def _detach(schedule: Optional[ObservationSchedule]) -> Optional[ObservationSchedule]:
    """A copy whose records are fresh copies: periods derived for this product never reach another (§4.2)."""
    if schedule is None:
        return None
    return replace(schedule, records=[replace(r) for r in schedule.records])


def _calendar_key(calendar):
    """``calendar.identity()``, cached on the calendar: it sorts every holiday, and the stale-terms check reads it
    on each resolver call. The cache is refreshed when the holiday count or the weekend changes."""
    if calendar is None:
        return None
    stamp = (len(calendar.holidays), tuple(sorted(calendar.weekend_days)))
    cached = getattr(calendar, "_accrual_identity_cache", None)
    if cached is None or cached[0] != stamp:
        cached = (stamp, calendar.identity())
        calendar._accrual_identity_cache = cached
    return cached[1]


def same_terms(period: AccrualPeriod, derived: tuple, current: tuple) -> bool:
    """Whether ``period``, derived under ``derived``, is still the period of a contract with terms ``current``.
    ``initial_date`` (the keys' first term) is the origin of DATE periods and decides NONE ones; a TIME or
    relative period never reads it, so setting it on a numeric contract changes none of them."""
    if period.kind is AccrualKind.TIME:
        return derived[1:] == current[1:]
    return derived == current


def accrual_terms_key(product) -> tuple:
    """The contract terms every derived period depends on, factors aside (ageing slices those by position): a
    period derived under another key is stale. ``initial_date`` comes first (see ``same_terms``)."""
    ac = product.accrual_config
    coupon = getattr(product, "coupon_config", None)
    convention = product.settlement_convention
    settlement = ac.accrual_end is AccrualEndType.SETTLEMENT
    return (
        product.initial_date, product.annualization_day_count.name,
        None if coupon is None else coupon.day_count_convention.name,
        None if coupon is None else coupon.fixed_coupon_year_fraction, ac.accrual_end.name,
        None if not settlement or convention is None else (
            convention.lag, convention.lag_unit.name, convention.business_day_convention.name,
            _calendar_key(convention.calendar)),
        _calendar_key(ac.accrual_calendar), ac.accrual_bus_days_in_year,
    )


class _Terms:
    """The contract terms every record's period is derived from."""

    def __init__(self, product):
        ac = product.accrual_config
        self.product = product
        self.accrued_offset = float(ac.accrued_offset)
        self.first_period_start = ac.first_period_start
        self.settlement = ac.accrual_end is AccrualEndType.SETTLEMENT
        self.initial_date = product.initial_date
        self.basis = AccrualBasis(product.annualization_day_count, ac.accrual_calendar, ac.accrual_bus_days_in_year)
        coupon = getattr(product, "coupon_config", None)
        self.period_basis = (self.basis if coupon is None else
                             AccrualBasis(coupon.day_count_convention, ac.accrual_calendar, ac.accrual_bus_days_in_year))
        self.ko_annualized = bool(product._effective_annualized_flag(ac.is_annualized_ko))
        self.coupon_annualized = coupon is not None and bool(product.is_coupon_rate_annualized)
        self.fixed = None if coupon is None else coupon.fixed_coupon_year_fraction
        if self.settlement and (ac.coupon_pay_type is CouponPayType.EXPIRY
                                or getattr(coupon, "coupon_pay_type", None) is CouponPayType.EXPIRY):
            raise ValidationError("an EXPIRY coupon under ACCRUAL_END_TYPE=SETTLEMENT: its accrual end is a "
                                  "term-sheet choice the contract must state (accrual time spec §4.5)")
        self.key = accrual_terms_key(product)

    # -- endpoints -----------------------------------------------------------------------------------------
    def fixing(self, rec: ObservationRecord):
        """(kind, origin, end) of a record's contractual fixing."""
        if rec.observation_date is not None:
            if self.initial_date is None:
                return AccrualKind.NONE, None, None
            return AccrualKind.DATE, self.initial_date, rec.observation_date
        return AccrualKind.TIME, 0.0, self.accrued_offset + float(rec.observation_time)

    def payment(self, rec: ObservationRecord):
        """(kind, origin, end) of a record's authoritative settlement endpoint (§4.5), own terms only."""
        kind, origin, end = self.fixing(rec)
        if kind is AccrualKind.NONE:
            return kind, origin, end
        raw = settlement_endpoint(self.product.settlement_convention, rec)
        return kind, origin, raw if kind is AccrualKind.DATE else self.accrued_offset + raw

    def end_of(self, rec: ObservationRecord):
        return self.payment(rec) if self.settlement else self.fixing(rec)

    def basis_for(self, kind: AccrualKind) -> Optional[AccrualBasis]:
        return None if kind is AccrualKind.NONE else self.basis

    # -- refusals ------------------------------------------------------------------------------------------
    def require(self, kind: AccrualKind, cumulative_needed: bool, coupon_needed: bool = False) -> None:
        if kind is AccrualKind.NONE and (cumulative_needed or coupon_needed):
            raise ValidationError(R5)
        if kind is AccrualKind.DATE and cumulative_needed and not self.basis.is_complete:
            raise ValidationError(BUSINESS_DAYS_TERMS)


def settlement_endpoint(convention: Optional[SettlementConvention], rec: ObservationRecord):
    """The settlement resolver's precedence, from the product's own terms (no environment), §4.5."""
    if rec.observation_date is not None:
        if rec.settlement_date is not None:
            return rec.settlement_date
        if rec.settlement_time is not None:
            raise ValidationError("a dated record's SETTLEMENT accrual needs a settlement_date, not a settlement_time")
        if convention is None or convention.lag == 0.0:
            return rec.observation_date
        unit, bdc, cal = convention.lag_unit, convention.business_day_convention, convention.calendar
        if unit is SettlementLagUnit.BUSINESS_DAYS:
            if cal is None:
                raise ValidationError("SETTLEMENT accrual on a business-day lag needs the settlement convention's own "
                                      "calendar (construction has no environment)")
            return cal.adjust_date(cal.add_business_days(rec.observation_date, int(convention.lag)), bdc)
        if unit is SettlementLagUnit.CALENDAR_DAYS:
            day = rec.observation_date + timedelta(days=int(convention.lag))
            if bdc is BusinessDayConvention.UNADJUSTED:
                return day
            if cal is None:
                raise ValidationError("SETTLEMENT accrual on an adjusted calendar-day lag needs the settlement "
                                      "convention's own calendar")
            return cal.adjust_date(day, bdc)
        raise ValidationError("a YEAR_FRACTION settlement lag cannot end a dated record's accrual")
    if rec.settlement_date is not None:
        raise ValidationError("a numeric record's SETTLEMENT accrual needs a settlement_time, not a settlement_date")
    if rec.settlement_time is not None:
        return float(rec.settlement_time)
    if convention is None or convention.lag == 0.0:
        return float(rec.observation_time)
    if convention.lag_unit is SettlementLagUnit.YEAR_FRACTION:
        return float(rec.observation_time) + float(convention.lag)
    raise ValidationError(f"a {convention.lag_unit.name} settlement lag cannot end a numeric record's accrual")


def _keep(rec: ObservationRecord, key: tuple, end=None) -> bool:
    """Whether a record keeps the period it carries (§4.2): an explicit one always; a derived one while the terms it
    was derived from are the contract's current terms and, for a DATE period, while ``end`` (its record's current
    endpoint) is still its end. Ageing and snapping never move a date, so only a corrected record fails that."""
    period = rec.accrual
    if period is None:
        return False
    if period.terms is None:
        return True
    if not same_terms(period, period.terms, key):
        return False
    return not (period.kind is AccrualKind.DATE and end is not None and period.end != end)


def _check_length(factors, records, name: str) -> None:
    if factors is not None and len(factors) != len(records):
        raise ValidationError(f"{name} length ({len(factors)}) must match its schedule's {len(records)} records")


def _fill_ko(records: List[ObservationRecord], terms: _Terms, factors, name: str, *, coupon: bool,
             factor_sets_coupon: bool) -> None:
    _check_length(factors, records, name)
    previous = None
    rederived = False
    for i, rec in enumerate(records):
        cumulative = None if factors is None else float(factors[i])
        key = terms.key + (name, cumulative)
        kind, origin, end = terms.end_of(rec)
        if not rederived and _keep(rec, key, end):
            previous = rec.accrual
            continue
        rederived = True   # the periods after a re-derived one chain from it
        old = rec.accrual
        period_fraction = cumulative if factor_sets_coupon else None
        if coupon and period_fraction is None:
            period_fraction = terms.fixed
        start = None
        if kind is not AccrualKind.NONE:
            if previous is None:
                if (old is not None and old.terms is not None and old.kind is kind and not old.relative
                        and old.start is not None and old.start != old.origin):
                    start = old.start   # an aged schedule's first remaining period keeps its contractual start
                else:
                    start = (terms.initial_date if kind is AccrualKind.DATE else
                             (None if terms.first_period_start is None else float(terms.first_period_start)))
            elif previous.kind is kind and not previous.relative:
                start = previous.end
            if start is not None and end < start:
                start = None   # a past record before the contract's time origin: its period start is unknown
        terms.require(kind, terms.ko_annualized and cumulative is None,
                      coupon and terms.coupon_annualized and period_fraction is None)
        rec.accrual = AccrualPeriod(kind, origin=origin, start=start, end=end, basis=terms.basis_for(kind),
                                    period_basis=None if kind is AccrualKind.NONE else terms.period_basis,
                                    cumulative_fraction=cumulative, period_fraction=period_fraction, terms=key)
        previous = rec.accrual


def _fill_ki(records: List[ObservationRecord], terms: _Terms, anchor_needed: bool) -> None:
    """Knock-in records accrue to their contractual fixing, always (§4.3): the end is a REBASED anchor only."""
    key = terms.key + ("knock-in",)
    for rec in records:
        kind, origin, end = terms.fixing(rec)
        if _keep(rec, key, end):
            continue
        terms.require(kind, anchor_needed)
        rec.accrual = AccrualPeriod(kind, origin=origin, end=end, basis=terms.basis_for(kind), terms=key)


def _fill_rebased(records: List[ObservationRecord], terms: _Terms, factors) -> None:
    """REBASED schedule 2: quoted offsets from the knock-in, its own rule (§4.3)."""
    _check_length(factors, records, "post_accrual_factors")
    previous_end = 0.0
    for i, rec in enumerate(records):
        factor = None if factors is None else float(factors[i])
        key = terms.key + ("rebased", factor)
        if _keep(rec, key):
            previous_end = rec.accrual.end
            continue
        if factor is not None:
            end = factor
        elif terms.settlement:
            end = settlement_endpoint(terms.product.settlement_convention, rec)
        else:
            end = float(rec.observation_time)
        rec.accrual = AccrualPeriod(AccrualKind.TIME, start=min(previous_end, end), end=end, basis=terms.basis,
                                    relative=True, terms=key)
        previous_end = end


def _schedule_terminal(records: List[ObservationRecord], terms: _Terms) -> Optional[AccrualPeriod]:
    """Origin to a schedule's last contractual fixing (KO-reset ``pre``, ABSOLUTE ``post``), §4.4."""
    if not records:
        return None
    dated = [r.observation_date for r in records if r.observation_date is not None]
    if dated:
        if terms.initial_date is None:
            return AccrualPeriod(AccrualKind.NONE)
        return AccrualPeriod(AccrualKind.DATE, origin=terms.initial_date, end=max(dated), basis=terms.basis)
    return AccrualPeriod(AccrualKind.TIME, origin=0.0, basis=terms.basis,
                         end=terms.accrued_offset + max(float(r.observation_time) for r in records))


def _contract_terminal(product, terms: _Terms) -> AccrualPeriod:
    """``BaseEquityOption.get_tenor``'s rule, frozen (§4.4)."""
    if product.tenor is not None:
        return AccrualPeriod(AccrualKind.NONE, cumulative_fraction=float(product.tenor))
    if product.initial_date is not None and product.get_tenor_end_date() is not None:
        return AccrualPeriod(AccrualKind.DATE, origin=product.initial_date, end=product.get_tenor_end_date(),
                             basis=terms.basis)
    if product.maturity is not None:
        return AccrualPeriod(AccrualKind.TIME, origin=0.0, end=terms.accrued_offset + float(product.maturity),
                             basis=terms.basis)
    return AccrualPeriod(AccrualKind.NONE)


def _terminals(product, terms: _Terms, stored: Optional[Dict[str, AccrualPeriod]]) -> Dict[str, AccrualPeriod]:
    """The terminal periods (§4.4). A stored terminal is kept while the one derived from today's terms agrees with
    it, and when its schedule has aged out (nothing left to derive it from); otherwise today's replaces it."""
    ac = product.accrual_config
    stored = stored or {}

    def settle(leg: str, candidate: Optional[AccrualPeriod]) -> Optional[AccrualPeriod]:
        old = stored.get(leg)
        if candidate is None:
            return old
        return old if old is not None and _same_terminal(old, candidate) else candidate

    out = {"contract": settle("contract", _contract_terminal(product, terms))}
    post_config = getattr(product, "post_barrier_config", None)
    if post_config is not None:
        out["pre"] = settle("pre", _schedule_terminal(product.barrier_config.ko_observation_schedule.records, terms)) \
            or out["contract"]
        post_records = post_config.ko_observation_schedule.records
        if product.post_ko_mode is PostKOScheduleMode.REBASED:
            candidate = None if not post_records else AccrualPeriod(
                AccrualKind.TIME, start=0.0, end=max(float(r.observation_time) for r in post_records),
                basis=terms.basis, relative=True)
            out["post"] = settle("post", candidate) or AccrualPeriod(
                AccrualKind.TIME, start=0.0, end=0.0, basis=terms.basis, relative=True)
        else:
            out["post"] = settle("post", _schedule_terminal(post_records, terms)) or out["pre"]
    v0 = "pre" if post_config is not None else "contract"
    v1 = "post" if post_config is not None else "contract"
    for leg, needed in ((v0, bool(ac.is_annualized_rebate)), (v1, bool(ac.is_annualized_ki))):
        period = out[leg]
        if period.cumulative_fraction is None and not period.relative:
            terms.require(period.kind, needed)
    return out


def _same_terminal(stored: AccrualPeriod, candidate: AccrualPeriod) -> bool:
    """Whether a stored terminal is still the contract's: the candidate derived from today's terms agrees with it.
    A numeric end is ``accrued_offset + maturity`` (or + the last fixing), which ageing keeps to round-off."""
    if (stored.kind, stored.relative, stored.cumulative_fraction) != (
            candidate.kind, candidate.relative, candidate.cumulative_fraction):
        return False
    if stored.kind is AccrualKind.TIME:
        return (abs(float(stored.end) - float(candidate.end)) <= _TERMINAL_TOL
                and (stored.relative or abs(float(stored.origin) - float(candidate.origin)) <= _TERMINAL_TOL)
                and (stored.basis is None or stored.basis.same_terms(candidate.basis)))
    if stored.kind is AccrualKind.DATE:
        return (stored.origin, stored.end) == (candidate.origin, candidate.end) and \
            stored.basis.same_terms(candidate.basis)
    return True


_TERMINAL_TOL = 1e-9   # years (about 30 ms): ageing round-off, never a contract change


def snapshot_terms(product) -> None:
    """Record the terms the product's periods were derived under, for the resolver's stale-terms check (§4.2)."""
    ac = product.accrual_config
    product._accrual_terms_key = accrual_terms_key(product)
    product._accrual_factor_snapshot = None if ac.accrual_factors is None else tuple(ac.accrual_factors)
    product._post_factor_snapshot = None if ac.post_accrual_factors is None else tuple(ac.post_accrual_factors)
    product._tenor_snapshot = product.tenor


def build_contract_accrual(product) -> None:
    """Detach the product's schedules and give every record its period, and the product its terminal periods
    (spec §4.2). Called last in ``validate()``."""
    terms = _Terms(product)
    ac = product.accrual_config
    bc = product.barrier_config
    ko, ki = _detach(bc.ko_observation_schedule), _detach(bc.ki_observation_schedule)
    phoenix = hasattr(product, "coupon_config")
    if ko is not None:
        _fill_ko(ko.records, terms, ac.accrual_factors, "accrual_factors", coupon=phoenix, factor_sets_coupon=True)
    post_config = getattr(product, "post_barrier_config", None)
    rebased = post_config is not None and product.post_ko_mode is PostKOScheduleMode.REBASED
    if ki is not None:
        # a REBASED contract reads its knock-in's accrual for the knock-out leg and the knocked-in (V1) tenor
        _fill_ki(ki.records, terms,
                 anchor_needed=rebased and (terms.ko_annualized or bool(ac.is_annualized_ki)))
    product.barrier_config = replace(bc, ko_observation_schedule=ko, ki_observation_schedule=ki)
    if post_config is not None:
        post = _detach(post_config.ko_observation_schedule)
        if post is not None:
            if rebased:
                _fill_rebased(post.records, terms, ac.post_accrual_factors)
            else:
                _fill_ko(post.records, terms, ac.post_accrual_factors, "post_accrual_factors", coupon=False,
                         factor_sets_coupon=False)
        product.post_barrier_config = replace(post_config, ko_observation_schedule=post)
    elif ac.post_accrual_factors is not None:
        raise ValidationError("post_accrual_factors belong to a KO-reset contract's schedule 2")
    if not getattr(product, "_terminal_accrual_explicit", False):
        product._terminal_accrual = _terminals(product, terms, getattr(product, "_terminal_accrual", None))
    snapshot_terms(product)
