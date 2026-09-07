"""Semantic identities: calendars, contracts (rolled in time), lifecycle states.

contract_fingerprint excludes the fields the lifecycle trackers change day to
day (spec §5.3); lifecycle_fingerprint is generic over every dataclass field
of a state so a new pricing-relevant field can never be missed (spec §8).

Normalisation is TYPE-TAGGED: ``1``, ``1.0``, ``True`` and an ``IntEnum``
member with value 1 are four different contract terms, and a mapping keyed by
``1`` differs from one keyed by ``"1"``. Values of a type this module cannot
serialise canonically are rejected (``ValidationError``) rather than fingerprinted
by ``repr``, whose default form carries object addresses.
"""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from quantark.asset.equity.lifecycle.cashflows import LifecycleCashflowLedger, ValuationPoint
from quantark.util.calendar.business_calendar import Calendar
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

LIFECYCLE_FINGERPRINT_VERSION = "v2"    # v2: type-tagged normalisation
ROLL_TOL = 1e-12        # spec §5.3 roll equation tolerance
ROLLED_FIELDS = frozenset({"maturity", "_otc_lifecycle_knocked_in"})
SCHEDULE_FIELDS = frozenset({
    "barrier_config", "post_barrier_config", "observation_schedule",
    "coupon_schedule", "ko_observation_schedule", "ki_observation_schedule",
})
_TIMING_TOKENS = ("date", "time")      # leaf names the trackers advance daily
MATURITY_FLOOR = 1e-8   # the trackers clamp a rolled float maturity here


def calendars_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if type(a) is not type(b):
        return False
    if getattr(a, "name", None) != getattr(b, "name", None):
        return False
    return set(getattr(a, "holidays", ())) == set(getattr(b, "holidays", ()))


def engines_equivalent(a: Any, b: Any) -> bool:
    """Identity, or the same class with equal, normalisable model fingerprints (patch spec §7).

    Engines without a fingerprint (``model_fingerprint()`` is None: MC, PDE, QUAD,
    vol-model engines) compare by identity; a fingerprint the normaliser cannot
    represent counts as NOT equivalent. Never raises.
    """
    if a is b:
        return True
    if a is None or b is None or type(a) is not type(b):
        return False
    fa, fb = getattr(a, "model_fingerprint", None), getattr(b, "model_fingerprint", None)
    if not callable(fa) or not callable(fb):
        return False
    try:
        # model_fingerprint() reads each declared attribute directly, so a subclass that
        # declares MODEL_FINGERPRINT_ATTRS without setting one raises AttributeError here;
        # an unrepresentable value raises ValidationError in _normalize. Neither may turn a
        # comparison into a hard failure (Kimi review 2026-09-03).
        ra, rb = fa(), fb()
        if ra is None or rb is None:
            return False
        return _normalize(ra) == _normalize(rb)
    except (AttributeError, ValidationError):
        return False


def _sorted_items(pairs: Any) -> tuple:
    """Deterministic order for normalised (key, value) pairs of mixed key types."""
    return tuple(sorted(pairs, key=repr))


def _normalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):             # before the primitives: IntEnum / str-mixin members ARE ints / strs
        return ("enum", type(value).__name__, _normalize(value.value))
    if isinstance(value, bool):             # before int: bool is an int subclass
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, datetime):
        return ("datetime", value.isoformat())
    if isinstance(value, date):
        return ("date", value.isoformat())
    if isinstance(value, timedelta):
        return ("timedelta", value.total_seconds())
    if isinstance(value, ValuationPoint):
        return ("point", "date", value.date.isoformat()) if value.date is not None \
            else ("point", "time", float(value.time))
    if isinstance(value, LifecycleCashflowLedger):
        return ("ledger", tuple(
            (cf.cashflow_id, cf.event_type.value, float(cf.amount),
             cf.determination_date.isoformat() if cf.determination_date is not None else cf.determination_time,
             cf.payment_date.isoformat() if cf.payment_date is not None else cf.payment_time,
             _sorted_items((_normalize(k), _normalize(v)) for k, v in cf.metadata.items()))
            for cf in value.cashflows
        ))
    if isinstance(value, Calendar):
        return ("calendar", type(value).__name__, value.name,
                tuple(sorted((_normalize(d) for d in value.holidays), key=repr)),
                tuple(sorted(int(d) for d in value.weekend_days)))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted((_normalize(v) for v in value), key=repr)))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_normalize(v) for v in value))
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, tuple(_normalize(v) for v in value.ravel().tolist()))
    if isinstance(value, Mapping):
        return ("map", _sorted_items((_normalize(k), _normalize(v)) for k, v in value.items()))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, tuple(
            (f.name, _normalize(getattr(value, f.name))) for f in dataclasses.fields(value)
        ))
    raise ValidationError(
        f"cannot fingerprint a {type(value).__module__}.{type(value).__name__} value: "
        "contract and lifecycle identity need a canonical normalisation, not repr()"
    )


def _public_fields(obj: Any) -> Tuple[Tuple[str, Any], ...]:
    """Dataclass fields plus every public instance attribute, by sorted name.

    Several products are decorated ``@dataclass`` without declaring annotated
    fields of their own (the contract terms are plain attributes set by the
    base class ``__init__``), so ``dataclasses.fields`` alone can be empty.
    Names are sorted: attribute insertion order is not contract semantics.
    """
    names = set()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        names.update(f.name for f in dataclasses.fields(obj))
    names.update(k for k in vars(obj) if not k.startswith("_") or k in ROLLED_FIELDS)
    return tuple((n, getattr(obj, n)) for n in sorted(names))


def _timing_name(name: str) -> bool:
    return any(tok in name for tok in _TIMING_TOKENS)


def _static_terms(value: Any) -> Any:
    """Schedule-bearing config -> its static terms, recursively.

    Only LEAF fields whose name marks them as timing (``observation_time``,
    ``settlement_date``, ``ko_observation_dates``, ``time_shift`` ...) are
    dropped: the trackers advance those every day. Containers are always
    entered, whatever they are called, so the per-observation barriers,
    payoffs and return rates inside a schedule's records stay part of the
    contract's identity.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, tuple(
            (f.name, _static_terms(getattr(value, f.name)))
            for f in dataclasses.fields(value)
            if not _timing_name(f.name)
        ))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_static_terms(v) for v in value))
    if isinstance(value, Mapping):
        return ("map", _sorted_items((_normalize(k), _static_terms(v)) for k, v in value.items()))
    return _normalize(value)


def contract_fingerprint(product: Any) -> tuple:
    items = []
    for name, val in _public_fields(product):
        if name in ROLLED_FIELDS:
            continue
        items.append((name, _static_terms(val) if name in SCHEDULE_FIELDS else _normalize(val)))
    return (type(product).__name__, tuple(items))


def _schedule_node(node: Any) -> bool:
    return isinstance(node, tuple) and len(node) == 2 and node[0] == "ObservationSchedule"


def _rolled_equal(a: Any, b: Any) -> bool:
    """Fingerprint equality where `b` may be `a` rolled forward in time.

    A rolled observation schedule keeps its non-record terms and its records
    are a SUFFIX of the original's (the shifter drops observations that have
    passed); a schedule whose observations have all passed becomes ``None``.
    Everything else must match exactly.
    """
    if _schedule_node(a):
        if b is None:
            return True                              # every observation has passed
        if not _schedule_node(b):
            return False
        fa, fb = dict(a[1]), dict(b[1])
        if set(fa) != set(fb):
            return False
        ra, rb = fa.pop("records"), fb.pop("records")
        if fa != fb:
            return False
        recs_a, recs_b = ra[1], rb[1]                # ("seq", (...))
        return len(recs_b) <= len(recs_a) and recs_a[len(recs_a) - len(recs_b):] == recs_b
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(_rolled_equal(x, y) for x, y in zip(a, b))
    return a == b


def _float_maturity(value: Any, what: str) -> float:
    try:
        m = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{what} product maturity must be a number, got {value!r}") from None
    if not np.isfinite(m):
        raise ValidationError(f"{what} product maturity must be finite, got {value!r}")
    return m


def check_contract_roll(
    product_t0: Any,
    product_alive_t1: Any,
    calendar_days: int,
    *,
    expected_decrement: Optional[float] = None,
) -> None:
    """Raise unless product_alive_t1 is product_t0 rolled forward calendar_days (spec §5.3).

    A float maturity must equal ``max(MATURITY_FLOOR, m0 - decrement)`` within
    ROLL_TOL, where the decrement defaults to ``calendar_days/365``: the
    trackers' floor is only reached when the roll would cross it, so a floored
    maturity is validated, not waved through. ``calendar_days == 0`` declares an
    unrolled contract (same maturity, same schedules); otherwise a schedule may
    have lost the observations that passed inside the step (its records are a
    suffix of the t0 records).

    ``expected_decrement`` overrides the calendar-year default for a contract
    whose float maturity is quoted on another clock. A trading-day contract ages
    by (trading days elapsed) / (days in a trading year), which is a different
    number from calendar days / 365, and the caller is the only one who knows
    the clock: a float maturity carries none of its own (see
    ``PricingEnvironment``, whose convention a float maturity bypasses).
    """
    fp0, fp1 = contract_fingerprint(product_t0), contract_fingerprint(product_alive_t1)
    same = fp0 == fp1 if int(calendar_days) == 0 else _rolled_equal(fp0, fp1)
    if not same:
        raise ValidationError("contract replacement is not a time step")
    date_based = getattr(product_t0, "exercise_date", None) is not None \
        or getattr(product_t0, "maturity_date", None) is not None
    if date_based:
        return                      # dates are in the fingerprint; the float is derived metadata
    m0 = getattr(product_t0, "maturity", None)
    m1 = getattr(product_alive_t1, "maturity", None)
    if m0 is None and m1 is None:
        return                      # no expiry on either side (spot, perpetual)
    if m0 is None or m1 is None:
        raise ValidationError(
            "alive product switches between an expiring and a non-expiring contract"
        )
    m0 = _float_maturity(m0, "t0")
    m1 = _float_maturity(m1, "alive-at-t1")
    days = int(calendar_days)
    if days < 0:
        raise ValidationError(f"calendar_days must be non-negative, got {calendar_days}")
    decrement = days / 365.0 if expected_decrement is None else float(expected_decrement)
    if decrement < 0.0:
        raise ValidationError(f"expected_decrement must be non-negative, got {decrement}")
    expected = m0 if days == 0 else max(MATURITY_FLOOR, m0 - decrement)
    if not is_close(m1, expected, rel_tol=0.0, abs_tol=ROLL_TOL):
        basis = ("calendar_days/365" if expected_decrement is None
                 else f"the environment's own clock ({decrement:.10g} of a year)")
        raise ValidationError(
            f"alive product maturity {m1} is not {m0} rolled by {days} days (expected {expected}; "
            f"a float-maturity contract must be supplied rolled by {basis}, "
            f"floored at {MATURITY_FLOOR})"
        )


BOOKKEEPING_FIELDS = frozenset({
    "observed_ko_indices", "observed_ki_indices", "observed_coupon_indices",
    "valuation_point", "pending_settlement_cashflow", "settled",
})   # change without an event; pricing-neutral (spec §8).
# ``valuation_point`` is the tracker's last-observation clock stamp: the snapshot
# carries its own valuation point, so the stamp is bookkeeping, not contract state.
# ``pending_settlement_cashflow`` / ``settled`` mirror the ledger for the replay
# engine and flip when a receivable is PAID (``state.settle()``), which is a
# time-step fact read from the ledger, not a lifecycle event.


def lifecycle_fingerprint(state: Any) -> tuple:
    if state is None:
        return (LIFECYCLE_FINGERPRINT_VERSION, None)
    if not dataclasses.is_dataclass(state):
        raise ValidationError(f"lifecycle state must be a dataclass, got {type(state).__name__}")
    fields = tuple(
        (f.name, _normalize(getattr(state, f.name)))
        for f in dataclasses.fields(state) if f.name not in BOOKKEEPING_FIELDS
    )
    return (LIFECYCLE_FINGERPRINT_VERSION, type(state).__name__, fields)
