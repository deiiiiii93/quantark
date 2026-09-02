"""Semantic identities: calendars, contracts (rolled in time), lifecycle states.

contract_fingerprint excludes the fields the lifecycle trackers change day to
day (spec §5.3); lifecycle_fingerprint is generic over every dataclass field
of a state so a new pricing-relevant field can never be missed (spec §8).
"""
from __future__ import annotations

import dataclasses
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping, Tuple

import numpy as np

from quantark.asset.equity.lifecycle.cashflows import LifecycleCashflowLedger, ValuationPoint
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

LIFECYCLE_FINGERPRINT_VERSION = "v1"
ROLL_TOL = 1e-12        # spec §5.3 roll equation tolerance
ROLLED_FIELDS = frozenset({"maturity", "_otc_lifecycle_knocked_in"})
SCHEDULE_FIELDS = frozenset({
    "barrier_config", "post_barrier_config", "observation_schedule",
    "coupon_schedule", "ko_observation_schedule", "ki_observation_schedule",
})
_TIMING_TOKENS = ("date", "time", "schedule", "record")
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


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return ("enum", type(value).__name__, value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, ValuationPoint):
        return ("date", value.date.isoformat()) if value.date is not None else ("time", float(value.time))
    if isinstance(value, LifecycleCashflowLedger):
        return ("ledger", tuple(
            (cf.cashflow_id, cf.event_type.value, float(cf.amount),
             cf.determination_date.isoformat() if cf.determination_date is not None else cf.determination_time,
             cf.payment_date.isoformat() if cf.payment_date is not None else cf.payment_time,
             tuple(sorted((k, _normalize(v)) for k, v in cf.metadata.items())))
            for cf in value.cashflows
        ))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_normalize(v) for v in value))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(v) for v in value)
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, tuple(value.ravel().tolist()))
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _normalize(v)) for k, v in value.items()))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, tuple(
            (f.name, _normalize(getattr(value, f.name))) for f in dataclasses.fields(value)
        ))
    return ("repr", type(value).__name__, repr(value))


def _public_fields(obj: Any) -> Tuple[Tuple[str, Any], ...]:
    """Dataclass fields first, then every public instance attribute.

    Several products are decorated ``@dataclass`` without declaring annotated
    fields of their own (the contract terms are plain attributes set by the
    base class ``__init__``), so ``dataclasses.fields`` alone can be empty.
    """
    names: list = []
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        names = [f.name for f in dataclasses.fields(obj)]
    names += [k for k in vars(obj) if not k.startswith("_") and k not in names]
    names += [k for k in vars(obj) if k in ROLLED_FIELDS and k not in names]
    return tuple((n, getattr(obj, n)) for n in names)


def _static_terms(value: Any) -> Any:
    """Schedule-bearing config -> its static terms (levels, rates, counts, indices)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, tuple(
            (f.name, _normalize(getattr(value, f.name)))
            for f in dataclasses.fields(value)
            if not any(tok in f.name for tok in _TIMING_TOKENS)
        ))
    return _normalize(value)


def contract_fingerprint(product: Any) -> tuple:
    items = []
    for name, val in _public_fields(product):
        if name in ROLLED_FIELDS:
            continue
        items.append((name, _static_terms(val) if name in SCHEDULE_FIELDS else _normalize(val)))
    return (type(product).__name__, tuple(items))


def check_contract_roll(product_t0: Any, product_alive_t1: Any, calendar_days: int) -> None:
    """Raise unless product_alive_t1 is product_t0 rolled forward calendar_days (spec §5.3)."""
    if contract_fingerprint(product_t0) != contract_fingerprint(product_alive_t1):
        raise ValidationError("contract replacement is not a time step")
    m0 = getattr(product_t0, "maturity", None)
    m1 = getattr(product_alive_t1, "maturity", None)
    if m0 is None or m1 is None:
        return
    date_based = getattr(product_t0, "exercise_date", None) is not None \
        or getattr(product_t0, "maturity_date", None) is not None
    if date_based:
        return                      # dates are in the fingerprint; the float is metadata
    m0, m1 = float(m0), float(m1)
    if m1 <= MATURITY_FLOOR + ROLL_TOL:
        return                      # clamped at the trackers' floor
    if not is_close(m0 - m1, calendar_days / 365.0, rel_tol=0.0, abs_tol=ROLL_TOL):
        raise ValidationError(
            f"alive product maturity {m1} is not {m0} rolled by {calendar_days} days "
            "(a float-maturity contract must be supplied rolled by calendar_days/365)"
        )


BOOKKEEPING_FIELDS = frozenset({
    "observed_ko_indices", "observed_ki_indices", "observed_coupon_indices",
})   # grow on every observation date without an event; pricing-neutral (spec §8)


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
