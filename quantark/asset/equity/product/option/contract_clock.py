"""The contract clock: one rule for "the same instant" (spec 2026-11-23-contract-clock-design.md).

Layer 1 -- ``schedule_times`` -- is the one generator every factory uses for evenly spaced numeric schedules. It
computes ``(i + 1) * maturity / n``, which rounds once: whenever ``(i + 1) * maturity`` is exact (every maturity with a
short binary expansion), the result is the correctly rounded value of the exact rational, so two schedules that name
the same calendar instant hold the same float. The old ``(i + 1) / n * maturity`` rounded twice and left 105 such
instants across common tenors one ulp apart (11/12 * 1.0 vs 11/18 * 1.5).

Layer 2 -- ``canonicalize_contract_clock`` -- decides once, at the contract, which of its NUMERIC event times are one
instant, so every consumer (every engine, the tracker, cash legs, replay, theta) reads the same floats and no engine
decides equality for contract times itself. The rule (owner decisions D1-D3, 2026-11-23):

- two times closer than ``Tolerance.ZERO`` (1e-10 y, ~3 ms) are one instant: above float noise, far below the
  intraday clock's one second;
- a run of such times becomes its LAST member (``merge_time_nodes``' rule, the MC grid's since W17a), except that a
  run holding an exact ``0.0`` becomes ``0.0`` (a theta landing, which ``time_shift`` books at exactly zero, keeps its
  settlement and maturity residues AT the valuation instant) and, next, a run within the tolerance of the maturity
  becomes the maturity exactly. A lone positive time, however small, stays ahead: only an exact zero is the valuation
  instant (``at_valuation_instant``, review 2026-09-21 R1);
- a run wider than the tolerance is ambiguous and refused, and so are two distinct observation floats of ONE schedule
  landing on one instant (no engine may pick one record's terms silently). Settlement times merge but never refuse.

Dated records are untouched: they resolve from their dates at pricing time. A REBASED knock-out-reset's schedule 2
holds offsets from the knock-in, a product term, and is not part of the instant set; its absolute times are snapped
onto the contract's instants when they are generated (``snap_to_instants``).
"""
from __future__ import annotations

import dataclasses
import functools
from math import isfinite
from numbers import Real
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, snap_to_anchors

from .observation_schedule import ObservationSchedule

__all__ = [
    "schedule_times",
    "canonicalize_contract_clock",
    "contract_instants",
    "snap_to_instants",
    "canonical_clock_after",
]

#: product-level float lists that hold event times
_PRODUCT_TIME_LISTS = ("observation_dates", "observation_times")
#: product-level lists of observation records that are one schedule (a range accrual's)
_PRODUCT_RECORD_LISTS = ("observation_records",)
#: product-level scalar event times besides the maturity
_PRODUCT_TIME_SCALARS = ("first_schedule_settlement_time",)


def schedule_times(maturity: float, num_observations: int) -> List[float]:
    """Evenly spaced observation times ``(i + 1) * maturity / num_observations``, the last exactly ``maturity``.

    Args:
        maturity: The schedule's horizon in years (positive, finite).
        num_observations: How many observations (at least one).

    Returns:
        ``num_observations`` increasing times in years.

    Raises:
        ValidationError: On a non-positive or non-finite maturity, or fewer than one observation.
    """
    maturity = float(maturity)
    if not isfinite(maturity) or maturity <= 0.0:
        raise ValidationError(f"schedule_times: maturity must be positive and finite, got {maturity}")
    if isinstance(num_observations, bool) or int(num_observations) != num_observations or num_observations < 1:
        raise ValidationError(f"schedule_times: num_observations must be a positive integer, got {num_observations!r}")
    n = int(num_observations)
    times = [(i + 1) * maturity / n for i in range(n)]
    times[-1] = maturity
    return times


# ---------------------------------------------------------------------------------------------------------------
# The instant set
# ---------------------------------------------------------------------------------------------------------------


def _is_time(value: Any) -> bool:
    # ``type is float`` first: the ABC check is the slow path and nearly every time is a plain float
    return type(value) is float or (isinstance(value, Real) and not isinstance(value, bool))


def _time_list(value: Any) -> bool:
    """A non-empty list/tuple of numeric times (not dates, not ``(time, spot)`` fixings)."""
    return isinstance(value, (list, tuple)) and bool(value) and all(_is_time(v) for v in value)


def _record_list(value: Any) -> bool:
    """A non-empty list of observation records (objects carrying an ``observation_time``)."""
    return isinstance(value, (list, tuple)) and bool(value) and all(hasattr(r, "observation_time") for r in value)


def _rebased_offsets(product: Any, config_name: str) -> bool:
    mode = getattr(product, "post_ko_mode", None)
    return config_name == "post_barrier_config" and getattr(mode, "name", None) == "REBASED"


#: a source of times: its path from the product (attribute, or attribute + config field) and its value
_Source = Tuple[Tuple[str, ...], Any]


def _sources(product: Any) -> Tuple[List[_Source], List[_Source]]:
    """The product's observation schedules and time lists, each with its path: ``(attr,)`` or ``(config, field)``."""
    schedules: List[_Source] = []
    lists: List[_Source] = []
    for name, value in vars(product).items():
        if name.startswith("_") or value is None:
            continue
        if isinstance(value, ObservationSchedule):
            schedules.append(((name,), value))
        elif name in _PRODUCT_RECORD_LISTS and _record_list(value):
            schedules.append(((name,), value))
        elif name in _PRODUCT_TIME_LISTS and _time_list(value):
            lists.append(((name,), value))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            if _rebased_offsets(product, name):
                continue                      # REBASED schedule 2: offsets from the knock-in, a product term
            for f in dataclasses.fields(value):
                member = getattr(value, f.name, None)
                if isinstance(member, ObservationSchedule):
                    schedules.append(((name, f.name), member))
                elif f.name.endswith("_observation_dates") and _time_list(member):
                    lists.append(((name, f.name), member))
    return schedules, lists


def _numeric_records(schedule: Any):
    """The schedule's records (an ``ObservationSchedule``, or a product's list of records) on the numeric clock (a
    dated record resolves from its date)."""
    records = schedule if isinstance(schedule, (list, tuple)) else schedule.records
    return [r for r in records if getattr(r, "observation_date", None) is None]


def _numeric_maturity(product: Any) -> Optional[float]:
    if getattr(product, "exercise_date", None) is not None:
        return None
    maturity = getattr(product, "maturity", None)
    return float(maturity) if _is_time(maturity) else None


def _gather(product: Any) -> Tuple[List[float], List[_Source], List[_Source], Optional[float]]:
    schedules, lists = _sources(product)
    values: List[float] = []
    for _, schedule in schedules:
        for record in _numeric_records(schedule):
            for name in ("observation_time", "settlement_time"):
                t = getattr(record, name, None)
                if _is_time(t):
                    values.append(float(t))
    for _, times in lists:
        values.extend(float(t) for t in times)
    maturity = _numeric_maturity(product)
    if maturity is not None:
        values.append(maturity)
    for name in _PRODUCT_TIME_SCALARS:
        t = getattr(product, name, None)
        if _is_time(t):
            values.append(float(t))
    return values, schedules, lists, maturity


# ---------------------------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------------------------


def _canonical_map(
    values: Iterable[float], maturity: Optional[float], tol: float = Tolerance.ZERO
) -> Optional[Dict[float, float]]:
    """Each time's canonical float, or ``None`` when every time already is one (nothing to move).

    Raises:
        ValidationError: A run of times, each step below ``tol``, spans ``tol`` or more.
    """
    u = np.unique(np.asarray(list(values), dtype=float))
    if u.size == 0:
        return None
    close = np.diff(u) < tol
    has_zero = bool(np.any(u == 0.0))
    near_zero = (np.abs(u) < tol) & (u != 0.0) & has_zero
    near_maturity = (
        (np.abs(u - maturity) < tol) & (u != maturity) if maturity is not None else np.zeros(u.size, dtype=bool)
    )
    if not close.any() and not near_zero.any() and not near_maturity.any():
        return None                           # the common case: every time is its own instant
    mapping: Dict[float, float] = {}
    start = 0
    for end in range(u.size):
        if end + 1 < u.size and close[end]:
            continue
        run = u[start:end + 1]
        if run[-1] - run[0] >= tol:
            raise ValidationError(
                f"contract clock: the times {[float(t) for t in run]} chain into a run wider than the "
                f"{tol:g}-year tolerance; which instant they are is ambiguous -- book them as one time or apart"
            )
        if np.any(run == 0.0):
            rep = 0.0
        elif maturity is not None and np.any(np.abs(run - maturity) < tol):
            rep = float(maturity)
        else:
            rep = float(run[-1])
        for t in run:
            mapping[float(t)] = rep
        start = end + 1
    if all(k == v for k, v in mapping.items()):
        return None
    return mapping


def _refuse_same_schedule_collisions(schedules: Sequence[_Source], mapping: Dict[float, float]) -> None:
    for path, schedule in schedules:
        seen: Dict[float, float] = {}
        for record in _numeric_records(schedule):
            t = getattr(record, "observation_time", None)
            if not _is_time(t):
                continue
            t = float(t)
            rep = mapping.get(t, t)
            other = seen.setdefault(rep, t)
            if other != t:
                raise ValidationError(
                    f"contract clock: two observations of {'.'.join(path)} are one instant ({other!r} and {t!r} are "
                    f"closer than {Tolerance.ZERO:g} years); an observation schedule may not observe one instant twice"
                )


# ---------------------------------------------------------------------------------------------------------------
# Rebinding (never in place: QUAD V2 validates a shallow copy and relies on the original staying untouched)
# ---------------------------------------------------------------------------------------------------------------


def _moved(t: Any, mapping: Dict[float, float]) -> Any:
    return mapping.get(float(t), t) if _is_time(t) else t


def _canonical_schedule(schedule: Any, mapping: Dict[float, float]) -> Any:
    records = []
    changed = False
    for record in (schedule if isinstance(schedule, (list, tuple)) else schedule.records):
        if getattr(record, "observation_date", None) is None:
            changes = {
                name: _moved(getattr(record, name), mapping)
                for name in ("observation_time", "settlement_time")
                if _is_time(getattr(record, name, None)) and _moved(getattr(record, name), mapping) != getattr(record, name)
            }
            if changes:
                record = dataclasses.replace(record, **changes)
                changed = True
        records.append(record)
    if not changed:
        return schedule
    if isinstance(schedule, (list, tuple)):
        return type(schedule)(records)
    return dataclasses.replace(schedule, records=records)


def _canonical_list(times: Sequence[float], mapping: Dict[float, float]) -> Sequence[float]:
    out = [_moved(t, mapping) for t in times]
    if all(a == b for a, b in zip(out, times)):
        return times
    return type(times)(out) if isinstance(times, tuple) else out


def canonicalize_contract_clock(product: Any) -> bool:
    """Snap the contract's numeric event times onto one float per instant (module docstring), in place.

    Configs, schedules, records and time lists that change are REBUILT and rebound on ``product``; nothing shared is
    written. A contract whose every time is already its own instant is left untouched (nothing is rebound).

    Returns:
        Whether any time moved. A second call returns ``False``.

    Raises:
        ValidationError: Two distinct observation times of one schedule are one instant, or a run of times is wider
            than the tolerance.
    """
    offsets_moved = _canonicalize_rebased_offsets(product)
    values, schedules, lists, maturity = _gather(product)
    mapping = _canonical_map(values, maturity)
    if mapping is None:
        return offsets_moved
    _refuse_same_schedule_collisions(schedules, mapping)

    product_changes: Dict[str, Any] = {}
    config_changes: Dict[str, Dict[str, Any]] = {}

    def record_change(path: Tuple[str, ...], old: Any, new: Any) -> None:
        if new is old:
            return
        if len(path) == 1:
            product_changes[path[0]] = new
        else:
            config_changes.setdefault(path[0], {})[path[1]] = new

    for path, schedule in schedules:
        record_change(path, schedule, _canonical_schedule(schedule, mapping))
    for path, times in lists:
        record_change(path, times, _canonical_list(times, mapping))
    for name, changes in config_changes.items():
        product_changes[name] = dataclasses.replace(getattr(product, name), **changes)
    if maturity is not None and mapping.get(maturity, maturity) != maturity:
        product_changes["maturity"] = mapping[maturity]
    for name in _PRODUCT_TIME_SCALARS:
        t = getattr(product, name, None)
        if _is_time(t) and _moved(t, mapping) != t:
            product_changes[name] = _moved(t, mapping)
    for name, value in product_changes.items():
        setattr(product, name, value)
    return bool(product_changes) or offsets_moved


def _canonicalize_rebased_offsets(product: Any) -> bool:
    """A REBASED schedule 2's offsets (observation and settlement) on one float per offset instant, its largest
    observation offset pinned (the REBASED maturity offset).

    The offsets are not contract instants, but every absolute schedule-2 time is ``t_ki + offset``: two spellings of
    one offset would become two spellings of one instant on the knocked-in copy (canonicalized as an ABSOLUTE contract)
    and stay two in the MC rows. Equal offsets give equal sums on every path (CP1 finding 1).
    """
    if not _rebased_offsets(product, "post_barrier_config"):
        return False
    config = getattr(product, "post_barrier_config", None)
    schedule = getattr(config, "ko_observation_schedule", None)
    if schedule is None:
        return False
    records = _numeric_records(schedule)
    observations = [float(r.observation_time) for r in records if _is_time(r.observation_time)]
    if not observations:
        return False                          # no numeric offset to place
    values = observations + [float(r.settlement_time) for r in records if _is_time(r.settlement_time)]
    legacy = getattr(config, "ko_observation_dates", None)
    if _time_list(legacy):
        values.extend(float(t) for t in legacy)
    mapping = _canonical_map(values, max(observations))
    if mapping is None:
        return False
    path = ("post_barrier_config", "ko_observation_schedule")
    _refuse_same_schedule_collisions([(path, schedule)], mapping)
    changes: Dict[str, Any] = {}
    new_schedule = _canonical_schedule(schedule, mapping)
    if new_schedule is not schedule:
        changes["ko_observation_schedule"] = new_schedule
    if _time_list(legacy):
        new_legacy = _canonical_list(legacy, mapping)
        if new_legacy is not legacy:
            changes["ko_observation_dates"] = new_legacy
    if not changes:
        return False
    product.post_barrier_config = dataclasses.replace(config, **changes)
    return True


def contract_instants(product: Any) -> np.ndarray:
    """The contract's numeric instants (observation, settlement, maturity times), sorted and unique."""
    values, _, _, _ = _gather(product)
    return np.unique(np.asarray(values, dtype=float))


def snap_to_instants(
    times: Iterable[float], instants: Iterable[float], tol: float = Tolerance.ZERO
) -> np.ndarray:
    """Each time within ``tol`` of an instant becomes that instant (the nearest; the earlier on a tie).

    For times a contract generates itself (a REBASED schedule 2: knock-in time + offset) and the nodes an engine
    adds: a contract instant is never displaced (D2).
    """
    return snap_to_anchors(times, instants, tol)


def canonical_clock_after(method: Callable) -> Callable:
    """Decorate a contract operation (``time_shift``) so the OUTERMOST call canonicalizes the clock afterwards.

    A subclass that extends the operation (a knock-out-reset shifts its schedule 2 after the snowball part) is
    canonicalized once, on the fully shifted contract, never on a half-shifted one.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        state = self.__dict__
        depth = state.get("_contract_clock_depth", 0)
        state["_contract_clock_depth"] = depth + 1
        try:
            result = method(self, *args, **kwargs)
        finally:
            if depth:
                state["_contract_clock_depth"] = depth
            else:
                state.pop("_contract_clock_depth", None)
        if not depth:
            canonicalize_contract_clock(self)
        return result

    return wrapper
