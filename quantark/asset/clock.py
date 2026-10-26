"""The date / numeric clock of an equity or FX option (user ruling 2026-10-26).

A product is on the DATE clock (every time it uses is a date) or on the NUMERIC clock (every time is a float, in
years from the valuation date), never on a mix of the two:

- Both clocks given completely: the dates win and the floats are ignored, with no consistency check. A float that
  only shadows a date given for the same time (``maturity`` beside ``exercise_date``) is not a violation.
- Partial or mixed input is a construction error: a date field on a numeric contract (``initial_date`` on a numeric
  note), or a numeric time on a dated contract that has no date for it (a record with only ``observation_time``).

Durations and counts are clock-neutral product terms: ``tenor``, settlement lags in any unit (user ruling
2026-10-26: a lag is a product term, like a settlement date or "2D"), accrual factors, coupon year fractions,
schedule generators, and the schedule-2 offsets of a REBASED knock-out-reset (a product term, measured from the
knock-in). Floats that aging writes on a dated copy and that no dated reader uses (``accrued_offset``,
``first_period_start``) are ignored.

The FX target-redemption products and the DCN keep their own clock rules (both already enforce this one).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterator, List, Tuple

#: Fields of a schedule record that are dates.
_RECORD_DATE_FIELDS = (
    "observation_date", "settlement_date", "initial_date", "maturity_date",
    "observation_timestamp", "settlement_timestamp",
)


DATE_FIELD = "a date field on the numeric clock"
NUMERIC_TIME = "a numeric time on the date clock with no date for it"


class ProductClock(Enum):
    """The clock a product's times are given on."""

    DATE = "date"
    NUMERIC = "numeric"


def _has_own_rule(product: Any) -> bool:
    """The DCN (date-only) and the FX target-redemption products enforce their own clock rule."""
    from quantark.asset.equity.product.option.dcn_option import DCNOption
    from quantark.asset.fx.product.option.target_redemption_base import TargetRedemptionProduct

    return isinstance(product, (DCNOption, TargetRedemptionProduct))


def _is_fx(product: Any) -> bool:
    from quantark.asset.fx.product.base_fx_product import BaseFxProduct

    return isinstance(product, BaseFxProduct)


def product_clock(product: Any) -> ProductClock:
    """The clock ``product`` is on, read off its maturity anchor.

    An FX option is dated when it has an ``expiry_date``; an equity option when it has an ``exercise_date`` or a
    ``maturity_date``. Otherwise it is numeric.
    """
    if _is_fx(product):
        dated = getattr(product, "expiry_date", None) is not None
    else:
        dated = getattr(product, "exercise_date", None) is not None or getattr(product, "maturity_date", None) is not None
    return ProductClock.DATE if dated else ProductClock.NUMERIC


def _is_rebased_schedule_two(product: Any, config_name: str) -> bool:
    mode = getattr(product, "post_ko_mode", None)
    return config_name == "post_barrier_config" and getattr(mode, "name", None) == "REBASED"


def _schedule_records(product: Any) -> Iterator[Tuple[str, Any, bool]]:
    """Every observation record of ``product``: (where, record, holds offsets rather than times)."""
    for config_name in ("barrier_config", "post_barrier_config"):
        config = getattr(product, config_name, None)
        if config is None:
            continue
        offsets = _is_rebased_schedule_two(product, config_name)
        for schedule_name in ("ko_observation_schedule", "ki_observation_schedule"):
            schedule = getattr(config, schedule_name, None)
            for record in getattr(schedule, "records", None) or ():
                yield f"{config_name}.{schedule_name}", record, offsets
    schedule = getattr(product, "observation_schedule", None)
    for record in getattr(schedule, "records", None) or ():
        yield "observation_schedule", record, False
    for record in getattr(product, "observation_records", None) or ():
        yield "observation_records", record, False


def _is_date(value: Any) -> bool:
    return hasattr(value, "year") and hasattr(value, "month")


def _float_list(values: Any) -> bool:
    """A non-empty list whose entries are numeric times (not dates)."""
    if not values:
        return False
    first = values[0]
    if isinstance(first, (tuple, list)):          # (time, spot) fixings
        first = first[0]
    return not _is_date(first)


def _all_dated(schedule: Any) -> bool:
    """A non-empty schedule (or record list) whose every record has an ``observation_date``."""
    records = getattr(schedule, "records", schedule) or ()
    return bool(records) and all(getattr(record, "observation_date", None) is not None for record in records)


def _date_list(values: Any) -> bool:
    if not values:
        return False
    first = values[0]
    if isinstance(first, (tuple, list)):          # (date, spot) fixings
        first = first[0]
    return _is_date(first)


def clock_violations(product: Any) -> List[Tuple[str, str]]:
    """Every field of ``product`` that breaks the clock rule, as ``(field, reason)``; empty when it is on one clock.

    Pure detection: nothing is changed.
    """
    if _has_own_rule(product):
        return []
    found: List[Tuple[str, str]] = []
    if product_clock(product) is ProductClock.NUMERIC:
        for name in ("initial_date", "settlement_date", "maturity_date", "delivery_date"):
            if getattr(product, name, None) is not None:
                found.append((name, DATE_FIELD))
        for name in ("observation_dates", "past_observations"):
            if _date_list(getattr(product, name, None)):
                found.append((name, DATE_FIELD))
        for where, record, _offsets in _schedule_records(product):
            for name in _RECORD_DATE_FIELDS:
                if getattr(record, name, None) is not None:
                    found.append((f"{where}.{name}", DATE_FIELD))
        return _unique(found)

    for where, record, offsets in _schedule_records(product):
        if offsets:
            continue                                # REBASED schedule 2: offsets from the knock-in, a product term
        if getattr(record, "observation_time", None) is not None and getattr(record, "observation_date", None) is None:
            found.append((f"{where}.observation_time", NUMERIC_TIME))
        if getattr(record, "settlement_time", None) is not None and getattr(record, "settlement_date", None) is None:
            found.append((f"{where}.settlement_time", NUMERIC_TIME))
    for config_name in ("barrier_config", "post_barrier_config"):
        config = getattr(product, config_name, None)
        if config is None or _is_rebased_schedule_two(product, config_name):
            continue
        for name, schedule_name in (("ko_observation_dates", "ko_observation_schedule"),
                                    ("ki_observation_dates", "ki_observation_schedule")):
            if getattr(config, name, None) and not _all_dated(getattr(config, schedule_name, None)):
                found.append((f"{config_name}.{name}", NUMERIC_TIME))
    dated_observations = (
        _date_list(getattr(product, "observation_dates", None))
        or _all_dated(getattr(product, "observation_schedule", None))
        or _all_dated(getattr(product, "observation_records", None))
    )
    for name in ("observation_dates", "observation_times"):
        if _float_list(getattr(product, name, None)) and not dated_observations:
            found.append((name, NUMERIC_TIME))       # a float list beside its dates is shadowed: the dates win
    if _float_list(getattr(product, "past_observations", None)):
        found.append(("past_observations", NUMERIC_TIME))
    if _is_fx(product) and getattr(product, "delivery", None) is not None \
            and getattr(product, "delivery_date", None) is None:
        found.append(("delivery", NUMERIC_TIME))
    return _unique(found)


def _unique(found: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
