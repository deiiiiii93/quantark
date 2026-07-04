"""
Aggregate position-level futures-tenor bucket rows by contract.

Input rows are already position-level PV sensitivities (position quantity
applied before aggregation). Additive fields are summed; non-additive fields
(maturity, future_price, delta_per_hand, extrapolated_tail) must agree across
rows for a contract — a mismatch means incompatible futures curves and raises
ValidationError instead of silently aggregating.
"""
from typing import Dict, List, Mapping, Sequence

from quantark.util.exceptions import ValidationError

FuturesBucketRow = Dict[str, object]

_ADDITIVE_KEYS = (
    "delta_bucket",
    "hedge_hands",
    "rhoq_bucket",
    "futures_rhoq_bucket",
    "net_delta_bucket",
    "net_rhoq_bucket",
)
_MATCH_KEYS = ("maturity", "future_price", "delta_per_hand", "extrapolated_tail")
_MATCH_TOL = 1e-9


def _values_match(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if b is None:
        return False
    fa, fb = float(a), float(b)
    return abs(fa - fb) <= _MATCH_TOL * max(1.0, abs(fa))


def _aggregate(
    rows_by_position: Mapping[str, Sequence[FuturesBucketRow]],
) -> List[FuturesBucketRow]:
    order: List[str] = []
    grouped: Dict[str, List[FuturesBucketRow]] = {}
    for rows in rows_by_position.values():
        for row in rows:
            contract = row["contract"]
            if contract not in grouped:
                grouped[contract] = []
                order.append(contract)
            grouped[contract].append(row)

    out: List[FuturesBucketRow] = []
    for contract in order:
        rows = grouped[contract]
        first = rows[0]
        agg: FuturesBucketRow = {"contract": contract}
        for key in _MATCH_KEYS:
            if key not in first:
                continue
            for row in rows[1:]:
                if key in row and not _values_match(first[key], row[key]):
                    raise ValidationError(
                        f"incompatible futures curves for contract {contract}: "
                        f"{key} mismatch ({first[key]} vs {row[key]})"
                    )
            agg[key] = first[key]
        for key in _ADDITIVE_KEYS:
            if any(key in row for row in rows):
                agg[key] = sum(float(row[key]) for row in rows if key in row)
        out.append(agg)
    return out


def aggregate_futures_delta_buckets(
    rows_by_position: Mapping[str, Sequence[FuturesBucketRow]],
) -> List[FuturesBucketRow]:
    return _aggregate(rows_by_position)


def aggregate_futures_rhoq_buckets(
    rows_by_position: Mapping[str, Sequence[FuturesBucketRow]],
) -> List[FuturesBucketRow]:
    return _aggregate(rows_by_position)
