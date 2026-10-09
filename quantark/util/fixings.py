"""Missing past fixings: decided at the current spot, with a warning.

A past fixing (an observation on or before the valuation instant) that carries no recorded outcome is decided at
the pricing environment's current spot for that pricing call, and the products say so with one
:class:`MissingFixingWarning` per call that names every fixing filled. The product itself is never changed: the
fill lives only in what ``resolve_observations`` returns.
"""

from __future__ import annotations

import warnings
from typing import Sequence


class MissingFixingWarning(UserWarning):
    """A past fixing without a recorded outcome was decided at the current spot."""


def fixing_label(index: int, record) -> str:
    """``#<index> t=<time>`` for a numeric record, ``#<index> <date>`` for a dated one (the time wins, as in
    ``resolve_time``)."""
    if record.observation_time is not None:
        return f"#{index} t={record.observation_time:.6g}"
    return f"#{index} {record.observation_date.date().isoformat()}"


def warn_filled_fixings(
    product_name: str, field: str, labels: Sequence[str], *, stacklevel: int = 3
) -> None:
    """One warning naming every past fixing ``product_name`` filled at the current spot (nothing when none)."""
    if not labels:
        return
    count = len(labels)
    noun = "fixing" if count == 1 else "fixings"
    warnings.warn(
        f"{product_name}: {count} past {noun} without {field} ({', '.join(labels)}) decided at the current spot "
        "for this pricing; the product is not changed - record the realised fixings to remove this warning",
        MissingFixingWarning,
        stacklevel=stacklevel,
    )
