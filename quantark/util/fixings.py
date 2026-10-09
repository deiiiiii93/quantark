"""Missing past fixings: decided at the current spot, with a warning.

A past fixing (an observation on or before the valuation instant) that carries no recorded outcome is decided at
the pricing environment's current spot for that pricing call, and the products say so with one
:class:`MissingFixingWarning` per call that names every fixing filled. The product itself is never changed: the
fill lives only in what ``resolve_observations`` returns.

Bump Greeks and scenario revaluations first take :func:`freeze_missing_fixings` at the BASE environment, so the
fill is made once at the base spot and does not follow a bumped or shocked spot (user decision 2026-10-09).
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


def freeze_missing_fixings(product, env):
    """``product`` with its missing past fixings recorded at ``env``'s spot, for every revaluation that follows.

    User decision 2026-10-09: a filled past fixing is history - every bump Greek and every scenario shock keeps it
    at the BASE spot. Call this once at the base environment, before bumping or shocking, and reprice the result:
    the fill and its one ``MissingFixingWarning`` happen here, and the repricings find nothing left to fill.
    Returns ``product`` itself when it has no such fixings (or no ``with_missing_fixings_filled``); never mutates it.
    """
    fill = getattr(product, "with_missing_fixings_filled", None)
    if fill is None:
        return product
    return fill(env)
