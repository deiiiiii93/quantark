"""Missing past fixings: decided at the current spot, with a warning.

A past fixing (an observation on or before the valuation instant) that carries no recorded outcome is decided at
the pricing environment's current spot for that pricing call, and the products say so with one
:class:`MissingFixingWarning` per call that names every fixing filled. The product itself is never changed: the
fill lives only in what ``resolve_observations`` returns.

Bump Greeks and scenario revaluations first take :func:`freeze_missing_fixings` at the BASE environment, so the
fill is made once at the base spot and does not follow a bumped or shocked spot (user decision 2026-10-09).
"""

from __future__ import annotations

import copy
import warnings
from typing import Sequence

from quantark.util.numerical import is_zero


class MissingFixingWarning(UserWarning):
    """A past fixing without a recorded outcome was decided at the current spot."""


def is_history(observation_time: float) -> bool:
    """True for a fixing STRICTLY before the valuation instant (``t < 0`` and not zero).

    Only history is frozen at the base spot. A fixing at exactly ``t = 0`` (today, not yet printed) is live: it
    moves with spot bumps and shocks (user decision 2026-10-09), though pricing still decides it at the current
    spot as ``resolve_observations`` does.
    """
    return observation_time < 0 and not is_zero(observation_time)


def snap_time(t: float) -> float:
    """``t`` with a time indistinguishable from zero (``is_zero``) made exactly ``0.0``: an aged numeric fixing lands
    on t = 0 on its own day, not an ulp either side (spec 2026-10-09 run fixings §3.2)."""
    return 0.0 if is_zero(t) else t


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


def freeze_missing_fixings(product, env, *, warn: bool = True):
    """``product`` with its missing past fixings recorded at ``env``'s spot, for every revaluation that follows.

    User decision 2026-10-09: a filled past fixing is history - every bump Greek and every scenario shock keeps it
    at the BASE spot. Call this once at the base environment, before bumping or shocking, and reprice the result:
    the fill and its one ``MissingFixingWarning`` happen here, and the repricings find nothing left to fill.
    Returns ``product`` itself when it has no such fixings (or no ``with_missing_fixings_filled``); never mutates it.
    ``warn=False`` for a caller whose base valuation has already warned.
    """
    fill = getattr(product, "with_missing_fixings_filled", None)
    if fill is None:
        return product
    return fill(env, warn=warn)


def freeze_portfolio_fixings(portfolio):
    """``portfolio`` with every position's missing past fixings recorded at its BASE environment's spot.

    For scenario revaluations (VaR, stress, dynamic scenarios, backtests): call once on the unshocked portfolio,
    before any environment is shocked or rolled, so a fill stays at the base spot for the whole run (user decision
    2026-10-09: history does not move). Positions are found in ``portfolio.positions`` and priced against
    ``portfolio.pricing_environments[position.underlying]``. Returns ``portfolio`` itself when nothing is filled;
    otherwise a shallow copy whose changed positions are shallow copies holding the frozen product. Neither the
    portfolio, its positions nor their products are mutated.
    """
    positions = getattr(portfolio, "positions", None)
    envs = getattr(portfolio, "pricing_environments", None)
    if not positions or envs is None:
        return portfolio
    frozen_positions = {}
    changed = False
    for key, position in positions.items():
        product = getattr(position, "product", None)
        env = envs.get(getattr(position, "underlying", None))
        frozen = product if product is None or env is None else freeze_missing_fixings(product, env)
        if frozen is not product:
            position = copy.copy(position)
            position.product = frozen
            changed = True
        frozen_positions[key] = position
    if not changed:
        return portfolio
    out = copy.copy(portfolio)
    out.positions = frozen_positions
    return out
