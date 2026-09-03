"""Per-product factor coordinate (spec §5.3)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, FrozenSet, Optional

from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.pnlexplain.base import MARKET_FACTORS, Factor
from quantark.pnlexplain.equity.fingerprints import MATURITY_FLOOR
from quantark.util.exceptions import NumericalError, ValidationError

_TERM_FACTORS = frozenset({Factor.VOL, Factor.RATE, Factor.DIVIDEND, Factor.BASIS})


@dataclass(frozen=True)
class FactorCoordinate:
    reference_strike: Optional[float]
    tenor_t1: Optional[float]
    applicable: FrozenSet[Factor]


def _level(product: Any, name: str) -> Optional[float]:
    """A positive contract level, or None only when the product has no such term.

    A present-but-malformed level is an error, never a fallback to the next
    candidate: a NaN strike must not be silently read at the spot.
    """
    raw = getattr(product, name, None)
    if raw is None:
        return None
    what = f"{type(product).__name__}.{name}"
    try:
        f = float(raw)
    except (TypeError, ValueError):
        raise ValidationError(f"{what} must be a number, got {raw!r}") from None
    if not math.isfinite(f):
        raise ValidationError(f"{what} must be finite, got {raw!r}")
    if f <= 0.0:
        raise ValidationError(f"{what} must be positive, got {raw!r}")
    return f


def _tenor(product: Any, env: Any) -> Optional[float]:
    """Remaining tenor of the alive-at-t1 product, or None when it has no expiry.

    The one known non-error case in which ``get_maturity`` raises is a
    date-based product valued on or after its expiry; that is detected here
    explicitly and read as tenor 0. Every other ``ValidationError`` (a
    malformed product or environment) propagates: no invented expiry.
    """
    expiry = getattr(product, "exercise_date", None)
    if expiry is None:
        expiry = getattr(product, "maturity_date", None)
    if getattr(product, "maturity", None) is None and expiry is None:
        return None
    if expiry is not None and env is not None and env.valuation_date >= expiry:
        return 0.0
    raw = product.get_maturity(env)
    try:
        tenor = float(raw)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{type(product).__name__}.get_maturity must return a number, got {raw!r}"
        ) from None
    if not math.isfinite(tenor):
        raise NumericalError(f"non-finite remaining tenor for {type(product).__name__}: {raw!r}")
    return tenor


def resolve_coordinate(product_t0: Any, spot_t0: float, product_alive_t1: Any, env_t1: Any
                       ) -> FactorCoordinate:
    if isinstance(product_t0, SpotInstrument):
        return FactorCoordinate(reference_strike=float(spot_t0), tenor_t1=None,
                                applicable=frozenset({Factor.SPOT, Factor.MODEL}))
    if isinstance(product_t0, Futures):
        applicable = frozenset({Factor.TIME, Factor.SPOT, Factor.RATE, Factor.DIVIDEND,
                                Factor.BASIS, Factor.MODEL})
        strike = float(spot_t0)
    else:
        applicable = frozenset(MARKET_FACTORS)
        strike = _level(product_t0, "strike")
        if strike is None:
            strike = _level(product_t0, "initial_price")
        if strike is None:
            strike = float(spot_t0)
    tenor = _tenor(product_alive_t1, env_t1)
    if tenor is not None and tenor <= MATURITY_FLOOR:     # at or below the trackers' 1e-8 floor = expired
        applicable = applicable - _TERM_FACTORS
    return FactorCoordinate(reference_strike=strike, tenor_t1=tenor, applicable=applicable)
