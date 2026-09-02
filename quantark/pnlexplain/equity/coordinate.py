"""Per-product factor coordinate (spec §5.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Optional

from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.pnlexplain.base import MARKET_FACTORS, Factor
from quantark.pnlexplain.equity.fingerprints import MATURITY_FLOOR
from quantark.util.exceptions import ValidationError

_TERM_FACTORS = frozenset({Factor.VOL, Factor.RATE, Factor.DIVIDEND, Factor.BASIS})


@dataclass(frozen=True)
class FactorCoordinate:
    reference_strike: Optional[float]
    tenor_t1: Optional[float]
    applicable: FrozenSet[Factor]


def _positive(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f > 0.0 else None


def _tenor(product: Any, env: Any) -> Optional[float]:
    if getattr(product, "maturity", None) is None and getattr(product, "maturity_date", None) is None \
            and getattr(product, "exercise_date", None) is None:
        return None
    try:
        return float(product.get_maturity(env))
    except ValidationError:
        return 0.0     # valuation on/after a date-based expiry


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
        strike = _positive(getattr(product_t0, "strike", None)) \
            or _positive(getattr(product_t0, "initial_price", None)) or float(spot_t0)
    tenor = _tenor(product_alive_t1, env_t1)
    if tenor is not None and tenor <= MATURITY_FLOOR:     # at or below the trackers' 1e-8 floor = expired
        applicable = applicable - _TERM_FACTORS
    return FactorCoordinate(reference_strike=strike, tenor_t1=tenor, applicable=applicable)
