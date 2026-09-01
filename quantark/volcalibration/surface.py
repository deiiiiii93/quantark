"""IV-surface construction: rectangular grid assembly and SABR smoothing.

The dict this module produces and consumes is the stage-02 surface schema the
artifact writer already serializes.  It is kept verbatim because artifact bytes
must not change (spec 5.3).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.quotes import QuoteSet

MIN_COMMON_STRIKES = 5
_STRIKE_MATCH_ATOL = 1e-6


def build_raw_surface(quotes: QuoteSet) -> Dict[str, Any]:
    """Assemble a rectangular strike x maturity IV grid from a QuoteSet.

    The common strike grid holds strikes quoted in at least two expiries; each
    expiry's row is linear interpolation of its own smile, flat past its wings.
    """
    expiries = list(quotes.expiries)
    if len(expiries) < 2:
        raise ValidationError("a surface needs at least 2 expiries")

    all_strikes = sorted({n.strike for e in expiries for n in e.nodes})

    def _count(k: float) -> int:
        return sum(
            any(abs(k - n.strike) < _STRIKE_MATCH_ATOL for n in e.nodes)
            for e in expiries
        )

    strikes = [k for k in all_strikes if _count(k) >= 2]
    if len(strikes) < MIN_COMMON_STRIKES:
        raise ValidationError(
            f"only {len(strikes)} common strikes across expiries "
            f"(need >= {MIN_COMMON_STRIKES})"
        )

    maturities = [e.T for e in expiries]
    grid = np.empty((len(maturities), len(strikes)), dtype=float)
    per_expiry: List[Dict[str, Any]] = []
    for i, e in enumerate(expiries):
        ks = np.asarray([n.strike for n in e.nodes], dtype=float)
        vs = np.asarray([n.iv for n in e.nodes], dtype=float)
        grid[i] = np.interp(strikes, ks, vs)
        rmse = e.diagnostics.get("parity_rmse_over_forward")
        per_expiry.append(
            {
                "expiry_date": e.expiry_date if e.expiry_date else e.expiry_label,
                "T": e.T,
                "r": e.r,
                "q": e.q,
                "forward": e.forward,
                "df": e.discount_factor,
                "pair_count": int(e.diagnostics.get("n_pairs", 0)),
                "parity_rmse_points": (
                    float(rmse) * e.forward if rmse is not None else None
                ),
                "points": [(n.strike, n.iv) for n in e.nodes],
            }
        )

    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValidationError("grid assembly produced non-positive or non-finite IVs")

    return {
        "s0": quotes.spot,
        "strikes": strikes,
        "maturities": maturities,
        "iv_grid": grid.tolist(),
        "per_expiry": per_expiry,
    }
