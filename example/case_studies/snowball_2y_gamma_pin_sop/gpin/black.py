"""Black-76 on a listed MO series (``gpin.board.SeriesInfo``): implied vol, value, index delta and gamma.

Units. Prices are in index points per option; ``index_delta`` is points of option value per index point of SPOT and
``index_gamma`` points per index point squared. Both assume the series' forward moves in proportion to spot
(``F(S) = F x S / spot``), which is the sizing convention of the overlay (spec C5): the delta is
``discount x forward-delta x F / spot`` and the gamma is ``discount x forward-gamma x (F / spot) ** 2``. The implied vol
reuses the library's Black-76 inversion (``quantark.param.vol.marketquotes``), the one the tail-stability study uses.
Ledger-side code: no engine, no market data of its own.
"""

from __future__ import annotations

from typing import Optional

from scipy.stats import norm

from quantark.param.vol.marketquotes import black_price, implied_vol_black
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero, safe_divide, safe_exp, safe_log, safe_sqrt

from s2ts import RATE

from .board import SeriesInfo


def _is_call(cp: str) -> bool:
    if cp not in ("C", "P"):
        raise ValidationError(f"option side must be 'C' or 'P', got {cp!r}")
    return cp == "C"


def implied_vol(quote_price: float, cp: str, strike: float, info: SeriesInfo) -> Optional[float]:
    """Black-76 implied vol of a settlement; ``None`` when it cannot be inverted: no time left, a price outside the
    no-arbitrage bounds (the library's bracket does not straddle it) or a failed root search."""
    if info.tenor <= 0.0 or is_zero(info.tenor):
        return None
    try:
        return float(implied_vol_black(float(quote_price), info.forward, float(strike), info.tenor, info.discount,
                                       _is_call(cp)))
    except (ValidationError, ValueError, RuntimeError):
        return None


def value(vol: float, cp: str, strike: float, info: SeriesInfo) -> float:
    """Black-76 value in index points."""
    return float(black_price(info.forward, float(strike), info.tenor, float(vol), info.discount, _is_call(cp)))


def _d1(vol: float, strike: float, forward: float, tenor: float) -> float:
    if tenor <= 0.0 or vol <= 0.0:
        raise ValidationError("Black-76 greeks need a positive tenor and vol")
    s = vol * safe_sqrt(tenor)
    return float((safe_log(forward / strike) + 0.5 * s * s) / s)


def index_delta(vol: float, cp: str, strike: float, info: SeriesInfo, spot: float) -> float:
    """``discount x forward delta x forward / spot`` (points per index point of spot)."""
    d1 = _d1(vol, strike, info.forward, info.tenor)
    forward_delta = norm.cdf(d1) if _is_call(cp) else norm.cdf(d1) - 1.0
    return float(info.discount * forward_delta * safe_divide(info.forward, spot))


def index_gamma(vol: float, cp: str, strike: float, info: SeriesInfo, spot: float) -> float:
    """Second derivative of the value with respect to spot under the proportional forward:
    ``discount x forward gamma x (forward / spot) ** 2``."""
    _is_call(cp)
    d1 = _d1(vol, strike, info.forward, info.tenor)
    forward_gamma = norm.pdf(d1) / (info.forward * vol * safe_sqrt(info.tenor))
    return float(info.discount * forward_gamma * safe_divide(info.forward, spot) ** 2)


def value_at(vol: float, cp: str, strike: float, info: SeriesInfo, spot_now: float, spot_scenario: float,
             tenor_left: float) -> float:
    """Value in a scenario: the forward scaled by ``spot_scenario / spot_now``, ``tenor_left`` years to the last
    trading day, discounted at the study's flat rate. No time left gives the payoff."""
    forward = info.forward * safe_divide(spot_scenario, spot_now)
    discount = float(safe_exp(-RATE * max(tenor_left, 0.0)))
    return float(black_price(forward, float(strike), float(tenor_left), float(vol), discount, _is_call(cp)))
