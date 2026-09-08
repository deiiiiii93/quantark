"""The term-structure dividend rule shared by the replay engine, the
q term-structure study and the simulated-path backtest.

One function, so a ``dividend_source`` / ``futures_curve_extrapolation``
pair means the same object wherever a chain is turned into the pricer's
dividend input.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.util.exceptions import ValidationError

EXTRAPOLATIONS = ("flat_q", "flat_forward_carry", "surface_forward_carry")


def term_dividend_yield(
    quotes: Sequence[IndexFuturesQuote],
    *,
    spot: float,
    rate_curve: Any,
    extrapolation: str,
    underlying: str = "index",
    artifact: Optional[Any] = None,
):
    """Dividend object for the ``futures_curve`` source from eligible quotes.

    ``surface_forward_carry``: the chain in log-forward space continued past
    its last node with ``artifact.implied_q_pillars``' forward carry.  One
    quote is the flat one-node limit of the other two conventions;
    ``flat_forward_carry`` continues the last segment's forward carry;
    ``flat_q`` holds the endpoint zero yield.  No quotes, an unknown
    convention or a missing artifact fail closed.
    """
    if extrapolation not in EXTRAPOLATIONS:
        raise ValidationError(
            f"futures_curve_extrapolation must be one of {EXTRAPOLATIONS}, got {extrapolation!r}"
        )
    if not quotes:
        raise ValidationError("term_dividend_yield needs at least one eligible futures quote")
    if extrapolation == "surface_forward_carry":
        if artifact is None:
            raise ValidationError("'surface_forward_carry' requires an artifact with implied_q_pillars")
        from quantark.backtest.replay.product_replay import surface_tail_carry_yield

        return surface_tail_carry_yield(
            spot=float(spot),
            forward_nodes=[(q.maturity, q.price) for q in quotes],
            artifact=artifact,
            rate_curve=rate_curve,
        )
    if len(quotes) == 1:
        q = quotes[0]
        return ContinuousDividendYield(
            float(rate_curve.get_rate(q.maturity)) - math.log(q.price / float(spot)) / q.maturity
        )
    curve = IndexFuturesCurve(underlying=underlying or "index", spot=float(spot), quotes=list(quotes))
    if extrapolation == "flat_forward_carry":
        return ForwardCarryCurve.from_index_futures(curve, rate_curve).to_dividend_yield(rate_curve)
    return curve.to_dividend_yield_curve(rate_curve)
