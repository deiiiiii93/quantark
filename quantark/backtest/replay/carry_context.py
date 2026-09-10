"""Immutable carry-curve scenarios over the day's actual futures quotes.

Every dividend object a bucket risk measurement prices against comes from
:meth:`CarryCurveContext.dividend`, which delegates to the one shared builder
in :mod:`quantark.backtest.replay.dividend_source`.  A scenario is a new
frozen context with transformed quotes, never a second interpolation routine:
a hand-rolled nodal bump written for ``flat_q`` would silently misprice
``flat_forward_carry``, and the audits of design section 6.3 would then be
validating the bump implementation rather than the hedge.

The transformations follow design section 6.1:

``with_spot``
    moves spot and PINS every listed quote, which is the pinned-futures
    direction ``D_F``;
``bump_future``
    adds index points to one quote at fixed spot, which is the bucket
    ``B_i``;
``bump_node_yield``
    multiplies one quote by ``exp(-T_i * shift)`` at fixed spot, which moves
    that node's zero yield by ``shift``;
``parallel_yield_shift``
    applies that transformation to every quote using its own tenor, which
    moves the whole rebuilt zero curve by ``shift``.

Only ``flat_q`` and ``flat_forward_carry`` are accepted.  Both satisfy the
two invariances of design section 2.2 -- proportional spot/quote scaling
leaves the zero curve unchanged, and the exponential quote transformation
shifts it by ``shift`` at every pricing time -- which is what establishes
``R_parallel = sum_i R_i`` and the chain rule the policies rely on.  A tail
or interpolation-shape stress is NOT a quote transformation and lives in
:mod:`quantark.backtest.replay.carry_stress` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Sequence, Tuple

from quantark.asset.equity.market import IndexFuturesQuote
from quantark.backtest.replay.dividend_source import term_dividend_yield
from quantark.util.exceptions import ValidationError

#: The conventions whose Jacobian the bucket policies are derived for.
SUPPORTED_EXTRAPOLATIONS: Tuple[str, ...] = ("flat_q", "flat_forward_carry")

__all__ = ["SUPPORTED_EXTRAPOLATIONS", "CarryCurveContext"]


def _finite_positive(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be finite and positive, got {value!r}")
    return value


@dataclass(frozen=True)
class CarryCurveContext:
    """One valuation date's spot, quote universe and curve convention."""

    quotes: Tuple[IndexFuturesQuote, ...]
    spot: float
    rate_curve: Any
    extrapolation: str
    underlying: str
    valuation_date: Any

    def __post_init__(self) -> None:
        quotes = tuple(self.quotes)
        if not quotes:
            raise ValidationError("a carry context needs at least one futures quote")
        for quote in quotes:
            if not isinstance(quote, IndexFuturesQuote):
                raise ValidationError(
                    f"quotes must be IndexFuturesQuote records, got {type(quote)!r}"
                )
            _finite_positive(quote.price, f"price[{quote.contract}]")
            _finite_positive(quote.maturity, f"maturity[{quote.contract}]")
            _finite_positive(quote.multiplier, f"multiplier[{quote.contract}]")
        contracts = [q.contract for q in quotes]
        if len(set(contracts)) != len(contracts):
            raise ValidationError(f"futures contracts must be unique: {contracts}")
        maturities = [q.maturity for q in quotes]
        if any(
            maturities[i] >= maturities[i + 1] for i in range(len(maturities) - 1)
        ):
            raise ValidationError(
                f"quotes must have strictly increasing maturities: {maturities}"
            )
        object.__setattr__(self, "quotes", quotes)
        object.__setattr__(self, "spot", _finite_positive(self.spot, "spot"))
        if self.extrapolation not in SUPPORTED_EXTRAPOLATIONS:
            raise ValidationError(
                f"carry scenarios support {SUPPORTED_EXTRAPOLATIONS}, got "
                f"{self.extrapolation!r}; an option-implied tail is not a "
                "tradable futures coordinate"
            )
        if not isinstance(self.underlying, str) or not self.underlying.strip():
            raise ValidationError("underlying must be a non-empty identifier")
        if not hasattr(self.rate_curve, "get_rate"):
            raise ValidationError("rate_curve must expose get_rate")

    # -- coordinates ----------------------------------------------------

    @property
    def contracts(self) -> Tuple[str, ...]:
        return tuple(q.contract for q in self.quotes)

    @property
    def prices(self) -> Dict[str, float]:
        return {q.contract: float(q.price) for q in self.quotes}

    @property
    def multipliers(self) -> Dict[str, float]:
        return {q.contract: float(q.multiplier) for q in self.quotes}

    def quote(self, contract: str) -> IndexFuturesQuote:
        for q in self.quotes:
            if q.contract == contract:
                return q
        raise ValidationError(f"unknown futures contract: {contract!r}")

    def coordinates(self) -> Tuple[Tuple[str, float, float, float], ...]:
        """``(contract, maturity, price, multiplier)`` in curve order.

        Two risk samples may only be added when this tuple, the spot, the
        convention and the valuation date all agree; the node COUNT alone is
        not evidence of the same coordinates.
        """
        return tuple(
            (q.contract, float(q.maturity), float(q.price), float(q.multiplier))
            for q in self.quotes
        )

    def same_coordinates_as(self, other: "CarryCurveContext") -> bool:
        return (
            self.coordinates() == other.coordinates()
            and self.spot == other.spot
            and self.extrapolation == other.extrapolation
            and self.underlying == other.underlying
            and self.valuation_date == other.valuation_date
        )

    # -- pricing input --------------------------------------------------

    def dividend(self) -> Any:
        """The dividend object this scenario prices against."""
        return term_dividend_yield(
            self.quotes,
            spot=self.spot,
            rate_curve=self.rate_curve,
            extrapolation=self.extrapolation,
            underlying=self.underlying,
        )

    def implied_yield(self, contract: str) -> float:
        """``q_i = r(T_i) - log(F_i/S)/T_i`` for one listed contract."""
        quote = self.quote(contract)
        return float(self.rate_curve.get_rate(quote.maturity)) - math.log(
            quote.price / self.spot
        ) / float(quote.maturity)

    # -- scenarios ------------------------------------------------------

    def with_spot(self, spot: float) -> "CarryCurveContext":
        """Move spot; every listed quote is pinned."""
        return replace(self, spot=float(spot))

    def _replace_price(
        self, contract: str, transform: Callable[[IndexFuturesQuote], float]
    ) -> "CarryCurveContext":
        if contract not in {q.contract for q in self.quotes}:
            raise ValidationError(f"unknown futures contract: {contract}")
        return replace(
            self,
            quotes=tuple(
                replace(q, price=float(transform(q))) if q.contract == contract else q
                for q in self.quotes
            ),
        )

    def bump_future(self, contract: str, points: float) -> "CarryCurveContext":
        """Add ``points`` index points to one quote at fixed spot."""
        points = float(points)
        if not math.isfinite(points) or points == 0.0:
            raise ValidationError("futures bump must be finite and non-zero")
        return self._replace_price(contract, lambda q: q.price + points)

    def bump_node_yield(
        self, contract: str, yield_shift: float
    ) -> "CarryCurveContext":
        """Move one node's zero yield by ``yield_shift`` at fixed spot."""
        yield_shift = _finite_shift(yield_shift, "yield_shift")
        return self._replace_price(
            contract, lambda q: q.price * math.exp(-q.maturity * yield_shift)
        )

    def parallel_yield_shift(self, yield_shift: float) -> "CarryCurveContext":
        """Move the whole rebuilt zero curve by ``yield_shift``."""
        yield_shift = _finite_shift(yield_shift, "yield_shift")
        return replace(
            self,
            quotes=tuple(
                replace(q, price=q.price * math.exp(-q.maturity * yield_shift))
                for q in self.quotes
            ),
        )


def _finite_shift(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return value


def quotes_from_rows(
    rows: Sequence[Any],
    *,
    valuation_date,
    day_count: float = 365.0,
) -> Tuple[IndexFuturesQuote, ...]:
    """Eligible chain rows as ``IndexFuturesQuote`` records, in curve order.

    Kept next to the context so replay and the study build the same
    coordinates from the same columns.
    """
    import pandas as pd

    quotes = [
        IndexFuturesQuote(
            contract=str(row["contract"]),
            maturity=(pd.Timestamp(row["expiry_date"]) - valuation_date).days
            / day_count,
            price=float(row["futures_price"]),
            multiplier=float(row["multiplier"]),
            expiry_date=pd.Timestamp(row["expiry_date"]).to_pydatetime(),
        )
        for row in rows
    ]
    return tuple(sorted(quotes, key=lambda q: (q.maturity, q.contract)))
