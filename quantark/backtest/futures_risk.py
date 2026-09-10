"""Currency-unit futures risk records for the bucket hedge.

This module holds the records, unit conversions and held-book algebra that the
bucket strategies, the replay recorder and the study analysis all share.  It
deliberately imports nothing from ``quantark.backtest.replay``: the strategy
package imports these records, and a strategy-to-replay import would close a
cycle through ``replay.__init__`` and ``replay.engine``.

Units (revised design sections 2.1 and 2.3):

* every Greek here is a CURRENCY sensitivity, never a hand count;
* ``bucket_currency`` is ``dV/dF_i`` in currency per index point of contract
  ``i``, already weighted by the book's position quantities;
* ``delta_q`` is ``dV/dS`` with the carry curve frozen;
* a nodal rhoq is ``R_i = -F_i T_i B_i``, currency per unit (decimal) yield;
* "per 1%" always means one ABSOLUTE percentage point of zero yield, so
  ``rhoq_bp_per_1pct`` multiplies by ``0.01`` and reports basis points of the
  run's fixed reporting notional.

The coordinate identities of design section 2.3 are the reason both
``delta_f_derived`` and a directly repriced ``D_F`` exist: the derived value
is an inexpensive sizing input, the direct one is the audit.  They are never
interchangeable, and :func:`held_book_risk` is algebra over already measured
Greeks -- it prices nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from quantark.util.exceptions import ValidationError

__all__ = [
    "CarryRiskSettings",
    "FuturesBookRisk",
    "FuturesBucket",
    "bucket_contract_equivalent",
    "gross_contractual_notional",
    "gross_of",
    "held_book_risk",
    "parallel_of",
    "rhoq_bp_per_1pct",
    "spot_1pct_bp",
    "spot_delta_hands",
]


def _finite(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return value


def _positive(value, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValidationError(f"{name} must be positive, got {value!r}")
    return value


def _non_negative(value, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0:
        raise ValidationError(f"{name} must be non-negative, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Risk coordinates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuturesBucket:
    """One tradable futures coordinate and the book's sensitivity to it.

    ``expiry_date`` may be ``None`` in a purely mathematical fixture; replay
    construction always supplies it, because a retired leg is identified by
    its expiry.  ``bucket_currency`` is signed and may be zero: a quoted node
    the book happens not to be exposed to is still a coordinate, and dropping
    it would hide an unhedged node from the report.
    """

    contract: str
    expiry_date: Optional[datetime]
    tenor_years: float
    price: float
    multiplier: float
    bucket_currency: float

    def __post_init__(self) -> None:
        if not isinstance(self.contract, str) or not self.contract.strip():
            raise ValidationError("contract must be a non-empty identifier")
        object.__setattr__(
            self, "tenor_years", _positive(self.tenor_years, "tenor_years")
        )
        object.__setattr__(self, "price", _positive(self.price, "price"))
        object.__setattr__(
            self, "multiplier", _positive(self.multiplier, "multiplier")
        )
        object.__setattr__(
            self, "bucket_currency", _finite(self.bucket_currency, "bucket_currency")
        )

    @property
    def nodal_rhoq(self) -> float:
        """``R_i = -F_i T_i B_i``: currency per unit of this node's yield."""
        return -self.bucket_currency * self.price * self.tenor_years

    @property
    def contract_equivalent(self) -> float:
        """``B_i / m_i``: this bucket expressed in contracts of its own size."""
        return self.bucket_currency / self.multiplier

    @property
    def spot_delta_share(self) -> float:
        """``(F_i / S) B_i`` needs the book's spot; this is the ``B_i F_i`` part."""
        return self.bucket_currency * self.price


@dataclass(frozen=True)
class FuturesBookRisk:
    """A book's spot and futures sensitivities on one common quote universe.

    One risk record means one valuation date, one spot, one curve convention
    and one quote universe.  Products with different maturities and pricing
    engines may share it; incompatible contexts must be separate records.
    """

    spot: float
    delta_q: float
    buckets: Tuple[FuturesBucket, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spot", _positive(self.spot, "spot"))
        object.__setattr__(self, "delta_q", _finite(self.delta_q, "delta_q"))
        buckets = tuple(self.buckets)
        if not buckets:
            raise ValidationError("a risk record needs at least one futures node")
        if any(not isinstance(b, FuturesBucket) for b in buckets):
            raise ValidationError("buckets must be FuturesBucket records")
        contracts = [b.contract for b in buckets]
        if len(set(contracts)) != len(contracts):
            raise ValidationError(f"futures contracts must be unique: {contracts}")
        tenors = [b.tenor_years for b in buckets]
        if any(tenors[i] >= tenors[i + 1] for i in range(len(tenors) - 1)):
            raise ValidationError(
                f"buckets must be ordered by strictly increasing tenor: {tenors}"
            )
        object.__setattr__(self, "buckets", buckets)

    # -- coordinates ----------------------------------------------------

    @property
    def contracts(self) -> Tuple[str, ...]:
        return tuple(b.contract for b in self.buckets)

    @property
    def tenors(self) -> Tuple[float, ...]:
        return tuple(b.tenor_years for b in self.buckets)

    def bucket(self, contract: str) -> FuturesBucket:
        for b in self.buckets:
            if b.contract == contract:
                return b
        raise ValidationError(f"unknown futures contract: {contract!r}")

    # -- derived product risks ------------------------------------------

    @property
    def delta_f_derived(self) -> float:
        """``D_F = D - sum_i (F_i/S) B_i``.

        A sizing input, not a diagnostic: it cancels by construction against
        the same identity, so only an independent reprice can audit it.
        """
        return self.delta_q - sum(
            b.price / self.spot * b.bucket_currency for b in self.buckets
        )

    @property
    def nodal_rhoq(self) -> Dict[str, float]:
        return {b.contract: b.nodal_rhoq for b in self.buckets}

    @property
    def parallel_rhoq(self) -> float:
        """Signed sum: ``R_parallel = sum_i R_i`` for the supported builders."""
        return sum(b.nodal_rhoq for b in self.buckets)

    @property
    def gross_nodal_rhoq(self) -> float:
        """Sum of absolute nodal rhoq; never compared with a signed total."""
        return sum(abs(b.nodal_rhoq) for b in self.buckets)


# ---------------------------------------------------------------------------
# Held-book algebra
# ---------------------------------------------------------------------------


def held_book_risk(
    risk: FuturesBookRisk, holdings: Mapping[str, float]
) -> Tuple[float, Dict[str, float]]:
    """Product-plus-hedge ``(D_book, R_book)`` for fixed futures holdings.

    ``D_book = D + sum_i m_i h_i F_i/S`` and
    ``R_book,i = R_i - m_i h_i F_i T_i`` (design section 2.3).  A holding on a
    contract with no risk coordinate is only acceptable when it is flat: a
    live position outside the risk universe would silently drop from both
    sums.
    """
    extra = set(holdings) - {b.contract for b in risk.buckets}
    for contract in sorted(extra):
        if not math.isfinite(float(holdings[contract])):
            raise ValidationError(f"holding for {contract!r} must be finite")
    if any(holdings[c] != 0.0 for c in extra):
        raise ValidationError(
            f"nonzero holding has no risk coordinate: {sorted(extra)}"
        )
    delta = risk.delta_q
    rho: Dict[str, float] = {}
    for b in risk.buckets:
        h = _finite(holdings.get(b.contract, 0.0), f"holding[{b.contract}]")
        delta += h * b.multiplier * b.price / risk.spot
        rho[b.contract] = (
            -b.bucket_currency * b.price * b.tenor_years
            - h * b.multiplier * b.price * b.tenor_years
        )
    return delta, rho


def hedge_spot_delta(
    risk: FuturesBookRisk, holdings: Mapping[str, float]
) -> float:
    """Hedge-only spot sensitivity ``sum_i h_i m_i F_i / S``."""
    total = 0.0
    for b in risk.buckets:
        h = _finite(holdings.get(b.contract, 0.0), f"holding[{b.contract}]")
        total += h * b.multiplier * b.price / risk.spot
    return total


def parallel_of(rhoq: Mapping[str, float]) -> float:
    """Signed sum of a nodal rhoq map."""
    return sum(float(v) for v in rhoq.values())


def gross_of(rhoq: Mapping[str, float]) -> float:
    """Sum of absolute nodal rhoq: a gross measure, never a signed total."""
    return sum(abs(float(v)) for v in rhoq.values())


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def spot_delta_hands(delta: float, reference_multiplier: float) -> float:
    """``delta / m_ref``: a book spot delta expressed in reference hands."""
    return _finite(delta, "delta") / _positive(
        reference_multiplier, "reference_multiplier"
    )


def bucket_contract_equivalent(bucket: FuturesBucket) -> float:
    """``B_i / m_i`` in contracts of that node's own size."""
    return bucket.contract_equivalent


def rhoq_bp_per_1pct(rhoq: float, reference_notional: float) -> float:
    """Basis points of ``N_ref`` per +1 percentage point of zero yield.

    ``rhoq * 0.01 / N_ref * 10_000 = 100 * rhoq / N_ref``.
    """
    if not math.isfinite(float(reference_notional)) or float(reference_notional) <= 0:
        raise ValidationError("reference_notional must be finite and positive")
    return 100.0 * _finite(rhoq, "rhoq") / float(reference_notional)


def spot_1pct_bp(delta: float, spot: float, reference_notional: float) -> float:
    """Basis points of ``N_ref`` moved by a 1% relative spot shock."""
    if not math.isfinite(float(reference_notional)) or float(reference_notional) <= 0:
        raise ValidationError("reference_notional must be finite and positive")
    return (
        100.0
        * _finite(delta, "delta")
        * _positive(spot, "spot")
        / float(reference_notional)
    )


def gross_contractual_notional(
    positions: Iterable[Tuple[float, float, float]]
) -> float:
    """``sum |Q_p| * initial_price_p * contract_multiplier_p``.

    The run's fixed reporting notional for Snowball/Phoenix books.  It is
    gross by construction, so a long-and-short book does not report a zero
    denominator, and it does not shrink when a product knocks out.
    """
    total = 0.0
    count = 0
    for quantity, initial_price, contract_multiplier in positions:
        count += 1
        total += (
            abs(_finite(quantity, "position quantity"))
            * _positive(initial_price, "initial_price")
            * _positive(contract_multiplier, "contract_multiplier")
        )
    if count == 0:
        raise ValidationError(
            "an empty product book has no contractual notional; supply an "
            "explicit positive reference_notional"
        )
    if total <= 0.0:
        raise ValidationError(
            "gross contractual notional is zero; supply an explicit positive "
            "reference_notional"
        )
    return total


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarryRiskSettings:
    """Numerical settings for carry risk measurement, audits and stresses.

    The two ``None`` fields are resolved once per run -- the notional from the
    initial book, the spot bump from the effective pricing bump settings --
    through :meth:`resolved`, so an artifact records the value actually used
    rather than the request to infer one.
    """

    reference_notional: Optional[float] = None
    reference_multiplier: float = 200.0
    futures_bump_points: float = 1.0
    audit_spot_bump_rel: Optional[float] = None
    audit_yield_bump: float = 1e-4
    delta_tolerance_hands: float = 0.01
    rhoq_tolerance_bp: float = 0.01
    stress_dates: Tuple[datetime, ...] = ()
    tail_shifts: Tuple[float, ...] = (-0.01, 0.01)
    shape_shifts: Tuple[float, ...] = (-0.01, 0.01)

    def __post_init__(self) -> None:
        if self.reference_notional is not None:
            object.__setattr__(
                self,
                "reference_notional",
                _positive(self.reference_notional, "reference_notional"),
            )
        if self.audit_spot_bump_rel is not None:
            object.__setattr__(
                self,
                "audit_spot_bump_rel",
                _positive(self.audit_spot_bump_rel, "audit_spot_bump_rel"),
            )
        object.__setattr__(
            self,
            "reference_multiplier",
            _positive(self.reference_multiplier, "reference_multiplier"),
        )
        object.__setattr__(
            self,
            "futures_bump_points",
            _positive(self.futures_bump_points, "futures_bump_points"),
        )
        object.__setattr__(
            self,
            "audit_yield_bump",
            _positive(self.audit_yield_bump, "audit_yield_bump"),
        )
        object.__setattr__(
            self,
            "delta_tolerance_hands",
            _non_negative(self.delta_tolerance_hands, "delta_tolerance_hands"),
        )
        object.__setattr__(
            self,
            "rhoq_tolerance_bp",
            _non_negative(self.rhoq_tolerance_bp, "rhoq_tolerance_bp"),
        )
        object.__setattr__(
            self, "stress_dates", _normalised_dates(self.stress_dates)
        )
        object.__setattr__(
            self, "tail_shifts", _finite_tuple(self.tail_shifts, "tail_shifts")
        )
        object.__setattr__(
            self, "shape_shifts", _finite_tuple(self.shape_shifts, "shape_shifts")
        )

    @property
    def is_resolved(self) -> bool:
        return (
            self.reference_notional is not None
            and self.audit_spot_bump_rel is not None
        )

    def resolved(
        self,
        *,
        reference_notional: Optional[float] = None,
        audit_spot_bump_rel: Optional[float] = None,
    ) -> "CarryRiskSettings":
        """These settings with any unset field filled in; never mutates self."""
        return replace(
            self,
            reference_notional=(
                self.reference_notional
                if self.reference_notional is not None
                else reference_notional
            ),
            audit_spot_bump_rel=(
                self.audit_spot_bump_rel
                if self.audit_spot_bump_rel is not None
                else audit_spot_bump_rel
            ),
        )

    def require_resolved(self) -> "CarryRiskSettings":
        missing = [
            name
            for name in ("reference_notional", "audit_spot_bump_rel")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValidationError(
                f"carry risk settings are unresolved: {', '.join(missing)}"
            )
        return self


def _normalised_dates(dates: Sequence[datetime]) -> Tuple[datetime, ...]:
    stamps = tuple(dates)
    for stamp in stamps:
        if not hasattr(stamp, "year"):
            raise ValidationError(f"expected a date-like value, got {stamp!r}")
    return tuple(sorted(set(stamps)))


def _finite_tuple(values: Sequence[float], name: str) -> Tuple[float, ...]:
    return tuple(_finite(v, name) for v in values)
