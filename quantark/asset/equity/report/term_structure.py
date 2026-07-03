"""Term-structure wrappers for bucketed risk analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, List

from quantark.param.div import DividendYield
from quantark.param.vol import VolatilitySurface
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_log, validate_positive


def _check_carry_bound(q: float) -> float:
    """Enforce the signed-carry policy bound |q| <= 1.0 on bumped outputs.

    Matches the TermStructureDividendYield node bound so a valid base plus a
    valid bump cannot produce a yield outside the library-wide sanity range.
    """
    if not math.isfinite(q) or abs(q) > 1.0:
        raise ValidationError(
            f"bumped dividend yield magnitude must be <= 1.0, got {q}"
        )
    return q


@dataclass(frozen=True)
class TenorBucket:
    label: str
    start: float
    end: float


def default_tenor_buckets(maturity: float) -> List[TenorBucket]:
    if maturity <= 0:
        raise ValidationError(f"maturity must be positive, got {maturity}")

    edges = [
        ("1M", 1.0 / 12.0),
        ("3M", 0.25),
        ("6M", 0.50),
        ("1Y", 1.00),
    ]
    buckets: List[TenorBucket] = []
    prev = 0.0
    for label, edge in edges:
        if maturity <= prev:
            break
        end = min(edge, maturity)
        buckets.append(TenorBucket(label=label, start=prev, end=end))
        prev = end

    if maturity > prev:
        buckets.append(TenorBucket(label="1Y+", start=prev, end=maturity))

    return buckets


@dataclass(frozen=True)
class BucketedVolSurface(VolatilitySurface):
    base: VolatilitySurface
    bucket_start: float
    bucket_end: float
    bump: float

    def __post_init__(self) -> None:
        if self.bucket_start < 0:
            raise ValidationError("bucket_start must be non-negative")
        if self.bucket_end <= self.bucket_start:
            raise ValidationError("bucket_end must be > bucket_start")

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        base_vol = float(self.base.get_vol(strike, time_to_maturity, spot))
        if self.bucket_start < time_to_maturity <= self.bucket_end:
            bumped = base_vol + float(self.bump)
            validate_positive(bumped, name="volatility")
            return bumped
        return base_vol


@dataclass(frozen=True)
class ScaledVolSurface(VolatilitySurface):
    base: VolatilitySurface
    scale: float

    def __post_init__(self) -> None:
        validate_positive(float(self.scale), name="scale")

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        base_vol = float(self.base.get_vol(strike, time_to_maturity, spot))
        scaled = base_vol * float(self.scale)
        validate_positive(scaled, name="volatility")
        return scaled


@dataclass(frozen=True)
class BucketedDividendYield(DividendYield):
    base: DividendYield
    bucket_start: float
    bucket_end: float
    bump: float

    def __post_init__(self) -> None:
        if self.bucket_start < 0:
            raise ValidationError("bucket_start must be non-negative")
        if self.bucket_end <= self.bucket_start:
            raise ValidationError("bucket_end must be > bucket_start")
        if not math.isfinite(float(self.bump)) or abs(float(self.bump)) > 1.0:
            raise ValidationError(
                f"bump must be finite with magnitude <= 1.0, got {self.bump}"
            )

    def get_yield(self, time_to_maturity: float) -> float:
        base_yield = float(self.base.get_yield(time_to_maturity))
        if self.bucket_start < time_to_maturity <= self.bucket_end:
            return _check_carry_bound(base_yield + float(self.bump))
        return base_yield


@dataclass(frozen=True)
class ShiftedDividendYield(DividendYield):
    base: DividendYield
    shift: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.shift)) or abs(float(self.shift)) > 1.0:
            raise ValidationError(
                f"shift must be finite with magnitude <= 1.0, got {self.shift}"
            )

    def get_yield(self, time_to_maturity: float) -> float:
        base_yield = float(self.base.get_yield(time_to_maturity))
        return _check_carry_bound(base_yield + float(self.shift))


@dataclass(frozen=True)
class SkewSmileVolSurface(VolatilitySurface):
    is_smile: ClassVar[bool] = True

    base: VolatilitySurface
    skew: float = 0.0
    smile: float = 0.0

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        base_vol = float(self.base.get_vol(strike, time_to_maturity, spot))
        moneyness = safe_log(float(strike) / float(spot))
        bumped = base_vol + float(self.skew) * moneyness + float(self.smile) * (
            moneyness * moneyness
        )
        validate_positive(bumped, name="volatility")
        return bumped
