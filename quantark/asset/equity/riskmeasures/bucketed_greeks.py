"""Contracts for consolidated equity bucketed Greek results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

from quantark.util.exceptions import ValidationError
from quantark.volmodels.risk import (
    HestonCalibrationSpec,
    MarketVegaRequest,
    ModelRiskRequest,
    SlvCalibrationSpec,
)

if TYPE_CHECKING:
    from quantark.asset.equity.report.term_structure import TenorBucket


class BucketedGreekCoordinate(Enum):
    FUTURES_DELTA = "futures_delta"
    CARRY_RHOQ = "carry_rhoq"
    VOL_TENOR_VEGA = "vol_tenor_vega"
    MARKET_IV_VEGA = "market_iv_vega"
    MODEL_ARTIFACT = "model_artifact"


class BucketedGreekDifferenceMode(Enum):
    COORDINATE_DEFAULT = "coordinate_default"
    CENTRAL = "central"
    ONE_SIDED_UP = "one_sided_up"


@dataclass(frozen=True)
class BucketedGreeksRequest:
    coordinates: Optional[Tuple[BucketedGreekCoordinate, ...]] = None
    difference_mode: BucketedGreekDifferenceMode = (
        BucketedGreekDifferenceMode.COORDINATE_DEFAULT
    )
    difference_mode_overrides: Mapping[
        BucketedGreekCoordinate, BucketedGreekDifferenceMode
    ] = field(default_factory=dict)
    futures_curve: Optional[object] = None
    futures_price_bump: float = 1.0
    carry_bump: Optional[float] = None
    vol_bump: float = 0.01
    tenor_buckets: Optional[Tuple["TenorBucket", ...]] = None
    futures_carry_mode: Optional[object] = None
    market_vega_request: Optional[MarketVegaRequest] = None
    model_risk_request: Optional[ModelRiskRequest] = None
    heston_calibration_spec: Optional[HestonCalibrationSpec] = None
    slv_calibration_spec: Optional[SlvCalibrationSpec] = None
    allow_partial: bool = False

    def __post_init__(self) -> None:
        if self.coordinates is not None:
            coordinates = tuple(self.coordinates)
            if len(coordinates) == 0:
                raise ValidationError("coordinates must be non-empty when provided")
            if not all(
                isinstance(coord, BucketedGreekCoordinate) for coord in coordinates
            ):
                raise ValidationError(
                    "coordinates must contain BucketedGreekCoordinate values"
                )
            object.__setattr__(self, "coordinates", coordinates)

        if not isinstance(self.difference_mode, BucketedGreekDifferenceMode):
            raise ValidationError(
                "difference_mode must be a BucketedGreekDifferenceMode"
            )

        overrides = dict(self.difference_mode_overrides)
        if not all(isinstance(key, BucketedGreekCoordinate) for key in overrides):
            raise ValidationError(
                "difference_mode_overrides keys must be BucketedGreekCoordinate values"
            )
        if not all(
            isinstance(value, BucketedGreekDifferenceMode)
            for value in overrides.values()
        ):
            raise ValidationError(
                "difference_mode_overrides values must be BucketedGreekDifferenceMode values"
            )
        object.__setattr__(self, "difference_mode_overrides", overrides)

        self._validate_positive_bump("futures_price_bump", self.futures_price_bump)
        self._validate_positive_bump("vol_bump", self.vol_bump)
        if self.carry_bump is not None:
            self._validate_positive_bump("carry_bump", self.carry_bump)

        if self.tenor_buckets is not None:
            from quantark.asset.equity.report.term_structure import TenorBucket

            tenor_buckets = tuple(self.tenor_buckets)
            if len(tenor_buckets) == 0:
                raise ValidationError("tenor_buckets must be non-empty when provided")
            if not all(isinstance(bucket, TenorBucket) for bucket in tenor_buckets):
                raise ValidationError(
                    "tenor_buckets must contain TenorBucket values"
                )
            object.__setattr__(self, "tenor_buckets", tenor_buckets)

    @staticmethod
    def _validate_positive_bump(name: str, value: float) -> None:
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValidationError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class BucketedGreekPoint:
    coordinate: BucketedGreekCoordinate
    name: str
    reported: Optional[float]
    derivative: Optional[float]
    pnl: Optional[float]
    bump_size: float
    convention_scale: float
    base_price: float
    up_price: Optional[float] = None
    down_price: Optional[float] = None
    difference_mode: str = BucketedGreekDifferenceMode.ONE_SIDED_UP.value
    status: str = "ok"
    bucket: Optional[str] = None
    contract: Optional[str] = None
    maturity: Optional[float] = None
    future_price: Optional[float] = None
    delta_per_hand: Optional[float] = None
    hedge_hands: Optional[float] = None
    extrapolated_tail: Optional[bool] = None
    model: Optional[str] = None
    unit: str = "price"
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate, BucketedGreekCoordinate):
            raise ValidationError("coordinate must be a BucketedGreekCoordinate")
        if not self.name:
            raise ValidationError("name must be non-empty")
        if self.status not in {"ok", "failed"}:
            raise ValidationError("status must be 'ok' or 'failed'")
        if isinstance(self.difference_mode, BucketedGreekDifferenceMode):
            object.__setattr__(self, "difference_mode", self.difference_mode.value)
        elif not isinstance(self.difference_mode, str):
            raise ValidationError("difference_mode must be a string")
        if self.difference_mode not in {
            mode.value for mode in BucketedGreekDifferenceMode
        } | {"one_sided_down"}:
            raise ValidationError("difference_mode is not supported")

        self._validate_finite("bump_size", self.bump_size)
        self._validate_finite("convention_scale", self.convention_scale)
        self._validate_finite("base_price", self.base_price)
        if float(self.bump_size) <= 0.0:
            raise ValidationError("bump_size must be positive")
        if float(self.convention_scale) <= 0.0:
            raise ValidationError("convention_scale must be positive")
        if self.maturity is not None:
            self._validate_finite("maturity", self.maturity)
        for name in (
            "up_price",
            "down_price",
            "future_price",
            "delta_per_hand",
            "hedge_hands",
        ):
            value = getattr(self, name)
            if value is not None:
                self._validate_finite(name, value)
        if self.extrapolated_tail is not None and not isinstance(
            self.extrapolated_tail, bool
        ):
            raise ValidationError("extrapolated_tail must be a bool")

        if self.status == "ok":
            for name in ("reported", "derivative", "pnl"):
                value = getattr(self, name)
                if value is None:
                    raise ValidationError(f"{name} is required for ok points")
                self._validate_finite(name, value)
        else:
            if not self.error:
                raise ValidationError("error is required for failed points")
            for name in ("reported", "derivative", "pnl"):
                value = getattr(self, name)
                if value is not None:
                    self._validate_finite(name, value)

        object.__setattr__(self, "metadata", dict(self.metadata))

    @staticmethod
    def _validate_finite(name: str, value: float) -> None:
        if not math.isfinite(float(value)):
            raise ValidationError(f"{name} must be finite")

    @classmethod
    def failed(
        cls,
        *,
        coordinate: BucketedGreekCoordinate,
        name: str,
        bump_size: float,
        base_price: float,
        error: str,
        convention_scale: float = 1.0,
        bucket: Optional[str] = None,
        maturity: Optional[float] = None,
        difference_mode: str = BucketedGreekDifferenceMode.ONE_SIDED_UP.value,
        up_price: Optional[float] = None,
        down_price: Optional[float] = None,
        unit: str = "price",
        model: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "BucketedGreekPoint":
        return cls(
            coordinate=coordinate,
            name=name,
            reported=None,
            derivative=None,
            pnl=None,
            bump_size=bump_size,
            convention_scale=convention_scale,
            base_price=base_price,
            up_price=up_price,
            down_price=down_price,
            status="failed",
            bucket=bucket,
            maturity=maturity,
            difference_mode=difference_mode,
            unit=unit,
            model=model,
            error=error,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class BucketedGreeksResult:
    points: Tuple[BucketedGreekPoint, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if not all(isinstance(point, BucketedGreekPoint) for point in points):
            raise ValidationError(
                "points must contain BucketedGreekPoint values"
            )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def successful_points(self) -> Tuple[BucketedGreekPoint, ...]:
        return tuple(point for point in self.points if point.status == "ok")

    @property
    def failed_points(self) -> Tuple[BucketedGreekPoint, ...]:
        return tuple(point for point in self.points if point.status == "failed")

    def by_coordinate(
        self, coordinate: BucketedGreekCoordinate
    ) -> Tuple[BucketedGreekPoint, ...]:
        return tuple(point for point in self.points if point.coordinate is coordinate)
