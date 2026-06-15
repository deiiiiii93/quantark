"""Public contracts for structured volatility-model risk."""

from __future__ import annotations

import math
from numbers import Integral
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston.params import HestonParams


class SurfaceBucketKind(Enum):
    """Sticky-strike surface bump shape."""

    PARALLEL = "parallel"
    MATURITY_ROW = "maturity_row"
    STRIKE_COLUMN = "strike_column"
    NODE = "node"


class SlvLeverageRiskMode(Enum):
    """Treatment of the SLV leverage surface under model-parameter bumps.

    ``RECALIBRATE`` performs one leverage calibration per scenario. Combining it with
    ``all_surface_bumps`` can therefore be expensive on large grids.
    """

    FROZEN = "frozen"
    RECALIBRATE = "recalibrate"


@dataclass(frozen=True)
class SurfaceBump:
    """A sticky-strike surface bucket and absolute bump magnitude.

    Quote and local-vol surfaces interpret ``bump_size`` as an absolute volatility
    change. Leverage surfaces interpret it as a relative multiplicative change.
    """

    kind: SurfaceBucketKind
    bump_size: float = 0.01
    maturity_index: Optional[int] = None
    strike_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SurfaceBucketKind):
            raise ValidationError("kind must be a SurfaceBucketKind")
        if not math.isfinite(self.bump_size) or self.bump_size <= 0:
            raise ValidationError("bump_size must be finite and positive")
        if self.kind in (SurfaceBucketKind.MATURITY_ROW, SurfaceBucketKind.NODE):
            if self.maturity_index is None or self.maturity_index < 0:
                raise ValidationError(f"{self.kind.value} bump requires maturity_index >= 0")
        elif self.maturity_index is not None:
            raise ValidationError(f"{self.kind.value} bump must not set maturity_index")
        if self.kind in (SurfaceBucketKind.STRIKE_COLUMN, SurfaceBucketKind.NODE):
            if self.strike_index is None or self.strike_index < 0:
                raise ValidationError(f"{self.kind.value} bump requires strike_index >= 0")
        elif self.strike_index is not None:
            raise ValidationError(f"{self.kind.value} bump must not set strike_index")

    @classmethod
    def parallel(cls, bump_size: float = 0.01) -> "SurfaceBump":
        return cls(SurfaceBucketKind.PARALLEL, bump_size)

    @classmethod
    def maturity(cls, maturity_index: int, bump_size: float = 0.01) -> "SurfaceBump":
        return cls.maturity_row(maturity_index, bump_size)

    @classmethod
    def strike(cls, strike_index: int, bump_size: float = 0.01) -> "SurfaceBump":
        return cls.strike_column(strike_index, bump_size)

    @classmethod
    def maturity_row(cls, maturity_index: int, bump_size: float = 0.01) -> "SurfaceBump":
        return cls(SurfaceBucketKind.MATURITY_ROW, bump_size, maturity_index=maturity_index)

    @classmethod
    def strike_column(cls, strike_index: int, bump_size: float = 0.01) -> "SurfaceBump":
        return cls(SurfaceBucketKind.STRIKE_COLUMN, bump_size, strike_index=strike_index)

    @classmethod
    def node(
        cls, maturity_index: int, strike_index: int, bump_size: float = 0.01
    ) -> "SurfaceBump":
        return cls(
            SurfaceBucketKind.NODE,
            bump_size,
            maturity_index=maturity_index,
            strike_index=strike_index,
        )

    @property
    def label(self) -> str:
        if self.kind == SurfaceBucketKind.PARALLEL:
            return "parallel"
        if self.kind == SurfaceBucketKind.MATURITY_ROW:
            return f"maturity_row[{self.maturity_index}]"
        if self.kind == SurfaceBucketKind.STRIKE_COLUMN:
            return f"strike_column[{self.strike_index}]"
        return f"node[{self.maturity_index},{self.strike_index}]"


@dataclass(frozen=True)
class VolRiskPoint:
    """One finite-difference structured-risk result."""

    name: str
    bump_size: float
    base_price: float
    derivative: Optional[float]
    pnl: Optional[float]
    up_price: Optional[float]
    down_price: Optional[float]
    status: str
    error: Optional[str] = None
    difference_mode: str = "central"

    def __post_init__(self) -> None:
        if not math.isfinite(self.bump_size) or self.bump_size <= 0:
            raise ValidationError("bump_size must be finite and positive")
        if not math.isfinite(self.base_price):
            raise ValidationError("base_price must be finite")
        for name, value in (
            ("derivative", self.derivative),
            ("pnl", self.pnl),
            ("up_price", self.up_price),
            ("down_price", self.down_price),
        ):
            if value is not None and not math.isfinite(value):
                raise ValidationError(f"{name} must be finite when provided")
        if self.status not in ("ok", "failed"):
            raise ValidationError("status must be 'ok' or 'failed'")

    @classmethod
    def success(
        cls,
        name: str,
        bump_size: float,
        base_price: float,
        up_price: Optional[float],
        down_price: Optional[float],
        difference_mode: str = "central",
    ) -> "VolRiskPoint":
        if bump_size <= 0:
            raise ValidationError("bump_size must be positive")
        if difference_mode == "central":
            if up_price is None or down_price is None:
                raise ValidationError("central result requires up_price and down_price")
            derivative = (up_price - down_price) / (2.0 * bump_size)
            pnl = (up_price - down_price) / 2.0
        elif difference_mode == "one_sided_up":
            if up_price is None:
                raise ValidationError("one_sided_up result requires up_price")
            derivative = (up_price - base_price) / bump_size
            pnl = up_price - base_price
        elif difference_mode == "one_sided_down":
            if down_price is None:
                raise ValidationError("one_sided_down result requires down_price")
            derivative = (base_price - down_price) / bump_size
            pnl = base_price - down_price
        else:
            raise ValidationError(f"unknown difference_mode: {difference_mode}")
        return cls(
            name=name,
            bump_size=float(bump_size),
            base_price=float(base_price),
            derivative=float(derivative),
            pnl=float(pnl),
            up_price=None if up_price is None else float(up_price),
            down_price=None if down_price is None else float(down_price),
            status="ok",
            difference_mode=difference_mode,
        )

    @classmethod
    def failed(
        cls,
        name: str,
        bump_size: float,
        base_price: float,
        error: Exception | str,
        up_price: Optional[float] = None,
        down_price: Optional[float] = None,
    ) -> "VolRiskPoint":
        return cls(
            name=name,
            bump_size=float(bump_size),
            base_price=float(base_price),
            derivative=None,
            pnl=None,
            up_price=None if up_price is None else float(up_price),
            down_price=None if down_price is None else float(down_price),
            status="failed",
            error=str(error),
        )


@dataclass(frozen=True)
class VolRiskResult:
    """Trade-level structured volatility-risk result.

    For SLV MC risk using a default engine with no precomputed leverage, ``base_price``
    is priced with the leverage surface materialized by ``SlvCalibrationSpec``. It may
    differ slightly from the engine's standalone on-the-fly-binning price.
    """

    base_price: float
    points: Tuple[VolRiskPoint, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_price):
            raise ValidationError("base_price must be finite")
        object.__setattr__(self, "points", tuple(self.points))
        if not all(isinstance(point, VolRiskPoint) for point in self.points):
            raise ValidationError("points must contain VolRiskPoint values")

    @property
    def successful_points(self) -> Tuple[VolRiskPoint, ...]:
        return tuple(point for point in self.points if point.status == "ok")

    @property
    def failed_points(self) -> Tuple[VolRiskPoint, ...]:
        return tuple(point for point in self.points if point.status != "ok")


@dataclass(frozen=True)
class ModelRiskRequest:
    """Request for raw model-artifact sensitivities.

    Central differences are the default even near parameter boundaries. Invalid sides
    produce a failed point; they switch to a one-sided result only when
    ``allow_one_sided`` is explicitly enabled.
    """

    parameter_names: Tuple[str, ...] = ("v0", "kappa", "theta", "sigma", "rho", "eta")
    surface_bumps: Tuple[SurfaceBump, ...] = field(
        default_factory=lambda: (SurfaceBump.parallel(),)
    )
    relative_parameter_bump: float = 0.01
    variance_bump_floor: float = 1e-4
    positive_parameter_bump_floor: float = 1e-3
    rho_bump: float = 1e-3
    allow_one_sided: bool = False
    slv_leverage_mode: Optional[SlvLeverageRiskMode] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(self, "surface_bumps", tuple(self.surface_bumps))
        allowed = {"v0", "kappa", "theta", "sigma", "rho", "eta"}
        unknown = tuple(name for name in self.parameter_names if name not in allowed)
        if unknown:
            raise ValidationError(f"unknown model parameter names: {unknown}")
        if not all(isinstance(bump, SurfaceBump) for bump in self.surface_bumps):
            raise ValidationError("surface_bumps must contain SurfaceBump values")
        if self.slv_leverage_mode is not None and not isinstance(
            self.slv_leverage_mode, SlvLeverageRiskMode
        ):
            raise ValidationError("slv_leverage_mode must be a SlvLeverageRiskMode")
        for name, value in (
            ("relative_parameter_bump", self.relative_parameter_bump),
            ("variance_bump_floor", self.variance_bump_floor),
            ("positive_parameter_bump_floor", self.positive_parameter_bump_floor),
            ("rho_bump", self.rho_bump),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValidationError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MarketVegaRequest:
    """Request for sticky-strike market-IV quote sensitivities."""

    surface_bumps: Tuple[SurfaceBump, ...] = field(
        default_factory=lambda: (SurfaceBump.parallel(),)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface_bumps", tuple(self.surface_bumps))
        if not all(isinstance(bump, SurfaceBump) for bump in self.surface_bumps):
            raise ValidationError("surface_bumps must contain SurfaceBump values")


@dataclass(frozen=True)
class HestonCalibrationSpec:
    """Configuration used for base and bumped Heston calibrations."""

    initial_params: Optional[HestonParams] = None
    bounds: Tuple[
        Tuple[float, float, float, float, float],
        Tuple[float, float, float, float, float],
    ] = (
        (1e-8, 1e-6, 1e-6, 1e-6, -0.999),
        (5.0, 50.0, 5.0, 5.0, 0.999),
    )
    regularize_feller: float = 1e-4
    method: str = "gatheral"
    max_nfev: int = 200
    xtol: float = 1e-6
    ftol: float = 1e-6
    gtol: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", tuple(tuple(side) for side in self.bounds))
        if self.initial_params is not None and not isinstance(self.initial_params, HestonParams):
            raise ValidationError("initial_params must be HestonParams when provided")
        if not math.isfinite(self.regularize_feller) or self.regularize_feller < 0:
            raise ValidationError("regularize_feller must be finite and non-negative")
        if len(self.bounds) != 2 or any(len(side) != 5 for side in self.bounds):
            raise ValidationError("bounds must contain lower and upper five-parameter tuples")
        if any(
            not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper
            for lower, upper in zip(self.bounds[0], self.bounds[1])
        ):
            raise ValidationError("each lower Heston bound must be finite and below its upper")
        if self.max_nfev <= 0:
            raise ValidationError("max_nfev must be positive")
        for name, value in (("xtol", self.xtol), ("ftol", self.ftol), ("gtol", self.gtol)):
            if not math.isfinite(value) or value <= 0:
                raise ValidationError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class SlvCalibrationSpec:
    """Configuration used to materialize SLV leverage surfaces."""

    num_paths: int = 50_000
    time_steps: int = 100
    num_bins: int = 20
    bin_method: str = "equal_weighted"
    seed: int = 42
    n_strike_nodes: int = 41
    strike_span_stds: float = 4.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bin_method", getattr(self.bin_method, "value", self.bin_method))
        if (
            not isinstance(self.num_paths, Integral)
            or not isinstance(self.time_steps, Integral)
            or not isinstance(self.num_bins, Integral)
            or self.num_paths <= 0
            or self.time_steps <= 0
            or self.num_bins < 2
        ):
            raise ValidationError("SLV calibration paths/steps must be positive and bins >= 2")
        if self.bin_method not in ("equidistant", "equal_weighted"):
            raise ValidationError("bin_method must be 'equidistant' or 'equal_weighted'")
        if not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValidationError("seed must be a non-negative integer")
        if (
            not isinstance(self.n_strike_nodes, Integral)
            or self.n_strike_nodes < 2
            or not math.isfinite(self.strike_span_stds)
            or self.strike_span_stds <= 0
        ):
            raise ValidationError("n_strike_nodes must be >= 2 and strike_span_stds positive")
