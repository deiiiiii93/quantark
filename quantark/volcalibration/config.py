"""Vol-model calibration configuration.

Canonical home for ``VolModelCalibrationConfig`` and the frozen Heston presets.
Both previously lived one layer up -- the config in
``quantark.backtest.replay.config`` and the presets in
``quantark.volmodels.calibration`` -- which made the calibration engine
duck-type a config owned by the backtest package.  They live here so the
dependency runs downward: backtest imports volcalibration, never the reverse.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.snapshot import (
    CONVENTION_FX_DELTA,
    CONVENTION_LISTED,
    CONVENTIONS,
    PRICE_FIELD_MID_OR_LAST,
    PRICE_FIELD_SETTLEMENT,
)

HESTON_PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")

# Frozen Heston calibration preset "mo_frozen".
# Provenance: values copied from the mo_volmodels suite —
# example/mo_volmodels/04_heston_calibration.py (HESTON_BOUNDS,
# REGULARIZE_FELLER, SOLVER_TOLERANCES, target="iv", method="lewis", single
# deterministic shortest-expiry ATM-variance start) and
# example/mo_volmodels/10_calibration_diagnostics.py (same bounds, frozen
# 2026-07 for the CFFEX MO cohort).  example/ scripts are NOT importable
# from quantark (canonical-import rule), so the frozen configuration is
# re-declared here; keep it in sync with the suite.
#
# DELIBERATE DIVERGENCE from those scripts: enforce_feller is True here,
# where they use False with only the soft regularize_feller penalty.  Every
# MO settlement surface sampled under the soft policy came back with
# kappa/sigma pinned on the bounds at 2*kappa*theta/sigma^2 ~ 0.29-0.48, and
# the 0.4.0 re-baseline design (§7A) traced a 2.5%-of-notional 2D-PDE vs
# QE-M-MC gap to exactly that degeneracy.  Feasibility is bought with smile
# accuracy on those dates, so runs MUST report per-date fit RMSE — the model
# under test is not the same one the diagnostics scripts fitted.  The soft
# penalty is kept: it still shapes the interior of the feasible region.
HESTON_PRESETS: Dict[str, Dict[str, Any]] = {
    "mo_frozen": {
        "bounds": (
            (1e-6, 1e-3, 1e-4, 1e-3, -0.95),
            (0.5, 3.0, 0.5, 0.7, 0.0),
        ),
        "regularize_feller": 0.05,
        "solver_tolerances": {"xtol": 1e-6, "ftol": 1e-6, "gtol": 1e-6},
        "target": "iv",
        "method": "lewis",
        "enforce_feller": True,
    }
}


@dataclass
class VolModelCalibrationConfig:
    """
    Calibration options for the vol-model variants (Task 2.4).

    ``cache_dir`` selects the persistent calibration cache (JSON files
    keyed by surface sha + variant + config fingerprint); ``None`` means
    in-memory caching only.  ``records_path`` (optional) makes the engine
    persist per-day calibration records as one JSON file per run.
    ``heston_preset`` selects the frozen Heston calibration configuration
    (only ``"mo_frozen"`` — the mo_volmodels suite values).  ``slv_eta`` /
    ``slv_n_steps`` / ``slv_n_x`` / ``slv_n_z`` steer the forward
    Fokker-Planck leverage calibration (suite defaults 1.0 / 40 / 161 /
    81).  ``heston_temporal_reference`` and
    ``heston_temporal_regularization`` opt into a structural-only temporal
    penalty for Heston; ``slv_heston_override`` supplies an explicit Heston
    vector for leverage calibration.  These advanced fields are normally
    orchestrated by the daily MO pipeline.  ``shared_store`` is an optional
    caller-held dict shared across runs as a common in-memory cache.
    """

    cache_dir: Optional[str] = None
    records_path: Optional[str] = None
    heston_preset: str = "mo_frozen"
    heston_max_nfev: int = 200
    slv_eta: float = 1.0
    slv_n_steps: int = 40
    slv_n_x: int = 161
    slv_n_z: int = 81
    heston_temporal_reference: Optional[
        tuple[float, float, float, float, float]
    ] = None
    heston_temporal_regularization: float = 0.0
    slv_heston_override: Optional[
        tuple[float, float, float, float, float]
    ] = None
    shared_store: Optional[MutableMapping] = None

    def __post_init__(self) -> None:
        if self.heston_preset not in HESTON_PRESETS:
            raise ValidationError(
                f"heston_preset must be one of {sorted(HESTON_PRESETS)}"
            )
        if not isinstance(self.heston_max_nfev, int) or self.heston_max_nfev <= 0:
            raise ValidationError("heston_max_nfev must be a positive integer")
        if not math.isfinite(float(self.slv_eta)) or float(self.slv_eta) < 0.0:
            raise ValidationError("slv_eta must be non-negative and finite")
        if int(self.slv_n_steps) < 1:
            raise ValidationError("slv_n_steps must be >= 1")
        if int(self.slv_n_x) < 3:
            raise ValidationError("slv_n_x must be >= 3")
        if int(self.slv_n_z) < 3:
            raise ValidationError("slv_n_z must be >= 3")
        if not math.isfinite(float(self.heston_temporal_regularization)) or (
            float(self.heston_temporal_regularization) < 0.0
        ):
            raise ValidationError(
                "heston_temporal_regularization must be non-negative and finite"
            )
        self.heston_temporal_reference = self._validate_heston_vector(
            self.heston_temporal_reference,
            "heston_temporal_reference",
        )
        self.slv_heston_override = self._validate_heston_vector(
            self.slv_heston_override,
            "slv_heston_override",
        )
        if (
            float(self.heston_temporal_regularization) > 0.0
            and self.heston_temporal_reference is None
        ):
            raise ValidationError(
                "heston_temporal_reference is required when "
                "heston_temporal_regularization > 0"
            )
        if self.shared_store is not None and not isinstance(
            self.shared_store, MutableMapping
        ):
            raise ValidationError("shared_store must be a mutable mapping")

    def _validate_heston_vector(
        self,
        value: Optional[tuple[float, float, float, float, float]],
        field_name: str,
    ) -> Optional[tuple[float, float, float, float, float]]:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            raise ValidationError(f"{field_name} must contain five parameters")
        try:
            normalized = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{field_name} must contain five finite parameters"
            ) from exc
        if len(normalized) != 5 or not all(math.isfinite(x) for x in normalized):
            raise ValidationError(
                f"{field_name} must contain five finite parameters"
            )
        lower, upper = HESTON_PRESETS[self.heston_preset]["bounds"]
        if any(
            parameter < lo or parameter > hi
            for parameter, lo, hi in zip(normalized, lower, upper)
        ):
            raise ValidationError(
                f"{field_name} parameters must lie within the "
                f"{self.heston_preset!r} bounds"
            )
        return normalized


# --------------------------------------------------------------- run config

TEMPORAL_SCHEME = "daily_v0_structural_ewma"
HESTON_STRUCTURAL_PARAMETER_NAMES = ("kappa", "theta", "sigma", "rho")

# Mirrors surface.ARTIFACT_SCHEMA_VERSION.  Declared as a literal rather than
# imported so that quantark.backtest.replay.config -- which imports this
# module -- does not pull numpy and the SABR stack through surface.py.
# test_runconfig.py asserts the two agree.
_ARTIFACT_SCHEMA_VERSION_IN_FINGERPRINT = 1


@dataclass(frozen=True)
class UnderlyingConfig:
    """What is being calibrated, and which quote convention its snapshots use."""

    symbol: str
    convention: str
    price_field: str

    def __post_init__(self) -> None:
        if not str(self.symbol).strip():
            raise ValidationError("underlying.symbol must be non-empty")
        if self.convention not in CONVENTIONS:
            raise ValidationError(
                f"underlying.convention must be one of {list(CONVENTIONS)}, "
                f"got {self.convention!r}"
            )
        if self.price_field not in (PRICE_FIELD_SETTLEMENT, PRICE_FIELD_MID_OR_LAST):
            raise ValidationError(
                "underlying.price_field must be "
                f"{PRICE_FIELD_SETTLEMENT!r} or {PRICE_FIELD_MID_OR_LAST!r}, "
                f"got {self.price_field!r}"
            )


@dataclass(frozen=True)
class SurfaceBuildConfig:
    """Every value stage 03 hard-coded, with that value as its default."""

    sabr_beta: float = 1.0
    min_expiries: int = 2
    min_strikes_per_expiry: int = 5
    min_common_strikes: int = 3
    extrapolation: str = "flat_total_variance"
    max_abs_implied_rate: float = 0.10
    max_rmse_over_forward: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.sabr_beta) <= 1.0:
            raise ValidationError("surface.sabr_beta must lie in [0, 1]")
        for name in ("min_expiries", "min_strikes_per_expiry", "min_common_strikes"):
            if int(getattr(self, name)) < 2:
                raise ValidationError(f"surface.{name} must be at least 2")
        if float(self.max_abs_implied_rate) <= 0.0:
            raise ValidationError(
                "surface.parity_gate.max_abs_implied_rate must be positive"
            )
        if float(self.max_rmse_over_forward) <= 0.0:
            raise ValidationError(
                "surface.parity_gate.max_rmse_over_forward must be positive"
            )

    def fingerprint_payload(self) -> Dict[str, Any]:
        """The canonical block ``builder_fingerprint`` hashes.

        The five legacy keys are the ones the 787 admitted artifacts were built
        with, so emitting exactly them keeps a default run's fingerprint equal
        to migrated history and nothing rebuilds.  A knob moved off its frozen
        default is appended, which changes the fingerprint and correctly
        invalidates -- so no knob is silently ignored.
        """
        payload: Dict[str, Any] = {
            "sabr_beta": float(self.sabr_beta),
            "min_expiries": int(self.min_expiries),
            "min_strikes_per_expiry": int(self.min_strikes_per_expiry),
            "min_common_strikes": int(self.min_common_strikes),
            "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION_IN_FINGERPRINT,
        }
        defaults = SurfaceBuildConfig()
        for name in ("extrapolation", "max_abs_implied_rate", "max_rmse_over_forward"):
            value = getattr(self, name)
            if value != getattr(defaults, name):
                payload[name] = value
        return payload


@dataclass(frozen=True)
class CalibrationRunConfig:
    """Which vol models to fit for each admitted surface, and with what settings."""

    variants: Tuple[str, ...] = ("localvol", "heston", "heston_slv")
    heston_preset: str = "mo_frozen"
    heston_max_nfev: int = 200
    slv_eta: float = 1.0
    slv_n_steps: int = 40
    slv_n_x: int = 161
    slv_n_z: int = 81
    temporal_smoothing: bool = False
    structural_ewma_span: int = 20
    heston_temporal_regularization: float = 0.0

    def __post_init__(self) -> None:
        # Imported lazily: calibrate.py imports this module, so a module-level
        # import here would close the cycle.
        from quantark.volcalibration.calibrate import VOL_MODEL_VARIANTS

        if not self.variants:
            raise ValidationError("calibration.variants must not be empty")
        unknown = [v for v in self.variants if v not in VOL_MODEL_VARIANTS]
        if unknown:
            raise ValidationError(
                f"calibration.variants: unknown {unknown}; "
                f"known variants are {list(VOL_MODEL_VARIANTS)}"
            )
        if self.heston_preset not in HESTON_PRESETS:
            raise ValidationError(
                f"calibration.heston_preset must be one of {sorted(HESTON_PRESETS)}, "
                f"got {self.heston_preset!r}"
            )
        if int(self.structural_ewma_span) < 1:
            raise ValidationError("calibration.structural_ewma_span must be at least 1")

    def manifest_payload(self) -> Dict[str, Any]:
        """The per-record ``config`` block, shaped like the live manifest's."""
        payload: Dict[str, Any] = {
            "variants": list(self.variants),
            "heston_preset": str(self.heston_preset),
            "heston_max_nfev": int(self.heston_max_nfev),
            "slv_eta": float(self.slv_eta),
            "slv_n_steps": int(self.slv_n_steps),
            "slv_n_x": int(self.slv_n_x),
            "slv_n_z": int(self.slv_n_z),
        }
        if self.temporal_smoothing:
            span = int(self.structural_ewma_span)
            payload["temporal_scheme"] = {
                "name": TEMPORAL_SCHEME,
                "structural_ewma_span": span,
                "structural_ewma_alpha": 2.0 / (span + 1.0),
                "heston_temporal_regularization": float(
                    self.heston_temporal_regularization
                ),
                "daily_parameters": ["v0"],
                "structural_parameters": list(HESTON_STRUCTURAL_PARAMETER_NAMES),
            }
        return payload

    def calibrator_config(self, cache_dir) -> "VolModelCalibrationConfig":
        """The engine-level config this run's calibrator is built from."""
        return VolModelCalibrationConfig(
            cache_dir=str(cache_dir),
            heston_preset=self.heston_preset,
            heston_max_nfev=int(self.heston_max_nfev),
            slv_eta=float(self.slv_eta),
            slv_n_steps=int(self.slv_n_steps),
            slv_n_x=int(self.slv_n_x),
            slv_n_z=int(self.slv_n_z),
        )


@dataclass(frozen=True)
class RunConfig:
    """One resolved, versionable calibration run."""

    name: str
    underlying: UnderlyingConfig
    history_dir: Path
    runtime_dir: Path
    spot_csv: Optional[Path]
    surface: SurfaceBuildConfig
    calibration: CalibrationRunConfig
    workers: int = 1

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValidationError("name must be non-empty")
        if int(self.workers) < 1:
            raise ValidationError("run.workers must be at least 1")

    def echo(self) -> Dict[str, Any]:
        """The resolved config, as echoed into both manifests and status."""
        return {
            "name": self.name,
            "underlying": {
                "symbol": self.underlying.symbol,
                "convention": self.underlying.convention,
                "price_field": self.underlying.price_field,
            },
            "surface": self.surface.fingerprint_payload(),
            "calibration": self.calibration.manifest_payload(),
            "workers": int(self.workers),
        }


