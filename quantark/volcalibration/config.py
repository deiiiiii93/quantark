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
from typing import Any, Dict, Optional

from quantark.util.exceptions import ValidationError

HESTON_PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")


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


