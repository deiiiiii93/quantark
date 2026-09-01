"""The producer/consumer handover: a directory of artifacts as one object.

A backtest should not glob directories, parse manifests or rebuild a pricing
environment from an artifact's per-expiry block -- three jobs it did before,
each written three times across the example suite.  ``CalibrationSet`` is the
one supported entry point: open a store, get dates, surfaces, calibrated models
and pricing environments.

This lives beside ``store.py`` rather than inside it because it imports the
pricing-environment stack, and ``store.py`` is deliberately import-light: the
backtest package depends on this module's *type* only through duck typing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quantark.param import GridVolSurface, SpotQuote
from quantark.param.div import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.param.vol.surface_history import IvSurfaceArtifact, VolSurfaceHistory
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import CalibrationRunConfig, RunConfig
from quantark.volcalibration.store import (
    StoreLayout,
    load_calibration_manifest,
    read_json,
)

MIN_CURVE_PILLARS = 2


@dataclass(frozen=True)
class CalibrationSet:
    """Read-only view of one calibration store."""

    layout: StoreLayout
    surface_history: VolSurfaceHistory
    calibration: CalibrationRunConfig
    records: Dict[str, dict] = field(default_factory=dict)

    # ----------------------------------------------------------------- open
    @classmethod
    def open(cls, root, *, runtime=None, calibration=None) -> "CalibrationSet":
        """Validate a store's manifests and return a reader. Fail-closed."""
        layout = StoreLayout(Path(root), Path(runtime) if runtime else Path(root))
        if not layout.surface_manifest.is_file():
            raise ValidationError(
                f"no surface manifest at {layout.surface_manifest}; "
                "this directory is not a calibration store"
            )
        history = VolSurfaceHistory(layout.history_dir)
        _payload, records = load_calibration_manifest(layout)
        return cls(
            layout=layout,
            surface_history=history,
            calibration=calibration or CalibrationRunConfig(),
            records=records,
        )

    @classmethod
    def from_config(cls, config: RunConfig) -> "CalibrationSet":
        return cls.open(
            config.history_dir,
            runtime=config.runtime_dir,
            calibration=config.calibration,
        )

    # ------------------------------------------------------------ accessors
    def dates(self) -> List[date]:
        """Admitted trading dates, ascending."""
        return self.surface_history.admitted_dates

    def surface_for(self, when: date) -> IvSurfaceArtifact:
        """The artifact in force on ``when`` (manifest carry-forward applied)."""
        return self.surface_history.surface_for(when)

    def status(self) -> Dict[str, Any]:
        """The store's last written status payload, or ``{}`` if it never ran."""
        return read_json(self.layout.status, default={})

    # --------------------------------------------------------------- models
    def model_for(self, when: date, variant: str):
        """The calibrated model for the surface in force on ``when``.

        Reads the warm cache keyed on (surface sha, variant, config
        fingerprint), so this is a lookup for a date the runner has already
        calibrated -- but it refuses a date with no record rather than fitting
        one silently, because an unrecorded fit is an unauditable one.
        """
        from quantark.volcalibration.calibrate import VolModelCalibrator

        artifact = self.surface_for(when)
        tag = _tag(artifact)
        record = self.records.get(tag, {})
        if not record:
            raise ValidationError(
                f"{when}: no calibration record for surface {artifact.sha256[:12]} "
                f"({tag}); run `python -m quantark.volcalibration run` first"
            )
        variants = record.get("variants", {})
        if variant not in variants:
            raise ValidationError(
                f"{when}: surface {artifact.sha256[:12]} has no {variant!r} "
                f"calibration; recorded variants are {sorted(variants)}"
            )
        if variants[variant].get("status") != "ok":
            raise ValidationError(
                f"{when}: calibration of {variant!r} for surface "
                f"{artifact.sha256[:12]} failed: {variants[variant].get('error')}"
            )
        calibrator = VolModelCalibrator(
            self.calibration.calibrator_config(self.layout.calibration_cache)
        )
        return calibrator.calibrate(variant, artifact)

    # --------------------------------------------------------- environments
    def environment_for(
        self, when: date
    ) -> Tuple[PricingEnvironment, GridVolSurface, float]:
        """``(PricingEnvironment, GridVolSurface, spot)`` for ``when``.

        The rate and dividend curves are built from the artifact's own parity
        pillars, so the environment carries the exact term structure the
        surface was calibrated against.

        The valuation date is ``when`` -- the date being priced -- not the
        artifact's trade date and not a constant.  ``_mo_common.build_env``
        hardcoded a single calendar day, which is harmless for a one-date demo
        and wrong for a history; and when the manifest carries a surface
        forward, the environment must still be dated to the day it prices.
        """
        artifact = self.surface_for(when)
        pillars = artifact.per_expiry
        if len(pillars) < MIN_CURVE_PILLARS:
            raise ValidationError(
                f"{when}: surface {artifact.sha256[:12]} has {len(pillars)} parity "
                f"pillar(s); both curve types need at least {MIN_CURVE_PILLARS}"
            )
        grid = artifact.grid_vol_surface()
        env = PricingEnvironment(
            rate_curve=LinearRateCurve(
                [(float(p["T"]), float(p["r"])) for p in pillars]
            ),
            valuation_date=datetime(when.year, when.month, when.day),
            spot_quote=SpotQuote(spot=float(artifact.s0)),
            vol_surface=grid,
            div_yield=TermStructureDividendYield(
                times=[float(p["T"]) for p in pillars],
                yields=[float(p["q"]) for p in pillars],
            ),
        )
        return env, grid, float(artifact.s0)


def _tag(artifact: IvSurfaceArtifact) -> str:
    return artifact.trade_date.strftime("%Y%m%d")
