"""YAML -> ``RunConfig``, fail-closed.

A run config is declarative and diffable: one versionable file reproduces a
run.  The loader validates structure and rejects unknown keys, because a typo
that silently fell back to a default would produce a run nobody asked for --
and the artifacts it wrote would carry a builder fingerprint nobody can
explain.  Every structural error names the YAML path that caused it, so a typo
in a config file diagnoses itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)

SCHEMA_VERSION = 1

TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "name", "underlying", "paths", "surface", "calibration", "run"}
)
UNDERLYING_KEYS = frozenset({"symbol", "convention", "price_field"})
PATHS_KEYS = frozenset({"root", "runtime", "spot_csv"})
SURFACE_KEYS = frozenset(
    {
        "sabr_beta",
        "min_expiries",
        "min_strikes_per_expiry",
        "min_common_strikes",
        "extrapolation",
        "parity_gate",
    }
)
PARITY_GATE_KEYS = frozenset({"max_abs_implied_rate", "max_rmse_over_forward"})
CALIBRATION_KEYS = frozenset(
    {
        "variants",
        "heston_preset",
        "heston_max_nfev",
        "slv",
        "temporal_smoothing",
        "structural_ewma_span",
        "heston_temporal_regularization",
    }
)
SLV_KEYS = frozenset({"eta", "n_steps", "n_x", "n_z"})
RUN_KEYS = frozenset({"workers"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(block: Mapping[str, Any], known: frozenset, path: str) -> None:
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValidationError(
            f"{path}: unknown key(s) {unknown}; known keys are {sorted(known)}"
        )


def _require(block: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in block:
        raise ValidationError(f"missing required key {path}.{key}")
    return block[key]


def load_run_config_text(text: str, *, base_dir) -> RunConfig:
    """Parse run-config YAML. Relative paths resolve against ``base_dir``."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError(f"run config is not valid YAML: {exc}") from exc
    document = _mapping(document, "run config")
    _reject_unknown(document, TOP_LEVEL_KEYS, "run config")

    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValidationError(
            f"run config schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )

    underlying_block = _mapping(
        _require(document, "underlying", "run config"), "underlying"
    )
    _reject_unknown(underlying_block, UNDERLYING_KEYS, "underlying")
    underlying = UnderlyingConfig(
        symbol=str(_require(underlying_block, "symbol", "underlying")),
        convention=str(_require(underlying_block, "convention", "underlying")),
        price_field=str(_require(underlying_block, "price_field", "underlying")),
    )

    paths_block = _mapping(_require(document, "paths", "run config"), "paths")
    _reject_unknown(paths_block, PATHS_KEYS, "paths")
    base = Path(base_dir)
    root = base / str(_require(paths_block, "root", "paths"))
    runtime = (
        (base / str(paths_block["runtime"])).resolve()
        if "runtime" in paths_block
        else root
    )
    spot_csv: Optional[Path] = (
        base / str(paths_block["spot_csv"]) if "spot_csv" in paths_block else None
    )

    surface_block = _mapping(document.get("surface"), "surface")
    _reject_unknown(surface_block, SURFACE_KEYS, "surface")
    gate = _mapping(surface_block.get("parity_gate"), "surface.parity_gate")
    _reject_unknown(gate, PARITY_GATE_KEYS, "surface.parity_gate")
    defaults = SurfaceBuildConfig()
    surface = SurfaceBuildConfig(
        sabr_beta=float(surface_block.get("sabr_beta", defaults.sabr_beta)),
        min_expiries=int(surface_block.get("min_expiries", defaults.min_expiries)),
        min_strikes_per_expiry=int(
            surface_block.get(
                "min_strikes_per_expiry", defaults.min_strikes_per_expiry
            )
        ),
        min_common_strikes=int(
            surface_block.get("min_common_strikes", defaults.min_common_strikes)
        ),
        extrapolation=str(surface_block.get("extrapolation", defaults.extrapolation)),
        max_abs_implied_rate=float(
            gate.get("max_abs_implied_rate", defaults.max_abs_implied_rate)
        ),
        max_rmse_over_forward=float(
            gate.get("max_rmse_over_forward", defaults.max_rmse_over_forward)
        ),
    )

    calibration_block = _mapping(document.get("calibration"), "calibration")
    _reject_unknown(calibration_block, CALIBRATION_KEYS, "calibration")
    slv = _mapping(calibration_block.get("slv"), "calibration.slv")
    _reject_unknown(slv, SLV_KEYS, "calibration.slv")
    cal_defaults = CalibrationRunConfig()
    calibration = CalibrationRunConfig(
        variants=tuple(calibration_block.get("variants", cal_defaults.variants)),
        heston_preset=str(
            calibration_block.get("heston_preset", cal_defaults.heston_preset)
        ),
        heston_max_nfev=int(
            calibration_block.get("heston_max_nfev", cal_defaults.heston_max_nfev)
        ),
        slv_eta=float(slv.get("eta", cal_defaults.slv_eta)),
        slv_n_steps=int(slv.get("n_steps", cal_defaults.slv_n_steps)),
        slv_n_x=int(slv.get("n_x", cal_defaults.slv_n_x)),
        slv_n_z=int(slv.get("n_z", cal_defaults.slv_n_z)),
        temporal_smoothing=bool(
            calibration_block.get(
                "temporal_smoothing", cal_defaults.temporal_smoothing
            )
        ),
        structural_ewma_span=int(
            calibration_block.get(
                "structural_ewma_span", cal_defaults.structural_ewma_span
            )
        ),
        heston_temporal_regularization=float(
            calibration_block.get(
                "heston_temporal_regularization",
                cal_defaults.heston_temporal_regularization,
            )
        ),
    )

    run_block = _mapping(document.get("run"), "run")
    _reject_unknown(run_block, RUN_KEYS, "run")

    return RunConfig(
        name=str(_require(document, "name", "run config")),
        underlying=underlying,
        history_dir=root,
        runtime_dir=runtime,
        spot_csv=spot_csv,
        surface=surface,
        calibration=calibration,
        workers=int(run_block.get("workers", 1)),
    )


def load_run_config(path) -> RunConfig:
    """Load a run config from a YAML file."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot read run config {source}: {exc}") from exc
    return load_run_config_text(text, base_dir=source.parent)
