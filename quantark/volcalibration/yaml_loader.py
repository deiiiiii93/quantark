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


# YAML scalars are typed, so they are validated rather than coerced.  bool()
# accepts every non-empty string -- a quoted `temporal_smoothing: "false"`
# would silently enable smoothing -- and int()/float() raise builtin ValueError
# or TypeError, which are not QuantArkException and so escape the CLI's handler
# as a traceback instead of the documented JSON error.


def _opt_bool(block: Mapping[str, Any], key: str, path: str, default: bool) -> bool:
    if key not in block:
        return default
    value = block[key]
    if not isinstance(value, bool):
        raise ValidationError(
            f"{path}.{key} must be a YAML boolean (true/false), got {value!r}"
        )
    return value


def _opt_int(block: Mapping[str, Any], key: str, path: str, default: int) -> int:
    if key not in block:
        return default
    value = block[key]
    # bool is an int subclass; `workers: true` is not a worker count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"{path}.{key} must be an integer, got {value!r}"
        )
    return int(value)


def _opt_float(block: Mapping[str, Any], key: str, path: str, default: float) -> float:
    if key not in block:
        return default
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{path}.{key} must be a number, got {value!r}")
    return float(value)


def _req_str(block: Mapping[str, Any], key: str, path: str) -> str:
    value = _require(block, key, path)
    if not isinstance(value, str):
        raise ValidationError(f"{path}.{key} must be a string, got {value!r}")
    return value


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
        symbol=_req_str(underlying_block, "symbol", "underlying"),
        convention=_req_str(underlying_block, "convention", "underlying"),
        price_field=_req_str(underlying_block, "price_field", "underlying"),
    )

    paths_block = _mapping(_require(document, "paths", "run config"), "paths")
    _reject_unknown(paths_block, PATHS_KEYS, "paths")
    base = Path(base_dir)
    root = base / _req_str(paths_block, "root", "paths")
    runtime = (
        (base / _req_str(paths_block, "runtime", "paths")).resolve()
        if "runtime" in paths_block
        else root
    )
    spot_csv: Optional[Path] = (
        base / _req_str(paths_block, "spot_csv", "paths")
        if "spot_csv" in paths_block
        else None
    )

    surface_block = _mapping(document.get("surface"), "surface")
    _reject_unknown(surface_block, SURFACE_KEYS, "surface")
    gate = _mapping(surface_block.get("parity_gate"), "surface.parity_gate")
    _reject_unknown(gate, PARITY_GATE_KEYS, "surface.parity_gate")
    defaults = SurfaceBuildConfig()
    surface = SurfaceBuildConfig(
        sabr_beta=_opt_float(surface_block, "sabr_beta", "surface", defaults.sabr_beta),
        min_expiries=_opt_int(
            surface_block, "min_expiries", "surface", defaults.min_expiries
        ),
        min_strikes_per_expiry=_opt_int(
            surface_block,
            "min_strikes_per_expiry",
            "surface",
            defaults.min_strikes_per_expiry,
        ),
        min_common_strikes=_opt_int(
            surface_block, "min_common_strikes", "surface", defaults.min_common_strikes
        ),
        extrapolation=(
            _req_str(surface_block, "extrapolation", "surface")
            if "extrapolation" in surface_block
            else defaults.extrapolation
        ),
        max_abs_implied_rate=_opt_float(
            gate,
            "max_abs_implied_rate",
            "surface.parity_gate",
            defaults.max_abs_implied_rate,
        ),
        max_rmse_over_forward=_opt_float(
            gate,
            "max_rmse_over_forward",
            "surface.parity_gate",
            defaults.max_rmse_over_forward,
        ),
    )

    calibration_block = _mapping(document.get("calibration"), "calibration")
    _reject_unknown(calibration_block, CALIBRATION_KEYS, "calibration")
    slv = _mapping(calibration_block.get("slv"), "calibration.slv")
    _reject_unknown(slv, SLV_KEYS, "calibration.slv")
    cal_defaults = CalibrationRunConfig()
    variants = calibration_block.get("variants", cal_defaults.variants)
    if not isinstance(variants, (list, tuple)) or not all(
        isinstance(v, str) for v in variants
    ):
        raise ValidationError(
            f"calibration.variants must be a list of strings, got {variants!r}"
        )
    calibration = CalibrationRunConfig(
        variants=tuple(variants),
        heston_preset=(
            _req_str(calibration_block, "heston_preset", "calibration")
            if "heston_preset" in calibration_block
            else cal_defaults.heston_preset
        ),
        heston_max_nfev=_opt_int(
            calibration_block,
            "heston_max_nfev",
            "calibration",
            cal_defaults.heston_max_nfev,
        ),
        slv_eta=_opt_float(slv, "eta", "calibration.slv", cal_defaults.slv_eta),
        slv_n_steps=_opt_int(
            slv, "n_steps", "calibration.slv", cal_defaults.slv_n_steps
        ),
        slv_n_x=_opt_int(slv, "n_x", "calibration.slv", cal_defaults.slv_n_x),
        slv_n_z=_opt_int(slv, "n_z", "calibration.slv", cal_defaults.slv_n_z),
        temporal_smoothing=_opt_bool(
            calibration_block,
            "temporal_smoothing",
            "calibration",
            cal_defaults.temporal_smoothing,
        ),
        structural_ewma_span=_opt_int(
            calibration_block,
            "structural_ewma_span",
            "calibration",
            cal_defaults.structural_ewma_span,
        ),
        heston_temporal_regularization=_opt_float(
            calibration_block,
            "heston_temporal_regularization",
            "calibration",
            cal_defaults.heston_temporal_regularization,
        ),
    )

    run_block = _mapping(document.get("run"), "run")
    _reject_unknown(run_block, RUN_KEYS, "run")

    return RunConfig(
        name=_req_str(document, "name", "run config"),
        underlying=underlying,
        history_dir=root,
        runtime_dir=runtime,
        spot_csv=spot_csv,
        surface=surface,
        calibration=calibration,
        workers=_opt_int(run_block, "workers", "run", 1),
    )


def load_run_config(path) -> RunConfig:
    """Load a run config from a YAML file."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot read run config {source}: {exc}") from exc
    return load_run_config_text(text, base_dir=source.parent)
