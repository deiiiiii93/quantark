"""
Preset profiles and factory helpers for engine parameter objects.
"""

from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from quantark.util.exceptions import ValidationError

from .engine_params import PDEParams, QuadParams

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

ProfileMap = Dict[str, Dict[str, Any]]

_PROFILE_NAMES = (
    "fast",
    "balanced",
    "accurate",
    "barrier_sensitive",
    "reverse_sensitive",
)

QUAD_PARAM_PRESETS: ProfileMap = {
    "fast": {
        "grid_points": 401,
        "num_std_devs": 8.0,
        "fft_padding_factor": 2,
        "fft_filter_alpha": 12.0,
        "fft_filter_power": 8,
    },
    "balanced": {},
    "accurate": {
        "grid_points": 1601,
        "num_std_devs": 12.0,
        "fft_padding_factor": 2,
        "fft_filter_alpha": 18.0,
        "fft_filter_power": 8,
    },
    "barrier_sensitive": {
        "grid_points": 1201,
        "num_std_devs": 10.0,
        "fft_padding_factor": 2,
        "fft_filter_alpha": 18.0,
        "fft_filter_power": 8,
        "event_smoothing_mode": "auto",
        "event_smoothing_kernel": "cosine",
        "event_smoothing_log_width": 0.002,
        "align_priority": "auto",
    },
    "reverse_sensitive": {
        "grid_points": 1201,
        "num_std_devs": 10.0,
        "fft_padding_factor": 2,
        "fft_filter_alpha": 18.0,
        "fft_filter_power": 8,
        "align_priority": "ko",
        "event_smoothing_mode": "reverse_aware",
        "event_smoothing_kernel": "cosine",
    },
}

# PDE presets target the declarative grid layer (0.4.0): "accuracy" selects a
# GridConfig profile; "grid_config" holds GridConfig kwargs (materialized in
# make_pde_params — GridConfig cannot be imported at param module load).
PDE_PARAM_PRESETS: ProfileMap = {
    "fast": {"accuracy": "fast"},
    "balanced": {},
    "accurate": {"accuracy": "high"},
    "barrier_sensitive": {
        "grid_config": {"points": 600, "steps_per_day": 6.0},
    },
    "reverse_sensitive": {
        "grid_config": {"points": 600, "steps_per_day": 6.0},
    },
}

ENGINE_PARAM_PRESETS: Dict[str, ProfileMap] = {
    "quad": QUAD_PARAM_PRESETS,
    "pde": PDE_PARAM_PRESETS,
}


def list_param_profiles(engine: str) -> Tuple[str, ...]:
    engine_key = _normalize_engine(engine)
    return tuple(ENGINE_PARAM_PRESETS[engine_key].keys())


def make_quad_params(
    profile: str = "balanced",
    product: Optional[Any] = None,
    reverse: bool = False,
    **overrides: Any,
) -> QuadParams:
    profile_name = _normalize_profile("quad", profile)
    preset = dict(QUAD_PARAM_PRESETS[profile_name])
    preset = _apply_quad_product_hints(preset, product, reverse)
    merged = _merge_overrides(preset, overrides, QuadParams)
    return QuadParams(**merged)


def make_pde_params(
    profile: str = "balanced",
    product: Optional[Any] = None,
    reverse: bool = False,
    **overrides: Any,
) -> PDEParams:
    # product/reverse are accepted for signature parity with make_quad_params;
    # the declarative grid layer concentrates on every product barrier by
    # construction, so no product-conditional knob hints remain.
    del product, reverse
    profile_name = _normalize_profile("pde", profile)
    preset = dict(PDE_PARAM_PRESETS[profile_name])
    grid_kwargs = preset.pop("grid_config", None)
    if grid_kwargs is not None:
        # Lazy import: param cannot pull engine.pde at module load (cycle).
        from quantark.asset.equity.engine.pde.grid.config import GridConfig

        preset["grid"] = GridConfig(**grid_kwargs)
    merged = _merge_overrides(preset, overrides, PDEParams)
    return PDEParams(**merged)


def make_engine_params(
    engine: str,
    profile: str = "balanced",
    product: Optional[Any] = None,
    reverse: bool = False,
    **overrides: Any,
) -> Union[QuadParams, PDEParams]:
    engine_key = _normalize_engine(engine)
    if engine_key == "quad":
        return make_quad_params(
            profile=profile, product=product, reverse=reverse, **overrides
        )
    if engine_key == "pde":
        return make_pde_params(
            profile=profile, product=product, reverse=reverse, **overrides
        )
    raise ValidationError(
        f"Unknown engine '{engine}'. Supported engines: {', '.join(ENGINE_PARAM_PRESETS)}"
    )


def load_param_config(config_or_path: Union[Mapping[str, Any], str, Path]) -> Dict[str, Any]:
    if isinstance(config_or_path, Mapping):
        return dict(config_or_path)

    path = Path(config_or_path)
    if not path.exists():
        raise ValidationError(f"Config path does not exist: {path}")

    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise ValidationError("PyYAML is required to load YAML configs.")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    elif ext == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        raise ValidationError(
            f"Unsupported config extension: {ext}. Use .yaml, .yml, or .json."
        )

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValidationError("Config must contain a top-level mapping.")
    return data


def parse_profile_config(
    config_or_path: Union[Mapping[str, Any], str, Path],
    engine: str,
    profile: Optional[str] = None,
    reverse: Optional[bool] = None,
) -> Tuple[str, Dict[str, Any], bool]:
    data = load_param_config(config_or_path)
    engine_key = _normalize_engine(engine)

    config_engine = data.get("engine")
    if config_engine is not None and str(config_engine).lower() != engine_key:
        raise ValidationError(
            f"Config engine '{config_engine}' does not match requested engine '{engine_key}'."
        )

    profile_name = profile or data.get("profile") or "balanced"
    reverse_flag = reverse if reverse is not None else bool(data.get("reverse", False))

    overrides = data.get("overrides")
    if overrides is None:
        overrides = data.get("params")
    if overrides is None:
        reserved = {"engine", "profile", "overrides", "params", "reverse"}
        overrides = {k: v for k, v in data.items() if k not in reserved}

    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValidationError("Config overrides must be a mapping.")

    return str(profile_name), dict(overrides), reverse_flag


def _normalize_engine(engine: str) -> str:
    if engine is None:
        raise ValidationError("Engine must be provided.")
    engine_key = str(engine).lower()
    if engine_key not in ENGINE_PARAM_PRESETS:
        raise ValidationError(
            f"Unknown engine '{engine}'. Supported engines: {', '.join(ENGINE_PARAM_PRESETS)}"
        )
    return engine_key


def _normalize_profile(engine: str, profile: str) -> str:
    engine_key = _normalize_engine(engine)
    if profile is None:
        return "balanced"
    profile_key = str(profile).lower()
    if profile_key not in ENGINE_PARAM_PRESETS[engine_key]:
        profiles = ", ".join(sorted(ENGINE_PARAM_PRESETS[engine_key]))
        raise ValidationError(
            f"Unknown profile '{profile}'. Supported profiles: {profiles}"
        )
    return profile_key


def _merge_overrides(
    preset: Dict[str, Any],
    overrides: Mapping[str, Any],
    param_cls: Any,
) -> Dict[str, Any]:
    merged = dict(preset)
    if overrides:
        _validate_keys(overrides, param_cls)
        merged.update(overrides)
    _validate_keys(merged, param_cls)
    return merged


def _validate_keys(values: Mapping[str, Any], param_cls: Any) -> None:
    allowed = {field.name for field in fields(param_cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValidationError(
            f"Unknown parameter keys for {param_cls.__name__}: {', '.join(unknown)}"
        )


def _apply_quad_product_hints(
    preset: Dict[str, Any],
    product: Optional[Any],
    reverse: bool,
) -> Dict[str, Any]:
    reverse_flag = reverse or bool(getattr(product, "is_reverse", False))
    if reverse_flag:
        preset.setdefault("align_priority", "ko")
        preset.setdefault("event_smoothing_mode", "reverse_aware")
    return preset



