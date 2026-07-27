"""GridConfig — the expert overlay — and the accuracy profiles (spec §4.7).

Every field is Optional: ``None`` means "inherit from the selected accuracy
profile"; an explicit value overrides that single field. ``resolve_config``
produces the fully-populated instance the builders consume; its ``key`` is the
fingerprint used in cache keys and ``Layout.config_key``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Tuple

from quantark.util.exceptions import ValidationError

#: Preset values per accuracy profile. steps_per_day never sits below 4 in any
#: preset (continuously monitored KI needs the resolution — convergence-gate
#: finding preserved); "fast" economizes on the spatial axis instead.
ACCURACY_PROFILES = {
    "fast": dict(points=200, steps_per_day=4.0, eps_crit=0.003),
    "standard": dict(points=400, steps_per_day=4.0, eps_crit=0.003),
    "high": dict(points=800, steps_per_day=8.0, eps_crit=0.002),
}

_SHARED_DEFAULTS = dict(
    num_std=4.0,
    bounds=(None, None),
    max_points=2000,
    max_steps=5000,
    day_count=252,
    terminal_damping_steps=1,
    event_damping_steps=2,
)


@dataclass(frozen=True)
class GridConfig:
    """Expert grid configuration; ``None`` fields inherit from the profile."""

    points: Optional[int] = None
    steps_per_day: Optional[float] = None
    eps_crit: Optional[float] = None
    num_std: Optional[float] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    max_points: Optional[int] = None
    max_steps: Optional[int] = None
    day_count: Optional[int] = None
    terminal_damping_steps: Optional[int] = None
    event_damping_steps: Optional[int] = None

    @property
    def key(self) -> tuple:
        """Value fingerprint (cache keys, Layout.config_key)."""
        return tuple(getattr(self, f.name) for f in fields(self))


def resolve_config(accuracy: str, override: Optional[GridConfig]) -> GridConfig:
    """Merge profile defaults with an explicit override and validate."""
    if accuracy not in ACCURACY_PROFILES:
        raise ValidationError(
            f"accuracy must be one of {sorted(ACCURACY_PROFILES)}, got {accuracy!r}"
        )
    merged = dict(_SHARED_DEFAULTS)
    merged.update(ACCURACY_PROFILES[accuracy])
    if override is not None:
        for f in fields(GridConfig):
            v = getattr(override, f.name)
            if v is not None:
                merged[f.name] = v
    cfg = GridConfig(**merged)  # type: ignore[arg-type]  # merged is per-field typed
    _validate_resolved(cfg)
    return cfg


def _validate_resolved(cfg: GridConfig) -> None:
    for name in ("points", "max_points", "max_steps", "day_count"):
        if getattr(cfg, name) <= 0:
            raise ValidationError(f"{name} must be positive, got {getattr(cfg, name)}")
    for name in ("steps_per_day", "eps_crit", "num_std"):
        if getattr(cfg, name) <= 0.0:
            raise ValidationError(f"{name} must be positive, got {getattr(cfg, name)}")
    for name in ("terminal_damping_steps", "event_damping_steps"):
        if getattr(cfg, name) < 0:
            raise ValidationError(
                f"{name} must be non-negative, got {getattr(cfg, name)}"
            )
    if cfg.points > cfg.max_points:
        raise ValidationError(
            f"points ({cfg.points}) exceeds max_points ({cfg.max_points})"
        )
    lo, hi = cfg.bounds
    for side, v in (("lower", lo), ("upper", hi)):
        if v is not None and float(v) <= 0.0:
            raise ValidationError(f"bounds {side} side must be positive, got {v}")
    if lo is not None and hi is not None and lo >= hi:
        raise ValidationError(f"bounds must be ordered, got ({lo}, {hi})")
