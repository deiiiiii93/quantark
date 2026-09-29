"""GridConfig — the expert overlay — and the accuracy profiles (spec §4.7).

Every field is Optional: ``None`` means "inherit from the selected accuracy
profile"; an explicit value overrides that single field. ``resolve_config``
produces the fully-populated instance the builders consume; its ``key`` is the
fingerprint used in cache keys and ``Layout.config_key``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional, Tuple

from quantark.asset.equity.param.pde_removed_names import keyword_only_init
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

#: Local-vol vanilla/barrier routes (spec §4.2, decision 16): today's plan at
#: standard (400 points, a 200-step budget); fast/high are new resolution
#: choices with no banked evidence. eps_crit only satisfies validation -- the
#: routes' kernels have no concentration (route applicability, spec §4.1).
LV_UNIFORM_ACCURACY_PROFILES = {
    "fast": dict(points=200, time_steps=200, eps_crit=0.003),
    "standard": dict(points=400, time_steps=200, eps_crit=0.003),
    "high": dict(points=800, time_steps=400, eps_crit=0.002),
}

#: DCN route (spec §4.2, decision 8): one fill step per observation interval
#: (steps_per_day=0.1 rounds every realistic calendar gap to one step);
#: accuracy scales spatial resolution only. Standard is today's plan exactly.
#: max_points is max(2000, points); the DCN engine derives it the same way
#: for an explicit points overlay.
DCN_ACCURACY_PROFILES = {
    name: dict(
        points=pts,
        steps_per_day=0.1,
        eps_crit=ACCURACY_PROFILES[name]["eps_crit"],
        max_points=max(2000, pts),
        terminal_damping_steps=2,
        event_damping_steps=2,
    )
    for name, pts in (("fast", 401), ("standard", 801), ("high", 1601))
}


@keyword_only_init(removal_map={})
@dataclass(frozen=True, kw_only=True)
class GridConfig:
    """Expert grid configuration; ``None`` fields inherit from the profile.

    Exactly one time basis is in force after resolution: ``steps_per_day``
    (a density per numerical day, ``day_count`` days per solver year) or
    ``time_steps`` (a total budget over the horizon).
    """

    #: NOTE: ``day_count`` is a STEP-DENSITY constant in the units of the
    #: engine's time axis (interval_length * day_count = "days" of fill),
    #: not a day-count convention: 252 per solver year, not exchange
    #: sessions. On the default ACT/365 calendar axis the 252 default yields
    #: ~0.69 steps-per-"day" per calendar day; see docs/trading-clock.md.
    #: It becomes the environment's business day in sub-project 3.
    points: Optional[int] = None
    steps_per_day: Optional[float] = None
    time_steps: Optional[int] = None
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

    def evidence(self) -> dict:
        """The certificate serialization: every SET field, in field order.

        An unset optional is omitted, so every banked resolved configuration
        (all ten pre-0.6 fields set, ``time_steps`` unset) serializes exactly
        as ``dataclasses.asdict`` did and keeps its identity (decision 14).
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    def with_steps_per_day(self, steps_per_day: float) -> "GridConfig":
        """Copy on the density basis (the budget cleared)."""
        return replace(self, steps_per_day=steps_per_day, time_steps=None)

    def with_time_steps(self, time_steps: int) -> "GridConfig":
        """Copy on the budget basis: the density and its day basis cleared."""
        return replace(
            self, time_steps=time_steps, steps_per_day=None, day_count=None
        )


def resolve_config(
    accuracy: str, override: Optional[GridConfig], profiles=ACCURACY_PROFILES
) -> GridConfig:
    """Merge a route's preset with an explicit overlay and validate (spec §4.2).

    Authority: explicit overlay > route preset > shared defaults. The overlay
    chooses the time basis: setting one basis clears the other.
    """
    if accuracy not in profiles:
        raise ValidationError(
            f"accuracy must be one of {sorted(profiles)}, got {accuracy!r}"
        )
    merged = dict(_SHARED_DEFAULTS)
    merged.update(profiles[accuracy])
    if override is not None:
        if override.steps_per_day is not None and override.time_steps is not None:
            raise ValidationError(
                "GridConfig sets both steps_per_day and time_steps; set one time "
                "basis (steps_per_day = steps per numerical day, time_steps = "
                "total steps over the horizon)"
            )
        for f in fields(GridConfig):
            v = getattr(override, f.name)
            if v is not None:
                merged[f.name] = v
        if override.time_steps is not None:
            merged["steps_per_day"] = None
        elif override.steps_per_day is not None:
            merged["time_steps"] = None
        if override.day_count is not None and merged.get("time_steps") is not None:
            raise ValidationError(
                "GridConfig.day_count applies only to the steps_per_day density; "
                "the time basis in force is time_steps. Set steps_per_day too, or "
                "drop day_count"
            )
    cfg = GridConfig(**merged)  # type: ignore[arg-type]  # merged is per-field typed
    _validate_resolved(cfg)
    return cfg


def _validate_resolved(cfg: GridConfig) -> None:
    for name in ("points", "max_points", "max_steps", "day_count"):
        if getattr(cfg, name) <= 0:
            raise ValidationError(f"{name} must be positive, got {getattr(cfg, name)}")
    if (cfg.steps_per_day is None) == (cfg.time_steps is None):
        raise ValidationError(
            "exactly one time basis must be in force after resolution "
            f"(steps_per_day={cfg.steps_per_day}, time_steps={cfg.time_steps})"
        )
    if cfg.steps_per_day is not None and cfg.steps_per_day <= 0.0:
        raise ValidationError(
            f"steps_per_day must be positive, got {cfg.steps_per_day}"
        )
    if cfg.time_steps is not None:
        n = cfg.time_steps
        if isinstance(n, bool) or int(n) != n or n <= 0:
            raise ValidationError(f"time_steps must be a positive integer, got {n}")
        if n > cfg.max_steps:
            raise ValidationError(
                f"time_steps ({n}) exceeds max_steps ({cfg.max_steps}); "
                "raise GridConfig(max_steps=...)"
            )
    for name in ("eps_crit", "num_std"):
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
