"""GridConfig — the expert overlay — and the accuracy profiles (spec §4.7).

Every field is Optional: ``None`` means "inherit from the selected accuracy
profile"; an explicit value overrides that single field. ``resolve_config``
produces the fully-populated instance the builders consume; its ``key`` is the
fingerprint used in cache keys and ``Layout.config_key``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional, Tuple

from quantark.asset.equity.param.pde_removed_names import REMOVED_GRID_NAMES, keyword_only_init
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
    max_steps=12000,
)

#: The built-in stepper's route defaults, ``(form, terminal count, event count)`` (spec 3A §5.3):
#: a ``None`` on ``CrankNicolson`` resolves to its route's row. The core has two rows chosen by the
#: contract alone (decision 12): a product whose knock-in is monitored continuously keeps
#: full-step; every other core product takes "core". DCN, the local-vol kernels and the 2D solvers
#: have their own rows, so a core flip cannot reach them. The local-vol vanilla kernel reads only
#: ``terminal > 0`` (one two-half-step startup); the barrier kernel reads none of them.
DAMPING_DEFAULTS = {
    "core": ("full_step", 1, 2),
    "core-continuous-ki": ("full_step", 1, 2),
    "DCN": ("half_step", 2, 2),
    "local-vol": ("half_step", 1, 2),
    "2D": ("full_step", 1, 2),
}

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
    )
    for name, pts in (("fast", 401), ("standard", 801), ("high", 1601))
}


@keyword_only_init(REMOVED_GRID_NAMES)
@dataclass(frozen=True, kw_only=True)
class GridConfig:
    """Expert grid configuration; ``None`` fields inherit from the profile.

    Exactly one time basis is in force after resolution: ``steps_per_day``
    (steps per business day of the pricing environment
    (``PricingEnvironment.bus_days_in_year``, carried on ``GridRequest.day_count``))
    or ``time_steps`` (a total budget over the horizon).
    """

    points: Optional[int] = None
    steps_per_day: Optional[float] = None
    time_steps: Optional[int] = None
    eps_crit: Optional[float] = None
    num_std: Optional[float] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    max_points: Optional[int] = None
    max_steps: Optional[int] = None

    @property
    def key(self) -> tuple:
        """Value fingerprint (cache keys, Layout.config_key)."""
        return tuple(getattr(self, f.name) for f in fields(self))

    def evidence(self) -> dict:
        """The certificate serialization: every SET field, in field order.

        An unset optional is omitted. The certificates' historical ten-key
        ``grid`` record adds the stepper's two damping counts
        (``resolved_grid_evidence``).
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
        """Copy on the budget basis (the density cleared)."""
        return replace(self, time_steps=time_steps, steps_per_day=None)


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
    cfg = GridConfig(**merged)  # type: ignore[arg-type]  # merged is per-field typed
    _validate_resolved(cfg)
    return cfg


def _validate_resolved(cfg: GridConfig) -> None:
    for name in ("points", "max_points", "max_steps"):
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


def resolved_grid_evidence(params, *, day_count: int) -> dict:
    """The certificates' ``grid`` record (spec 3A §5.8): the resolved GridConfig fields (with the
    environment's ``day_count`` after ``max_steps`` on the density basis, where 0.7.0 had it), then the
    damping of a product with continuous knock-in under its historical keys, ``damping`` only when
    that form is half-step, and ``discrete_monitoring_damping`` only when every other core solve
    resolves differently. A full-step (1, 2) configuration still serializes to the 0.7.0 record.
    It is the numerical fact, not a serialization of GridConfig."""
    from quantark.asset.equity.engine.pde.core.components import CrankNicolson, resolve_stepper

    record = dict(resolve_config(params.accuracy, params.grid).evidence())
    if record.get("time_steps") is None:
        record["day_count"] = int(day_count)
    cont = resolve_stepper(params, DAMPING_DEFAULTS["core-continuous-ki"])
    disc = resolve_stepper(params, DAMPING_DEFAULTS["core"])
    if type(cont) is not CrankNicolson or type(disc) is not CrankNicolson:
        raise ValidationError("resolved_grid_evidence: only the built-in stepper is certifiable")
    record["terminal_damping_steps"] = int(cont.terminal_damping_steps)
    record["event_damping_steps"] = int(cont.event_damping_steps)
    if cont.damping == "half_step":
        record["damping"] = "half_step"

    def as_row(st):
        return [st.damping, int(st.terminal_damping_steps), int(st.event_damping_steps)]

    if as_row(disc) != as_row(cont):
        record["discrete_monitoring_damping"] = as_row(disc)
    return record
