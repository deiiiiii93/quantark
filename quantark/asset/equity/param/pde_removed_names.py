"""Names removed from the 1D PDE parameter surface in 0.6.0 (spec §4.4).

One map supplies the replacement text for every removed name; it is also
rendered into docs/engine_param_guide.md and the CHANGELOG. Removed names
raise ValidationError, never TypeError, so the user is told what replaces
them. Names not in the map raise the same error without a replacement line.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Mapping

from quantark.util.exceptions import ValidationError

_TERMINAL = "grid=GridConfig(terminal_damping_steps=N); 0 disables"
_EVENT = "grid=GridConfig(event_damping_steps=N); 0 disables"
_CACHE = "cache_enabled=True/False is the only cache control"

REMOVED_PDE_NAMES: Mapping[str, str] = {
    "method": (
        "Crank-Nicolson is the only built-in scheme; "
        "PDEEngine(params=PDEParams(...)). theta=1.0 gives backward Euler."
    ),
    "grid_size": "grid=GridConfig(points=...) on every route.",
    "time_steps": (
        "grid=GridConfig(time_steps=N): the same meaning, a total step budget "
        "over the horizon. steps_per_day= is the per-day alternative."
    ),
    "s_max": (
        "grid=GridConfig(bounds=(None, hi)) on the shared-grid solvers. On the two "
        "local-vol solvers an explicit bound is rejected until sub-project 3: the "
        "kernel sets its own domain and aligns the top, so there is no hard edge "
        "to honour."
    ),
    "s_min": "removed; it was read by nothing.",
    "use_rannacher": _TERMINAL,
    "rannacher_steps": _TERMINAL,
    "rannacher_at_events": _EVENT,
    "event_rannacher_steps": _EVENT,
    "boundary_mode": "removed; asymptotic boundaries are the only treatment",
    "continuous_ki_correction": (
        "removed; the first-passage correction always applies to "
        "continuously monitored KI"
    ),
    "ki_monitoring_mode": (
        "removed from the PDE; each barrier is monitored as the product "
        "declares it (discrete dates, continuous, or maturity-only). To "
        "price a dense discrete KI as continuous, declare it continuous on "
        "the product. QuadParams.ki_monitoring_mode is unchanged."
    ),
    "cache_strategy": _CACHE,
    "grid_cache_max_entries": _CACHE,
    "banded_cache_max_entries": _CACHE,
    "use_banded_solver": "removed; the linear solver is chosen by the solver family",
}

REMOVED_DCN_NAMES: Mapping[str, str] = {
    "num_space_nodes": "DCNPDEEngine(params=PDEParams(grid=GridConfig(points=...)))",
    "rannacher_steps": (
        "grid=GridConfig(terminal_damping_steps=N, event_damping_steps=N) "
        "(DCN applies both in its half-step form)"
    ),
    "s_min_mult": "removed; never read",
    "s_max_mult": "removed; never read",
    "concentration": "removed; never read",
}


def removed_name_error(
    owner: str, name: str, removal_map: Mapping[str, str] = REMOVED_PDE_NAMES
) -> ValidationError:
    """The error for keyword ``name`` passed to ``owner``."""
    text = removal_map.get(name)
    if text is None:
        return ValidationError(f"{owner} got an unexpected keyword argument {name!r}")
    return ValidationError(f"{owner}({name}=...) was removed in 0.6.0: {text}")


def reject_removed_kwargs(
    owner: str, kwargs: Mapping, removal_map: Mapping[str, str] = REMOVED_PDE_NAMES
) -> None:
    """Raise for the first keyword in ``kwargs`` (a constructor's ``**removed``)."""
    for name in kwargs:
        raise removed_name_error(owner, name, removal_map)


def keyword_only_init(removal_map: Mapping[str, str] = REMOVED_PDE_NAMES):
    """Class decorator for a dataclass: keyword arguments only, and an unknown
    name raises ValidationError with its replacement (spec §4.4, decision 11).

    ``kw_only=True`` alone leaves fields inherited from a non-kw_only base
    positional (``PDEParams("high")`` would set ``EngineParams.bump_size``);
    the wrapper is what closes that.
    """

    def deco(cls):
        init = cls.__init__
        allowed = frozenset(f.name for f in dataclasses.fields(cls) if f.init)

        @functools.wraps(init)
        def __init__(self, *args, **kwargs):
            if args:
                raise TypeError(
                    f"{cls.__name__} takes keyword arguments only "
                    f"({len(args)} positional given)"
                )
            for name in kwargs:
                if name not in allowed:
                    raise removed_name_error(cls.__name__, name, removal_map)
            init(self, **kwargs)

        cls.__init__ = __init__
        return cls

    return deco
