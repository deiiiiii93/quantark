"""Identities the core keys on (spec 2 §5.9, decisions 8, 13, 17, 18).

Every key is a value fingerprint (``quantark.execution.cache.fingerprint``);
an input without one returns ``None`` and disables whatever it would key.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = ["layout_fingerprint", "component_fingerprint", "market_fingerprints", "coefficient_key",
           "group_schedule", "factorization_key", "plain", "problem_identity"]


# The fingerprint module lives in quantark.execution, whose package imports the
# engines: imported inside each function so loading the PDE package stays acyclic.

def layout_fingerprint(layout) -> str:
    """sha256 over what the march consumes (decision 17): x, dx, t, dt, the event and
    extra node indices, config_key."""
    from quantark.execution.cache.fingerprint import fingerprint

    step_of = layout.time.step_of
    return fingerprint((
        np.asarray(layout.spatial.x), np.asarray(layout.spatial.dx),
        np.asarray(layout.time.t), np.asarray(layout.time.dt),
        tuple(sorted(int(step_of[e]) for e in layout.request.event_times)),
        tuple(sorted(int(step_of[e]) for e in layout.request.extra_times)),
        tuple(layout.config_key),
    ))


def component_fingerprint(obj) -> Optional[str]:
    """Decision 8: the one identity function for a component."""
    from quantark.execution.cache.fingerprint import try_fingerprint

    return try_fingerprint(obj)


def market_fingerprints(env, *, spot: bool = True) -> Dict[str, Optional[str]]:
    """Value fingerprints of the environment's market inputs; ``None`` = none exists."""
    from quantark.execution.cache.fingerprint import try_fingerprint

    fps = {
        "rate": try_fingerprint(env.rate_curve),
        "div": try_fingerprint(env.div_yield),
        "vol": try_fingerprint(env.vol_surface),
    }
    if spot:
        fps["spot"] = float(env.spot).hex()
    return fps


def coefficient_key(layout_fp, market_fps, ref_strike, operator_fp, lv_surface_fp=None):
    """The coefficients' identity: the delivered layout plus every sampler input (decision 18)."""
    parts = (layout_fp, market_fps["rate"], market_fps["div"], market_fps["vol"],
             market_fps["spot"], float(ref_strike).hex(), operator_fp, lv_surface_fp)
    return None if any(p is None for p in parts[:7]) else parts


def group_schedule(prepared, group) -> np.ndarray:
    """The θ a group steps with: the schedule, θ=1 on zero-diffusion sets when the
    group's policy overrides them (decision 17)."""
    theta = np.array(prepared.theta_schedule, dtype=float)
    if group.policy.zero_diffusion_theta_one:
        coeffs = prepared.coefficients
        for j in range(len(theta)):
            if coeffs.is_zero_diffusion(coeffs.set_at(j)):
                theta[j] = 1.0
    return theta


def factorization_key(prepared) -> Optional[str]:
    """What every factorization of the march depends on: the coefficients, the
    stepper, the groups' backends and their θ schedules."""
    from quantark.execution.cache.fingerprint import try_fingerprint

    parts = (prepared.identity["coefficient_key"], component_fingerprint(prepared.stepper),
             tuple(g.backend.value for g in prepared.groups),
             tuple(group_schedule(prepared, g) for g in prepared.groups))
    return None if parts[0] is None or parts[1] is None else try_fingerprint(parts)


def plain(value):
    """``value`` as canonicalizable data: enums by value, sets as sorted tuples,
    datetimes as ISO strings, mappings as sorted (str key, value) pairs."""
    import dataclasses
    from datetime import date, datetime
    from enum import Enum

    if isinstance(value, Enum):
        return (type(value).__name__, plain(value.value))
    if isinstance(value, (frozenset, set)):
        return tuple(sorted((plain(v) for v in value), key=repr))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return tuple(sorted((str(k), plain(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(plain(v) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain({f.name: getattr(value, f.name) for f in dataclasses.fields(value)})
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def problem_identity(prepared) -> Optional[str]:
    """The march's identity: every input a march of this problem reads (spec 2 §5.9).

    A product token that falls back to ``id(...)`` makes it process-local, which
    is acceptable: the identity keys nothing persistent.
    """
    from quantark.asset.equity.settlement import calendar_identity, effective_settlement_calendar
    from quantark.execution.cache.fingerprint import try_fingerprint

    fk = factorization_key(prepared)
    rd = component_fingerprint(prepared.readout)
    if fk is None or rd is None:
        return None
    env, product, solver = prepared.env, prepared.product, prepared.solver
    events = prepared.events
    return try_fingerprint((
        fk, rd,
        plain(solver._product_cache_token(product, "standard")),
        plain(prepared.solve_state),
        events.evidence() if hasattr(events, "evidence") else plain(events),
        (plain(env.valuation_date), plain(env.day_count_convention), int(env.bus_days_in_year)),
        plain(calendar_identity(effective_settlement_calendar(product, env))),
        plain(solver.params.event_projection),
        plain(prepared.request),
    ))
