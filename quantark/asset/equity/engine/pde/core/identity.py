"""Identities the core keys on (spec 2 §5.9, decisions 8, 13, 17, 18).

Every key is a value fingerprint (``quantark.execution.cache.fingerprint``);
an input without one returns ``None`` and disables whatever it would key.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = ["layout_fingerprint", "component_fingerprint", "market_fingerprints", "coefficient_key"]


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
