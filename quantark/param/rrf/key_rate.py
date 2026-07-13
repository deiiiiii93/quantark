"""Key-rate (pillar) bump of an interpolated zero curve (spec WP3.3).

Bumps exactly one pillar's zero rate; between pillars the curve's own
interpolation produces the decay to neighboring pillars ("triangle" under
linear interpolation), and the curve's flat extrapolation carries the end
pillars outward. Returns a new curve of the same class; the input curve is
not mutated.
"""
from __future__ import annotations

import numpy as np

from quantark.util.exceptions import ValidationError

_PILLAR_MATCH_ATOL = 1e-12  # pillar lookup is exact-by-construction


def key_rate_bumped_zero_curve(curve, pillar_tenor: float, bump: float):
    pillars = getattr(curve, "pillars", None)
    if not pillars:
        raise ValidationError(
            "key_rate_bumped_zero_curve needs an InterpolatedRateCurve-style "
            "curve with pillars"
        )
    tenors = np.array([p[0] for p in pillars], dtype=float)
    hits = np.where(
        np.isclose(tenors, float(pillar_tenor), rtol=0.0, atol=_PILLAR_MATCH_ATOL)
    )[0]
    if hits.size != 1:
        raise ValidationError(
            f"pillar {pillar_tenor} is not a node of the curve "
            f"(nodes: {tenors.tolist()})"
        )
    i = int(hits[0])
    new_pillars = [
        (t, r + float(bump)) if j == i else (t, r)
        for j, (t, r) in enumerate(pillars)
    ]
    return type(curve)(
        new_pillars,
        node_roles=getattr(curve, "node_roles", None),
        last_observable_tenor=getattr(curve, "last_observable_tenor", None),
    )
