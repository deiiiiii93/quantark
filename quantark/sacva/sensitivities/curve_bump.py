"""Key-rate (tenor-localised) curve bumps for SA-CVA sensitivities (spec §3.4).

The existing ``with_hazard_shift`` / ``with_rate_shift`` are parallel-only; SA-CVA
needs per-tenor pillar bumps. ``bump_hazard_pillar`` adds ``Δλ = spread_bp·1e-4 /
ELGD`` (chain rule ``s = λ(1−R)``, MAR50.63) to a single hazard pillar and
re-derives survival. The credit curve must satisfy a pillar protocol
(``tenors``, ``hazards``, ``get_survival_probability(t)``, ``recovery_rate``).
"""

import copy

import numpy as np

from quantark.util.exceptions import ValidationError


def bump_hazard_pillar(curve, tenor, spread_bp, elgd):
    if not (elgd > 0):
        raise ValidationError("ELGD must be > 0 for spread->hazard conversion")
    if not hasattr(curve, "tenors") or not hasattr(curve, "hazards"):
        raise ValidationError(
            "credit curve must expose pillar 'tenors'/'hazards' (PillarCreditCurve)")
    dlam = (spread_bp * 1e-4) / elgd
    bumped = copy.deepcopy(curve)
    tenors = np.asarray(bumped.tenors, dtype=float)
    idx = int(np.argmin(np.abs(tenors - tenor)))
    if not np.isclose(tenors[idx], tenor):
        raise ValidationError(f"tenor {tenor} not a curve pillar {tenors.tolist()}")
    hazards = np.asarray(bumped.hazards, dtype=float)
    hazards[idx] += dlam
    if np.any(hazards < 0):
        raise ValidationError("bumped hazard negative")
    bumped.hazards = hazards
    S = np.array([float(bumped.get_survival_probability(float(t))) for t in tenors])
    if np.any(np.diff(S) > 1e-12):
        raise ValidationError("bumped survival not monotone non-increasing")
    return bumped
