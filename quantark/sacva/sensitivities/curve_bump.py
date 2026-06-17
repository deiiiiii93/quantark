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
    if not (np.isfinite(tenor) and np.isfinite(spread_bp)):
        raise ValidationError("tenor and spread_bp must be finite")
    if not hasattr(curve, "tenors") or not hasattr(curve, "hazards"):
        raise ValidationError(
            "credit curve must expose pillar 'tenors'/'hazards' (PillarCreditCurve)")
    if not callable(getattr(curve, "get_survival_probability", None)):
        raise ValidationError("credit curve must expose get_survival_probability(t)")
    if not hasattr(curve, "recovery_rate"):
        raise ValidationError("credit curve must expose recovery_rate")
    if not np.isclose(elgd, 1.0 - float(curve.recovery_rate)):
        raise ValidationError(
            "elgd must equal 1 - curve.recovery_rate (chain rule s = lambda(1-R))")
    dlam = (spread_bp * 1e-4) / elgd
    bumped = copy.deepcopy(curve)
    tenors = np.asarray(bumped.tenors, dtype=float)
    hazards = np.asarray(bumped.hazards, dtype=float)
    if tenors.ndim != 1 or hazards.ndim != 1 or tenors.shape != hazards.shape:
        raise ValidationError("tenors/hazards must be equal-length 1-D arrays")
    if not (np.all(np.isfinite(tenors)) and np.all(np.isfinite(hazards))):
        raise ValidationError("tenors/hazards must be finite")
    if np.any(np.diff(tenors) <= 0):
        raise ValidationError("tenors must be strictly increasing (sorted, unique)")
    idx = int(np.argmin(np.abs(tenors - tenor)))
    if not np.isclose(tenors[idx], tenor):
        raise ValidationError(f"tenor {tenor} not a curve pillar {tenors.tolist()}")
    hazards[idx] += dlam
    if np.any(hazards < 0):
        raise ValidationError("bumped hazard negative")
    bumped.hazards = hazards
    S = np.array([float(bumped.get_survival_probability(float(t))) for t in tenors])
    if not np.all(np.isfinite(S)) or np.any(S < -1e-12) or np.any(S > 1.0 + 1e-12):
        raise ValidationError("bumped survival must be finite in [0, 1]")
    if np.any(np.diff(S) > 1e-12):
        raise ValidationError("bumped survival not monotone non-increasing")
    return bumped
