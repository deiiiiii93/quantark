"""Pillar (bootstrapped) hazard curve for SA-CVA counterparty credit (spec §3.4).

Piecewise-constant forward hazards on the regulatory tenor grid, with an exact
survival integral. Exposes ``tenors`` / ``hazards`` / ``recovery_rate`` so the
key-rate bump (``bump_hazard_pillar``) can shift one tenor's hazard and re-derive
survival, giving the per-tenor counterparty credit-spread delta (MAR50.63).

``hazards[i]`` is the constant forward hazard on ``(tenors[i-1], tenors[i]]`` (with
``hazards[0]`` covering ``(0, tenors[0]]``); beyond the last pillar the hazard is
held flat. ``S(t) = exp(-∫_0^t λ(u) du)``.
"""

import numpy as np

from quantark.util.exceptions import ValidationError


class PillarHazardCurve:
    def __init__(self, tenors, hazards, recovery_rate=0.40):
        self.tenors = np.asarray(tenors, dtype=float)
        self.hazards = np.asarray(hazards, dtype=float)
        self.recovery_rate = float(recovery_rate)
        if self.tenors.ndim != 1 or self.tenors.shape != self.hazards.shape:
            raise ValidationError("tenors/hazards must be equal-length 1-D arrays")
        if self.tenors.size == 0:
            raise ValidationError("at least one pillar required")
        if not (np.all(np.isfinite(self.tenors)) and np.all(np.isfinite(self.hazards))):
            raise ValidationError("tenors/hazards must be finite")
        if np.any(self.tenors <= 0) or np.any(np.diff(self.tenors) <= 0):
            raise ValidationError("tenors must be positive and strictly increasing")
        if np.any(self.hazards < 0):
            raise ValidationError("hazards must be non-negative")
        if not (0.0 <= self.recovery_rate < 1.0):
            raise ValidationError("recovery_rate must be in [0, 1)")

    def _cumulative_hazard(self, t: float) -> float:
        t = float(t)
        if t <= 0:
            return 0.0
        prev = 0.0
        acc = 0.0
        for i, pillar in enumerate(self.tenors):
            seg_end = min(t, float(pillar))
            if seg_end > prev:
                acc += self.hazards[i] * (seg_end - prev)
            prev = float(pillar)
            if t <= pillar:
                return acc
        # beyond the last pillar: hold the last forward hazard flat
        acc += self.hazards[-1] * (t - prev)
        return acc

    def get_hazard_rate(self, t: float) -> float:
        idx = int(np.searchsorted(self.tenors, float(t), side="left"))
        idx = min(idx, len(self.hazards) - 1)
        return float(self.hazards[idx])

    def get_survival_probability(self, t: float) -> float:
        return float(np.exp(-self._cumulative_hazard(t)))
