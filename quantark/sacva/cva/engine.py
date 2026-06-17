"""Regulatory unilateral CVA (spec §3.3, MAR50.32).

CVA = ELGD · Σ_i trapezoid(EE*_disc) · (S(t_{i-1}) − S(t_i))

where EE*_disc is the discounted expected positive exposure (from a
regulatory-eligible, risk-neutral ExposureProfile) and S is the counterparty
survival probability. Unilateral, bank default-free; CVA ≥ 0 (positive-loss
sign). ELGD = 1 − recovery, the same value used to bootstrap the PD curve.
"""

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_valid_number


class RegulatoryCVAEngine:
    def compute(self, credit_curve, profile) -> float:
        from quantark.sacva.exposure.engine import Measure

        if not profile.regulatory_eligible or profile.measure is not Measure.RISK_NEUTRAL:
            raise ValidationError(
                "CVA requires a regulatory-eligible RISK_NEUTRAL ExposureProfile")
        elgd = 1.0 - float(credit_curve.recovery_rate)
        if not (elgd > 0):
            raise ValidationError("ELGD must be > 0")

        t = np.asarray(profile.times, dtype=float)
        ee = np.asarray(profile.epe_discounted, dtype=float)
        S = np.array([float(credit_curve.get_survival_probability(float(x))) for x in t])
        if np.any(~np.isfinite(S)) or np.any(S < -1e-12) or np.any(S > 1.0 + 1e-12):
            raise ValidationError("survival probabilities must be finite in [0, 1]")
        if np.any(np.diff(S) > 1e-12):
            raise ValidationError("survival probabilities must be non-increasing")

        cva = 0.0
        for i in range(1, len(t)):
            ee_bar = 0.5 * (ee[i - 1] + ee[i])
            cva += ee_bar * (S[i - 1] - S[i])
        cva *= elgd
        if not is_valid_number(cva) or cva < -1e-9:
            raise ValidationError(f"invalid CVA {cva}")
        return float(max(cva, 0.0))
