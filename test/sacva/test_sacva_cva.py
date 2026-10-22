"""Regulatory CVA engine tests (spec §3.3, MAR50.32)."""

import numpy as np
import pytest

from quantark.sacva.cva.engine import RegulatoryCVAEngine
from quantark.sacva.exposure.engine import ExposureProfile, Measure
from quantark.util.exceptions import ValidationError


class FlatCurve:
    def __init__(self, hazard, recovery=0.4):
        self.h = hazard
        self.R = recovery

    def get_survival_probability(self, t):
        return float(np.exp(-self.h * t))

    @property
    def recovery_rate(self):
        return self.R


def test_cva_matches_analytic_flat():
    times = np.array([0.0, 1.0, 2.0])
    epe = np.array([0.0, 100.0, 100.0])
    prof = ExposureProfile(times, epe, Measure.RISK_NEUTRAL, True)
    cur = FlatCurve(hazard=0.02, recovery=0.4)
    cva = RegulatoryCVAEngine().compute(cur, prof)
    elgd = 0.6
    S = [np.exp(-0.02 * t) for t in times]
    expect = elgd * (0.5 * (0 + 100) * (S[0] - S[1])
                     + 0.5 * (100 + 100) * (S[1] - S[2]))
    assert cva == pytest.approx(expect, rel=1e-12)


def test_cva_zero_when_hazard_zero():
    times = np.array([0.0, 1.0, 2.0])
    prof = ExposureProfile(times, np.array([0.0, 100.0, 100.0]),
                           Measure.RISK_NEUTRAL, True)
    assert RegulatoryCVAEngine().compute(FlatCurve(0.0), prof) == pytest.approx(0.0)


def test_cva_scales_with_elgd_when_curve_fixed():
    # holding the survival curve fixed, CVA scales linearly with ELGD
    times = np.array([0.0, 1.0])
    prof = ExposureProfile(times, np.array([0.0, 100.0]), Measure.RISK_NEUTRAL, True)
    c40 = RegulatoryCVAEngine().compute(FlatCurve(0.02, recovery=0.40), prof)
    c70 = RegulatoryCVAEngine().compute(FlatCurve(0.02, recovery=0.70), prof)
    assert c70 / c40 == pytest.approx((1 - 0.70) / (1 - 0.40), rel=1e-12)


def test_cva_rejects_non_regulatory_profile():
    prof = ExposureProfile(np.array([0., 1.]), np.array([0., 1.]),
                           Measure.REAL_WORLD, False)
    with pytest.raises(ValidationError):
        RegulatoryCVAEngine().compute(FlatCurve(0.02), prof)


def test_cva_rejects_full_recovery():
    prof = ExposureProfile(np.array([0., 1.]), np.array([0., 1.]),
                           Measure.RISK_NEUTRAL, True)
    with pytest.raises(ValidationError):
        RegulatoryCVAEngine().compute(FlatCurve(0.02, recovery=1.0), prof)


def test_cva_rejects_negative_recovery():
    prof = ExposureProfile(np.array([0., 1.]), np.array([0., 1.]),
                           Measure.RISK_NEUTRAL, True)
    with pytest.raises(ValidationError):  # negative recovery => ELGD > 1
        RegulatoryCVAEngine().compute(FlatCurve(0.02, recovery=-0.1), prof)


def test_cva_rejects_survival_not_unity_at_origin():
    class BadOrigin:
        recovery_rate = 0.4

        def get_survival_probability(self, t):
            return 0.9 * float(np.exp(-0.02 * t))   # S(0) = 0.9, not 1

    prof = ExposureProfile(np.array([0., 1.]), np.array([0., 100.]),
                           Measure.RISK_NEUTRAL, True)
    with pytest.raises(ValidationError):
        RegulatoryCVAEngine().compute(BadOrigin(), prof)
