"""SA-CVA sensitivity tests (spec §3.4)."""

import numpy as np
import pytest

from quantark.sacva.sensitivities.curve_bump import bump_hazard_pillar
from quantark.util.exceptions import ValidationError


class PillarCreditCurve:
    """Pillar-style credit curve satisfying the bump protocol."""

    def __init__(self, tenors, hazards, recovery=0.4):
        self.tenors = np.array(tenors, dtype=float)
        self.hazards = np.array(hazards, dtype=float)
        self.recovery = recovery

    def get_survival_probability(self, t):
        h = float(np.interp(t, self.tenors, self.hazards))
        return float(np.exp(-h * t))

    @property
    def recovery_rate(self):
        return self.recovery


def test_bump_hazard_pillar_localised_and_monotone():
    c = PillarCreditCurve([1.0, 3.0, 5.0], [0.02, 0.02, 0.02])
    elgd = 1 - 0.4
    bumped = bump_hazard_pillar(c, tenor=3.0, spread_bp=1.0, elgd=elgd)
    assert bumped.get_survival_probability(3.0) < c.get_survival_probability(3.0)
    assert bumped.get_survival_probability(0.25) == pytest.approx(
        c.get_survival_probability(0.25), rel=1e-6)
    # delta-lambda magnitude: 1bp / elgd added at the pillar
    assert bumped.hazards[1] == pytest.approx(0.02 + 1e-4 / elgd, rel=1e-12)


def test_bump_rejects_zero_elgd():
    c = PillarCreditCurve([1.0, 3.0], [0.02, 0.02])
    with pytest.raises(ValidationError):
        bump_hazard_pillar(c, tenor=1.0, spread_bp=1.0, elgd=0.0)


def test_bump_rejects_non_pillar_tenor():
    c = PillarCreditCurve([1.0, 3.0], [0.02, 0.02])
    with pytest.raises(ValidationError):
        bump_hazard_pillar(c, tenor=2.0, spread_bp=1.0, elgd=0.6)


def test_bump_rejects_non_pillar_curve():
    class NoPillars:
        recovery_rate = 0.4

        def get_survival_probability(self, t):
            return 1.0

    with pytest.raises(ValidationError):
        bump_hazard_pillar(NoPillars(), tenor=1.0, spread_bp=1.0, elgd=0.6)


def test_bump_rejects_elgd_inconsistent_with_recovery():
    c = PillarCreditCurve([1.0, 3.0], [0.02, 0.02], recovery=0.4)  # 1-R = 0.6
    with pytest.raises(ValidationError):
        bump_hazard_pillar(c, tenor=1.0, spread_bp=1.0, elgd=0.55)


def test_bump_rejects_unsorted_pillars():
    c = PillarCreditCurve([3.0, 1.0], [0.02, 0.02], recovery=0.4)
    with pytest.raises(ValidationError):
        bump_hazard_pillar(c, tenor=3.0, spread_bp=1.0, elgd=0.6)


def test_bump_rejects_nonfinite_tenor_and_spread():
    c = PillarCreditCurve([1.0, 3.0], [0.02, 0.02], recovery=0.4)
    with pytest.raises(ValidationError):
        bump_hazard_pillar(c, tenor=float("nan"), spread_bp=1.0, elgd=0.6)
    with pytest.raises(ValidationError):
        bump_hazard_pillar(c, tenor=1.0, spread_bp=float("inf"), elgd=0.6)


def test_bump_rejects_curve_without_survival_method():
    class OnlyPillars:
        tenors = np.array([1.0, 3.0])
        hazards = np.array([0.02, 0.02])
        recovery_rate = 0.4

    with pytest.raises(ValidationError):
        bump_hazard_pillar(OnlyPillars(), tenor=1.0, spread_bp=1.0, elgd=0.6)
