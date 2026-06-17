"""The MAR50.34(1) eligibility guard. A historical (REAL_WORLD) exposure profile
must NEVER feed the SA-CVA capital path. The guard lives at the capital-path
entry point (MC-owned ``RegulatoryCVAEngine`` / adapter / facade); the stub here
is a PROVISIONAL stand-in so the guard can be tested from this worktree.
"""
from __future__ import annotations

from quantark.util.exceptions import ValidationError
from quantark.sacva.exposure._contract_provisional import Measure


def assert_regulatory_eligible(profile) -> None:
    """Raise unless ``profile`` is risk-neutral, regulatory-eligible, and carries the
    regulatory discounted-EE field (MAR50.34(1): historical drifts not allowed)."""
    if profile.measure is not Measure.RISK_NEUTRAL:
        raise ValidationError(
            "historical/real-world exposure is not SA-CVA eligible "
            "(MAR50.34(1): historical drifts not allowed)")
    if not profile.regulatory_eligible:
        raise ValidationError("exposure profile is not regulatory_eligible")
    if profile.epe_discounted is None:
        raise ValidationError("regulatory CVA reads epe_discounted; it is None")


class ProvisionalRegulatoryCVAStub:
    """PROVISIONAL stand-in for the MC-owned RegulatoryCVAEngine entry point;
    exists only to test the guard from this worktree. Deleted at merge."""

    def compute(self, counterparty, exposure_profile):
        assert_regulatory_eligible(exposure_profile)
        # field-identity audit: only epe_discounted may feed the integral
        return float(exposure_profile.epe_discounted.sum())
