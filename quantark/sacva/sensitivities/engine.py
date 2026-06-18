"""Bump-and-revalue SA-CVA sensitivity engine (spec §3.4).

Counterparty credit-spread delta (MAR50.63, per entity x tenor): the regulatory
exposure (EE) is invariant to the counterparty hazard, so each tenor's sensitivity
re-runs only the (cheap) regulatory CVA after a one-sided 1bp key-rate hazard bump:

    s_cva(tenor) = (CVA(spread + 1bp) - CVA) / 1e-4

with the spread->hazard chain rule handled by ``bump_hazard_pillar`` (Δλ = 1bp/ELGD).
Equity/IR market delta and vega (which DO move the exposure and need an MC re-run with
common random numbers, plus per-underlying bucket metadata) are a later extension.
"""

from quantark.sacva.cva.engine import RegulatoryCVAEngine
from quantark.sacva.models.enums import RiskClass, RiskType
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.sensitivities.curve_bump import bump_hazard_pillar
from quantark.util.exceptions import ValidationError

_CREDIT_DIVISOR = 1e-4  # 1bp absolute spread shift (MAR50.63)


class CVASensitivityEngine:
    def __init__(self, cva_engine=None):
        self.cva_engine = cva_engine or RegulatoryCVAEngine()

    def counterparty_spread_deltas(self, counterparty, profile):
        """Per-tenor counterparty credit-spread delta ``CVASensitivity`` records."""
        curve = counterparty.credit_curve
        if not (hasattr(curve, "tenors") and hasattr(curve, "recovery_rate")):
            raise ValidationError(
                f"{counterparty.name}: counterparty credit-spread delta requires a "
                "pillar hazard curve (tenors/hazards/recovery_rate)")
        elgd = 1.0 - float(curve.recovery_rate)
        base_cva = self.cva_engine.compute(curve, profile)
        out = []
        for tenor in [float(t) for t in curve.tenors]:
            bumped = bump_hazard_pillar(curve, tenor=tenor, spread_bp=1.0, elgd=elgd)
            cva_up = self.cva_engine.compute(bumped, profile)
            s_cva = (cva_up - base_cva) / _CREDIT_DIVISOR
            out.append(CVASensitivity(
                risk_class=RiskClass.COUNTERPARTY_CREDIT,
                risk_type=RiskType.DELTA,
                bucket=counterparty.bucket,
                risk_factor=f"{counterparty.name}@{tenor:g}",
                s_cva=s_cva,
                name=counterparty.name,
                tenor=tenor,
                credit_quality=counterparty.credit_quality,
                sub_bucket=counterparty.sub_bucket,
                legal_entity_group=counterparty.legal_entity_group,
            ))
        return out
