"""
Credit stress application logic.

Mirrors the FX stress applicator but operates on
:class:`CreditPricingEnvironment`, whose risk surface is the issuer hazard
intensity and the discount rate. Stress parameter names:

* ``hazard``                       - raw hazard-intensity shock (canonical)
* ``spread`` / ``credit_spread``   - quoted CDS-spread shock, converted to a
  hazard shock via the credit triangle ``s = lambda (1 - R)``
* ``rate`` / ``interest_rate`` / ``discount_rate`` - discount curve

A *quoted-spread* shock is recovery-dependent, while the hazard curve is shared
across positions that may carry different recoveries. Absolute / target-value
spread shocks therefore require a recovery convention: an explicit
``recovery_rate`` in the stress metadata, otherwise the (single) recovery shared
by the affected positions, otherwise the ISDA standard. A percentage spread
shock equals the same percentage hazard shock and needs no recovery.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from quantark.asset.credit.conventions import (
    STANDARD_RECOVERY,
    spread_shift_to_hazard_shift,
)
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.param.rrf.rate_curve import InterpolatedRateCurve, RateCurve
from quantark.priceenv import CreditPricingEnvironment
from quantark.stresstest.stress.stress_types import StressLevel, StressType
from quantark.util.exceptions import ValidationError

if TYPE_CHECKING:
    from quantark.stresstest.scenario.scenario import Scenario, Stress

_SPREAD_PARAMS = ("spread", "credit_spread")


class CreditStressApplicator:
    """Applies scenario stresses to credit pricing environments (per entity)."""

    @staticmethod
    def apply_scenario_to_portfolio(
        portfolio, scenario: "Scenario"
    ) -> Dict[str, CreditPricingEnvironment]:
        stressed: Dict[str, CreditPricingEnvironment] = {
            entity: dataclasses.replace(env)
            for entity, env in portfolio.pricing_environments.items()
        }

        for stress in scenario.stresses:
            recovery = CreditStressApplicator._resolve_spread_recovery(portfolio, stress)
            if stress.level == StressLevel.PORTFOLIO:
                for entity in stressed:
                    stressed[entity] = CreditStressApplicator._apply(
                        stressed[entity], stress, recovery
                    )
            elif stress.level == StressLevel.UNDERLYING:
                if stress.target not in stressed:
                    raise ValidationError(
                        f"Stress target entity '{stress.target}' not found in portfolio"
                    )
                stressed[stress.target] = CreditStressApplicator._apply(
                    stressed[stress.target], stress, recovery
                )
            elif stress.level == StressLevel.POSITION:
                # Credit pricing environments are shared per reference entity, so
                # there is no per-position risk surface to stress in isolation:
                # shocking the entity hazard/discount curve would move *every*
                # trade on that issuer, not just the targeted position. Reject the
                # request rather than silently applying an entity-wide stress.
                position = portfolio.get_position(stress.target)
                if position is None:
                    raise ValidationError(
                        f"Stress target position '{stress.target}' not found in portfolio"
                    )
                raise ValidationError(
                    "Position-level credit stresses are not supported: credit "
                    "environments are keyed per reference entity, so a single "
                    "position cannot be stressed without affecting every trade on "
                    f"issuer '{position.reference_entity}'. Use an UNDERLYING-level "
                    "stress targeting the reference entity instead."
                )

        return stressed

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_spread_recovery(portfolio, stress: "Stress") -> Optional[float]:
        """
        Recovery convention for a quoted-spread stress (``None`` when irrelevant).

        Only absolute / target-value spread shocks depend on recovery; percentage
        spread shocks and hazard/rate shocks do not. When a recovery is needed it
        is taken from the stress metadata, else the single recovery shared by the
        affected positions, else the ISDA standard — but a mix of recoveries among
        the affected positions is rejected as ambiguous.
        """
        if stress.parameter.lower() not in _SPREAD_PARAMS:
            return None
        if stress.stress_type == StressType.PERCENTAGE:
            return None  # percentage spread == percentage hazard, recovery-free

        meta_recovery = stress.metadata.get("recovery_rate")
        if meta_recovery is not None:
            if not 0.0 <= float(meta_recovery) < 1.0:
                raise ValidationError(
                    f"Stress metadata recovery_rate must be in [0, 1), got {meta_recovery}"
                )
            return float(meta_recovery)

        recoveries = CreditStressApplicator._affected_recoveries(portfolio, stress)
        if not recoveries:
            return STANDARD_RECOVERY
        if len(recoveries) > 1:
            raise ValidationError(
                f"Absolute/target spread stress on '{stress.parameter}' is "
                "ambiguous: the affected positions carry mixed recovery rates "
                f"{sorted(recoveries)}. Supply an explicit recovery_rate in the "
                "stress metadata, or stress the 'hazard' factor directly."
            )
        return next(iter(recoveries))

    @staticmethod
    def _affected_recoveries(portfolio, stress: "Stress") -> set:
        """Distinct recovery rates of the positions a stress applies to."""
        positions: List = list(portfolio.positions.values())
        if stress.level == StressLevel.UNDERLYING:
            positions = [p for p in positions if p.reference_entity == stress.target]
        return {
            float(getattr(p.product, "recovery_rate", STANDARD_RECOVERY))
            for p in positions
        }

    @staticmethod
    def _apply(
        env: CreditPricingEnvironment, stress: "Stress", recovery: Optional[float]
    ) -> CreditPricingEnvironment:
        handler = _HANDLERS.get(stress.parameter.lower())
        if handler is None:
            raise ValidationError(
                f"Unknown credit stress parameter '{stress.parameter}'. Supported: "
                f"{sorted(_HANDLERS)}"
            )
        return handler(env, stress, recovery)

    @staticmethod
    def _stress_hazard(
        env: CreditPricingEnvironment, stress: "Stress", recovery: Optional[float]
    ) -> CreditPricingEnvironment:
        """Raw hazard-intensity shock (the canonical curve-level factor)."""
        curve = env.hazard_curve
        if not isinstance(curve, FlatHazardCurve):
            raise ValidationError(
                f"Cannot stress hazard curve type {type(curve).__name__}; "
                "only FlatHazardCurve is supported."
            )
        new_rate = stress.stress_type.apply(curve.hazard_rate, stress.stress_value)
        return CreditStressApplicator._with_hazard(env, new_rate)

    @staticmethod
    def _stress_spread(
        env: CreditPricingEnvironment, stress: "Stress", recovery: Optional[float]
    ) -> CreditPricingEnvironment:
        """Quoted-spread shock, converted to a hazard shock via ``s = lambda(1-R)``."""
        curve = env.hazard_curve
        if not isinstance(curve, FlatHazardCurve):
            raise ValidationError(
                f"Cannot stress hazard curve type {type(curve).__name__}; "
                "only FlatHazardCurve is supported."
            )
        lam = curve.hazard_rate
        if stress.stress_type == StressType.PERCENTAGE:
            # At fixed recovery s is proportional to lambda, so a percentage spread
            # move is the same percentage hazard move (recovery-independent).
            new_rate = lam * (1.0 + stress.stress_value)
        elif stress.stress_type == StressType.ABSOLUTE:
            new_rate = lam + spread_shift_to_hazard_shift(
                stress.stress_value, recovery if recovery is not None else STANDARD_RECOVERY
            )
        elif stress.stress_type == StressType.VALUE:
            # Target a quoted spread level s -> hazard lambda = s / (1 - R).
            new_rate = spread_shift_to_hazard_shift(
                stress.stress_value, recovery if recovery is not None else STANDARD_RECOVERY
            )
        else:
            raise ValidationError(f"Unsupported stress type {stress.stress_type}")
        return CreditStressApplicator._with_hazard(env, new_rate)

    @staticmethod
    def _with_hazard(
        env: CreditPricingEnvironment, new_rate: float
    ) -> CreditPricingEnvironment:
        if new_rate < 0:
            raise ValidationError(
                f"Stressed hazard rate must be non-negative, got {new_rate}"
            )
        return dataclasses.replace(env, hazard_curve=FlatHazardCurve(hazard_rate=new_rate))

    @staticmethod
    def _stress_rate(
        env: CreditPricingEnvironment, stress: "Stress", recovery: Optional[float]
    ) -> CreditPricingEnvironment:
        return dataclasses.replace(
            env, discount_curve=CreditStressApplicator._shift_curve(
                env.discount_curve, stress
            )
        )

    @staticmethod
    def _shift_curve(curve: RateCurve, stress: "Stress") -> RateCurve:
        current = curve.get_rate(1.0)
        new_rate = stress.stress_type.apply(current, stress.stress_value)
        if isinstance(curve, FlatRateCurve):
            return FlatRateCurve(rate=new_rate)
        if isinstance(curve, InterpolatedRateCurve):
            delta = new_rate - current
            return curve.__class__([(t, r + delta) for t, r in curve.pillars])
        raise ValidationError(
            f"Cannot stress credit discount curve type {type(curve).__name__}; "
            "only FlatRateCurve or interpolated curves are supported."
        )


_HANDLERS: Dict[
    str,
    Callable[
        [CreditPricingEnvironment, "Stress", Optional[float]], CreditPricingEnvironment
    ],
] = {
    "hazard": CreditStressApplicator._stress_hazard,
    "spread": CreditStressApplicator._stress_spread,
    "credit_spread": CreditStressApplicator._stress_spread,
    "rate": CreditStressApplicator._stress_rate,
    "interest_rate": CreditStressApplicator._stress_rate,
    "discount_rate": CreditStressApplicator._stress_rate,
}
