"""
Credit stress application logic.

Mirrors the FX stress applicator but operates on
:class:`CreditPricingEnvironment`, whose risk surface is the issuer hazard
intensity (credit spread) and the discount rate. Stress parameter names:

* ``spread`` / ``hazard`` / ``credit_spread`` - issuer hazard intensity
* ``rate`` / ``interest_rate`` / ``discount_rate`` - discount curve
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Dict

from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.param.rrf.rate_curve import InterpolatedRateCurve, RateCurve
from quantark.priceenv import CreditPricingEnvironment
from quantark.stresstest.stress.stress_types import StressLevel
from quantark.util.exceptions import ValidationError

if TYPE_CHECKING:
    from quantark.stresstest.scenario.scenario import Scenario, Stress


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
            if stress.level == StressLevel.PORTFOLIO:
                for entity in stressed:
                    stressed[entity] = CreditStressApplicator._apply(
                        stressed[entity], stress
                    )
            elif stress.level == StressLevel.UNDERLYING:
                if stress.target not in stressed:
                    raise ValidationError(
                        f"Stress target entity '{stress.target}' not found in portfolio"
                    )
                stressed[stress.target] = CreditStressApplicator._apply(
                    stressed[stress.target], stress
                )
            elif stress.level == StressLevel.POSITION:
                position = portfolio.get_position(stress.target)
                if position is None:
                    raise ValidationError(
                        f"Stress target position '{stress.target}' not found in portfolio"
                    )
                entity = position.reference_entity
                stressed[entity] = CreditStressApplicator._apply(
                    stressed[entity], stress
                )

        return stressed

    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply(
        env: CreditPricingEnvironment, stress: "Stress"
    ) -> CreditPricingEnvironment:
        handler = _HANDLERS.get(stress.parameter.lower())
        if handler is None:
            raise ValidationError(
                f"Unknown credit stress parameter '{stress.parameter}'. Supported: "
                f"{sorted(_HANDLERS)}"
            )
        return handler(env, stress)

    @staticmethod
    def _stress_spread(
        env: CreditPricingEnvironment, stress: "Stress"
    ) -> CreditPricingEnvironment:
        curve = env.hazard_curve
        if not isinstance(curve, FlatHazardCurve):
            raise ValidationError(
                f"Cannot stress hazard curve type {type(curve).__name__}; "
                "only FlatHazardCurve is supported."
            )
        new_rate = stress.stress_type.apply(curve.hazard_rate, stress.stress_value)
        if new_rate < 0:
            raise ValidationError(
                f"Stressed hazard rate must be non-negative, got {new_rate}"
            )
        return dataclasses.replace(env, hazard_curve=FlatHazardCurve(hazard_rate=new_rate))

    @staticmethod
    def _stress_rate(
        env: CreditPricingEnvironment, stress: "Stress"
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
    str, Callable[[CreditPricingEnvironment, "Stress"], CreditPricingEnvironment]
] = {
    "spread": CreditStressApplicator._stress_spread,
    "hazard": CreditStressApplicator._stress_spread,
    "credit_spread": CreditStressApplicator._stress_spread,
    "rate": CreditStressApplicator._stress_rate,
    "interest_rate": CreditStressApplicator._stress_rate,
    "discount_rate": CreditStressApplicator._stress_rate,
}
