"""
FX stress application logic.

Mirrors :class:`~quantark.stresstest.stress.stress_applicator.StressApplicator`
but operates on :class:`FxPricingEnvironment`, whose risk surface is spot, vol,
and *two* rate curves (domestic / foreign). Stress parameter names:

* ``spot``                          - FX spot rate
* ``vol`` / ``volatility``          - FX volatility
* ``domestic_rate`` / ``rate_dom``  - domestic (quote-currency) curve
* ``foreign_rate``  / ``rate_for``  - foreign (base-currency) curve
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Dict

from quantark.param import (
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
    TermStructureVolSurface,
)
from quantark.param.rrf.rate_curve import InterpolatedRateCurve, RateCurve
from quantark.priceenv import FxPricingEnvironment
from quantark.stresstest.stress.stress_types import StressLevel
from quantark.util.exceptions import ValidationError

if TYPE_CHECKING:
    from quantark.stresstest.scenario.scenario import Scenario, Stress


class FXStressApplicator:
    """Applies scenario stresses to FX pricing environments (per currency pair)."""

    @staticmethod
    def apply_scenario_to_portfolio(
        portfolio, scenario: "Scenario"
    ) -> Dict[str, FxPricingEnvironment]:
        """Return a new dict of stressed FX environments keyed by currency pair."""
        stressed: Dict[str, FxPricingEnvironment] = {
            pair: FXStressApplicator._clone_env(env)
            for pair, env in portfolio.pricing_environments.items()
        }

        for stress in scenario.stresses:
            if stress.level == StressLevel.PORTFOLIO:
                for pair in stressed:
                    stressed[pair] = FXStressApplicator._apply(stressed[pair], stress)
            elif stress.level == StressLevel.UNDERLYING:
                if stress.target not in stressed:
                    raise ValidationError(
                        f"Stress target pair '{stress.target}' not found in portfolio"
                    )
                stressed[stress.target] = FXStressApplicator._apply(
                    stressed[stress.target], stress
                )
            elif stress.level == StressLevel.POSITION:
                position = portfolio.get_position(stress.target)
                if position is None:
                    raise ValidationError(
                        f"Stress target position '{stress.target}' not found in portfolio"
                    )
                pair = position.underlying
                stressed[pair] = FXStressApplicator._apply(stressed[pair], stress)

        return stressed

    # ------------------------------------------------------------------ #
    @staticmethod
    def _clone_env(env: FxPricingEnvironment) -> FxPricingEnvironment:
        return dataclasses.replace(env)

    @staticmethod
    def _apply(env: FxPricingEnvironment, stress: "Stress") -> FxPricingEnvironment:
        param = stress.parameter.lower()
        handler = _HANDLERS.get(param)
        if handler is None:
            raise ValidationError(
                f"Unknown FX stress parameter '{stress.parameter}'. Supported: "
                f"{sorted(_HANDLERS)}"
            )
        return handler(env, stress)

    # -- individual parameter handlers --------------------------------- #
    @staticmethod
    def _stress_spot(env: FxPricingEnvironment, stress: "Stress") -> FxPricingEnvironment:
        current = env.spot_quote.spot
        new_spot = stress.stress_type.apply(current, stress.stress_value)
        if new_spot <= 0:
            raise ValidationError(f"Stressed FX spot must be positive, got {new_spot}")
        new_quote = SpotQuote(
            spot=new_spot,
            timestamp=env.spot_quote.timestamp,
            asset_name=env.spot_quote.asset_name,
        )
        return dataclasses.replace(env, spot_quote=new_quote)

    @staticmethod
    def _stress_vol(env: FxPricingEnvironment, stress: "Stress") -> FxPricingEnvironment:
        # A pair with no vol surface carries no vega (e.g. delta-one only book),
        # so a portfolio-wide vol shock is a genuine no-op on that pair.
        if env.vol_surface is None:
            return env
        surface = env.vol_surface
        if isinstance(surface, FlatVolSurface):
            new_vol = stress.stress_type.apply(surface.volatility, stress.stress_value)
            if new_vol <= 0:
                raise ValidationError(f"Stressed FX vol must be positive, got {new_vol}")
            return dataclasses.replace(env, vol_surface=FlatVolSurface(volatility=new_vol))
        if isinstance(surface, TermStructureVolSurface):
            new_vols = [
                stress.stress_type.apply(float(v), stress.stress_value)
                for v in surface.vols
            ]
            if any(v <= 0 for v in new_vols):
                raise ValidationError("Stressed FX term-structure vol must be positive.")
            return dataclasses.replace(
                env,
                vol_surface=TermStructureVolSurface(
                    times=list(surface.times), vols=new_vols
                ),
            )
        raise ValidationError(
            f"Stressing vol surface type {type(surface).__name__} not supported."
        )

    @staticmethod
    def _stress_domestic(
        env: FxPricingEnvironment, stress: "Stress"
    ) -> FxPricingEnvironment:
        return dataclasses.replace(
            env, domestic_curve=FXStressApplicator._shift_curve(env.domestic_curve, stress)
        )

    @staticmethod
    def _stress_foreign(
        env: FxPricingEnvironment, stress: "Stress"
    ) -> FxPricingEnvironment:
        return dataclasses.replace(
            env, foreign_curve=FXStressApplicator._shift_curve(env.foreign_curve, stress)
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
            f"Cannot stress FX curve type {type(curve).__name__}; "
            "only FlatRateCurve or interpolated curves are supported."
        )


_HANDLERS: Dict[str, Callable[[FxPricingEnvironment, "Stress"], FxPricingEnvironment]] = {
    "spot": FXStressApplicator._stress_spot,
    "vol": FXStressApplicator._stress_vol,
    "volatility": FXStressApplicator._stress_vol,
    "domestic_rate": FXStressApplicator._stress_domestic,
    "rate_dom": FXStressApplicator._stress_domestic,
    "foreign_rate": FXStressApplicator._stress_foreign,
    "rate_for": FXStressApplicator._stress_foreign,
}
