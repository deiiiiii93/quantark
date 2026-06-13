"""
Credit risk measures by bump-and-reprice.

All sensitivities are computed by central finite differences on the
:class:`~quantark.priceenv.CreditPricingEnvironment` (for market shifts) or on
the product (for recovery), using the position holder's signed price.
"""
from dataclasses import replace
from typing import Any, Dict

from quantark.priceenv import CreditPricingEnvironment

# Default bump sizes.
_HAZARD_BUMP = 1e-4   # 1 bp of hazard intensity (CS01)
_RATE_BUMP = 1e-4     # 1 bp of interest rate (IR01)
_REC_BUMP = 0.01      # 1 percentage point of recovery (Rec01)


class CreditGreeksCalculator:
    """Finite-difference credit Greeks for CDS-type products."""

    def __init__(
        self,
        hazard_bump: float = _HAZARD_BUMP,
        rate_bump: float = _RATE_BUMP,
        recovery_bump: float = _REC_BUMP,
    ):
        self.hazard_bump = hazard_bump
        self.rate_bump = rate_bump
        self.recovery_bump = recovery_bump

    def cs01(self, product: Any, env: CreditPricingEnvironment, engine: Any) -> float:
        """Credit spread sensitivity: PV change for a +1bp hazard shift."""
        h = self.hazard_bump
        up = engine.price(product, env.with_hazard_shift(+h))
        down = engine.price(product, env.with_hazard_shift(-h))
        return (up - down) / 2.0

    def ir01(self, product: Any, env: CreditPricingEnvironment, engine: Any) -> float:
        """Interest-rate sensitivity: PV change for a +1bp parallel rate shift."""
        h = self.rate_bump
        up = engine.price(product, env.with_rate_shift(+h))
        down = engine.price(product, env.with_rate_shift(-h))
        return (up - down) / 2.0

    def rec01(self, product: Any, env: CreditPricingEnvironment, engine: Any) -> float:
        """Recovery sensitivity: PV change for a +1 percentage-point recovery."""
        h = self.recovery_bump
        r = product.recovery_rate
        up = engine.price(replace(product, recovery_rate=min(1.0, r + h)), env)
        down = engine.price(replace(product, recovery_rate=max(0.0, r - h)), env)
        span = min(1.0, r + h) - max(0.0, r - h)
        return (up - down) / span * h

    def calculate(
        self, product: Any, env: CreditPricingEnvironment, engine: Any
    ) -> Dict[str, float]:
        """All credit risk measures plus the present value."""
        return {
            "price": engine.price(product, env),
            "cs01": self.cs01(product, env, engine),
            "ir01": self.ir01(product, env, engine),
            "rec01": self.rec01(product, env, engine),
        }
