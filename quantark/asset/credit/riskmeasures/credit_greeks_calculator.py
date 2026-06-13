"""
Credit risk measures by bump-and-reprice.

All sensitivities are computed by central finite differences on the
:class:`~quantark.priceenv.CreditPricingEnvironment` (for market shifts) or on
the product (for recovery), using the position holder's signed price.

Two distinct spread-risk measures are reported (the credit triangle
``s = lambda (1 - R)`` links them):

* ``hazard01`` - PV change for a 1bp shift of the **hazard intensity**. This is
  the canonical sensitivity for the shared hazard curve and is consistent with
  the curve-level shocks applied by the VaR, stress and dynamic-scenario engines.
* ``cs01`` - PV change for a 1bp shift of the **quoted CDS spread**, obtained
  from a ``1bp / (1 - R)`` hazard move. This is the product-level / SIMM-facing
  measure. At fixed recovery ``cs01 = hazard01 / (1 - R)``.
"""
from dataclasses import replace
from typing import Any, Dict

from quantark.asset.credit.conventions import STANDARD_RECOVERY
from quantark.priceenv import CreditPricingEnvironment

# Default bump sizes.
_HAZARD_BUMP = 1e-4   # 1 bp of hazard intensity (Hazard01)
_SPREAD_BUMP = 1e-4   # 1 bp of quoted CDS spread (CS01)
_RATE_BUMP = 1e-4     # 1 bp of interest rate (IR01)
_REC_BUMP = 0.01      # 1 percentage point of recovery (Rec01)


def _recovery_rate(product: Any) -> float:
    """Recovery used for the spread->hazard chain rule (mirrors the SIMM path)."""
    return getattr(product, "recovery_rate", STANDARD_RECOVERY)


class CreditGreeksCalculator:
    """Finite-difference credit Greeks for CDS-type products."""

    def __init__(
        self,
        hazard_bump: float = _HAZARD_BUMP,
        spread_bump: float = _SPREAD_BUMP,
        rate_bump: float = _RATE_BUMP,
        recovery_bump: float = _REC_BUMP,
    ):
        self.hazard_bump = hazard_bump
        self.spread_bump = spread_bump
        self.rate_bump = rate_bump
        self.recovery_bump = recovery_bump

    def hazard01(
        self, product: Any, env: CreditPricingEnvironment, engine: Any
    ) -> float:
        """Hazard sensitivity: PV change for a +1bp hazard-intensity shift."""
        h = self.hazard_bump
        up = engine.price(product, env.with_hazard_shift(+h))
        down = engine.price(product, env.with_hazard_shift(-h))
        return (up - down) / 2.0

    def cs01(self, product: Any, env: CreditPricingEnvironment, engine: Any) -> float:
        """
        Credit-spread CS01: PV change for a +1bp quoted CDS-spread shift.

        The spread bump is converted to a hazard-intensity bump through the
        credit triangle ``s = lambda (1 - R)``: a 1bp spread move is a
        ``1bp / (1 - R)`` hazard move. This matches the spread->hazard
        conversion used when generating SIMM credit-delta sensitivities. (The
        curve-level VaR / stress / dynamic-scenario engines shock the hazard
        intensity directly and so correspond to ``hazard01``, not ``cs01``.)
        """
        lgd = 1.0 - _recovery_rate(product)
        if lgd <= 1e-6:
            # Full recovery: the spread s = lambda (1 - R) is pinned at zero, so a
            # CDS-spread bump is undefined and the contract carries no spread risk.
            return 0.0
        h = self.spread_bump / lgd
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
            "hazard01": self.hazard01(product, env, engine),
            "cs01": self.cs01(product, env, engine),
            "ir01": self.ir01(product, env, engine),
            "rec01": self.rec01(product, env, engine),
        }
