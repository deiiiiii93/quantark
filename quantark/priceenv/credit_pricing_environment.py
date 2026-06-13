"""
Credit pricing environment bundling discount and hazard-rate market data.
"""
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import List

from quantark.param import RateCurve
from quantark.param.credit import HazardCurve, ParallelShiftHazardCurve
from quantark.param.rrf import ParallelShiftRateCurve
from quantark.util.exceptions import MarketDataError


@dataclass
class CreditPricingEnvironment:
    """
    Pricing environment for credit derivatives (CDS, basket CDS).

    Bundles the risk-free discount curve and the issuer hazard-rate
    (default-intensity) curve. Recovery is a contract/entity attribute and
    lives on the product, not here.

    The environment is treated as immutable; ``with_*`` helpers return shifted
    copies for finite-difference risk (CS01, IR01) and scenario revaluation.

    Attributes:
        valuation_date: Date of valuation (required)
        discount_curve: Risk-free discount curve (required)
        hazard_curve: Issuer hazard-rate curve (required)
    """

    valuation_date: datetime
    discount_curve: RateCurve
    hazard_curve: HazardCurve

    def __post_init__(self) -> None:
        if self.valuation_date is None:
            raise MarketDataError("Valuation date is required")
        if self.discount_curve is None:
            raise MarketDataError("Discount curve is required")
        if self.hazard_curve is None:
            raise MarketDataError("Hazard curve is required")

    # ------------------------------------------------------------------ #
    # Market-data accessors
    # ------------------------------------------------------------------ #
    def get_discount_factor(self, time: float) -> float:
        """Risk-free discount factor to ``time``."""
        return self.discount_curve.get_discount_factor(time)

    def get_rate(self, time: float) -> float:
        """Risk-free zero rate for ``time``."""
        return self.discount_curve.get_rate(time)

    def get_survival_probability(self, time: float) -> float:
        """Issuer survival probability to ``time``."""
        return self.hazard_curve.get_survival_probability(time)

    def get_default_probability(self, time: float) -> float:
        """Cumulative issuer default probability by ``time``."""
        return self.hazard_curve.get_default_probability(time)

    def get_hazard_rate(self, time: float) -> float:
        """Issuer hazard rate (default intensity) at ``time``."""
        return self.hazard_curve.get_hazard_rate(time)

    def get_default_density(self, time: float) -> float:
        """Unconditional issuer default density at ``time``."""
        return self.hazard_curve.get_default_density(time)

    # ------------------------------------------------------------------ #
    # Shift helpers (return new environments; original untouched)
    # ------------------------------------------------------------------ #
    def with_hazard_shift(self, shift: float) -> "CreditPricingEnvironment":
        """Copy with the hazard intensity shifted by ``shift`` (additive)."""
        return replace(
            self, hazard_curve=ParallelShiftHazardCurve(self.hazard_curve, shift)
        )

    def with_rate_shift(self, shift: float) -> "CreditPricingEnvironment":
        """Copy with the discount curve shifted by ``shift`` (parallel, additive)."""
        return replace(
            self, discount_curve=ParallelShiftRateCurve(self.discount_curve, shift)
        )

    def __repr__(self) -> str:
        return (
            f"CreditPricingEnvironment(valuation_date={self.valuation_date.date()}, "
            f"r={self.get_rate(1.0):.2%}, lambda={self.get_hazard_rate(1.0):.2%})"
        )


@dataclass
class BasketCreditPricingEnvironment:
    """
    Pricing environment for a basket of credit reference entities.

    Bundles a shared discount curve with one hazard-rate curve per reference
    entity. Used by basket-CDS Monte-Carlo engines.

    Attributes:
        valuation_date: Date of valuation (required).
        discount_curve: Shared risk-free discount curve (required).
        hazard_curves: One issuer hazard curve per basket entity (required).
    """

    valuation_date: datetime
    discount_curve: RateCurve
    hazard_curves: List[HazardCurve] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.valuation_date is None:
            raise MarketDataError("Valuation date is required")
        if self.discount_curve is None:
            raise MarketDataError("Discount curve is required")
        if not self.hazard_curves:
            raise MarketDataError("At least one entity hazard curve is required")

    @property
    def n_entities(self) -> int:
        return len(self.hazard_curves)

    def get_discount_factor(self, time: float) -> float:
        return self.discount_curve.get_discount_factor(time)

    def entity_env(self, index: int) -> CreditPricingEnvironment:
        """Single-name pricing environment for entity ``index``."""
        return CreditPricingEnvironment(
            valuation_date=self.valuation_date,
            discount_curve=self.discount_curve,
            hazard_curve=self.hazard_curves[index],
        )

    def with_hazard_shift(self, shift: float) -> "BasketCreditPricingEnvironment":
        """Copy with every entity hazard curve shifted by ``shift``."""
        return replace(
            self,
            hazard_curves=[
                ParallelShiftHazardCurve(c, shift) for c in self.hazard_curves
            ],
        )

    def with_rate_shift(self, shift: float) -> "BasketCreditPricingEnvironment":
        """Copy with the shared discount curve shifted by ``shift``."""
        return replace(
            self, discount_curve=ParallelShiftRateCurve(self.discount_curve, shift)
        )
