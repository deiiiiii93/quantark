"""Portfolio of CVA sensitivities for SA-CVA aggregation."""

from dataclasses import dataclass, field
from typing import List

from quantark.sacva.models.enums import RiskClass
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.util.exceptions import ValidationError


@dataclass
class CVAPortfolio:
    """A set of CVA + hedge sensitivities with a reporting currency."""

    sensitivities: List[CVASensitivity] = field(default_factory=list)
    reporting_currency: str = "USD"

    def __post_init__(self):
        if not self.sensitivities:
            raise ValidationError("Portfolio must contain at least one sensitivity")
        for s in self.sensitivities:
            if not isinstance(s, CVASensitivity):
                raise ValidationError(f"Expected CVASensitivity, got {s!r}")
            if s.risk_class == RiskClass.FX and s.currency == self.reporting_currency:
                raise ValidationError(
                    "FX sensitivity must not reference the reporting currency "
                    f"({self.reporting_currency})")
