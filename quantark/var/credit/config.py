"""Configuration for credit risk factors used by the credit VaR engines."""

from dataclasses import dataclass


@dataclass
class CreditRiskFactorConfig:
    """
    Which credit risk factors to include in VaR.

    Credit positions carry a default-intensity (hazard / credit-spread) factor
    and an interest-rate factor per reference entity.
    """

    include_spread: bool = True
    include_rate: bool = True
