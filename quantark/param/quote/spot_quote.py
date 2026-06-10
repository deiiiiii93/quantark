"""
Spot price quote representation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import validate_positive


@dataclass
class SpotQuote:
    """
    Represents the current spot price of an underlying asset.

    Attributes:
        spot: Current spot price of the underlying asset
        timestamp: Optional timestamp of the quote
        asset_name: Optional name/identifier of the asset
    """

    spot: float
    timestamp: Optional[datetime] = None
    asset_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate spot price."""
        if not isinstance(self.spot, (int, float)):
            raise ValidationError(f"Spot price must be numeric, got {self.spot}")
        validate_positive(float(self.spot), name="spot")

    def __repr__(self):
        return f"SpotQuote(spot={self.spot:.4f}, asset={self.asset_name})"
