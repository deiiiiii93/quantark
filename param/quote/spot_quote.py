"""
Spot price quote representation.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from util.exceptions import ValidationError


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
    
    def __post_init__(self):
        """Validate spot price."""
        if self.spot <= 0:
            raise ValidationError(f"Spot price must be positive, got {self.spot}")
    
    def __repr__(self):
        return f"SpotQuote(spot={self.spot:.4f}, asset={self.asset_name})"

