"""
Base class for bond forward and futures products.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from util.exceptions import ValidationError


@dataclass
class BaseBondForward(ABC):
    """
    Abstract base class for bond forwards and futures.
    
    This class defines the common interface for bond derivative products
    that involve future delivery of bonds.
    
    Attributes:
        delivery_date: Date when the bond is delivered
        contract_size: Notional amount per contract (default: 100,000)
    """
    delivery_date: datetime
    contract_size: float = 100_000.0
    
    @abstractmethod
    def get_delivery_date(self) -> datetime:
        """
        Get the delivery date of the contract.
        
        Returns:
            Delivery date
        """
        pass
    
    @abstractmethod
    def validate(self) -> None:
        """
        Validate contract parameters.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        pass
    
    def get_time_to_delivery(self, valuation_date: datetime) -> float:
        """
        Calculate time to delivery in years.
        
        Args:
            valuation_date: Current valuation date
            
        Returns:
            Time to delivery in years (ACT/365 basis)
        """
        delta = self.delivery_date - valuation_date
        return delta.days / 365.0
    
    def is_expired(self, valuation_date: datetime) -> bool:
        """
        Check if the contract has expired.
        
        Args:
            valuation_date: Date to check
            
        Returns:
            True if contract has expired, False otherwise
        """
        return valuation_date >= self.delivery_date
    
    def days_to_delivery(self, valuation_date: datetime) -> int:
        """
        Calculate days to delivery.
        
        Args:
            valuation_date: Current valuation date
            
        Returns:
            Number of days to delivery
        """
        delta = self.delivery_date - valuation_date
        return max(0, delta.days)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"delivery={self.delivery_date.date()}, "
                f"size={self.contract_size:,.0f})")

