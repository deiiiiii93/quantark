"""
Base class for bond products.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from asset.bond.schedule.cashflow import CashFlow


class BaseBondProduct(ABC):
    """
    Abstract base class for all bond products.
    
    This class defines the common interface that all bond derivatives must implement.
    Bond products generate cashflows and provide methods for calculating accrued interest.
    """
    
    @abstractmethod
    def get_cashflows(self, valuation_date: datetime) -> List[CashFlow]:
        """
        Get all future cashflows from valuation date.
        
        Args:
            valuation_date: Date to value the bond
            
        Returns:
            List of future cashflows
        """
        pass
    
    @abstractmethod
    def get_maturity_date(self) -> datetime:
        """
        Get the maturity date of the bond.
        
        Returns:
            Maturity date
        """
        pass
    
    @abstractmethod
    def get_issue_date(self) -> datetime:
        """
        Get the issue date of the bond.
        
        Returns:
            Issue date
        """
        pass
    
    @abstractmethod
    def get_notional(self) -> float:
        """
        Get the notional/face value of the bond.
        
        Returns:
            Notional amount
        """
        pass
    
    @abstractmethod
    def calculate_accrued_interest(self, settlement_date: datetime) -> float:
        """
        Calculate accrued interest as of settlement date.
        
        Args:
            settlement_date: Settlement date for the calculation
            
        Returns:
            Accrued interest amount
        """
        pass
    
    @abstractmethod
    def validate(self) -> None:
        """
        Validate bond product parameters.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        pass
    
    def time_to_maturity(self, valuation_date: datetime) -> float:
        """
        Calculate time to maturity in years.
        
        Args:
            valuation_date: Valuation date
            
        Returns:
            Time to maturity in years
        """
        maturity = self.get_maturity_date()
        delta = maturity - valuation_date
        return delta.days / 365.0
    
    def is_expired(self, valuation_date: datetime) -> bool:
        """
        Check if bond has matured.
        
        Args:
            valuation_date: Date to check
            
        Returns:
            True if bond has matured, False otherwise
        """
        return valuation_date >= self.get_maturity_date()
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"maturity={self.get_maturity_date().date()}, "
                f"notional={self.get_notional():.2f})")

