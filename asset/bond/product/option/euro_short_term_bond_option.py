"""
European short-term bond option product.

A European option on a fixed-rate bond, priced using the Black '76 model.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from asset.bond.product.couponbond.fixed_bond import FixedBond
from util.enum import OptionType
from util.exceptions import ValidationError


@dataclass
class EuroShortTermBondOption:
    """
    European option on a fixed-rate bond.
    
    This product represents a European-style option where the holder has the right
    (but not the obligation) to buy (call) or sell (put) a bond at a specified
    strike price on the expiry date.
    
    The option is typically priced using the Black '76 model, which models the
    forward bond price as a lognormal process.
    
    Attributes:
        underlying: The fixed bond that is the underlying asset
        strike: Strike price (clean or dirty based on strike_is_clean)
        expiry_date: Option expiration date
        option_type: CALL or PUT
        notional: Number of bonds (contract multiplier), default 1.0
        strike_is_clean: If True, strike is clean price; if False, dirty price
        settlement_days: Days between exercise and settlement (default: 0)
    """
    underlying: FixedBond
    strike: float
    expiry_date: datetime
    option_type: OptionType
    notional: float = 1.0
    strike_is_clean: bool = True
    settlement_days: int = 0
    
    def __post_init__(self):
        """Validate option parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate option parameters.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        if self.underlying is None:
            raise ValidationError("Underlying bond is required")
        
        if not isinstance(self.underlying, FixedBond):
            raise ValidationError(
                f"Underlying must be a FixedBond, got {type(self.underlying).__name__}"
            )
        
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")
        
        if self.expiry_date is None:
            raise ValidationError("Expiry date is required")
        
        if self.expiry_date >= self.underlying.maturity_date:
            raise ValidationError(
                f"Option expiry ({self.expiry_date.date()}) must be before "
                f"bond maturity ({self.underlying.maturity_date.date()})"
            )
        
        if not isinstance(self.option_type, OptionType):
            raise ValidationError(f"Invalid option type: {self.option_type}")
        
        if self.notional <= 0:
            raise ValidationError(f"Notional must be positive, got {self.notional}")
        
        if self.settlement_days < 0:
            raise ValidationError(
                f"Settlement days must be non-negative, got {self.settlement_days}"
            )
    
    def get_time_to_expiry(self, valuation_date: datetime) -> float:
        """
        Calculate time to option expiry in years.
        
        Args:
            valuation_date: Current valuation date
            
        Returns:
            Time to expiry in years (ACT/365 basis)
            
        Raises:
            ValidationError: If valuation date is after expiry
        """
        if valuation_date >= self.expiry_date:
            raise ValidationError(
                f"Valuation date ({valuation_date.date()}) must be before "
                f"expiry date ({self.expiry_date.date()})"
            )
        
        delta = self.expiry_date - valuation_date
        return delta.days / 365.0
    
    def is_expired(self, valuation_date: datetime) -> bool:
        """
        Check if option has expired.
        
        Args:
            valuation_date: Date to check
            
        Returns:
            True if option has expired
        """
        return valuation_date >= self.expiry_date
    
    def get_payoff(self, bond_price: float) -> float:
        """
        Calculate the option payoff at expiry.
        
        For a call: max(bond_price - strike, 0) * notional
        For a put:  max(strike - bond_price, 0) * notional
        
        Args:
            bond_price: Bond price at expiry (clean or dirty based on strike_is_clean)
            
        Returns:
            Option payoff
            
        Raises:
            ValidationError: If bond price is invalid
        """
        if bond_price < 0:
            raise ValidationError(f"Bond price must be non-negative, got {bond_price}")
        
        if self.is_call():
            intrinsic = max(bond_price - self.strike, 0.0)
        else:
            intrinsic = max(self.strike - bond_price, 0.0)
        
        return intrinsic * self.notional
    
    def intrinsic_value(self, bond_price: float) -> float:
        """
        Calculate the intrinsic value of the option.
        
        Args:
            bond_price: Current bond price
            
        Returns:
            Intrinsic value (same as payoff for European options)
        """
        return self.get_payoff(bond_price)
    
    def is_call(self) -> bool:
        """Check if option is a call."""
        return self.option_type == OptionType.CALL
    
    def is_put(self) -> bool:
        """Check if option is a put."""
        return self.option_type == OptionType.PUT
    
    def get_settlement_date(self) -> datetime:
        """
        Get settlement date after exercise.
        
        Returns:
            Settlement date (expiry + settlement_days)
        """
        from datetime import timedelta
        return self.expiry_date + timedelta(days=self.settlement_days)
    
    def get_underlying_time_to_maturity(self, valuation_date: datetime) -> float:
        """
        Get time to maturity of the underlying bond.
        
        Args:
            valuation_date: Current valuation date
            
        Returns:
            Time to bond maturity in years
        """
        return self.underlying.time_to_maturity(valuation_date)
    
    def __repr__(self):
        return (
            f"EuroShortTermBondOption("
            f"{self.option_type.name}, "
            f"K={self.strike:.2f}, "
            f"expiry={self.expiry_date.date()}, "
            f"underlying={self.underlying.maturity_date.date()})"
        )


def create_bond_option(
    underlying: FixedBond,
    strike: float,
    expiry_date: datetime,
    option_type: OptionType,
    notional: float = 1.0,
    strike_is_clean: bool = True
) -> EuroShortTermBondOption:
    """
    Create a European bond option with standard conventions.
    
    This is a convenience function for creating a bond option.
    
    Args:
        underlying: The fixed bond that is the underlying asset
        strike: Strike price
        expiry_date: Option expiration date
        option_type: CALL or PUT
        notional: Number of bonds (default: 1.0)
        strike_is_clean: If True, strike is clean price (default: True)
        
    Returns:
        EuroShortTermBondOption object
    """
    return EuroShortTermBondOption(
        underlying=underlying,
        strike=strike,
        expiry_date=expiry_date,
        option_type=option_type,
        notional=notional,
        strike_is_clean=strike_is_clean,
        settlement_days=0
    )

