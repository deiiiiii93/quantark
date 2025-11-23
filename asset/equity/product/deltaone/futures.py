"""
Futures contract implementation with basis handling.
"""

import math
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from .base_deltaone_product import BaseDeltaOneProduct
from util.enum.deltaone_enums import DeltaOneType
from util.exceptions import ValidationError


@dataclass
class Futures(BaseDeltaOneProduct):
    """
    Futures contract with basis handling and mark-to-market support.
    
    A futures contract is an agreement to buy/sell an asset at a future date
    at a predetermined price. The contract has a multiplier and optional
    observed market price for mark-to-market valuation.
    
    Theoretical forward pricing with basis:
        F(t,T) = S(t) * exp((r - q) * (T - t)) + basis(t) * exp(-λ * (T - t))
    
    Where:
        - S(t): Current spot price
        - r: Risk-free rate
        - q: Dividend yield
        - basis(t): Current basis (difference from theoretical forward)
        - λ: Basis decay rate (how fast basis converges to 0 at maturity)
    
    Attributes:
        underlying: Identifier for the underlying asset
        multiplier: Contract multiplier (e.g., 50 for E-mini S&P)
        maturity: Time to maturity in years (optional if maturity_date provided)
        maturity_date: Date when futures contract expires
        basis: Current basis (futures_price - theoretical_forward)
        basis_decay_rate: Rate at which basis converges to zero (default: 1.0)
        market_price: Optional observed futures price for mark-to-market valuation
    
    Example:
        >>> # E-mini S&P 500 futures
        >>> future = Futures(
        ...     underlying="ES",
        ...     multiplier=50.0,
        ...     maturity=0.25,  # 3 months
        ...     basis=2.5,
        ...     basis_decay_rate=2.0
        ... )
        >>> 
        >>> # With market price for mark-to-market
        >>> future_mtm = Futures(
        ...     underlying="ES",
        ...     multiplier=50.0,
        ...     maturity=0.25,
        ...     basis=2.5,
        ...     market_price=4525.0  # Observed market price
        ... )
    """
    
    multiplier: float = 1.0
    basis: float = 0.0
    basis_decay_rate: float = 1.0
    market_price: Optional[float] = None
    
    def __init__(
        self,
        underlying: str,
        multiplier: float = 1.0,
        maturity: Optional[float] = None,
        maturity_date: Optional[datetime] = None,
        basis: float = 0.0,
        basis_decay_rate: float = 1.0,
        market_price: Optional[float] = None,
    ):
        """
        Initialize futures contract.
        
        Args:
            underlying: Identifier for the underlying asset
            multiplier: Contract multiplier (must be positive)
            maturity: Time to maturity in years (optional if maturity_date provided)
            maturity_date: Date when contract expires (optional if maturity provided)
            basis: Current basis (futures_price - theoretical_forward)
            basis_decay_rate: Rate at which basis converges to zero (default: 1.0)
            market_price: Optional observed futures price for mark-to-market
        
        Raises:
            ValidationError: If parameters are invalid
        
        Note:
            Either maturity OR maturity_date must be provided (not both).
        """
        # Initialize parent with futures type
        super().__init__(
            underlying=underlying,
            deltaone_type=DeltaOneType.FUTURES,
            maturity=maturity,
            maturity_date=maturity_date,
        )
        
        self.multiplier = multiplier
        self.basis = basis
        self.basis_decay_rate = basis_decay_rate
        self.market_price = market_price
        
        # Validate futures-specific parameters
        self._validate_futures_params()
    
    def _validate_futures_params(self) -> None:
        """
        Validate futures-specific parameters.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        if self.multiplier <= 0:
            raise ValidationError(f"Multiplier must be positive, got {self.multiplier}")
        
        if not math.isfinite(self.basis):
            raise ValidationError(f"Basis must be finite, got {self.basis}")
        
        if self.basis_decay_rate <= 0:
            raise ValidationError(f"Basis decay rate must be positive, got {self.basis_decay_rate}")
        
        if self.market_price is not None and self.market_price <= 0:
            raise ValidationError(f"Market price must be positive if provided, got {self.market_price}")
        
        # Futures must have a maturity
        if self.maturity is None and self.maturity_date is None:
            raise ValidationError("Futures contract must have a maturity or maturity_date")
    
    def get_forward_price(
        self,
        spot: float,
        rate: float,
        div_yield: float,
        time_to_maturity: float
    ) -> float:
        """
        Calculate the theoretical forward price with basis.
        
        Forward price with basis:
            F(t,T) = S(t) * exp((r - q) * (T - t)) + basis(t) * exp(-λ * (T - t))
        
        The basis decays exponentially to zero as the contract approaches maturity,
        ensuring convergence to the spot price at expiration.
        
        Args:
            spot: Current spot price
            rate: Risk-free rate (continuously compounded)
            div_yield: Dividend yield (continuously compounded)
            time_to_maturity: Time to maturity in years
        
        Returns:
            Theoretical forward price with basis
        
        Raises:
            ValidationError: If inputs are invalid
        """
        if spot <= 0:
            raise ValidationError(f"Spot price must be positive, got {spot}")
        if time_to_maturity < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {time_to_maturity}")
        
        # Theoretical forward price (cost-of-carry)
        carry_cost = (rate - div_yield) * time_to_maturity
        theoretical_forward = spot * math.exp(carry_cost)
        
        # Add decaying basis
        basis_decay = math.exp(-self.basis_decay_rate * time_to_maturity)
        forward_price = theoretical_forward + self.basis * basis_decay
        
        return forward_price
    
    def get_theoretical_price(
        self,
        spot: float,
        rate: float,
        div_yield: float,
        time_to_maturity: float
    ) -> float:
        """
        Get theoretical futures contract value (per contract, not notional).
        
        For futures contracts, the value is the forward price times multiplier.
        
        Args:
            spot: Current spot price
            rate: Risk-free rate
            div_yield: Dividend yield
            time_to_maturity: Time to maturity in years
        
        Returns:
            Theoretical contract value
        """
        forward_price = self.get_forward_price(spot, rate, div_yield, time_to_maturity)
        return forward_price * self.multiplier
    
    def get_mark_to_market_price(self) -> Optional[float]:
        """
        Get mark-to-market price based on observed market price.
        
        If market_price is set, returns the contract value based on observed
        market price. Otherwise returns None.
        
        Returns:
            Mark-to-market contract value, or None if no market price available
        """
        if self.market_price is None:
            return None
        return self.market_price * self.multiplier
    
    def get_notional_value(self, price: float) -> float:
        """
        Calculate notional value of the futures contract.
        
        Notional value = price * multiplier
        
        Args:
            price: Futures price (per unit)
        
        Returns:
            Notional value
        """
        return price * self.multiplier
    
    def update_market_price(self, new_market_price: float) -> None:
        """
        Update the observed market price for mark-to-market.
        
        Args:
            new_market_price: New observed market price
        
        Raises:
            ValidationError: If price is invalid
        """
        if new_market_price <= 0:
            raise ValidationError(f"Market price must be positive, got {new_market_price}")
        self.market_price = new_market_price
    
    def get_basis(
        self,
        spot: float,
        rate: float,
        div_yield: float,
        time_to_maturity: float,
        observed_futures_price: Optional[float] = None
    ) -> float:
        """
        Calculate or update the basis.
        
        If observed_futures_price is provided, calculates the basis as:
            basis = observed_futures_price - theoretical_forward
        
        Otherwise returns the current basis attribute.
        
        Args:
            spot: Current spot price
            rate: Risk-free rate
            div_yield: Dividend yield
            time_to_maturity: Time to maturity in years
            observed_futures_price: Optional observed market price
        
        Returns:
            Basis value
        """
        if observed_futures_price is not None:
            # Calculate theoretical forward without basis
            carry_cost = (rate - div_yield) * time_to_maturity
            theoretical_forward = spot * math.exp(carry_cost)
            
            # Basis is the difference
            return observed_futures_price - theoretical_forward
        else:
            return self.basis
    
    def __repr__(self):
        mtm_str = f", mtm=${self.market_price:.2f}" if self.market_price else ""
        if self.maturity_date:
            return (f"Futures({self.underlying}, mult={self.multiplier:.1f}, "
                   f"maturity_date={self.maturity_date.date()}, basis={self.basis:.4f}{mtm_str})")
        else:
            return (f"Futures({self.underlying}, mult={self.multiplier:.1f}, "
                   f"T={self.maturity:.4f}, basis={self.basis:.4f}{mtm_str})")

