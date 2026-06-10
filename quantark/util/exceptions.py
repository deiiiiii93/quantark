"""
Custom exception classes for the QuantArk pricing library.

This module defines a hierarchy of exceptions to handle various error conditions
that may arise during derivative pricing and risk calculations.
"""


class QuantArkException(Exception):
    """
    Base exception class for all QuantArk library exceptions.
    
    All custom exceptions in the library inherit from this base class,
    allowing for easy catching of library-specific errors.
    """
    pass


class ValidationError(QuantArkException):
    """
    Exception raised when input parameters fail validation.
    
    Examples:
        - Negative spot price, strike, or volatility
        - Invalid time to maturity (negative or zero)
        - Invalid option type or exercise style
    """
    pass


class NumericalError(QuantArkException):
    """
    Exception raised when numerical computations encounter instability.
    
    Examples:
        - Overflow/underflow in exponential calculations
        - Division by zero or near-zero values
        - Non-convergence of iterative algorithms
        - Extreme parameter values causing numerical instability
    """
    pass


class MarketDataError(QuantArkException):
    """
    Exception raised when market data is missing or invalid.
    
    Examples:
        - Missing required market data (spot, vol, rate, etc.)
        - Inconsistent market data (e.g., invalid calendar dates)
        - Market data outside reasonable bounds
    """
    pass


class PricingError(QuantArkException):
    """
    Exception raised when pricing calculation fails.
    
    Examples:
        - Unsupported product/engine combination
        - Missing required pricing parameters
        - General pricing failures not covered by other exceptions
    """
    pass

