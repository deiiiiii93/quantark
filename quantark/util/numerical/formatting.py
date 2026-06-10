"""
Number formatting utilities for financial display.

This module provides standardized formatters for displaying financial
quantities such as prices, rates, percentages, and Greeks.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto


class SignDisplay(Enum):
    """Sign display options for formatted numbers."""
    AUTO = auto()       # Show sign only for negative
    ALWAYS = auto()     # Always show + or -
    NEVER = auto()      # Never show sign (absolute value)
    SPACE = auto()      # Space for positive, - for negative


@dataclass
class NumberFormatter:
    """
    Configurable number formatter for financial quantities.
    
    Provides consistent formatting across the library with customizable
    precision, sign display, and thousands separators.
    
    Attributes:
        decimals: Number of decimal places.
        sign_display: How to display the sign.
        thousands_separator: Whether to use thousands separator.
        prefix: String prefix (e.g., "$" for currency).
        suffix: String suffix (e.g., "%" for percentage).
        nan_display: String to display for NaN values.
        inf_display: String to display for infinite values.
    
    Examples:
        >>> fmt = NumberFormatter(decimals=2, prefix="$", thousands_separator=True)
        >>> fmt.format(1234567.89)
        '$1,234,567.89'
        
        >>> fmt = NumberFormatter(decimals=2, suffix="%")
        >>> fmt.format(5.5)
        '5.50%'
    """
    
    decimals: int = 2
    sign_display: SignDisplay = SignDisplay.AUTO
    thousands_separator: bool = False
    prefix: str = ""
    suffix: str = ""
    nan_display: str = "N/A"
    inf_display: str = "∞"
    
    def format(self, value: float) -> str:
        """Format a number according to the formatter settings."""
        import math
        
        # Handle special values
        if math.isnan(value):
            return self.nan_display
        if math.isinf(value):
            sign = "-" if value < 0 else ""
            return f"{sign}{self.inf_display}"
        
        # Determine sign string
        if self.sign_display == SignDisplay.ALWAYS:
            sign_str = "+" if value >= 0 else "-"
            abs_value = abs(value)
        elif self.sign_display == SignDisplay.NEVER:
            sign_str = ""
            abs_value = abs(value)
        elif self.sign_display == SignDisplay.SPACE:
            sign_str = " " if value >= 0 else "-"
            abs_value = abs(value)
        else:  # AUTO
            sign_str = ""
            abs_value = value
        
        # Format the number
        if self.thousands_separator:
            formatted = f"{abs_value:,.{self.decimals}f}"
        else:
            formatted = f"{abs_value:.{self.decimals}f}"
        
        return f"{self.prefix}{sign_str}{formatted}{self.suffix}"


def format_currency(
    value: float,
    decimals: int = 2,
    currency: str = "$",
    thousands_separator: bool = True,
    show_sign: bool = False,
) -> str:
    """
    Format a value as currency.
    
    Args:
        value: Numeric value to format.
        decimals: Number of decimal places (default: 2).
        currency: Currency symbol (default: "$").
        thousands_separator: Use thousands separator (default: True).
        show_sign: Always show +/- sign (default: False).
    
    Returns:
        Formatted currency string.
    
    Examples:
        >>> format_currency(1234567.89)
        '$1,234,567.89'
        >>> format_currency(-1000.5, show_sign=True)
        '$-1,000.50'
        >>> format_currency(1000.5, currency="€")
        '€1,000.50'
    """
    formatter = NumberFormatter(
        decimals=decimals,
        prefix=currency,
        thousands_separator=thousands_separator,
        sign_display=SignDisplay.ALWAYS if show_sign else SignDisplay.AUTO,
    )
    return formatter.format(value)


def format_percentage(
    value: float,
    decimals: int = 2,
    multiply_by_100: bool = True,
    show_sign: bool = False,
) -> str:
    """
    Format a value as percentage.
    
    Args:
        value: Numeric value to format.
        decimals: Number of decimal places (default: 2).
        multiply_by_100: If True, multiply by 100 (e.g., 0.05 -> 5%).
        show_sign: Always show +/- sign (default: False).
    
    Returns:
        Formatted percentage string.
    
    Examples:
        >>> format_percentage(0.0525)
        '5.25%'
        >>> format_percentage(5.25, multiply_by_100=False)
        '5.25%'
        >>> format_percentage(0.03, show_sign=True)
        '+3.00%'
    """
    display_value = value * 100 if multiply_by_100 else value
    formatter = NumberFormatter(
        decimals=decimals,
        suffix="%",
        sign_display=SignDisplay.ALWAYS if show_sign else SignDisplay.AUTO,
    )
    return formatter.format(display_value)


def format_basis_points(
    value: float,
    decimals: int = 1,
    from_decimal: bool = True,
    show_sign: bool = False,
) -> str:
    """
    Format a value as basis points.
    
    Args:
        value: Numeric value to format.
        decimals: Number of decimal places (default: 1).
        from_decimal: If True, convert from decimal (e.g., 0.0025 -> 25bp).
        show_sign: Always show +/- sign (default: False).
    
    Returns:
        Formatted basis points string.
    
    Examples:
        >>> format_basis_points(0.0025)
        '25.0bp'
        >>> format_basis_points(25, from_decimal=False)
        '25.0bp'
        >>> format_basis_points(0.001, show_sign=True)
        '+10.0bp'
    """
    display_value = value * 10000 if from_decimal else value
    formatter = NumberFormatter(
        decimals=decimals,
        suffix="bp",
        sign_display=SignDisplay.ALWAYS if show_sign else SignDisplay.AUTO,
    )
    return formatter.format(display_value)


def format_price(
    value: float,
    decimals: int = 4,
    show_sign: bool = False,
) -> str:
    """
    Format a value as a price (high precision).
    
    Args:
        value: Numeric value to format.
        decimals: Number of decimal places (default: 4).
        show_sign: Always show +/- sign (default: False).
    
    Returns:
        Formatted price string.
    
    Examples:
        >>> format_price(123.456789)
        '123.4568'
        >>> format_price(-5.5, show_sign=True)
        '-5.5000'
    """
    formatter = NumberFormatter(
        decimals=decimals,
        sign_display=SignDisplay.ALWAYS if show_sign else SignDisplay.AUTO,
    )
    return formatter.format(value)


def format_greeks(
    value: float,
    decimals: int = 6,
    show_sign: bool = True,
) -> str:
    """
    Format a Greek value (delta, gamma, vega, etc.).
    
    Args:
        value: Greek value to format.
        decimals: Number of decimal places (default: 6).
        show_sign: Always show +/- sign (default: True).
    
    Returns:
        Formatted Greek string.
    
    Examples:
        >>> format_greeks(0.543210)
        '+0.543210'
        >>> format_greeks(-0.00123)
        '-0.001230'
    """
    formatter = NumberFormatter(
        decimals=decimals,
        sign_display=SignDisplay.ALWAYS if show_sign else SignDisplay.AUTO,
    )
    return formatter.format(value)


def format_rate(
    value: float,
    decimals: int = 4,
    as_percentage: bool = True,
    show_sign: bool = False,
) -> str:
    """
    Format an interest rate.
    
    Args:
        value: Rate value (as decimal, e.g., 0.05 for 5%).
        decimals: Number of decimal places (default: 4).
        as_percentage: Display as percentage (default: True).
        show_sign: Always show +/- sign (default: False).
    
    Returns:
        Formatted rate string.
    
    Examples:
        >>> format_rate(0.05)
        '5.0000%'
        >>> format_rate(0.05, as_percentage=False)
        '0.0500'
    """
    if as_percentage:
        return format_percentage(value, decimals=decimals, show_sign=show_sign)
    
    formatter = NumberFormatter(
        decimals=decimals,
        sign_display=SignDisplay.ALWAYS if show_sign else SignDisplay.AUTO,
    )
    return formatter.format(value)


def format_volatility(
    value: float,
    decimals: int = 2,
    as_percentage: bool = True,
    show_sign: bool = False,
) -> str:
    """
    Format a volatility value.
    
    Args:
        value: Volatility value (as decimal, e.g., 0.20 for 20%).
        decimals: Number of decimal places (default: 2).
        as_percentage: Display as percentage (default: True).
        show_sign: Always show +/- sign (default: False).
    
    Returns:
        Formatted volatility string.
    
    Examples:
        >>> format_volatility(0.25)
        '25.00%'
        >>> format_volatility(0.25, as_percentage=False, decimals=4)
        '0.2500'
    """
    if as_percentage:
        return format_percentage(value, decimals=decimals, show_sign=show_sign)
    
    formatter = NumberFormatter(
        decimals=decimals,
        sign_display=SignDisplay.ALWAYS if show_sign else SignDisplay.AUTO,
    )
    return formatter.format(value)


def format_with_sign(
    value: float,
    decimals: int = 2,
) -> str:
    """
    Format a number always showing the sign.
    
    Args:
        value: Numeric value to format.
        decimals: Number of decimal places (default: 2).
    
    Returns:
        Formatted string with sign.
    
    Examples:
        >>> format_with_sign(100.5)
        '+100.50'
        >>> format_with_sign(-50.25)
        '-50.25'
    """
    formatter = NumberFormatter(
        decimals=decimals,
        sign_display=SignDisplay.ALWAYS,
    )
    return formatter.format(value)


def format_scientific(
    value: float,
    decimals: int = 4,
    show_sign: bool = False,
) -> str:
    """
    Format a number in scientific notation.
    
    Args:
        value: Numeric value to format.
        decimals: Number of significant decimals (default: 4).
        show_sign: Always show +/- sign (default: False).
    
    Returns:
        Formatted string in scientific notation.
    
    Examples:
        >>> format_scientific(0.00001234)
        '1.2340e-05'
        >>> format_scientific(1234567.89)
        '1.2346e+06'
    """
    import math
    
    if math.isnan(value):
        return "N/A"
    if math.isinf(value):
        return f"{'-' if value < 0 else ''}∞"
    
    sign = "+" if show_sign and value >= 0 else ""
    return f"{sign}{value:.{decimals}e}"
