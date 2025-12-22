"""
Asian option product definition.

Asian options are path-dependent options where the payoff depends on the average
price of the underlying asset over a specified observation period.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union
import numpy as np

from util.enum import (
    OptionType,
    ExerciseType,
    AveragingType,
    AsianStrikeType,
)
from util.exceptions import ValidationError
from util.numerical import is_zero, safe_log, safe_exp

from .base_equity_option import BaseEquityOption


@dataclass
class AsianOption(BaseEquityOption):
    """
    Asian option with arithmetic or geometric averaging.

    Asian options are path-dependent options where the payoff depends on the
    average price of the underlying asset over a specified observation period.

    Two main variants:
    - Fixed strike (average price option): payoff based on average vs strike
    - Floating strike (average strike option): payoff based on final spot vs average

    Attributes:
        strike: Strike price (required for fixed strike, optional for floating)
        option_type: CALL or PUT
        asian_strike_type: FIXED or FLOATING
        averaging_type: ARITHMETIC or GEOMETRIC
        observation_times: List of observation times as year fractions
        observation_dates: List of observation dates (alternative to times)
        num_observations: Number of observations for uniform schedule (default: 12)
        maturity: Time to maturity in years
    """

    # Asian-specific parameters
    asian_strike_type: AsianStrikeType = AsianStrikeType.FIXED
    averaging_type: AveragingType = AveragingType.ARITHMETIC
    observation_times: Optional[List[float]] = None
    observation_dates: Optional[List[datetime]] = None
    num_observations: int = 12

    def __init__(
        self,
        strike: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        asian_strike_type: AsianStrikeType = AsianStrikeType.FIXED,
        averaging_type: AveragingType = AveragingType.ARITHMETIC,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        observation_times: Optional[List[float]] = None,
        observation_dates: Optional[List[datetime]] = None,
        num_observations: int = 12,
        initial_price: float = 0.0,
        notional: Optional[float] = None,
        quantity: float = 1.0,
    ):
        """
        Initialize Asian option.

        Args:
            strike: Strike price (required for fixed strike options)
            option_type: CALL or PUT
            asian_strike_type: FIXED (average price) or FLOATING (average strike)
            averaging_type: ARITHMETIC or GEOMETRIC
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Date when option expires (optional if maturity provided)
            observation_times: List of observation times as year fractions
            observation_dates: List of observation dates (alternative to times)
            num_observations: Number of observations for uniform schedule (default: 12)
            initial_price: Reference/initial underlying price
            notional: Notional principal amount (optional)
            quantity: Number of option contracts (default: 1.0)
        """
        self.asian_strike_type = asian_strike_type
        self.averaging_type = averaging_type
        self.observation_times = observation_times
        self.observation_dates = observation_dates
        self.num_observations = num_observations

        # For floating strike, strike can be zero (strike is the average)
        if asian_strike_type == AsianStrikeType.FLOATING and strike == 0.0:
            strike = 1.0  # Placeholder, won't be used in payoff

        super().__init__(
            strike=strike,
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,  # Asian options are European-style
            maturity=maturity,
            exercise_date=exercise_date,
            initial_price=initial_price,
            notional=notional,
            quantity=quantity,
        )

    def validate(self) -> None:
        """
        Validate Asian option parameters.

        Raises:
            ValidationError: If any parameter is invalid
        """
        # Skip strike validation for floating strike options
        if self.asian_strike_type == AsianStrikeType.FIXED:
            self._validate_strike()

        self._validate_maturity_dates()
        self._validate_date_ordering()
        self._validate_types()
        self._validate_tenor_end()
        self._validate_notional_quantity()
        self._validate_asian_parameters()

    def _validate_asian_parameters(self) -> None:
        """Validate Asian-specific parameters."""
        # Validate asian_strike_type
        if not isinstance(self.asian_strike_type, AsianStrikeType):
            raise ValidationError(f"Invalid asian_strike_type: {self.asian_strike_type}")

        # Validate averaging_type
        if not isinstance(self.averaging_type, AveragingType):
            raise ValidationError(f"Invalid averaging_type: {self.averaging_type}")

        # Validate observation times if provided
        if self.observation_times is not None:
            if len(self.observation_times) == 0:
                raise ValidationError("observation_times cannot be empty")
            if any(t < 0 for t in self.observation_times):
                raise ValidationError("observation_times must be non-negative")
            if self.observation_times != sorted(self.observation_times):
                raise ValidationError("observation_times must be sorted in ascending order")

        # Validate observation dates if provided
        if self.observation_dates is not None:
            if len(self.observation_dates) == 0:
                raise ValidationError("observation_dates cannot be empty")
            if self.observation_dates != sorted(self.observation_dates):
                raise ValidationError("observation_dates must be sorted in ascending order")

        # Validate num_observations
        if self.num_observations < 1:
            raise ValidationError(f"num_observations must be >= 1, got {self.num_observations}")

    def get_observation_times(self, maturity: Optional[float] = None) -> List[float]:
        """
        Get observation times as year fractions.

        If observation_times is specified, returns it directly.
        Otherwise, generates uniform observations from 0 to maturity.

        Args:
            maturity: Time to maturity (used if generating uniform schedule)

        Returns:
            List of observation times as year fractions
        """
        if self.observation_times is not None:
            return self.observation_times

        # Generate uniform schedule
        T = maturity if maturity is not None else self.maturity
        if T is None or T <= 0:
            raise ValidationError("Cannot generate observation schedule: maturity not set")

        # Generate n equally-spaced observations from 0 to T (inclusive of T)
        return list(np.linspace(0, T, self.num_observations + 1)[1:])

    def get_average(
        self,
        prices: Union[List[float], np.ndarray],
    ) -> float:
        """
        Compute the average of observed prices.

        Args:
            prices: List or array of observed prices

        Returns:
            Average price (arithmetic or geometric based on averaging_type)

        Raises:
            ValidationError: If prices is empty or contains invalid values
        """
        if len(prices) == 0:
            raise ValidationError("prices cannot be empty")

        prices_arr = np.asarray(prices)

        if self.averaging_type == AveragingType.ARITHMETIC:
            return float(np.mean(prices_arr))
        else:  # GEOMETRIC
            if np.any(prices_arr <= 0):
                raise ValidationError("All prices must be positive for geometric averaging")
            # Use log-sum-exp for numerical stability
            log_prices = np.log(prices_arr)
            return float(safe_exp(np.mean(log_prices)))

    def get_payoff(
        self,
        spot: float,
        observed_prices: Optional[Union[List[float], np.ndarray]] = None,
        average: Optional[float] = None,
    ) -> float:
        """
        Calculate the option payoff at maturity.

        For fixed strike (average price option):
            Call: max(average - K, 0)
            Put:  max(K - average, 0)

        For floating strike (average strike option):
            Call: max(spot - average, 0)
            Put:  max(average - spot, 0)

        Args:
            spot: Spot price at maturity
            observed_prices: List of observed prices (used to compute average)
            average: Pre-computed average (alternative to observed_prices)

        Returns:
            Option payoff

        Raises:
            ValidationError: If neither observed_prices nor average is provided
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        # Get average price
        if average is not None:
            avg = average
        elif observed_prices is not None:
            avg = self.get_average(observed_prices)
        else:
            raise ValidationError("Either observed_prices or average must be provided")

        # Calculate payoff based on strike type
        if self.asian_strike_type == AsianStrikeType.FIXED:
            # Fixed strike (average price option)
            if self.is_call():
                return max(avg - self.strike, 0.0)
            else:
                return max(self.strike - avg, 0.0)
        else:
            # Floating strike (average strike option)
            if self.is_call():
                return max(spot - avg, 0.0)
            else:
                return max(avg - spot, 0.0)

    def intrinsic_value(
        self,
        spot: float,
        observed_prices: Optional[Union[List[float], np.ndarray]] = None,
        average: Optional[float] = None,
    ) -> float:
        """
        Calculate the intrinsic value of the option.

        For Asian options, intrinsic value equals payoff since they're European-style.

        Args:
            spot: Current spot price
            observed_prices: List of observed prices
            average: Pre-computed average

        Returns:
            Intrinsic value
        """
        return self.get_payoff(spot, observed_prices, average)

    def is_fixed_strike(self) -> bool:
        """Check if this is a fixed strike (average price) option."""
        return self.asian_strike_type == AsianStrikeType.FIXED

    def is_floating_strike(self) -> bool:
        """Check if this is a floating strike (average strike) option."""
        return self.asian_strike_type == AsianStrikeType.FLOATING

    def is_arithmetic(self) -> bool:
        """Check if this uses arithmetic averaging."""
        return self.averaging_type == AveragingType.ARITHMETIC

    def is_geometric(self) -> bool:
        """Check if this uses geometric averaging."""
        return self.averaging_type == AveragingType.GEOMETRIC

    def __repr__(self) -> str:
        """Return string representation of the Asian option."""
        strike_str = f"K={self.strike:.2f}" if self.is_fixed_strike() else "floating"
        maturity_str = (
            f"T={self.maturity:.4f}"
            if self.maturity is not None
            else f"exp={self.exercise_date}"
        )
        return (
            f"AsianOption({self.option_type}, {strike_str}, {maturity_str}, "
            f"{self.asian_strike_type}, {self.averaging_type})"
        )
