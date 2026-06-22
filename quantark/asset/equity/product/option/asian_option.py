"""
Asian option product definition.

Asian options are path-dependent options where the payoff depends on the average
price of the underlying asset over a specified observation period.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

from quantark.util.enum import (
    OptionType,
    ExerciseType,
    AveragingType,
    AsianStrikeType,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero, safe_log, safe_exp

from .base_equity_option import BaseEquityOption

if TYPE_CHECKING:
    from quantark.priceenv import PricingEnvironment

try:
    from quantark.util.calendar import calculate_year_fraction
except ImportError:
    calculate_year_fraction = None


@dataclass
class AsianObservationRecord:
    """
    Single observation entry for Asian option averaging.

    For past observations (before valuation_date), observed_price should be set.
    For future observations, observed_price should be None.

    Attributes:
        observation_time: Observation time as year fraction from initial date
        observation_date: Observation date (alternative to observation_time)
        observed_price: Actual observed price (for past observations only)
        weight: Relative averaging weight for this fixing. ``None`` means
            unweighted; weights are normalized across all observations so they
            sum to 1 (see ``AsianOption.resolve_observations``). Weights must be
            supplied for every record or for none.
    """

    observation_time: Optional[float] = None
    observation_date: Optional[datetime] = None
    observed_price: Optional[float] = None
    weight: Optional[float] = None

    def resolve_time(self, pricing_env: "PricingEnvironment") -> float:
        """
        Resolve observation time relative to valuation date.

        Args:
            pricing_env: Pricing environment with valuation date

        Returns:
            Year fraction from valuation date to observation date
            (negative if observation is in the past)

        Raises:
            ValidationError: If observation time/date cannot be resolved
        """
        if self.observation_time is not None:
            return self.observation_time
        if self.observation_date is not None:
            if pricing_env is None or calculate_year_fraction is None:
                raise ValidationError(
                    "PricingEnvironment is required to resolve observation_date."
                )

            # Handle past, current and future dates correctly
            if self.observation_date == pricing_env.valuation_date:
                return 0.0
            elif self.observation_date > pricing_env.valuation_date:
                # Future date: positive year fraction
                return calculate_year_fraction(
                    pricing_env.valuation_date,
                    self.observation_date,
                    pricing_env.day_count_convention,
                    pricing_env.bus_days_in_year,
                    calendar=getattr(pricing_env, "calendar", None),
                )
            else:
                # Past date: negative year fraction
                # We reverse dates for calculate_year_fraction and negate result
                return -calculate_year_fraction(
                    self.observation_date,
                    pricing_env.valuation_date,
                    pricing_env.day_count_convention,
                    pricing_env.bus_days_in_year,
                    calendar=getattr(pricing_env, "calendar", None),
                )
        raise ValidationError(
            "AsianObservationRecord requires observation_time or observation_date."
        )

    def is_observed(self) -> bool:
        """Check if this observation has an observed price."""
        return self.observed_price is not None

    def validate(self) -> None:
        """Validate the observation record."""
        if self.observation_time is None and self.observation_date is None:
            raise ValidationError(
                "AsianObservationRecord must provide observation_time or observation_date."
            )
        if self.observed_price is not None and self.observed_price <= 0:
            raise ValidationError(
                f"observed_price must be positive, got {self.observed_price}"
            )
        if self.weight is not None and self.weight <= 0:
            raise ValidationError(f"weight must be positive, got {self.weight}")


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
        observation_times: List of observation times as year fractions (legacy)
        observation_dates: List of observation dates (legacy, alternative to times)
        observation_records: List of AsianObservationRecord (preferred, supports historical prices)
        num_observations: Number of observations for uniform schedule (default: 12).
                          Set to None for continuous averaging.
        maturity: Time to maturity in years
    """

    # Asian-specific parameters
    asian_strike_type: AsianStrikeType = AsianStrikeType.FIXED
    averaging_type: AveragingType = AveragingType.ARITHMETIC
    observation_times: Optional[List[float]] = None
    observation_dates: Optional[List[datetime]] = None
    observation_records: Optional[List[AsianObservationRecord]] = None
    num_observations: Optional[int] = 12

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
        observation_records: Optional[List[AsianObservationRecord]] = None,
        num_observations: Optional[int] = 12,
        initial_price: float = 0.0,
        contract_multiplier: float = 1.0,
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
            observation_times: List of observation times as year fractions (legacy)
            observation_dates: List of observation dates (legacy, alternative to times)
            observation_records: List of AsianObservationRecord with optional observed prices
            num_observations: Number of observations for uniform schedule (default: 12)
            initial_price: Reference/initial underlying price
            contract_multiplier: Underlying units represented by one contract
        """
        self.asian_strike_type = asian_strike_type
        self.averaging_type = averaging_type
        self.observation_times = observation_times
        self.observation_dates = observation_dates
        self.observation_records = observation_records
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
            contract_multiplier=contract_multiplier,
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
        self._validate_contract_multiplier()
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

        # Validate observation records if provided
        if self.observation_records is not None:
            if len(self.observation_records) == 0:
                raise ValidationError("observation_records cannot be empty")
            for rec in self.observation_records:
                rec.validate()

        # Validate num_observations
        if self.num_observations is not None and self.num_observations < 1:
            raise ValidationError(f"num_observations must be >= 1, got {self.num_observations}")

    def get_observation_times(self, maturity: Optional[float] = None) -> List[float]:
        """
        Get observation times as year fractions (legacy method).

        If observation_times is specified, returns it directly.
        Otherwise, generates uniform observations from 0 to maturity.

        Note: For options with historical observations, use resolve_observations() instead.

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

        # If num_observations is None, signify continuous schedule
        if self.num_observations is None:
            return []

        # Generate n equally-spaced observations from 0 to T (inclusive of T)
        return list(np.linspace(0, T, self.num_observations + 1)[1:])

    def resolve_observations(
        self,
        pricing_env: "PricingEnvironment",
    ) -> Tuple[List[float], List[float], List[float], List[float], int]:
        """
        Resolve observation records into past and future observations.

        This method separates already-observed prices from future observation times,
        accounting for the valuation date in pricing_env.

        For observations with observation_time (year fractions from initial date):
        - Times <= 0 are considered past (already observed)
        - Times > 0 are considered future (to be simulated)

        Averaging weights are normalized across ALL observations (past + future)
        so that they sum to 1. Unweighted schedules yield uniform 1/n weights.

        Args:
            pricing_env: Pricing environment with valuation date

        Returns:
            Tuple of:
            - past_prices: List of already-observed prices
            - past_weights: Normalized weights aligned with past_prices
            - future_times: List of future observation times (relative to valuation date)
            - future_weights: Normalized weights aligned with future_times
            - total_observations: Total number of observations for averaging
        """
        if self.observation_records is not None:
            return self._resolve_from_records(pricing_env)
        else:
            return self._resolve_from_legacy(pricing_env)

    @staticmethod
    def _normalize_weights(raw_weights: List[Optional[float]]) -> List[float]:
        """Normalize a list of raw weights to sum to 1.

        Weights must be supplied for every observation or for none. ``None`` for
        all entries yields uniform 1/n weights; any mix raises ValidationError.
        """
        n = len(raw_weights)
        if n == 0:
            return []
        specified = [w is not None for w in raw_weights]
        if any(specified) and not all(specified):
            raise ValidationError(
                "Asian averaging weights must be supplied for every observation "
                "or for none."
            )
        if not any(specified):
            return [1.0 / n] * n
        total = float(sum(w for w in raw_weights))  # type: ignore[arg-type]
        if total <= 0:
            raise ValidationError("Sum of Asian averaging weights must be positive.")
        return [float(w) / total for w in raw_weights]  # type: ignore[arg-type]

    def _resolve_from_records(
        self,
        pricing_env: "PricingEnvironment",
    ) -> Tuple[List[float], List[float], List[float], List[float], int]:
        """Resolve observations from observation_records."""
        past_prices: List[float] = []
        future_times: List[float] = []
        # Track each observation's slot so normalized weights stay aligned.
        is_past: List[bool] = []
        raw_weights: List[Optional[float]] = []

        for rec in self.observation_records:
            t = rec.resolve_time(pricing_env)

            if t <= 0:
                # Past observation - must have observed price
                if rec.observed_price is None:
                    raise ValidationError(
                        f"Past observation (t={t}) must have observed_price set"
                    )
                past_prices.append(rec.observed_price)
                is_past.append(True)
            else:
                # Future observation
                if rec.observed_price is not None:
                    raise ValidationError(
                        f"Future observation (t={t}) should not have observed_price set"
                    )
                future_times.append(t)
                is_past.append(False)
            raw_weights.append(rec.weight)

        norm = self._normalize_weights(raw_weights)
        past_weights = [w for w, past in zip(norm, is_past) if past]
        future_weights = [w for w, past in zip(norm, is_past) if not past]

        total_observations = len(past_prices) + len(future_times)
        return past_prices, past_weights, future_times, future_weights, total_observations

    def _resolve_from_legacy(
        self,
        pricing_env: "PricingEnvironment",
    ) -> Tuple[List[float], List[float], List[float], List[float], int]:
        """Resolve observations from legacy observation_times/observation_dates."""
        T = self.get_maturity(pricing_env)
        obs_times = self.get_observation_times(T)

        # All observations are future (no historical prices in legacy mode)
        past_prices: List[float] = []
        past_weights: List[float] = []

        if self.num_observations is None:
            # Continuous mode
            future_times: List[float] = []
            future_weights: List[float] = []
            total_observations = 0  # Signifies continuous
        else:
            # Legacy mode has no historical prices, so times at valuation (t=0)
            # are treated as known-at-start observations (deterministic under spot).
            future_times = [t for t in obs_times if t >= 0]
            total_observations = len(obs_times)
            future_weights = self._normalize_weights([None] * len(future_times))

        return past_prices, past_weights, future_times, future_weights, total_observations

    def has_past_observations(self, pricing_env: "PricingEnvironment") -> bool:
        """
        Check if there are any past observations with recorded prices.

        Args:
            pricing_env: Pricing environment with valuation date

        Returns:
            True if there are past observations, False otherwise
        """
        past_prices, _, _, _, _ = self.resolve_observations(pricing_env)
        return len(past_prices) > 0

    def get_past_average(self, pricing_env: "PricingEnvironment") -> Optional[float]:
        """
        Get the average of past observations if any exist.

        Args:
            pricing_env: Pricing environment with valuation date

        Returns:
            Average of past observations, or None if no past observations
        """
        past_prices, past_weights, _, _, _ = self.resolve_observations(pricing_env)
        if len(past_prices) == 0:
            return None
        return self.get_average(past_prices, weights=past_weights)

    def get_average(
        self,
        prices: Union[List[float], np.ndarray],
        weights: Optional[Union[List[float], np.ndarray]] = None,
    ) -> float:
        """
        Compute the (optionally weighted) average of observed prices.

        Args:
            prices: List or array of observed prices
            weights: Optional per-price weights. ``None`` gives an equal-weight
                average. Weights are renormalized to sum to 1, so any positive
                scaling (e.g. a sub-list of globally-normalized weights) works.

        Returns:
            Average price (arithmetic or geometric based on averaging_type)

        Raises:
            ValidationError: If prices is empty or contains invalid values
        """
        if len(prices) == 0:
            raise ValidationError("prices cannot be empty")

        prices_arr = np.asarray(prices, dtype=float)

        if weights is None:
            weights_arr = np.full(len(prices_arr), 1.0 / len(prices_arr))
        else:
            if len(weights) != len(prices_arr):
                raise ValidationError(
                    "weights length must match prices length for averaging"
                )
            weights_arr = np.asarray(weights, dtype=float)
            total = float(weights_arr.sum())
            if total <= 0:
                raise ValidationError("Sum of averaging weights must be positive")
            weights_arr = weights_arr / total

        if self.averaging_type == AveragingType.ARITHMETIC:
            return float(np.dot(weights_arr, prices_arr))
        else:  # GEOMETRIC
            if np.any(prices_arr <= 0):
                raise ValidationError("All prices must be positive for geometric averaging")
            # Weighted log-mean for numerical stability
            log_prices = np.log(prices_arr)
            return float(safe_exp(np.dot(weights_arr, log_prices)))

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
                payoff = max(avg - self.strike, 0.0)
            else:
                payoff = max(self.strike - avg, 0.0)
        else:
            # Floating strike (average strike option)
            if self.is_call():
                payoff = max(spot - avg, 0.0)
            else:
                payoff = max(avg - spot, 0.0)

        return payoff * self.contract_multiplier

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

    def has_custom_weights(self) -> bool:
        """True if any observation carries an explicit (non-uniform) weight."""
        if self.observation_records is None:
            return False
        return any(rec.weight is not None for rec in self.observation_records)

    def is_continuous(self) -> bool:
        """Check if this uses continuous averaging."""
        return self.num_observations is None and self.observation_times is None and self.observation_dates is None and self.observation_records is None

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
