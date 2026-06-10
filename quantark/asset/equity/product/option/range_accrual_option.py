"""
Range Accrual option product definition.

Range Accrual options pay based on the proportion of observations where the
underlying stays within a defined price range. Supports weighted observations
and historical tracking for partially observed products.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, TYPE_CHECKING

from quantark.util.enum import ExerciseType, OptionType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

from .base_equity_option import BaseEquityOption
from .range_accrual_config import RangeAccrualConfig, RangeAccrualObservationRecord

if TYPE_CHECKING:
    from quantark.priceenv import PricingEnvironment



@dataclass
class RangeAccrualOption(BaseEquityOption):
    """
    Range Accrual option with weighted observations.

    Range Accrual options pay based on the proportion of observations where the
    underlying stays within a defined price range. The payoff formula is:

        Payoff = initial_price * contract_multiplier * accrual_rate
                 * (sum_in_range_weights / sum_total_weights) * year_fraction

    Key features:
    - Historical observations: Track past observations with observed_in_range flag
    - Weighted observations: Support calendar day weights (e.g., Friday = 3 for weekend carry)
    - Pay-at-expiry: Full accrual information maintained until maturity
    - Reverse mode: Pay when OUTSIDE range instead of inside

    Attributes:
        initial_price: Reference/initial underlying price
        range_config: Configuration with barriers, accrual rate, etc.
        observation_records: List of RangeAccrualObservationRecord with weights & history
        observation_times: Legacy support - list of observation times (weight=1.0 each)
        contract_multiplier: Scaling factor for payoff
        maturity: Time to maturity in years
    """

    # Range accrual specific parameters
    range_config: Optional[RangeAccrualConfig] = None
    observation_records: Optional[List[RangeAccrualObservationRecord]] = None
    observation_times: Optional[List[float]] = None
    num_observations: Optional[int] = None

    def __init__(
        self,
        initial_price: float,
        range_config: RangeAccrualConfig,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        observation_records: Optional[List[RangeAccrualObservationRecord]] = None,
        observation_times: Optional[List[float]] = None,
        num_observations: Optional[int] = None,
        contract_multiplier: float = 1.0,
    ):
        """
        Initialize Range Accrual option.

        Args:
            initial_price: Reference/initial underlying price
            range_config: Configuration with barriers, accrual rate, etc.
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Date when option expires (optional if maturity provided)
            observation_records: List of RangeAccrualObservationRecord with weights & history
            observation_times: Legacy - list of observation times as year fractions
            num_observations: Number of observations for uniform schedule generation
            contract_multiplier: Underlying units represented by one contract
        """
        self.range_config = range_config
        self.observation_records = observation_records
        self.observation_times = observation_times
        self.num_observations = num_observations

        # RangeAccrualOption doesn't use strike in the traditional sense
        # We use initial_price as the reference for barrier percentages
        # Set a dummy strike to pass BaseEquityOption validation
        super().__init__(
            strike=initial_price,  # Use initial_price as strike for base class
            option_type=OptionType.CALL,  # Dummy, not used for payoff
            exercise_type=ExerciseType.EUROPEAN,
            maturity=maturity,
            exercise_date=exercise_date,
            initial_price=initial_price,
            contract_multiplier=contract_multiplier,
        )

    def validate(self) -> None:
        """
        Validate Range Accrual option parameters.

        Raises:
            ValidationError: If any parameter is invalid
        """
        # Skip strike validation (we use initial_price as reference, not strike)
        self._validate_maturity_dates()
        self._validate_date_ordering()
        self._validate_tenor_end()
        self._validate_contract_multiplier()
        self._validate_range_accrual_parameters()

    def _validate_range_accrual_parameters(self) -> None:
        """Validate range accrual specific parameters."""
        # Validate initial_price
        if self.initial_price <= 0:
            raise ValidationError(
                f"initial_price must be positive, got {self.initial_price}"
            )

        # Validate range_config
        if self.range_config is None:
            raise ValidationError("range_config is required")

        # Validate observation inputs are mutually exclusive
        has_records = self.observation_records is not None
        has_times = self.observation_times is not None
        has_num = self.num_observations is not None

        if has_records and has_times:
            raise ValidationError(
                "Cannot provide both observation_records and observation_times"
            )

        if not has_records and not has_times and not has_num:
            raise ValidationError(
                "Must provide observation_records, observation_times, or num_observations"
            )

        # Validate observation_times if provided
        if self.observation_times is not None:
            if len(self.observation_times) == 0:
                raise ValidationError("observation_times cannot be empty")
            if any(t < 0 for t in self.observation_times):
                raise ValidationError("observation_times must be non-negative")
            if self.observation_times != sorted(self.observation_times):
                raise ValidationError(
                    "observation_times must be sorted in ascending order"
                )

        # Validate observation_records if provided
        if self.observation_records is not None:
            if len(self.observation_records) == 0:
                raise ValidationError("observation_records cannot be empty")
            for rec in self.observation_records:
                rec.validate()

        # Validate num_observations if provided
        if self.num_observations is not None and self.num_observations < 1:
            raise ValidationError(
                f"num_observations must be >= 1, got {self.num_observations}"
            )

        # Validate barrier list lengths match observation count if applicable
        self._validate_barrier_lengths()

    def _validate_barrier_lengths(self) -> None:
        """Validate barrier list lengths match observation count."""
        if self.range_config is None:
            return

        num_obs = self._get_expected_observation_count()
        if num_obs is None:
            return

        upper_is_list = isinstance(self.range_config.upper_barrier, list)
        lower_is_list = isinstance(self.range_config.lower_barrier, list)

        if upper_is_list:
            upper_list = list(self.range_config.upper_barrier)  # type: ignore[arg-type]
            if len(upper_list) != num_obs:
                raise ValidationError(
                    f"upper_barrier list length ({len(upper_list)}) must match "
                    f"observation count ({num_obs})"
                )

        if lower_is_list:
            lower_list = list(self.range_config.lower_barrier)  # type: ignore[arg-type]
            if len(lower_list) != num_obs:
                raise ValidationError(
                    f"lower_barrier list length ({len(lower_list)}) must match "
                    f"observation count ({num_obs})"
                )

    def _get_expected_observation_count(self) -> Optional[int]:
        """Get expected observation count from available inputs."""
        if self.observation_records is not None:
            return len(self.observation_records)
        if self.observation_times is not None:
            return len(self.observation_times)
        if self.num_observations is not None:
            return self.num_observations
        return None

    def _build_observation_records(self) -> List[RangeAccrualObservationRecord]:
        """
        Convert legacy observation_times to observation records.

        Returns:
            List of RangeAccrualObservationRecord with default weight=1.0
        """
        if self.observation_records is not None:
            return self.observation_records

        if self.observation_times is not None:
            return [
                RangeAccrualObservationRecord(observation_time=t, weight=1.0)
                for t in self.observation_times
            ]

        # Generate uniform schedule from num_observations
        if self.num_observations is not None and self.maturity is not None:
            times = [
                (i + 1) / self.num_observations * self.maturity
                for i in range(self.num_observations)
            ]
            return [
                RangeAccrualObservationRecord(observation_time=t, weight=1.0)
                for t in times
            ]

        raise ValidationError(
            "Cannot build observation records: provide observation_records, "
            "observation_times, or num_observations with maturity"
        )

    def get_observation_records(self) -> List[RangeAccrualObservationRecord]:
        """
        Get observation records, building from legacy inputs if needed.

        Returns:
            List of RangeAccrualObservationRecord
        """
        return self._build_observation_records()

    def is_in_range(
        self,
        spot: float,
        obs_idx: int,
        record: Optional[RangeAccrualObservationRecord] = None,
    ) -> bool:
        """
        Check if spot is within range for a specific observation.

        Uses per-observation barriers from record if available,
        otherwise falls back to config barriers.

        Args:
            spot: Spot price to check
            obs_idx: Observation index (for config barrier lookup)
            record: Optional observation record with per-observation barriers

        Returns:
            True if spot is in range (or out of range for reverse mode)
        """
        if self.range_config is None:
            raise ValidationError("range_config is required")

        # Get barriers - prefer per-observation if available
        if record is not None and record.upper_barrier is not None:
            upper = record.upper_barrier
        else:
            upper = self.range_config.get_upper_barrier(obs_idx)

        if record is not None and record.lower_barrier is not None:
            lower = record.lower_barrier
        else:
            lower = self.range_config.get_lower_barrier(obs_idx)

        # Check if in range
        in_range = lower <= spot <= upper

        # Reverse mode: pay when outside range
        if self.range_config.is_reverse:
            return not in_range
        return in_range

    def get_weighted_accrual_ratio(
        self,
        in_range_weights: float,
        total_weights: float,
    ) -> float:
        """
        Calculate weighted accrual ratio.

        Args:
            in_range_weights: Sum of weights for in-range observations
            total_weights: Sum of all observation weights

        Returns:
            Accrual ratio (in_range_weights / total_weights)
        """
        if is_zero(total_weights):
            return 0.0
        return in_range_weights / total_weights

    def get_year_fraction(self, pricing_env: Optional["PricingEnvironment"] = None) -> float:
        """
        Calculate year fraction for payoff.

        If rate is not annualized, returns 1.0.
        Otherwise calculates based on day count convention.

        Args:
            pricing_env: Pricing environment (optional)

        Returns:
            Year fraction for payoff calculation
        """
        if self.range_config is None:
            return 1.0

        if not self.range_config.is_rate_annualized:
            return 1.0

        # For annualized rate, use maturity as year fraction
        if self.maturity is not None:
            return self.maturity

        # If using dates, calculate from pricing_env
        if pricing_env is not None and self.exercise_date is not None:
            return self.get_maturity(pricing_env)

        return 1.0

    def get_payoff(
        self,
        spot: float,  # noqa: ARG002 - kept for API consistency with base class
        in_range_weights: Optional[float] = None,
        total_weights: Optional[float] = None,
        pricing_env: Optional["PricingEnvironment"] = None,
    ) -> float:
        """
        Calculate the option payoff at maturity.

        Payoff = initial_price * contract_multiplier * accrual_rate
                 * (in_range_weights / total_weights) * year_fraction

        Args:
            spot: Spot price at maturity (reserved for future use)
            in_range_weights: Sum of weights for in-range observations
            total_weights: Sum of all observation weights
            pricing_env: Optional pricing environment for date calculations

        Returns:
            Option payoff

        Raises:
            ValidationError: If weights are not provided
        """
        _ = spot  # Kept for API consistency; may be used in future extensions
        if self.range_config is None:
            raise ValidationError("range_config is required")

        if in_range_weights is None or total_weights is None:
            raise ValidationError(
                "in_range_weights and total_weights must be provided for payoff calculation"
            )

        accrual_ratio = self.get_weighted_accrual_ratio(in_range_weights, total_weights)
        year_fraction = self.get_year_fraction(pricing_env)

        payoff = (
            self.initial_price
            * self.contract_multiplier
            * self.range_config.accrual_rate
            * accrual_ratio
            * year_fraction
        )

        return payoff

    def resolve_observations(
        self,
        pricing_env: "PricingEnvironment",
    ) -> Tuple[
        List[Tuple[float, bool]],  # past: (weight, in_range)
        List[Tuple[float, float, int]],  # future: (weight, time, obs_idx)
        float,  # total_weights
    ]:
        """
        Resolve observation records into past and future observations.

        This method separates already-observed outcomes from future observations,
        accounting for the valuation date in pricing_env.

        Args:
            pricing_env: Pricing environment with valuation date

        Returns:
            Tuple of:
            - past_observations: List of (weight, in_range) for historical observations
            - future_observations: List of (weight, time, obs_idx) for future observations
            - total_weights: Sum of all observation weights
        """
        records = self.get_observation_records()

        past_observations: List[Tuple[float, bool]] = []
        future_observations: List[Tuple[float, float, int]] = []
        total_weights = 0.0

        for idx, rec in enumerate(records):
            t = rec.resolve_time(pricing_env)
            total_weights += rec.weight

            if t <= 0:
                # Past observation - must have observed_in_range set
                if rec.observed_in_range is None:
                    raise ValidationError(
                        f"Past observation at index {idx} (t={t}) must have "
                        "observed_in_range set"
                    )
                past_observations.append((rec.weight, rec.observed_in_range))
            else:
                # Future observation
                if rec.observed_in_range is not None:
                    raise ValidationError(
                        f"Future observation at index {idx} (t={t}) should not have "
                        "observed_in_range set"
                    )
                future_observations.append((rec.weight, t, idx))

        return past_observations, future_observations, total_weights

    def has_past_observations(self, pricing_env: "PricingEnvironment") -> bool:
        """
        Check if there are any past observations with recorded outcomes.

        Args:
            pricing_env: Pricing environment with valuation date

        Returns:
            True if there are past observations, False otherwise
        """
        past_obs, _, _ = self.resolve_observations(pricing_env)
        return len(past_obs) > 0

    def get_past_accrual(
        self, pricing_env: "PricingEnvironment"
    ) -> Tuple[float, float]:
        """
        Get accumulated accrual from historical observations.

        Args:
            pricing_env: Pricing environment with valuation date

        Returns:
            Tuple of (in_range_weights, past_total_weights) from historical observations
        """
        past_obs, _, _ = self.resolve_observations(pricing_env)

        in_range_weights = 0.0
        past_total_weights = 0.0

        for weight, in_range in past_obs:
            past_total_weights += weight
            if in_range:
                in_range_weights += weight

        return in_range_weights, past_total_weights

    def get_total_weights(self) -> float:
        """
        Get sum of all observation weights.

        Returns:
            Total sum of weights across all observations
        """
        records = self.get_observation_records()
        return sum(rec.weight for rec in records)

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate intrinsic value.

        For range accrual options, intrinsic value requires observation data.
        Returns 0 as a placeholder - use get_payoff with observation data instead.

        Args:
            spot: Current spot price (unused for range accrual)

        Returns:
            0.0 (placeholder - use get_payoff for actual value)
        """
        _ = spot  # Not used for range accrual; requires observation weights
        return 0.0

    def __repr__(self) -> str:
        """Return string representation of the Range Accrual option."""
        if self.range_config is None:
            return f"RangeAccrualOption(initial={self.initial_price:.2f})"

        upper = self.range_config.upper_barrier
        lower = self.range_config.lower_barrier
        rate = self.range_config.accrual_rate

        if isinstance(upper, list):
            upper_str = f"[{upper[0]:.2f}...{upper[-1]:.2f}]"
        else:
            upper_str = f"{upper:.2f}"

        if isinstance(lower, list):
            lower_str = f"[{lower[0]:.2f}...{lower[-1]:.2f}]"
        else:
            lower_str = f"{lower:.2f}"

        maturity_str = (
            f"T={self.maturity:.4f}"
            if self.maturity is not None
            else f"exp={self.exercise_date}"
        )

        return (
            f"RangeAccrualOption(initial={self.initial_price:.2f}, "
            f"range=[{lower_str}, {upper_str}], rate={rate:.2%}, {maturity_str})"
        )
