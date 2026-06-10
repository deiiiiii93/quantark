"""
Enumeration types for option specifications.
"""

from enum import Enum, auto


class OptionType(Enum):
    """Type of option: Call or Put."""

    CALL = auto()
    PUT = auto()

    def __str__(self):
        return self.name.capitalize()


class ExerciseType(Enum):
    """Exercise style of the option."""

    EUROPEAN = auto()
    AMERICAN = auto()
    BERMUDAN = auto()

    def __str__(self):
        return self.name.capitalize()


class BarrierType(Enum):
    """Barrier option type: knock-in or knock-out, up or down."""

    UP_IN = auto()  # Knock-in when price goes UP
    UP_OUT = auto()  # Knock-out when price goes UP
    DOWN_IN = auto()  # Knock-in when price goes DOWN
    DOWN_OUT = auto()  # Knock-out when price goes DOWN

    def __str__(self):
        return self.name.replace("_", " ").title()

    @property
    def is_up(self) -> bool:
        """Check if this is an up barrier."""
        return self in (BarrierType.UP_IN, BarrierType.UP_OUT)

    @property
    def is_down(self) -> bool:
        """Check if this is a down barrier."""
        return self in (BarrierType.DOWN_IN, BarrierType.DOWN_OUT)

    @property
    def is_knock_in(self) -> bool:
        """Check if this is a knock-in barrier."""
        return self in (BarrierType.UP_IN, BarrierType.DOWN_IN)

    @property
    def is_knock_out(self) -> bool:
        """Check if this is a knock-out barrier."""
        return self in (BarrierType.UP_OUT, BarrierType.DOWN_OUT)


class DoubleBarrierType(Enum):
    """Double barrier option type: knock-in or knock-out."""

    KNOCK_IN = auto()  # Knock-in when either barrier is hit
    KNOCK_OUT = auto()  # Knock-out when either barrier is hit

    def __str__(self):
        return self.name.replace("_", " ").title()


class BarrierDirection(Enum):
    """Direction of barrier relative to spot."""

    UP = auto()  # Barrier is above current spot
    DOWN = auto()  # Barrier is below current spot

    def __str__(self):
        return self.name.capitalize()


class ObservationType(Enum):
    """Observation type for barrier monitoring."""

    CONTINUOUS = auto()  # Barrier monitored continuously
    DISCRETE = auto()  # Barrier monitored at specific times only
    EXPIRY = auto()  # Barrier observed only at expiry/exercise

    def __str__(self):
        return self.name.capitalize()


class ObservationAggregation(Enum):
    """Aggregation mode for discrete barrier observations."""

    STOP_FIRST_HIT = "stop-first-hit"
    ACCUMULATE = "accumulate"
    BEST = "best"
    WORST = "worst"

    def __str__(self):
        return self.value


class TouchType(Enum):
    """Type of touch option."""

    ONE_TOUCH = auto()  # Pays if barrier is touched
    NO_TOUCH = auto()  # Pays if barrier is NOT touched
    DOUBLE_ONE_TOUCH = auto()  # Pays if either barrier is touched
    DOUBLE_NO_TOUCH = auto()  # Pays if neither barrier is touched

    def __str__(self):
        return self.name.replace("_", " ").title()


class CouponPayType(Enum):
    """Coupon payment timing for autocallable products."""

    INSTANT = auto()  # Pay at knock-out date
    EXPIRY = auto()  # Discount to final maturity

    def __str__(self):
        return self.name.capitalize()


class ProtectionType(Enum):
    """Protection type for downside in autocallable products."""

    NONE = auto()  # No protection
    PARTIAL = auto()  # Floor at (1 - protection_rate) × N
    FULL = auto()  # Floor at principal (or zero if principal excluded)

    def __str__(self):
        return self.name.capitalize()


class TenorEnd(Enum):
    """Tenor end-point selection for product-level accruals."""

    EXERCISE = auto()  # Use exercise_date as tenor end
    SETTLEMENT = auto()  # Use settlement_date as tenor end
    MATURITY = auto()  # Use maturity_date as tenor end

    def __str__(self):
        return self.name.capitalize()


class ObservationFrequency(Enum):
    """Frequency of observations for barrier monitoring."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"
    CUSTOM = "custom"

    def __str__(self):
        return self.value.replace("_", " ").title()

    def to_year_fraction(self, use_business_days: bool = False, days_in_year: float = 365.0) -> float:
        """
        Converts the observation frequency to a year fraction (dt).

        Args:
            use_business_days: If True, uses 252-day year (trading days).
                             If False, uses 365-day year (calendar days).
            days_in_year: Number of days in a year (default: 365.0).

        Raises:
            ValueError: If the frequency is CUSTOM, as it implies irregular spacing.
        """
        if self == ObservationFrequency.DAILY:
            return 1.0 / days_in_year
        elif self == ObservationFrequency.WEEKLY:
            days_per_week = 5.0 if use_business_days else 7.0
            return days_per_week / days_in_year
        elif self == ObservationFrequency.MONTHLY:
            return 1.0 / 12.0
        elif self == ObservationFrequency.QUARTERLY:
            return 1.0 / 4.0
        elif self == ObservationFrequency.SEMI_ANNUALLY:
            return 1.0 / 2.0
        elif self == ObservationFrequency.ANNUALLY:
            return 1.0
        elif self == ObservationFrequency.CUSTOM:
            raise ValueError(
                "Cannot convert CUSTOM observation frequency to year fraction, as it implies irregular spacing."
            )
        raise ValueError(f"Unknown ObservationFrequency: {self}")


class PostKOScheduleMode(Enum):
    """Mode for applying post-KI KO schedules."""

    ABSOLUTE = auto()
    REBASED = auto()

    def __str__(self):
        return self.name.capitalize()


class AveragingType(Enum):
    """Averaging method for Asian options."""

    ARITHMETIC = auto()  # Average = sum(prices) / n
    GEOMETRIC = auto()  # Average = (product(prices))^(1/n)

    def __str__(self):
        return self.name.capitalize()


class AsianStrikeType(Enum):
    """Strike type for Asian options."""

    FIXED = auto()  # Fixed strike (average price option): payoff based on avg vs K
    FLOATING = auto()  # Floating strike (average strike option): payoff based on S_T vs avg

    def __str__(self):
        return self.name.capitalize()
