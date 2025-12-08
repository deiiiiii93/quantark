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
