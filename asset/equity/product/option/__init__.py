"""
Equity option products.
"""

from .base_equity_option import BaseEquityOption
from .european_vanilla_option import EuropeanVanillaOption
from .american_option import AmericanOption
from .asian_option import AsianOption, AsianObservationRecord
from .digital_option import CashOrNothingDigitalOption
from .barrier_option import BarrierOption
from .double_barrier_option import DoubleBarrierOption
from .one_touch_option import OneTouchOption
from .double_one_touch_option import DoubleOneTouchOption
from .observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
    ResolvedObservationRecord,
)
from .snowball_option import SnowballOption
from .snowball_config import BarrierConfig, PayoffConfig, AccrualConfig
from .snowball_helpers import (
    create_standard_snowball,
    create_stepdown_snowball,
    create_european_ki_snowball,
    create_parachute_snowball,
    create_airbag_snowball,
    generate_ko_observation_dates,
    generate_stepdown_barriers,
)
from .phoenix_option import PhoenixOption
from .phoenix_config import CouponBarrierConfig
from .phoenix_helpers import (
    create_standard_phoenix,
    create_stepdown_phoenix,
    create_reverse_phoenix,
    create_memory_phoenix,
    create_non_memory_phoenix,
)

__all__ = [
    "BaseEquityOption",
    "EuropeanVanillaOption",
    "AmericanOption",
    "AsianOption",
    "AsianObservationRecord",
    "CashOrNothingDigitalOption",
    "BarrierOption",
    "DoubleBarrierOption",
    "OneTouchOption",
    "DoubleOneTouchOption",
    "ObservationRecord",
    "ObservationSchedule",
    "ResolvedObservationRecord",
    "SnowballOption",
    "BarrierConfig",
    "PayoffConfig",
    "AccrualConfig",
    # Snowball helpers
    "create_standard_snowball",
    "create_stepdown_snowball",
    "create_european_ki_snowball",
    "create_parachute_snowball",
    "create_airbag_snowball",
    "generate_ko_observation_dates",
    "generate_stepdown_barriers",
    # Phoenix option
    "PhoenixOption",
    "CouponBarrierConfig",
    # Phoenix helpers
    "create_standard_phoenix",
    "create_stepdown_phoenix",
    "create_reverse_phoenix",
    "create_memory_phoenix",
    "create_non_memory_phoenix",
]
