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
from .ko_reset_snowball_option import KnockOutResetSnowballOption
from .snowball_config import BarrierConfig, PayoffConfig, AccrualConfig
from .snowball_helpers import (
    create_standard_snowball,
    create_stepdown_snowball,
    create_european_ki_snowball,
    create_parachute_snowball,
    create_airbag_snowball,
    create_ko_reset_snowball,
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
from .range_accrual_option import RangeAccrualOption
from .range_accrual_config import RangeAccrualConfig, RangeAccrualObservationRecord
from .range_accrual_helpers import (
    create_standard_range_accrual,
    create_reverse_range_accrual,
    create_stepdown_range_accrual,
    generate_range_observation_records,
    assign_calendar_day_weights,
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
    "KnockOutResetSnowballOption",
    "BarrierConfig",
    "PayoffConfig",
    "AccrualConfig",
    # Snowball helpers
    "create_standard_snowball",
    "create_stepdown_snowball",
    "create_european_ki_snowball",
    "create_parachute_snowball",
    "create_airbag_snowball",
    "create_ko_reset_snowball",
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
    # Range Accrual option
    "RangeAccrualOption",
    "RangeAccrualConfig",
    "RangeAccrualObservationRecord",
    # Range Accrual helpers
    "create_standard_range_accrual",
    "create_reverse_range_accrual",
    "create_stepdown_range_accrual",
    "generate_range_observation_records",
    "assign_calendar_day_weights",
]
