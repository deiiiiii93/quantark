"""
FX option products.
"""
from .fx_vanilla_option import FxVanillaOption
from .fx_digital_option import FxDigitalOption
from .fx_quanto_vanilla_option import FxQuantoVanillaOption
from .fx_quanto_digital_option import FxQuantoDigitalOption
from .fx_one_touch_option import FxOneTouchOption
from .fx_barrier_option import FxBarrierOption
from .fx_sharkfin_option import FxSharkfinOption
from .fx_range_accrual_option import FxRangeAccrualOption
from .fx_quanto_range_accrual_option import FxQuantoRangeAccrualOption
from .fx_foreign_range_accrual_option import FxForeignRangeAccrualOption
from .fx_range_accrual_config import (
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
)

__all__ = [
    'FxVanillaOption',
    'FxDigitalOption',
    'FxQuantoVanillaOption',
    'FxQuantoDigitalOption',
    'FxOneTouchOption',
    'FxBarrierOption',
    'FxSharkfinOption',
    'FxRangeAccrualOption',
    'FxQuantoRangeAccrualOption',
    'FxForeignRangeAccrualOption',
    'FxRangeAccrualConfig',
    'FxRangeAccrualObservationRecord',
]
