"""
FX derivative products.
"""
from .currency_pair import CurrencyPair
from .base_fx_product import BaseFxProduct
from .option import (
    FxVanillaOption,
    FxDigitalOption,
    FxQuantoVanillaOption,
    FxQuantoDigitalOption,
    FxRangeAccrualOption,
    FxQuantoRangeAccrualOption,
    FxForeignRangeAccrualOption,
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
)
from .deltaone import BaseFxDeltaOneProduct, FxSpot, FxForward, FxSwap

__all__ = [
    'CurrencyPair',
    'BaseFxProduct',
    'FxVanillaOption',
    'FxDigitalOption',
    'FxQuantoVanillaOption',
    'FxQuantoDigitalOption',
    'FxRangeAccrualOption',
    'FxQuantoRangeAccrualOption',
    'FxForeignRangeAccrualOption',
    'FxRangeAccrualConfig',
    'FxRangeAccrualObservationRecord',
    'BaseFxDeltaOneProduct',
    'FxSpot',
    'FxForward',
    'FxSwap',
]
