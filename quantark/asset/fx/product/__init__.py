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
)
from .deltaone import BaseFxDeltaOneProduct, FxSpot, FxForward, FxSwap

__all__ = [
    'CurrencyPair',
    'BaseFxProduct',
    'FxVanillaOption',
    'FxDigitalOption',
    'FxQuantoVanillaOption',
    'FxQuantoDigitalOption',
    'BaseFxDeltaOneProduct',
    'FxSpot',
    'FxForward',
    'FxSwap',
]
