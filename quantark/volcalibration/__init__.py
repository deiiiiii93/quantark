"""Market-data-aware vol-model calibration: quotes -> IV surface -> LV/Heston/SLV.

Composes the asset-neutral kernels in ``quantark.volmodels`` with market data
from ``quantark.param``.  Nothing here imports ``quantark.backtest`` or
``quantark.asset``.
"""

from quantark.volcalibration.quotes import ExpiryQuotes, IvNode, QuoteSet
from quantark.volcalibration.snapshot import (
    PRICE_FIELD_MID_OR_LAST,
    PRICE_FIELD_SETTLEMENT,
    QuoteSnapshot,
)

from quantark.volcalibration.calibrate import (
    VOL_MODEL_VARIANTS,
    CalibratedVolModel,
    VolModelCalibrator,
)
from quantark.volcalibration.config import HESTON_PRESETS, VolModelCalibrationConfig

# The producer/consumer contract lives in the shared param layer so backtest
# and modelvalidation can depend on it without depending on this package.
from quantark.param.vol.surface_history import IvSurfaceArtifact, VolSurfaceHistory

__all__ = [
    "QuoteSnapshot",
    "PRICE_FIELD_SETTLEMENT",
    "PRICE_FIELD_MID_OR_LAST",
    "QuoteSet",
    "ExpiryQuotes",
    "IvNode",
    "VolModelCalibrator",
    "CalibratedVolModel",
    "VolModelCalibrationConfig",
    "HESTON_PRESETS",
    "VOL_MODEL_VARIANTS",
    "IvSurfaceArtifact",
    "VolSurfaceHistory",
]
