"""DEPRECATED: moved to :mod:`quantark.volcalibration.calibrate` in 0.4.0.

``quantark.volmodels`` is asset-neutral and imports no pricing environments or
market-data containers; per-day calibration needs both, so it moved one layer
up into ``quantark.volcalibration``.  This shim is kept until 0.5.0.

The disk cache key is ``sha256(surface_sha|variant|config_fingerprint)`` and
contains no module paths, so entries written through either import path
interoperate -- a cache warmed before the move is still a hit after it.
"""

from quantark.volcalibration.calibrate import (  # noqa: F401
    VOL_MODEL_HESTON,
    VOL_MODEL_HESTON_SLV,
    VOL_MODEL_LOCALVOL,
    VOL_MODEL_VARIANTS,
    CalibratedVolModel,
    VolModelCalibrator,
    _atomic_write_json,
    _CACHE_SCHEMA_VERSION,
    _json_safe,
)
from quantark.volcalibration.config import HESTON_PRESETS  # noqa: F401

__all__ = [
    "VolModelCalibrator",
    "CalibratedVolModel",
    "HESTON_PRESETS",
    "VOL_MODEL_VARIANTS",
    "VOL_MODEL_LOCALVOL",
    "VOL_MODEL_HESTON",
    "VOL_MODEL_HESTON_SLV",
]
