"""Deprecated shim — moved to quantark.volmodels.calibration (0.5.0 removes this)."""
import warnings

from quantark.volmodels.calibration import *  # noqa: F401,F403
from quantark.volmodels.calibration import (  # noqa: F401
    HESTON_PRESETS,
    VOL_MODEL_HESTON,
    VOL_MODEL_HESTON_SLV,
    VOL_MODEL_LOCALVOL,
    VOL_MODEL_VARIANTS,
    CalibratedVolModel,
    VolModelCalibrator,
    _atomic_write_json,
    _json_safe,
)

warnings.warn(
    "quantark.backtest.otc.vol_calibrators moved to "
    "quantark.volmodels.calibration; this alias is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)
