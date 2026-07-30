"""Deprecated shim — moved to quantark.param.vol.surface_history (0.5.0 removes this)."""
import warnings

from quantark.param.vol.surface_history import *  # noqa: F401,F403
from quantark.param.vol.surface_history import (  # noqa: F401
    IvSurfaceArtifact,
    VolSurfaceHistory,
)

warnings.warn(
    "quantark.backtest.otc.vol_history moved to "
    "quantark.param.vol.surface_history; this alias is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)
