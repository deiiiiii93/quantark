"""Deprecated shim — moved to quantark.backtest.replay.single (0.5.0 removes this)."""
import warnings

from quantark.backtest.replay.single import *  # noqa: F401,F403

warnings.warn(
    "quantark.backtest.otc.engine moved to quantark.backtest.replay.single; "
    "this alias is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)
