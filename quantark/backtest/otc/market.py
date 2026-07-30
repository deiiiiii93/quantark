"""Deprecated shim — moved to quantark.backtest.replay.market (0.5.0 removes this)."""
import warnings

from quantark.backtest.replay.market import *  # noqa: F401,F403

warnings.warn(
    "quantark.backtest.otc.market moved to quantark.backtest.replay.market; "
    "this alias is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)
