"""Deprecated shim — moved to quantark.backtest.replay.results (0.5.0 removes this)."""
import warnings

from quantark.backtest.replay.results import *  # noqa: F401,F403

warnings.warn(
    "quantark.backtest.otc.results moved to quantark.backtest.replay.results; "
    "this alias is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)
