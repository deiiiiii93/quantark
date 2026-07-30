"""Deprecated shim — moved to quantark.backtest.replay.engine (0.5.0 removes this)."""
import warnings

from quantark.backtest.replay.engine import *  # noqa: F401,F403

warnings.warn(
    "quantark.backtest.otc.book_engine moved to quantark.backtest.replay.engine; "
    "this alias is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)
