"""Deprecated shim — moved to quantark.backtest.replay.engine_factory (0.5.0 removes this)."""
import warnings

from quantark.backtest.replay.engine_factory import *  # noqa: F401,F403

warnings.warn(
    "quantark.backtest.otc.engine_factory moved to quantark.backtest.replay.engine_factory; "
    "this alias is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)
