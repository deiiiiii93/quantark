"""Deprecated shim — moved to quantark.backtest.replay.product_replay (0.5.0 removes this)."""
import warnings

from quantark.backtest.replay.product_replay import *  # noqa: F401,F403

warnings.warn(
    "quantark.backtest.otc._replay moved to quantark.backtest.replay.product_replay; "
    "this alias is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)
