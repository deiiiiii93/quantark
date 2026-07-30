"""Deprecated shim — moved to quantark.backtest.replay (0.5.0 removes this).

The engine lives in ``quantark.backtest.replay.engine``; the config/product
classes in ``.replay.config``; the results in ``.replay.results``.
"""
import warnings

from quantark.backtest.replay.config import (  # noqa: F401
    BookAutocallableBacktestConfig,
    BookProduct,
    HedgeSpec,
    ReplayBacktestConfig,
    ReplayProduct,
)
from quantark.backtest.replay.engine import (  # noqa: F401
    BookAutocallableBacktestEngine,
    ReplayBacktestEngine,
)
from quantark.backtest.replay.results import (  # noqa: F401
    BookBacktestResults,
    ReplayBacktestResults,
)

warnings.warn(
    "quantark.backtest.otc.book_engine moved to quantark.backtest.replay; "
    "this alias is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)
