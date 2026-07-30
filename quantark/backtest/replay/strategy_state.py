"""
State and strategy objects for OTC autocallable backtests.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy  # noqa: F401  (canonical home)
from quantark.backtest.futures_ledger import FuturesHedgePosition  # noqa: F401  (canonical home)
from quantark.util.exceptions import ValidationError
