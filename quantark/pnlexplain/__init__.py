"""PnL explain: waterfall + Taylor attribution of PV changes (spec 2026-09-02)."""
from quantark.pnlexplain.base import (  # noqa: F401
    FRAME_COLUMNS, MARKET_FACTORS, MOVE_KEYS, ExplainMethod, ExplainRow, Factor,
    PnLExplainResult, RowKind, ValueBreakdown, component_sum, make_total_row,
    rows_to_frame,
)
from quantark.pnlexplain.config import PnLExplainConfig  # noqa: F401
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, value  # noqa: F401
from quantark.pnlexplain.equity.coordinate import FactorCoordinate  # noqa: F401
from quantark.pnlexplain.equity.factor_diff import FactorMoves  # noqa: F401
from quantark.pnlexplain.equity.fingerprints import contract_fingerprint, lifecycle_fingerprint  # noqa: F401
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition  # noqa: F401
from quantark.pnlexplain.equity.explain import explain  # noqa: F401

__all__ = [
    "ExplainMethod", "ExplainRow", "FRAME_COLUMNS", "Factor", "FactorCoordinate", "FactorMoves",
    "LifecycleTransition", "MARKET_FACTORS", "MOVE_KEYS", "PnLExplainConfig", "PnLExplainResult",
    "RowKind", "ValuationSnapshot", "ValueBreakdown", "component_sum", "contract_fingerprint",
    "explain", "lifecycle_fingerprint", "make_total_row", "rows_to_frame", "value",
]
