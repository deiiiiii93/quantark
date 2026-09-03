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
from quantark.pnlexplain.equity.fingerprints import (  # noqa: F401
    contract_fingerprint, engines_equivalent, lifecycle_fingerprint,
)
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition  # noqa: F401
from quantark.pnlexplain.equity.explain import explain  # noqa: F401
from quantark.pnlexplain.equity.trades import ExplainTrade  # noqa: F401
from quantark.pnlexplain.equity.portfolio import (  # noqa: F401
    BookSnapshot, PortfolioExplainResult, PositionExplainResult, PositionSnapshot, QuotedLegSnapshot,
    explain_portfolio, explain_position, explain_quoted_leg,
)
from quantark.pnlexplain.equity.recorder import (  # noqa: F401
    RECON_COLUMNS, PnLExplainRecorder, ReplayPnLExplainRecorder,
)

__all__ = [
    "BookSnapshot", "ExplainMethod", "ExplainRow", "ExplainTrade", "FRAME_COLUMNS", "Factor",
    "FactorCoordinate", "FactorMoves", "LifecycleTransition", "MARKET_FACTORS", "MOVE_KEYS",
    "PnLExplainConfig", "PnLExplainRecorder", "PnLExplainResult", "PortfolioExplainResult",
    "PositionExplainResult", "PositionSnapshot", "QuotedLegSnapshot", "RECON_COLUMNS",
    "ReplayPnLExplainRecorder", "RowKind", "ValuationSnapshot", "ValueBreakdown", "component_sum",
    "contract_fingerprint", "engines_equivalent", "explain", "explain_portfolio", "explain_position",
    "explain_quoted_leg", "lifecycle_fingerprint", "make_total_row", "rows_to_frame", "value",
]
