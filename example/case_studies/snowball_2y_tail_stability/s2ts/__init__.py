"""2Y snowball carry-tail stability study: converge-to-anchor vs hold-forward.

Package name ``s2ts`` is unique on purpose (several studies own a ``_common.py``).
Spec: ``.ship/2026-11-06-snowball-2y-tail-stability/spec.md`` (untracked run directory).
"""

from __future__ import annotations

from pathlib import Path

from quantark.util.exceptions import MarketDataError

STUDY_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = STUDY_DIR / "results"

#: Shared input cache -- READ-ONLY for this study.
HISTORY_DIR = Path("/Users/fuxinyao/quant-ark/example/case_studies/mo_volmodels/data/history")
#: Untracked outputs (gitignored via ``example/output/``): extensions, chain pickle, runs.
OUTPUT_DIR = Path("/Users/fuxinyao/quant-ark/example/output/snowball_2y_tail_stability")
CFFEX_EXTENSION_DIR = OUTPUT_DIR / "cffex_extension"
SPOT_EXTENSION_PATH = OUTPUT_DIR / "spot_extension.csv"
CHAIN_PATH = OUTPUT_DIR / "daily_chain.pkl"

DASHBOARD_DB = Path("/Users/fuxinyao/index-dashboard-codex/data/basis.sqlite3")
INDEX_CODE = "000852.SH"
CFFEX_URL = "http://www.cffex.com.cn/sj/hqsj/rtj/{yyyymm}/{dd}/{yyyymmdd}_1.csv"

#: Flat continuously-compounded rate; ACT/365F year fractions.
RATE = 0.014
ACT = 365.0
NOTIONAL = 10_000_000.0
IM_MULT = 200.0  # CNY per index point per IM lot
MO_MULT = 100.0  # CNY per index point per MO option
MIN_TENOR_DAYS = 7
HALF_LIVES = (0.25, 0.5, 1.0)
ANCHOR_WINDOW = 63
ANCHOR_WINDOW_LONG = 252


class StudyDataError(MarketDataError):
    """A market-data row the study needs is missing or malformed (fail closed)."""
