"""2Y snowball gamma pin-risk study: techniques and SOPs, told from the snowball seller's side.

Package name ``gpin`` is unique on purpose. The study imports the sibling
tail-stability study's ``s2ts`` package read-only (term sheet, chain, carry
tails, engine handle); stage scripts and tests put both study directories on
``sys.path``.

Everything a reader could tune lives in ``results/preregistration.json``, which
is committed before any ledger run; this module only holds paths, the spot
ladder and the loader. Spec: ``.ship/2026-11-12-snowball-2y-gamma-pin-sop/spec.md``
(untracked run directory).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

STUDY_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = STUDY_DIR / "results"
S2TS_DIR = STUDY_DIR.parent / "snowball_2y_tail_stability"
if str(S2TS_DIR) not in sys.path:
    sys.path.insert(0, str(S2TS_DIR))

from s2ts import CFFEX_EXTENSION_DIR as S2TS_CFFEX_EXTENSION_DIR  # noqa: E402
from s2ts import HISTORY_DIR, SPOT_EXTENSION_PATH, StudyDataError  # noqa: E402

__all__ = [
    "BOARD_PATH", "CFFEX_EXTENSION_DIR", "CHAIN_PATH", "HISTORY_DIR", "LADDER", "OUTPUT_DIR", "PREREG_PATH",
    "RESULTS_DIR", "RUNS_DIR", "SPOT_EXTENSION_PATH", "STATES_DIR", "STUDY_DIR", "StudyDataError", "TAIL",
    "load_prereg", "prereg_hash",
]

#: Untracked outputs (gitignored via ``example/output/``): boards, chain, recorded states, ledger runs.
OUTPUT_DIR = Path("/Users/fuxinyao/quant-ark/example/output/snowball_2y_gamma_pin_sop")
BOARD_PATH = OUTPUT_DIR / "board.pkl"
CHAIN_PATH = OUTPUT_DIR / "chain.pkl"
STATES_DIR = OUTPUT_DIR / "states"
RUNS_DIR = OUTPUT_DIR / "runs"

#: Read-only inputs: the shared history cache and the tail-stability study's CFFEX extension
#: (2026-10-22 .. 2026-11-06). The data end is frozen; this study downloads nothing.
CFFEX_EXTENSION_DIR = S2TS_CFFEX_EXTENSION_DIR

PREREG_PATH = RESULTS_DIR / "preregistration.json"

#: Carry tail of every mark and hedge (the tail-stability study's adopted tail).
TAIL = "converge_hl100"

#: Spot ladder of the pricing pass, relative to the day's spot: -12% .. +12% in 1% steps.
LADDER: Tuple[float, ...] = tuple(round(0.01 * k, 2) for k in range(-12, 13))


def load_prereg(path: Optional[Path] = None) -> Dict[str, Any]:
    """The pre-registration (technique grid, windows, thresholds, statistics, verdict rule); fails closed."""
    path = PREREG_PATH if path is None else Path(path)
    if not path.exists():
        raise StudyDataError(f"{path} missing: the pre-registration is committed before any ledger run")
    return json.loads(path.read_text(encoding="utf-8"))


def prereg_hash(path: Optional[Path] = None) -> str:
    """sha256 of the pre-registration's canonical JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(load_prereg(path), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
