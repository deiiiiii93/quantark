"""Stage 1: the daily chain and the full CFFEX boards, from the files already on disk (nothing is downloaded).

Reads the shared ``mo_volmodels/data/history`` cache, the tail-stability study's CFFEX extension and its spot
extension (all READ-ONLY), clips at the pre-registered data end, and writes ``OUTPUT_DIR/chain.pkl`` and
``OUTPUT_DIR/board.pkl`` atomically.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpin import (  # noqa: E402
    BOARD_PATH, CFFEX_EXTENSION_DIR, CHAIN_PATH, HISTORY_DIR, SPOT_EXTENSION_PATH, StudyDataError, load_prereg,
)
from gpin.board import build_boards, write_boards, write_pickle_atomic  # noqa: E402
from s2ts.chain import build_all_chains, sse_trading_days  # noqa: E402


def main() -> None:
    data = load_prereg()["data"]
    end = date.fromisoformat(data["end"])
    calendar = sse_trading_days(2023, 2030)
    chains = [dc for dc in build_all_chains(HISTORY_DIR, CFFEX_EXTENSION_DIR, calendar, SPOT_EXTENSION_PATH)
              if dc.day <= end]
    if chains[-1].day != end or chains[0].day != date.fromisoformat(data["start"]):
        raise StudyDataError(
            f"the on-disk data covers {chains[0].day}..{chains[-1].day}, the pre-registration says "
            f"{data['start']}..{data['end']}")
    boards = build_boards(chains, HISTORY_DIR, CFFEX_EXTENSION_DIR, calendar)
    write_pickle_atomic(chains, CHAIN_PATH)
    write_boards(boards, BOARD_PATH)
    n_series = sum(len(b.series) for b in boards.values())
    n_quotes = sum(len(q) for b in boards.values() for q in b.mo.values())
    print(f"daily chain: {chains[0].day} .. {chains[-1].day}, {len(chains)} days -> {CHAIN_PATH}")
    print(f"boards: {len(boards)} days, {n_series} series-days with a forward, {n_quotes} option quotes -> {BOARD_PATH}")


if __name__ == "__main__":
    main()
