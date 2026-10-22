"""Unification tests: the book engine gains per-day vol-model calibration
(plan Task 8) and a book-of-one matches the single engine.

Until Task 9 lands per-product recording, the single and book engines emit
different greeks/states column sets (per-unit vs position-level, surface
provenance), so this test compares the frames whose semantics already align
exactly — trades, actions, rebalances, event frames, and the shared value
columns of states — plus calibration records. Task 9's golden gate then
enforces full-schema byte-identity for the single API.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.backtest.replay import (  # noqa: E402
    ReplayBacktestConfig,
    ReplayBacktestEngine,
    ReplayProduct,
    HedgeSpec,
)
from quantark.backtest.replay.single import AutocallableBacktestEngine  # noqa: E402
from quantark.backtest.transaction_costs import ZeroCostModel  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

SHARED_STATE_COLUMNS = [
    "date", "portfolio_value", "product_mtm", "hedge_mtm", "cash", "cashflows",
    "transaction_costs", "product_pnl", "hedge_pnl", "total_pnl", "spot",
    "volatility", "rate", "basis_yield", "implied_q", "pricing_q",
    "active_contract", "futures_price", "futures_ttm", "futures_multiplier",
    "futures_contracts", "alive", "knocked_in", "knocked_out", "matured",
]


def _book_of_one_localvol(tmp_root: Path) -> ReplayBacktestConfig:
    single_cfg = fixtures.make_localvol_config(tmp_root)
    return ReplayBacktestConfig(
        products=[
            ReplayProduct(
                product=single_cfg.product,
                quantity=single_cfg.product_quantity,
                position_id=0,
                has_lifecycle=True,
            )
        ],
        market_data=single_cfg.market_data,
        hedge=HedgeSpec(kind="futures"),
        engine_config=single_cfg.engine_config,
        transaction_cost_model=ZeroCostModel(),
        calculate_surfaces=False,
        calculate_event_probabilities=True,
    )


def test_book_of_one_localvol_matches_single(tmp_path):
    single_results = AutocallableBacktestEngine(
        fixtures.make_localvol_config(tmp_path / "a")
    ).run()
    book_results = ReplayBacktestEngine(_book_of_one_localvol(tmp_path / "b")).run()

    single_frames = fixtures.result_frames(single_results)
    book_frames = fixtures.result_frames(book_results)

    for name in ("trades", "actions", "rebalances", "daily_event_summary",
                 "event_probabilities"):
        pd.testing.assert_frame_equal(
            book_frames[name], single_frames[name], check_exact=True
        )

    pd.testing.assert_frame_equal(
        book_frames["states"][SHARED_STATE_COLUMNS],
        single_frames["states"][SHARED_STATE_COLUMNS],
        check_exact=True,
    )

    single_records = fixtures.calibration_records(single_results)
    book_records = fixtures.calibration_records(book_results)
    assert single_records, "single localvol run must calibrate"
    assert book_records == single_records


def test_phoenix_book_with_vol_model_rejected(tmp_path):
    cfg = _book_of_one_localvol(tmp_path)
    with pytest.raises(ValidationError, match="SnowballOption"):
        ReplayBacktestConfig(
            products=[
                ReplayProduct(
                    product=fixtures._phoenix_product(),
                    quantity=-1.0,
                    position_id=0,
                    has_lifecycle=True,
                )
            ],
            market_data=cfg.market_data,
            engine_config=cfg.engine_config,
        )


def test_mixed_book_with_vol_model_rejected(tmp_path):
    cfg = _book_of_one_localvol(tmp_path)
    with pytest.raises(ValidationError, match="SnowballOption"):
        ReplayBacktestConfig(
            products=[
                ReplayProduct(product=fixtures._snowball_product(), quantity=-1.0,
                              position_id=0, has_lifecycle=True),
                ReplayProduct(product=fixtures._phoenix_product(), quantity=-1.0,
                              position_id=1, has_lifecycle=True),
            ],
            market_data=cfg.market_data,
            engine_config=cfg.engine_config,
        )


def test_mixed_book_with_bsm_constructs():
    cfg = fixtures.make_book_config()
    mixed = ReplayBacktestConfig(
        products=[
            ReplayProduct(product=fixtures._snowball_product(), quantity=-1.0,
                          position_id=0, has_lifecycle=True),
            ReplayProduct(product=fixtures._phoenix_product(), quantity=-1.0,
                          position_id=1, has_lifecycle=True),
        ],
        market_data=cfg.market_data,
        engine_config=cfg.engine_config,
    )
    assert len(mixed.products) == 2
