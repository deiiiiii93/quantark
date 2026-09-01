import json
from datetime import date
from pathlib import Path

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.snapshot import (
    PRICE_FIELD_MID_OR_LAST,
    PRICE_FIELD_SETTLEMENT,
    QuoteSnapshot,
)

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"
SETTLE = ROOT / "example/mo_volmodels/data/mo_settlement_snapshot_20260430.json"


def test_lift_live_snapshot():
    snap = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()),
        trade_date=date(2026, 7, 6),
        symbol="000852.SH",
    )
    assert snap.convention == "listed_strike"
    assert snap.price_field == PRICE_FIELD_MID_OR_LAST
    assert snap.spot == 6000.0
    assert len(snap.expiries) == 4
    assert all(e["T_years"] > 0.0 for e in snap.expiries)


def test_lift_settlement_snapshot_requires_spot():
    payload = json.loads(SETTLE.read_text())
    snap = QuoteSnapshot.from_legacy_settlement(
        payload, trade_date=date(2026, 4, 30), spot=6000.0, symbol="000852.SH"
    )
    assert snap.price_field == PRICE_FIELD_SETTLEMENT
    assert snap.spot == 6000.0
    with pytest.raises(TypeError):
        QuoteSnapshot.from_legacy_settlement(
            payload, trade_date=date(2026, 4, 30), symbol="000852.SH"
        )


def test_price_field_resolution():
    live = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="X"
    )
    assert live.quote_price({"bid": 2.0, "ask": 4.0, "last": 99.0}) == 3.0
    assert live.quote_price({"bid": 0.0, "ask": 4.0, "last": 99.0}) == 99.0

    settle = QuoteSnapshot.from_legacy_settlement(
        json.loads(SETTLE.read_text()),
        trade_date=date(2026, 4, 30),
        spot=6000.0,
        symbol="X",
    )
    assert settle.quote_price({"settlement": 12.5, "close": 99.0}) == 12.5


def test_declared_price_field_absent_is_rejected():
    snap = QuoteSnapshot.from_legacy_settlement(
        json.loads(SETTLE.read_text()),
        trade_date=date(2026, 4, 30),
        spot=6000.0,
        symbol="X",
    )
    with pytest.raises(ValidationError, match="price_field_mismatch"):
        snap.quote_price({"last": 1.0, "bid": 0.9, "ask": 1.1})


def test_from_payload_rejects_unknown_convention():
    with pytest.raises(ValidationError, match="convention"):
        QuoteSnapshot.from_payload(
            {
                "schema_version": 1,
                "convention": "nonsense",
                "trade_date": "2026-07-06",
                "underlying": {"symbol": "X", "spot": 1.0},
                "source": {"price_field": PRICE_FIELD_MID_OR_LAST},
                "expiries": [],
            }
        )
