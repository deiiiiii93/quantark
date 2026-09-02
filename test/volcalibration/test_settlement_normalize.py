import json
from datetime import date
from pathlib import Path

import pytest

from quantark.volcalibration.admission import AdmissionError, AdmissionReason
from quantark.volcalibration.normalize.settlement import SettlementNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot

ROOT = Path(__file__).resolve().parents[2]
SETTLE = ROOT / "example/mo_volmodels/data/mo_settlement_snapshot_20260430.json"


def _snapshot(spot=6000.0):
    return QuoteSnapshot.from_legacy_settlement(
        json.loads(SETTLE.read_text()),
        trade_date=date(2026, 4, 30),
        spot=spot,
        symbol="000852.SH",
    )


def test_settlement_normalizer_builds_a_quoteset_with_a_universe():
    qs = SettlementNormalizer().normalize(_snapshot())
    assert qs.convention == "listed_strike"
    assert len(qs.expiries) >= 2
    assert [e.T for e in qs.expiries] == sorted(e.T for e in qs.expiries)
    assert set(qs.universe) >= {
        "node_count",
        "expiry_count",
        "filtered_quote_counts",
        "excluded_expiries",
    }
    assert qs.universe["node_count"] == sum(len(e.nodes) for e in qs.expiries)


def test_maturity_is_calendar_days_over_365_not_the_snapshot_field():
    qs = SettlementNormalizer().normalize(_snapshot())
    for e in qs.expiries:
        days = round(e.T * 365.0)
        assert e.T == pytest.approx(days / 365.0, abs=1e-15)
        assert 7 <= days <= 365


def test_both_otm_wings_are_present_in_every_admitted_expiry():
    qs = SettlementNormalizer().normalize(_snapshot())
    for e in qs.expiries:
        strikes = [n.strike for n in e.nodes]
        assert min(strikes) < e.forward, f"{e.expiry_label}: no put wing"
        assert max(strikes) >= e.forward, f"{e.expiry_label}: no call wing"


def test_zero_open_interest_quotes_are_filtered_and_counted():
    payload = json.loads(SETTLE.read_text())
    for quote in payload["expiries"][0]["quotes"]:
        quote["oi"] = 0
    snap = QuoteSnapshot.from_legacy_settlement(
        payload, trade_date=date(2026, 4, 30), spot=6000.0, symbol="X"
    )
    qs = SettlementNormalizer().normalize(snap)
    counts = qs.universe["filtered_quote_counts"]
    assert counts.get("zero_selected_open_interest", 0) > 0
    excluded = {row["expiry_date"] for row in qs.universe["excluded_expiries"]}
    assert excluded  # that expiry dropped out entirely


def test_third_friday_mismatch_is_rejected():
    payload = json.loads(SETTLE.read_text())
    payload["expiries"][0]["expiry_date"] = "2026-05-14"  # a Thursday
    snap = QuoteSnapshot.from_legacy_settlement(
        payload, trade_date=date(2026, 4, 30), spot=6000.0, symbol="X"
    )
    with pytest.raises(AdmissionError) as exc:
        SettlementNormalizer().normalize(snap)
    assert exc.value.reason is AdmissionReason.INVALID_SNAPSHOT
