"""Contract tests for the offline CFETS USD/CNY snapshot boundary."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example" / "fx_volmodels"
sys.path.insert(0, str(EXAMPLE))

import _fx_common as fx  # noqa: E402


def _load_numbered_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load_numbered_module("01_fetch_cfets_snapshot.py", "fx_fetch_snapshot_contract")


_DAYS = {
    "1D": 1,
    "1W": 7,
    "2W": 14,
    "3W": 21,
    "1M": 30,
    "2M": 60,
    "3M": 90,
    "6M": 180,
    "9M": 270,
    "1Y": 365,
    "18M": 548,
    "2Y": 730,
    "3Y": 1095,
}


def _payload(records: list[dict], *, error: str | None = None) -> dict:
    payload = {"head": {"rep_code": "200"}, "records": records}
    if error is not None:
        payload["data"] = {"error": error}
    return payload


def _synthetic_cfets_replies() -> tuple[dict[str, dict], dict]:
    spot = 7.0
    domestic_rate = 0.03
    foreign_rate = 0.02
    atm_iv = 0.04
    delta_records = []
    curve_records = {pillar: [] for pillar in fx.PILLAR_ORDER}

    for tenor in fx.TENOR_ORDER:
        days = _DAYS[tenor]
        maturity = days / 365.0
        forward = spot * math.exp((domestic_rate - foreign_rate) * maturity)
        foreign_df = math.exp(-foreign_rate * maturity)
        # ATMF has d1 = 0.5 sigma sqrt(T).  CFETS spot call and put
        # deltas must differ by exactly the USD discount factor.
        atm_call = foreign_df * NormalDist().cdf(0.5 * atm_iv * math.sqrt(maturity))
        delta_records.append(
            {
                "tenor": tenor,
                "expiryDays": str(days),
                "strikePrice": f"{forward:.12f}",
                "ccy2Rate": f"{domestic_rate * 100:.10f}",
                "ccy1Rate": f"{foreign_rate * 100:.10f}",
                "spotPrice": f"{spot:.10f}",
                "callDelta": f"{atm_call:.12f}",
                "putDelta": f"{atm_call - foreign_df:.12f}",
                "expiryDateCn": "2026-12-31",
                "swapRate": f"{(forward - spot) * 10_000:.8f}",
            }
        )
        for pillar in fx.PILLAR_ORDER:
            bid, mid, ask = 3.9, 4.0, 4.1
            if tenor == "1M" and pillar == "25P":
                # CFETS public composites can publish a mid outside the
                # displayed bid/ask.  It is evidence, not an executable book.
                bid, mid, ask = 3.9, 4.2, 4.1
            curve_records[pillar].append(
                {
                    "tenor": tenor,
                    "bidVolatilityStr": str(bid),
                    "midVolatilityStr": str(mid),
                    "askVolatilityStr": str(ask),
                }
            )

    curves = {pillar: _payload(records) for pillar, records in curve_records.items()}
    return curves, _payload(delta_records)


@pytest.mark.parametrize("delta", [-0.10, -0.25, 0.25, 0.10])
def test_spot_delta_identity_and_exact_strike_roundtrip(delta: float) -> None:
    forward = 7.08
    maturity = 0.75
    foreign_rate = 0.021
    iv = 0.047
    strike = fx.strike_from_spot_delta(forward, iv, maturity, foreign_rate, delta)
    recovered = fx.spot_delta_from_strike(
        forward,
        strike,
        iv,
        maturity,
        foreign_rate,
        is_call=delta > 0.0,
    )
    assert recovered == pytest.approx(delta, abs=2e-14)

    call = fx.spot_delta_from_strike(
        forward, strike, iv, maturity, foreign_rate, is_call=True
    )
    put = fx.spot_delta_from_strike(
        forward, strike, iv, maturity, foreign_rate, is_call=False
    )
    assert call - put == pytest.approx(math.exp(-foreign_rate * maturity), abs=2e-15)


def test_standardisation_keeps_atmf_and_non_executable_public_band(tmp_path: Path) -> None:
    curves, delta = _synthetic_cfets_replies()
    snapshot = fetch.standardise_snapshot("2026-07-20", "16:00", curves, delta)

    for row in snapshot["slices"]:
        atm = next(quote for quote in row["quotes"] if quote["pillar"] == "ATM")
        assert atm["strike"] == row["forward"]
        assert row["atm_call_delta"] - row["atm_put_delta"] == pytest.approx(
            row["foreign_discount_factor_from_deltas"], abs=1e-15
        )

    one_month = next(row for row in snapshot["slices"] if row["tenor"] == "1M")
    published = next(quote for quote in one_month["quotes"] if quote["pillar"] == "25P")
    assert published["mid_iv"] > published["ask_iv"]
    assert published["mid_outside_displayed_band"] is True

    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    loaded = fx.load_snapshot(path)
    loaded_quote = next(
        quote
        for row in loaded["slices"]
        if row["tenor"] == "1M"
        for quote in row["quotes"]
        if quote["pillar"] == "25P"
    )
    assert loaded_quote["mid_iv"] == published["mid_iv"]
    assert loaded_quote["mid_outside_displayed_band"] is True


def test_snapshot_rejects_noncanonical_pillar_order(tmp_path: Path) -> None:
    curves, delta = _synthetic_cfets_replies()
    snapshot = fetch.standardise_snapshot("2026-07-20", "16:00", curves, delta)
    broken = copy.deepcopy(snapshot)
    broken["slices"][0]["quotes"][0], broken["slices"][0]["quotes"][1] = (
        broken["slices"][0]["quotes"][1],
        broken["slices"][0]["quotes"][0],
    )
    path = tmp_path / "bad-order.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="quotes must follow"):
        fx.load_snapshot(path)


def test_cfets_backend_error_is_rejected_even_with_http_success() -> None:
    response = _payload([{"tenor": "1M"}], error="temporary backend failure")
    with pytest.raises(RuntimeError, match="backend error"):
        fetch._validate_payload(response, "synthetic")


def test_legacy_raw_delta_filename_replays_without_renaming(tmp_path: Path) -> None:
    date = "2026-07-20"
    legacy = tmp_path / f"cfets_delta_{date}.json"
    expected = _payload([{"tenor": "1M"}])
    legacy.write_text(json.dumps(expected), encoding="utf-8")
    assert fetch._load_raw(tmp_path, date, "delta") == expected
