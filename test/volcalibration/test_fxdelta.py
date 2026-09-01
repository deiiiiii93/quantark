import json
from pathlib import Path

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize.fxdelta import (
    FxDeltaNormalizer,
    spot_delta_from_strike,
    strike_from_spot_delta,
)
from quantark.volcalibration.quotes import QuoteSet
from quantark.volcalibration.snapshot import QuoteSnapshot

SAMPLE = Path("example/fx_volmodels/data/cfets_usdcny_snapshot_20260430.json")


def _snapshot():
    return QuoteSnapshot.from_legacy_fx(json.loads(SAMPLE.read_text(encoding="utf-8")))


def test_delta_and_strike_round_trip():
    forward, iv, maturity, rf = 7.15, 0.045, 0.25, 0.045
    for delta in (-0.10, -0.25, 0.25, 0.10):
        strike = strike_from_spot_delta(forward, iv, maturity, rf, delta)
        recovered = spot_delta_from_strike(
            forward, strike, iv, maturity, rf, is_call=delta > 0.0
        )
        assert recovered == pytest.approx(delta, abs=1e-12)


@pytest.mark.parametrize("delta", [1.5, 0.0, -1.0, float("nan")])
def test_an_impossible_delta_is_refused(delta):
    with pytest.raises(ValidationError):
        strike_from_spot_delta(7.15, 0.045, 0.25, 0.045, delta)


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_the_fx_lifter_declares_a_vol_quoted_snapshot():
    snapshot = _snapshot()
    assert snapshot.convention == "fx_delta"
    assert snapshot.price_field == "mid_iv"
    assert snapshot.symbol == "USD.CNY"
    with pytest.raises(ValidationError, match="price_field_mismatch"):
        snapshot.quote_price({"mid_iv": 0.05})


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_the_fx_normalizer_emits_a_structurally_identical_quote_set():
    quotes = FxDeltaNormalizer(tenor_set="core").normalize(_snapshot())
    assert isinstance(quotes, QuoteSet)
    assert len(quotes.expiries) == 6
    for slice_ in quotes.expiries:
        assert slice_.expiry_label
        assert len(slice_.nodes) == 5
        assert slice_.forward > 0.0
        assert 0.0 < slice_.discount_factor <= 1.0
        assert [n.strike for n in slice_.nodes] == sorted(
            n.strike for n in slice_.nodes
        )
    assert quotes.universe["node_count"] == 30
    assert quotes.universe["delta_round_trips_verified"] > 0


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_the_recorded_carry_reproduces_the_published_forward():
    """A consumer rebuilding curves from the artifact must land on the forward
    the artifact records.  foreign_rate would miss it by ~1 pip at 1M."""
    import math

    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    snapshot = QuoteSnapshot.from_legacy_fx(payload)
    quotes = FxDeltaNormalizer(tenor_set="core").normalize(snapshot)
    published = {
        row["tenor"]: row for row in payload["slices"]
    }
    for slice_ in quotes.expiries:
        row = published[slice_.expiry_label]
        rebuilt = snapshot.spot * math.exp((slice_.r - slice_.q) * slice_.T)
        assert rebuilt == pytest.approx(float(row["forward"]), abs=1e-12)
        # ... and it is not simply the published deposit rate
        assert slice_.diagnostics["published_deposit_rate"] == float(
            row["foreign_rate"]
        )


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_an_internally_inconsistent_slice_is_refused():
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for row in payload["slices"]:
        if row["tenor"] == "3M":
            row["pricing_foreign_rate"] = float(row["pricing_foreign_rate"]) + 1e-3
    with pytest.raises(ValidationError, match="internally inconsistent"):
        FxDeltaNormalizer(tenor_set="core").normalize(
            QuoteSnapshot.from_legacy_fx(payload)
        )


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_a_tampered_strike_fails_the_round_trip_check():
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for row in payload["slices"]:
        if row["tenor"] == "3M":
            row["quotes"][0]["strike"] *= 1.01
    snapshot = QuoteSnapshot.from_legacy_fx(payload)
    with pytest.raises(ValidationError, match="published strike implies spot delta"):
        FxDeltaNormalizer(tenor_set="core").normalize(snapshot)


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_a_slice_without_the_delta_rate_is_refused_not_skipped():
    """The round trip is the FX parity gate; it must not be waivable."""
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for row in payload["slices"]:
        if row["tenor"] == "3M":
            row.pop("effective_foreign_rate_for_delta", None)
    with pytest.raises(ValidationError, match="cannot be verified"):
        FxDeltaNormalizer(tenor_set="core").normalize(
            QuoteSnapshot.from_legacy_fx(payload)
        )


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_the_fx_snapshot_has_one_stable_identity():
    """Per-pillar digests fold into one, or the resume rule cannot tell two
    CFETS snapshots apart."""
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    first = QuoteSnapshot.from_legacy_fx(payload).sha256
    assert first and len(first) == 64
    assert QuoteSnapshot.from_legacy_fx(payload).sha256 == first

    changed = json.loads(SAMPLE.read_text(encoding="utf-8"))
    changed["provenance"]["payload_sha256"]["ATM"] = "0" * 64
    assert QuoteSnapshot.from_legacy_fx(changed).sha256 != first


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_an_unknown_tenor_set_is_refused():
    with pytest.raises(ValidationError, match="tenor_set"):
        FxDeltaNormalizer(tenor_set="nonsense").normalize(_snapshot())


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_a_listed_snapshot_is_refused():
    from datetime import date

    listed = QuoteSnapshot.from_legacy_settlement(
        json.loads(
            Path(
                "example/mo_volmodels/data/mo_settlement_snapshot_20260430.json"
            ).read_text(encoding="utf-8")
        ),
        trade_date=date(2026, 4, 30),
        spot=8381.947,
        symbol="000852.SH",
    )
    with pytest.raises(ValidationError, match="fx_delta"):
        FxDeltaNormalizer().normalize(listed)


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_show_emits_valid_json_for_a_surface_with_no_parity_diagnostics(tmp_path):
    """FX artifacts carry no put-call-parity residual; NaN is not JSON."""
    from quantark.volcalibration.cli import _expiry_diagnostics, _emit
    from quantark.volcalibration.surface import build_artifact

    snapshot = _snapshot()
    quotes = FxDeltaNormalizer(tenor_set="core").normalize(snapshot)
    artifact = build_artifact(quotes, snapshot)
    rows = _expiry_diagnostics(artifact)
    assert rows and rows[0]["parity_rmse_points"] is None
    assert rows[0]["parity_rmse_over_forward"] is None
    # json.dump(allow_nan=False) raises on NaN; this must not.
    json.dumps({"per_expiry": rows}, allow_nan=False)


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_one_end_to_end_fx_surface_builds_and_admits():
    from quantark.volcalibration.surface import build_artifact

    snapshot = _snapshot()
    quotes = FxDeltaNormalizer(tenor_set="core").normalize(snapshot)
    artifact = build_artifact(quotes, snapshot)
    assert len(artifact["maturities"]) == 6
    assert artifact["price_field"] == "mid_iv"
    assert artifact["admission"]["static_arbitrage_validation"]
    assert len(artifact["strikes"]) >= 3
