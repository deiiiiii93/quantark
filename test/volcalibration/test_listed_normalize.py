import json
import math
from datetime import date
from pathlib import Path

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize.listed import (
    ExpirySlice,
    ListedNormalizer,
    OtmQuote,
    iter_expiries,
    otm_implied_vol,
    select_otm,
)
from quantark.volcalibration.snapshot import QuoteSnapshot

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "test/volcalibration/data/mo_snapshot_sample.json"


def _snapshot():
    return QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )


def test_select_otm_splits_at_the_forward():
    sl = ExpirySlice(
        "x",
        "x",
        0.25,
        calls={5000.0: 1050.0, 6000.0: 200.0, 7000.0: 5.0},
        puts={5000.0: 5.0, 6000.0: 200.0, 7000.0: 1050.0},
        volume={(k, s): 10 for k in (5000.0, 6000.0, 7000.0) for s in ("C", "P")},
    )
    picked = select_otm(sl, forward=6000.0)
    by_strike = {q.strike: q.kind for q in picked}
    assert by_strike == {5000.0: "P", 6000.0: "C", 7000.0: "C"}


def test_select_otm_drops_illiquid_and_non_positive():
    sl = ExpirySlice(
        "x",
        "x",
        0.25,
        calls={6000.0: 200.0, 7000.0: 0.0},
        puts={5000.0: 5.0},
        volume={(5000.0, "P"): 0, (6000.0, "C"): 3, (7000.0, "C"): 3},
    )
    picked = select_otm(sl, forward=6000.0, min_volume=1)
    assert [q.strike for q in picked] == [6000.0]  # 5000 illiquid, 7000 zero-priced


def test_otm_put_inverts_through_its_call_equivalent():
    s0, r, q, T = 6000.0, 0.02, 0.01, 0.25
    fwd = s0 * math.exp((r - q) * T)
    df = math.exp(-r * T)
    from quantark.volmodels.black_scholes import bs_call_price

    K = 5600.0
    call = bs_call_price(s0, K, T, 0.23, r, q)
    put = call - df * (fwd - K)
    iv = otm_implied_vol(OtmQuote(K, "P", put), s0, r, q, fwd, df, T)
    assert iv == pytest.approx(0.23, abs=1e-6)


def test_uninvertible_quote_returns_none_not_a_fabricated_vol():
    s0, r, q, T = 6000.0, 0.02, 0.01, 0.25
    fwd = s0 * math.exp((r - q) * T)
    df = math.exp(-r * T)
    assert otm_implied_vol(OtmQuote(6000.0, "C", 9e9), s0, r, q, fwd, df, T) is None


def test_normalizer_builds_a_quoteset_from_the_sample():
    qs = ListedNormalizer().normalize(_snapshot())
    assert qs.convention == "listed_strike"
    assert qs.spot == 6000.0
    assert len(qs.expiries) >= 2
    times = [e.T for e in qs.expiries]
    assert times == sorted(times)
    first = qs.expiries[0]
    assert len(first.nodes) >= 5
    assert all(0.0 < n.iv < 2.0 for n in first.nodes)
    assert first.diagnostics["n_pairs"] >= 5
    atm = min(first.nodes, key=lambda n: abs(n.strike - first.forward))
    assert atm.weight_hint == max(n.weight_hint for n in first.nodes)


def test_normalizer_rejects_a_snapshot_with_too_few_usable_expiries():
    payload = json.loads(LIVE.read_text())
    payload["expiries"] = payload["expiries"][:1]
    snap = QuoteSnapshot.from_legacy_live(
        payload, trade_date=date(2026, 7, 6), symbol="X"
    )
    with pytest.raises(ValidationError, match="usable expiries"):
        ListedNormalizer().normalize(snap)


def test_otm_inversion_recovers_the_generating_smile_of_the_sample_snapshot():
    # sample smile: iv = 0.22 - 0.35 m + 0.6 m^2 + 0.02 T ; r=0.02 q=0.01 S0=6000
    s0, r, q_carry, T = 6000.0, 0.02, 0.01, 0.20
    forward = s0 * math.exp((r - q_carry) * T)
    discount_factor = math.exp(-r * T)
    picked = select_otm(iter_expiries(_snapshot())[1], forward)
    assert picked
    for oq in picked:
        iv = otm_implied_vol(oq, s0, r, q_carry, forward, discount_factor, T)
        m = math.log(oq.strike / s0)
        assert iv == pytest.approx(0.22 - 0.35 * m + 0.6 * m * m + 0.02 * T, abs=2e-3)
