import json
import math
from datetime import date
from pathlib import Path

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize.listed import (
    ExpirySlice,
    imply_forward_and_rate,
    iter_expiries,
)
from quantark.volcalibration.snapshot import QuoteSnapshot

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _snapshot():
    return QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )


def test_parity_recovers_rate_and_forward():
    # sample built with r=0.02, q=0.01, S0=6000 -> F = S0*exp((r-q)*T)
    s0 = 6000.0
    snap = _snapshot()
    slices = iter_expiries(snap)
    assert len(slices) == 4
    for sl in slices:
        res = imply_forward_and_rate(sl, s0)
        assert res.r == pytest.approx(0.02, abs=1e-3)
        assert res.q == pytest.approx(0.01, abs=1e-3)
        assert res.forward == pytest.approx(
            s0 * math.exp((0.02 - 0.01) * sl.T), rel=1e-3
        )
        assert res.n_pairs >= 5
        assert res.rmse_over_forward < 0.01


def test_parity_too_few_pairs():
    sl = ExpirySlice("x", "x", 0.2, {6000.0: 10.0}, {6000.0: 9.0}, {})
    with pytest.raises(ValidationError, match="paired strikes"):
        imply_forward_and_rate(sl, 6000.0)


def test_parity_rejects_non_positive_discount_factor():
    # C - P increasing in K implies a negative discount factor: arbitrage.
    calls = {5000.0: 1.0, 6000.0: 5.0, 7000.0: 9.0}
    puts = {5000.0: 9.0, 6000.0: 5.0, 7000.0: 1.0}
    sl = ExpirySlice("x", "x", 0.5, calls, puts, {})
    with pytest.raises(ValidationError, match="discount factor"):
        imply_forward_and_rate(sl, 6000.0)


def test_parity_rate_gate_rejects_absurd_rate():
    # DF far below 1 at short T => |implied rate| >> 10%
    calls = {5000.0: 600.0, 6000.0: 100.0, 7000.0: 10.0}
    puts = {5000.0: 10.0, 6000.0: 100.0, 7000.0: 600.0}
    sl = ExpirySlice("x", "x", 0.02, calls, puts, {})
    with pytest.raises(ValidationError, match="implied rate"):
        imply_forward_and_rate(sl, 6000.0)
