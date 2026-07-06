import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc  # noqa: E402

SNAP = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _slices():
    return mc.iter_expiries(mc.load_snapshot(SNAP))


def test_parity_recovers_rate_and_forward():
    # sample built with r=0.02, q=0.01, S0=6000 -> forward = S0*exp((r-q)*T)
    s0 = 6000.0
    for sl in _slices():
        res = mc.imply_forward_and_rate(sl, s0)
        assert res.r == pytest.approx(0.02, abs=1e-3)
        assert res.q == pytest.approx(0.01, abs=1e-3)
        assert res.forward == pytest.approx(s0 * math.exp((0.02 - 0.01) * sl.T), rel=1e-3)
        assert res.n_pairs >= 5


def test_parity_too_few_pairs():
    sl = mc.ExpirySlice("x", 0.2, {6000.0: 10.0}, {6000.0: 9.0})
    with pytest.raises(ValueError):
        mc.imply_forward_and_rate(sl, 6000.0)
