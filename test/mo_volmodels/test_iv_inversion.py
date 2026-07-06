import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc  # noqa: E402

SNAP = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def test_call_equiv_inversion_recovers_smile():
    # sample smile: iv = 0.22 - 0.35 m + 0.6 m^2 + 0.02 T ; r=0.02 q=0.01 S0=6000
    s0, r, qc, T = 6000.0, 0.02, 0.01, 0.20
    fwd = s0 * math.exp((r - qc) * T)
    DF = math.exp(-r * T)
    sl = mc.iter_expiries(mc.load_snapshot(SNAP))[1]
    for oq in mc.select_otm(sl, fwd):
        iv = mc.otm_implied_vol(oq, s0, r, qc, fwd, DF, T)
        m = math.log(oq.strike / s0)
        expected = 0.22 - 0.35 * m + 0.6 * m * m + 0.02 * T
        assert iv == pytest.approx(expected, abs=2e-3)
