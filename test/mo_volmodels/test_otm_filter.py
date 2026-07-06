import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc  # noqa: E402

SNAP = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def test_otm_selection_splits_at_forward():
    sl = mc.iter_expiries(mc.load_snapshot(SNAP))[1]
    fwd = 6000.0
    otm = mc.select_otm(sl, fwd)
    assert otm, "should select some OTM quotes"
    for q in otm:
        if q.strike < fwd:
            assert q.kind == "P"
        else:
            assert q.kind == "C"
        assert q.price > 0


def test_otm_drops_illiquid():
    sl = mc.ExpirySlice(
        "x",
        0.2,
        calls={6100.0: 5.0, 6200.0: 3.0},
        puts={5900.0: 4.0},
        volume={(6100.0, "C"): 0, (6200.0, "C"): 100, (5900.0, "P"): 100},
    )
    otm = mc.select_otm(sl, 6000.0, min_volume=1)
    strikes = {q.strike for q in otm}
    assert 6100.0 not in strikes  # zero volume dropped
    assert 6200.0 in strikes and 5900.0 in strikes
