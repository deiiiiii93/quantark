import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc  # noqa: E402

SNAP = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def test_load_and_iter():
    snap = mc.load_snapshot(SNAP)
    assert snap["underlying"]["spot"] == 6000.0
    slices = mc.iter_expiries(snap)
    assert len(slices) >= 3
    s0 = slices[0]
    assert s0.T > 0
    # paired strikes present in both call and put maps
    common = set(s0.calls) & set(s0.puts)
    assert len(common) >= 5


def test_load_missing_key(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"underlying": {}}))
    with pytest.raises(ValueError):
        mc.load_snapshot(bad)
