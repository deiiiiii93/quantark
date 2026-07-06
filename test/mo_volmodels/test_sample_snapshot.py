import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def test_sample_snapshot_schema():
    if not SNAP.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "example/mo_volmodels/make_sample_snapshot.py")],
            check=True,
        )
    snap = json.loads(SNAP.read_text())
    assert set(snap) >= {"fetched_at", "market_open", "underlying", "expiries"}
    assert snap["underlying"]["code"] == "000852.SH"
    assert snap["underlying"]["spot"] > 0
    assert len(snap["expiries"]) >= 3
    for exp in snap["expiries"]:
        assert exp["T_years"] > 0
        types = {q["type"] for q in exp["quotes"]}
        assert types == {"C", "P"}  # both wings present for parity
        assert all(q["strike"] > 0 and q["last"] > 0 for q in exp["quotes"])
