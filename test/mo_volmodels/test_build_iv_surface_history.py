"""Contracts for the CFFEX settlement IV-surface history build (stage 03-history).

Offline: real frozen settlement CSVs from ``data/history/settlement_csv`` plus
tiny synthetic GB18030 fixtures; no akshare, no network.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quantark.util.exceptions import NumericalError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example" / "mo_volmodels"
sys.path.insert(0, str(EXAMPLE))


def _load_numbered(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage03h = _load_numbered(
    "03_build_iv_surface_history.py", "mo_iv_surface_history_contract"
)

REAL_CSV_DIR = EXAMPLE / "data" / "history" / "settlement_csv"
SPOT_CSV = EXAMPLE / "data" / "history" / "csi1000_spot.csv"
SCRIPT = EXAMPLE / "03_build_iv_surface_history.py"
SMOKE_DATE = "20230504"

pytestmark = pytest.mark.skipif(
    not (REAL_CSV_DIR / f"{SMOKE_DATE}_1.csv").is_file(),
    reason="frozen settlement CSVs not present",
)


def _csv_bytes(rows: list[str]) -> bytes:
    header = (
        "合约代码,今开盘,最高价,最低价,成交量,成交金额,持仓量,持仓变化,"
        "今收盘,今结算,前结算,涨跌1,涨跌2,Delta"
    )
    return ("\n".join([header, *rows]) + "\n").encode("gb18030")


def _arb_violating_csv() -> bytes:
    """Calls strictly INCREASING in strike -> non-positive parity discount factor."""
    rows = []
    for i, strike in enumerate((6000, 6200, 6400, 6600, 6800)):
        call = 100.0 + 10.0 * i  # arbitrage: call prices rise with strike
        put = 50.0
        rows.append(
            f"MO2608-C-{strike},{call},{call},{call},500,1,300,0,"
            f"{call},{call},{call},0,0,0.5"
        )
        rows.append(
            f"MO2608-P-{strike},{put},{put},{put},500,1,300,0,"
            f"{put},{put},{put},0,0,-0.5"
        )
    return _csv_bytes(rows)


def _run_cli(csv_dir: Path, output_dir: Path, manifest: Path, *extra: str):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--csv-dir",
            str(csv_dir),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
            "--spot-csv",
            str(SPOT_CSV),
            *extra,
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def one_date_csv_dir(tmp_path: Path) -> Path:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    shutil.copy(REAL_CSV_DIR / f"{SMOKE_DATE}_1.csv", csv_dir / f"{SMOKE_DATE}_1.csv")
    return csv_dir


def test_artifact_schema_and_atm_pillars() -> None:
    artifact = stage03h.build_surface_artifact(
        SMOKE_DATE,
        csv_dir=REAL_CSV_DIR,
        s0=6733.969,
    )
    required = {
        "schema_version",
        "trade_date",
        "source_class",
        "price_field",
        "source_sha256",
        "s0",
        "strikes",
        "maturities",
        "iv_grid",
        "per_expiry",
        "atm_pillars",
        "target_smoothing",
        "extrapolation_policy",
        "admission",
    }
    assert required <= set(artifact)
    assert artifact["source_class"] == "official_cffex_eod_settlement"
    assert artifact["trade_date"] == "2023-05-04"

    strikes = artifact["strikes"]
    maturities = artifact["maturities"]
    assert all(strikes[i] < strikes[i + 1] for i in range(len(strikes) - 1))
    assert all(maturities[i] < maturities[i + 1] for i in range(len(maturities) - 1))
    grid = np.asarray(artifact["iv_grid"], dtype=float)
    assert grid.shape == (len(maturities), len(strikes))
    assert np.all(np.isfinite(grid)) and np.all(grid > 0.0)

    pillars = artifact["atm_pillars"]
    assert len(pillars) == len(artifact["per_expiry"]) >= 2
    from quantark.param.vol.sabr.hagan import sabr_implied_vol_black
    from quantark.util.numerical import is_close

    by_expiry = {pe["expiry_date"]: pe for pe in artifact["per_expiry"]}
    for pillar in pillars:
        pe = by_expiry[pillar["expiry_date"]]
        assert pillar["T"] == pe["T"]
        params = pe["sabr_params"]
        expected = float(
            sabr_implied_vol_black(
                pe["forward"],
                [pe["forward"]],
                [pe["T"]],
                params["alpha"],
                params["beta"],
                params["rho"],
                params["nu"],
                shift=params["shift"],
            )[0]
        )
        assert math.isfinite(pillar["atm_vol"]) and pillar["atm_vol"] > 0.0
        assert is_close(pillar["atm_vol"], expected)
        # raw nodes are preserved beside the smoothed ones
        assert len(pe["raw_points"]) == len(pe["points"]) >= 5

    policy = artifact["extrapolation_policy"]
    assert policy["beyond_last_listed_expiry"] == "flat_total_variance"
    assert policy["max_listed_T"] == max(maturities)

    # deterministic serialization
    first = stage03h.serialize_artifact(artifact)
    again = stage03h.build_surface_artifact(SMOKE_DATE, csv_dir=REAL_CSV_DIR, s0=6733.969)
    second = stage03h.serialize_artifact(again)
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_fail_closed_exclusion(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csv"
    out_dir = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    csv_dir.mkdir()
    # garbage: not a CFFEX statistics file at all
    (csv_dir / "20260720_1.csv").write_bytes(b"\xff\xfe not a csv \xff")
    # arb-violating: synthetic smile with calls increasing in strike
    (csv_dir / "20260721_1.csv").write_bytes(_arb_violating_csv())

    _run_cli(csv_dir, out_dir, manifest)

    assert not list(out_dir.glob("*.json")), "excluded dates must get no artifact"
    records = {
        record["date"]: record
        for record in json.loads(manifest.read_text())["records"]
    }
    assert records["20260720"]["status"] == "excluded"
    assert records["20260720"]["reason"] == "parse_failed"
    assert records["20260720"]["artifact_sha256"] is None
    assert records["20260721"]["status"] == "excluded"
    assert records["20260721"]["reason"] == "parity_gating_failed"
    assert records["20260721"]["artifact_sha256"] is None


def test_missing_spot_is_excluded_not_fabricated(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csv"
    out_dir = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    spot_csv = tmp_path / "spot.csv"
    csv_dir.mkdir()
    shutil.copy(REAL_CSV_DIR / f"{SMOKE_DATE}_1.csv", csv_dir / f"{SMOKE_DATE}_1.csv")
    spot_csv.write_text("date,spot\n1999-01-04,1000.0\n", encoding="utf-8")

    _run_cli(csv_dir, out_dir, manifest, "--spot-csv", str(spot_csv))

    assert not (out_dir / f"mo_iv_surface_{SMOKE_DATE}.json").exists()
    record = json.loads(manifest.read_text())["records"][0]
    assert record["status"] == "excluded"
    assert record["reason"] == "missing_spot"


def test_resume_skips_existing_artifact(one_date_csv_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    _run_cli(one_date_csv_dir, out_dir, manifest)
    artifact = out_dir / f"mo_iv_surface_{SMOKE_DATE}.json"
    assert artifact.is_file()
    first_bytes = artifact.read_bytes()

    artifact.write_bytes(b'{"tampered": true}\n')
    result = _run_cli(one_date_csv_dir, out_dir, manifest)
    assert "skip" in result.stdout
    assert artifact.read_bytes() == b'{"tampered": true}\n', (
        "resume must skip, not rebuild, an existing artifact"
    )

    _run_cli(one_date_csv_dir, out_dir, manifest, "--force")
    assert artifact.read_bytes() == first_bytes, "force rebuild must be deterministic"


def test_manifest_schema_and_atomic_write(tmp_path: Path) -> None:
    records = {
        "20230504": {
            "date": "20230504",
            "status": "ok",
            "reason": None,
            "detail": None,
            "n_expiries": 6,
            "artifact_sha256": "ab" * 32,
        }
    }
    manifest = tmp_path / "sub" / "surface_manifest.json"
    stage03h.save_manifest(manifest, records, window={"start": "20230504", "end": "20230504"})
    payload = json.loads(manifest.read_text())
    assert payload["schema_version"] == 1
    assert payload["source"] == "official_cffex_eod_settlement"
    assert payload["gap_policy"] == "consumers carry forward previous admitted surface"
    assert payload["window"] == {"start": "20230504", "end": "20230504"}
    assert isinstance(payload["generated_at"], str)
    assert payload["records"] == [records["20230504"]]
    # atomic write leaves no temp debris
    assert not list(manifest.parent.glob("*.tmp"))

    # simulated crash during serialization: previous manifest stays intact
    before = manifest.read_bytes()

    class _Boom:
        def __iter__(self):
            raise RuntimeError("simulated crash")

    with pytest.raises(TypeError):
        stage03h.save_manifest(
            manifest, {"x": {"date": _Boom()}}, window={"start": "x", "end": "x"}
        )
    assert manifest.read_bytes() == before
    assert not list(manifest.parent.glob("*.tmp"))


def test_cli_end_to_end_ok_and_determinism(one_date_csv_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    result = _run_cli(one_date_csv_dir, out_dir, manifest, "--workers", "2")
    assert f"{SMOKE_DATE}: ok" in result.stdout
    record = json.loads(manifest.read_text())["records"][0]
    assert record["status"] == "ok"
    assert record["n_expiries"] >= 2
    digest = hashlib.sha256(
        (out_dir / f"mo_iv_surface_{SMOKE_DATE}.json").read_bytes()
    ).hexdigest()
    assert record["artifact_sha256"] == digest


# ---------------------------------------------------------------------------
# _validate_static_arbitrage: direct synthetic-grid contracts
# ---------------------------------------------------------------------------

_SYNTH_STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]


def _synthetic_surface(maturities, iv_grid, *, s0=100.0, r=0.02, q=0.01):
    """Minimal surface dict as consumed by _validate_static_arbitrage."""
    return {
        "s0": s0,
        "strikes": list(_SYNTH_STRIKES),
        "maturities": [float(t) for t in maturities],
        "iv_grid": [list(row) for row in iv_grid],
        "per_expiry": [
            {
                "expiry_date": "2023-06-16",
                "T": float(t),
                "r": r,
                "q": q,
                "forward": s0 * math.exp((r - q) * t),
                "df": math.exp(-r * t),
            }
            for t in maturities
        ],
    }


def test_validate_two_maturity_admits_clean_grid() -> None:
    smile_lo = [0.25, 0.22, 0.20, 0.22, 0.25]
    smile_hi = [0.26, 0.23, 0.21, 0.23, 0.26]  # total variance rising in T
    surface = _synthetic_surface([0.1, 0.2], [smile_lo, smile_hi])
    method = stage03h._validate_static_arbitrage(surface)
    assert method == "reduced_form_dupire_checks_2_maturities"


def test_validate_two_maturity_calendar_rejection() -> None:
    # iv drops fast enough that total variance DECREASES across the 2 rows.
    surface = _synthetic_surface([0.1, 0.2], [[0.30] * 5, [0.20] * 5])
    with pytest.raises(NumericalError, match="calendar arbitrage"):
        stage03h._validate_static_arbitrage(surface)


def test_validate_two_maturity_butterfly_rejection() -> None:
    # Identical rows (calendar-clean) with an inverted spike -> negative butterfly.
    kinked = [0.10, 0.10, 0.60, 0.10, 0.10]
    surface = _synthetic_surface([0.1, 0.2], [kinked, kinked])
    with pytest.raises(NumericalError, match="butterfly arbitrage"):
        stage03h._validate_static_arbitrage(surface)


def test_validate_three_maturity_path_admits_and_rejects() -> None:
    rows = [
        [0.25, 0.22, 0.20, 0.22, 0.25],
        [0.26, 0.23, 0.21, 0.23, 0.26],
        [0.27, 0.24, 0.22, 0.24, 0.27],
    ]
    clean = _synthetic_surface([0.1, 0.2, 0.3], rows)
    method = stage03h._validate_static_arbitrage(clean)
    assert method == "build_dupire_local_vol(validate_arbitrage=True)"

    # Middle row dips below the front row in total variance -> build_dupire_local_vol
    # itself raises the calendar rejection the manifest records as static_arbitrage.
    bad = _synthetic_surface([0.1, 0.2, 0.3], [[0.30] * 5, [0.20] * 5, [0.28] * 5])
    with pytest.raises(NumericalError, match="calendar arbitrage"):
        stage03h._validate_static_arbitrage(bad)


def test_static_arbitrage_exclusion_reason_is_pinned(monkeypatch) -> None:
    def _boom(surface):
        raise NumericalError("calendar arbitrage: synthetic")

    monkeypatch.setattr(stage03h, "_validate_static_arbitrage", _boom)
    with pytest.raises(stage03h.AdmissionError) as excinfo:
        stage03h.build_surface_artifact(SMOKE_DATE, csv_dir=REAL_CSV_DIR, s0=6733.969)
    assert excinfo.value.reason == "static_arbitrage"


def test_force_reexclusion_deletes_stale_artifact(
    one_date_csv_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    artifact = out_dir / f"mo_iv_surface_{SMOKE_DATE}.json"
    _run_cli(one_date_csv_dir, out_dir, manifest)
    assert artifact.is_file()

    # Same date now fails admission under --force: the stale artifact must go.
    (one_date_csv_dir / f"{SMOKE_DATE}_1.csv").write_bytes(b"\xff\xfe garbage \xff")
    _run_cli(one_date_csv_dir, out_dir, manifest, "--force")
    assert not artifact.exists(), "re-excluded date must not keep a stale artifact"
    record = json.loads(manifest.read_text())["records"][0]
    assert record["status"] == "excluded"
    assert record["reason"] == "parse_failed"
    assert record["artifact_sha256"] is None
