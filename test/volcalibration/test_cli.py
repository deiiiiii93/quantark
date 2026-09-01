import json
from datetime import date
from pathlib import Path

import pytest

from quantark.volcalibration.cli import main
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.store import (
    StoreLayout,
    atomic_write_json,
    surface_record,
)

SAMPLE = Path("example/mo_volmodels/data/mo_settlement_snapshot_20260430.json")

CONFIG_YAML = """
schema_version: 1
name: test-mo
underlying:
  symbol: "000852.SH"
  convention: listed_strike
  price_field: settlement
paths:
  root: history
calibration:
  variants: [localvol]
"""


def _write_snapshot(layout, tag, trade_date, spot):
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    snap = QuoteSnapshot.from_legacy_settlement(
        payload,
        trade_date=trade_date,
        spot=spot,
        symbol="000852.SH",
        source_sha256=payload["source_sha256"],
        source_url=payload.get("source_url"),
    )
    layout.snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.snapshot_path(tag).write_text(json.dumps(snap.to_payload()), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path):
    """A config file plus a store holding one snapshot."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(CONFIG_YAML, encoding="utf-8")
    layout = StoreLayout(tmp_path / "history", tmp_path / "history")
    _write_snapshot(layout, "20260430", date(2026, 4, 30), 8381.947)
    return config_path, layout


def test_json_stdout_is_exactly_one_object(workspace, capsys):
    config_path, _ = workspace
    code = main(["status", str(config_path), "--as-of", "2026-04-30", "--json"])
    out, err = capsys.readouterr()
    payload = json.loads(out)  # parses whole: no log line may leak into stdout
    assert payload["module"] == "quantark.volcalibration"
    assert code == 2  # nothing built yet
    assert isinstance(err, str)


def test_plan_writes_nothing(workspace, capsys):
    config_path, layout = workspace
    history = layout.history_dir
    before = {p: p.stat().st_mtime_ns for p in history.rglob("*") if p.is_file()}
    code = main(["run", str(config_path), "--as-of", "2026-04-30", "--plan", "--json"])
    out, _ = capsys.readouterr()
    plan = json.loads(out)
    assert plan["surfaces_to_build"] == ["20260430"]
    assert plan["calibrations_to_run"] == ["20260430"]
    assert code == 2
    after = {p: p.stat().st_mtime_ns for p in history.rglob("*") if p.is_file()}
    assert after == before
    assert not layout.surface_dir.exists()
    assert not layout.surface_manifest.exists()


def test_run_is_idempotent_and_reports_current(workspace, capsys):
    config_path, layout = workspace
    first = main(["run", str(config_path), "--as-of", "2026-04-30"])
    capsys.readouterr()
    artifact = layout.artifact_path("20260430")
    bytes_before = artifact.read_bytes()
    second = main(["run", str(config_path), "--as-of", "2026-04-30", "--json"])
    out, _ = capsys.readouterr()
    status = json.loads(out)
    assert first == 0 and second == 0
    assert status["overall_status"] == "current"
    assert artifact.read_bytes() == bytes_before


def test_a_plan_after_a_run_is_empty_and_exits_zero(workspace, capsys):
    config_path, _ = workspace
    main(["run", str(config_path), "--as-of", "2026-04-30"])
    capsys.readouterr()
    code = main(["run", str(config_path), "--as-of", "2026-04-30", "--plan", "--json"])
    out, _ = capsys.readouterr()
    plan = json.loads(out)
    assert plan["surfaces_to_build"] == []
    assert plan["calibrations_to_run"] == []
    assert code == 0


def test_an_excluded_expected_date_exits_two_not_one(tmp_path, capsys):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(CONFIG_YAML, encoding="utf-8")
    layout = StoreLayout(tmp_path / "history", tmp_path / "history")
    layout.snapshots_dir.mkdir(parents=True)
    (layout.snapshots_dir / "20240930.json").write_text("{}")
    atomic_write_json(
        layout.surface_manifest,
        {
            "schema_version": 1,
            "records": [
                surface_record(
                    "20240930",
                    status="excluded",
                    reason="insufficient_expiries",
                    n_expiries=1,
                )
            ],
        },
    )
    code = main(["status", str(config_path), "--as-of", "2024-09-30", "--json"])
    out, _ = capsys.readouterr()
    assert json.loads(out)["overall_status"] == "surface_excluded"
    assert code == 2, "an excluded date is 'not yet', not a failure to retry-loop on"


def test_a_locked_store_exits_75(workspace, capsys):
    config_path, layout = workspace
    from quantark.volcalibration.runner import acquire_lock

    with acquire_lock(layout.lock):
        code = main(["run", str(config_path), "--as-of", "2026-04-30", "--json"])
    out, _ = capsys.readouterr()
    assert code == 75
    assert json.loads(out)["reason"] == "locked"


def test_an_unreadable_config_exits_one(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 9\n", encoding="utf-8")
    code = main(["status", str(bad), "--json"])
    out, err = capsys.readouterr()
    assert code == 1
    assert json.loads(out)["reason"] == "invalid_config"
    assert "schema_version" in err


def test_show_reports_one_date_s_verdict_and_diagnostics(workspace, capsys):
    config_path, _ = workspace
    main(["run", str(config_path), "--as-of", "2026-04-30"])
    capsys.readouterr()
    code = main(["show", str(config_path), "--date", "2026-04-30", "--json"])
    out, _ = capsys.readouterr()
    payload = json.loads(out)
    assert code == 0
    assert payload["surface"]["status"] == "ok"
    assert payload["surface"]["admission"]["static_arbitrage_validation"]
    per_expiry = payload["surface"]["per_expiry"]
    assert len(per_expiry) >= 2
    assert per_expiry[0]["parity_rmse_over_forward"] > 0.0
    assert per_expiry[0]["sabr_fit_rmse_vol_points"] > 0.0
    assert payload["calibration"]["variants"]["localvol"]["status"] == "ok"
    assert payload["surface"]["node_universe"]["node_count"] > 0


def test_show_reports_an_unbuilt_date_as_non_current(workspace, capsys):
    config_path, _ = workspace
    code = main(["show", str(config_path), "--date", "2026-04-30", "--json"])
    out, _ = capsys.readouterr()
    payload = json.loads(out)
    assert code == 2
    assert payload["surface"]["status"] == "absent"


def test_list_without_a_config_reports_discoverable_configs(capsys):
    code = main(["list", "--json"])
    out, _ = capsys.readouterr()
    assert code == 0
    assert isinstance(json.loads(out)["configs"], list)


def test_list_with_a_config_reports_per_date_rows(workspace, capsys):
    config_path, _ = workspace
    main(["run", str(config_path), "--as-of", "2026-04-30"])
    capsys.readouterr()
    code = main(["list", "--config", str(config_path), "--json"])
    out, _ = capsys.readouterr()
    rows = json.loads(out)["dates"]
    assert code == 0
    assert rows[0]["date"] == "20260430"
    assert rows[0]["surface_status"] == "ok"
    assert rows[0]["calibrated_variants"] == ["localvol"]


def test_human_output_carries_no_json(workspace, capsys):
    config_path, _ = workspace
    main(["status", str(config_path), "--as-of", "2026-04-30"])
    out, _ = capsys.readouterr()
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
    assert "surface_pending" in out
