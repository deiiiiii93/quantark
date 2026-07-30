"""Contracts for the bulk CFFEX settlement downloader (offline, no akshare)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example" / "mo_volmodels"
sys.path.insert(0, str(EXAMPLE))


def _load_numbered(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bulk = _load_numbered(
    "01_bulk_fetch_settlement_history.py", "mo_bulk_settlement_contract"
)

DATES = ["20260715", "20260716", "20260717"]
PAYLOAD = "合约代码,今收盘,今结算\nMO2608-C-6400,681.2,681.4\n".encode("gb18030")
REQUIRED_RECORD_KEYS = {"date", "status", "sha256", "bytes", "attempts", "reason"}


def _ok_download(payload: bytes = PAYLOAD):
    def fake(trade_date: str):
        return payload, bulk.official_url(trade_date)

    return fake


def _not_found_download(trade_date: str):
    raise urllib.error.HTTPError(
        bulk.official_url(trade_date), 404, "Not Found", hdrs=None, fp=None
    )


def _run(tmp_path: Path, dates, download, **overrides) -> dict[str, dict]:
    raw_dir = tmp_path / "raw"
    manifest = tmp_path / "settlement_manifest.json"
    options = {
        "raw_dir": raw_dir,
        "manifest_path": manifest,
        "delay": 0.0,
        "sleep": lambda _seconds: None,
        "download": download,
        "verbose": False,
    }
    options.update(overrides)
    return bulk.process_dates(dates, **options)


def test_resume_skips_dates_with_existing_csv(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    seeded = b"pre-existing official bytes"
    bulk.csv_path(raw_dir, "20260715").write_bytes(seeded)
    calls: list[str] = []

    def tracking_download(trade_date: str):
        calls.append(trade_date)
        return PAYLOAD, bulk.official_url(trade_date)

    monkeypatch.setattr(bulk, "download_one", tracking_download)
    records = _run(tmp_path, DATES, download=None)

    assert calls == ["20260716", "20260717"]
    assert bulk.csv_path(raw_dir, "20260715").read_bytes() == seeded
    assert bulk.csv_path(raw_dir, "20260716").read_bytes() == PAYLOAD
    assert records["20260716"]["status"] == "ok"
    assert records["20260716"]["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    # the seeded date was not attempted; its provenance was rebuilt from the CSV
    rebuilt = records["20260715"]
    assert rebuilt["status"] == "ok"
    assert rebuilt["attempts"] is None
    assert rebuilt["reason"] == "rebuilt from csv"
    assert rebuilt["sha256"] == hashlib.sha256(seeded).hexdigest()
    assert rebuilt["bytes"] == len(seeded)


def test_csv_without_manifest_record_is_refetched(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    bulk.csv_path(raw_dir, "20260715").write_bytes(b"torn write from a crashed run")
    # the manifest exists but has no record for the CSV -> no provenance
    (tmp_path / "settlement_manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": []}), encoding="utf-8"
    )
    calls: list[str] = []

    def tracking_download(trade_date: str):
        calls.append(trade_date)
        return PAYLOAD, bulk.official_url(trade_date)

    records = _run(tmp_path, ["20260715"], tracking_download)

    assert calls == ["20260715"]
    assert bulk.csv_path(raw_dir, "20260715").read_bytes() == PAYLOAD
    record = records["20260715"]
    assert record["status"] == "ok"
    assert record["attempts"] == 1
    assert record["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()


def test_missing_manifest_is_rebuilt_from_frozen_csvs(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    other = "合约代码,今收盘\nMO2609-P-6000,12.0\n".encode("gb18030")
    bulk.csv_path(raw_dir, "20260715").write_bytes(PAYLOAD)
    bulk.csv_path(raw_dir, "20260716").write_bytes(other)
    calls: list[str] = []

    def tracking_download(trade_date: str):
        calls.append(trade_date)
        return PAYLOAD, bulk.official_url(trade_date)

    records = _run(tmp_path, ["20260715", "20260716"], tracking_download)

    assert calls == []
    for tag, payload in (("20260715", PAYLOAD), ("20260716", other)):
        record = records[tag]
        assert record["status"] == "ok"
        assert record["attempts"] is None
        assert record["reason"] == "rebuilt from csv"
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        assert record["bytes"] == len(payload)
    on_disk = json.loads(
        (tmp_path / "settlement_manifest.json").read_text(encoding="utf-8")
    )
    assert [record["date"] for record in on_disk["records"]] == ["20260715", "20260716"]


def test_corrupt_manifest_aborts_with_recovery_instructions(tmp_path) -> None:
    manifest = tmp_path / "settlement_manifest.json"
    manifest.write_text("{ not valid json", encoding="utf-8")

    with pytest.raises(bulk.ManifestError, match="corrupt"):
        bulk.load_manifest(manifest)
    with pytest.raises(bulk.ManifestError, match="delete the file and rerun"):
        bulk.load_manifest(manifest)


def test_missing_date_records_http_evidence_and_writes_no_csv(tmp_path) -> None:
    records = _run(tmp_path, ["20260715"], _not_found_download)

    record = records["20260715"]
    assert record["status"] == "missing"
    assert record["attempts"] == 1
    assert "404" in record["reason"]
    assert bulk.official_url("20260715") in record["reason"]
    assert record["sha256"] is None
    assert record["bytes"] is None
    assert not bulk.csv_path(tmp_path / "raw", "20260715").exists()

    on_disk = json.loads((tmp_path / "settlement_manifest.json").read_text())
    (disk_record,) = on_disk["records"]
    assert disk_record["status"] == "missing"
    assert disk_record["reason"] == record["reason"]


def test_error_after_all_attempts_writes_no_csv(tmp_path) -> None:
    calls: list[str] = []

    def always_fails(trade_date: str):
        calls.append(trade_date)
        raise urllib.error.URLError("connection reset by peer")

    records = _run(tmp_path, ["20260715"], always_fails, max_attempts=3)

    record = records["20260715"]
    assert record["status"] == "error"
    assert record["attempts"] == 3
    assert len(calls) == 3
    assert "URLError" in record["reason"]
    assert record["sha256"] is None
    assert record["bytes"] is None
    assert not bulk.csv_path(tmp_path / "raw", "20260715").exists()


def test_retry_then_success_records_the_actual_attempt_count(tmp_path) -> None:
    calls: list[str] = []

    def flaky(trade_date: str):
        calls.append(trade_date)
        if len(calls) < 3:
            raise urllib.error.URLError("transient")
        return PAYLOAD, bulk.official_url(trade_date)

    records = _run(tmp_path, ["20260715"], flaky, max_attempts=3)

    record = records["20260715"]
    assert record["status"] == "ok"
    assert record["attempts"] == 3
    assert len(calls) == 3
    assert bulk.csv_path(tmp_path / "raw", "20260715").read_bytes() == PAYLOAD


def test_manifest_records_carry_the_required_schema(tmp_path) -> None:
    outcomes = {
        "20260715": _ok_download(),
        "20260716": _not_found_download,
        "20260717": lambda tag: (_ for _ in ()).throw(
            urllib.error.URLError("no route to host")
        ),
    }
    raw_dir = tmp_path / "raw"
    manifest = tmp_path / "settlement_manifest.json"
    for trade_date in DATES:
        bulk.process_dates(
            [trade_date],
            raw_dir=raw_dir,
            manifest_path=manifest,
            delay=0.0,
            sleep=lambda _seconds: None,
            download=outcomes[trade_date],
            verbose=False,
        )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    records = payload["records"]
    assert [record["date"] for record in records] == DATES  # sorted by date
    by_date = {record["date"]: record for record in records}
    for record in records:
        assert REQUIRED_RECORD_KEYS <= set(record)
        assert record["status"] in {"ok", "missing", "error"}
        assert isinstance(record["attempts"], int) and record["attempts"] >= 1
        if record["status"] == "ok":
            assert record["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
            assert record["bytes"] == len(PAYLOAD)
            assert record["reason"] is None
        else:
            assert record["sha256"] is None
            assert record["bytes"] is None
            assert isinstance(record["reason"], str) and record["reason"]
    assert by_date["20260716"]["status"] == "missing"
    assert by_date["20260717"]["status"] == "error"

    reloaded = bulk.load_manifest(manifest)
    assert set(reloaded) == set(DATES)
    assert reloaded["20260715"]["status"] == "ok"


def test_empty_payload_is_never_written(tmp_path) -> None:
    records = _run(tmp_path, ["20260715"], _ok_download(payload=b""), max_attempts=2)

    record = records["20260715"]
    assert record["status"] == "error"
    assert record["attempts"] == 2
    assert record["reason"] == "empty response body"
    assert not bulk.csv_path(tmp_path / "raw", "20260715").exists()


def test_non_cffex_payload_is_an_error_and_never_written(tmp_path) -> None:
    html = "<html><body>not a statistics csv</body></html>".encode("gb18030")
    records = _run(tmp_path, ["20260715"], _ok_download(payload=html), max_attempts=2)

    record = records["20260715"]
    assert record["status"] == "error"
    assert record["attempts"] == 2
    assert "合约代码" in record["reason"]
    assert record["sha256"] is None
    assert not bulk.csv_path(tmp_path / "raw", "20260715").exists()


def test_non_gb18030_payload_is_an_error_and_never_written(tmp_path) -> None:
    garbage = b"\xff\xfe\x00\x81 not decodable \xff\xff"
    records = _run(tmp_path, ["20260715"], _ok_download(payload=garbage), max_attempts=2)

    record = records["20260715"]
    assert record["status"] == "error"
    assert record["attempts"] == 2
    assert "GB18030" in record["reason"]
    assert not bulk.csv_path(tmp_path / "raw", "20260715").exists()


def test_load_trading_dates_reads_the_spot_calendar(tmp_path) -> None:
    spot_csv = tmp_path / "csi1000_spot.csv"
    spot_csv.write_text(
        "date,spot\n2026-07-15,7000.0\n2026-07-16,7100.0\n2026-07-16,7100.0\n",
        encoding="utf-8",
    )

    assert bulk.load_trading_dates(spot_csv) == ["20260715", "20260716"]

    with pytest.raises(FileNotFoundError, match="01_refresh_market_cache"):
        bulk.load_trading_dates(tmp_path / "missing.csv")
