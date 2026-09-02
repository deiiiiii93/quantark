import os
import subprocess
import sys
import textwrap
from datetime import date

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)
from quantark.volcalibration.runner import (
    EXIT_CURRENT,
    EXIT_FAILED,
    EXIT_LOCKED,
    EXIT_NON_CURRENT,
    LockBusy,
    acquire_lock,
    build_status,
    load_trading_dates,
    status_exit_code,
)
from quantark.volcalibration.store import (
    StoreLayout,
    atomic_write_json,
    surface_record,
)


def _config(tmp_path, spot_csv=None):
    return RunConfig(
        name="mo-test",
        underlying=UnderlyingConfig("S", "listed_strike", "settlement"),
        history_dir=tmp_path,
        runtime_dir=tmp_path,
        spot_csv=spot_csv,
        surface=SurfaceBuildConfig(),
        calibration=CalibrationRunConfig(),
        workers=1,
    )


def test_the_lock_is_exclusive_within_one_process(tmp_path):
    path = tmp_path / "pipeline.lock"
    with acquire_lock(path):
        with pytest.raises(LockBusy):
            with acquire_lock(path):
                pass


def test_the_lock_is_exclusive_across_processes(tmp_path):
    """flock is per-file-description; a same-process re-acquire is not proof."""
    path = tmp_path / "pipeline.lock"
    script = textwrap.dedent(
        f"""
        import sys
        from quantark.volcalibration.runner import LockBusy, acquire_lock
        try:
            with acquire_lock({str(path)!r}):
                sys.exit(0)
        except LockBusy:
            sys.exit(75)
        """
    )
    with acquire_lock(path):
        env = dict(os.environ, PYTHONPATH=os.getcwd())
        result = subprocess.run([sys.executable, "-c", script], env=env)
    assert result.returncode == EXIT_LOCKED


def test_the_lock_records_its_owner(tmp_path):
    path = tmp_path / "pipeline.lock"
    with acquire_lock(path):
        assert str(os.getpid()) in path.read_text()
    assert path.read_text() == "", "a released lock must not look held"


@pytest.mark.parametrize(
    "overall,expected",
    [
        ("current", EXIT_CURRENT),
        ("snapshot_pending", EXIT_NON_CURRENT),
        ("surface_excluded", EXIT_NON_CURRENT),
        ("surface_pending", EXIT_NON_CURRENT),
        ("calibration_pending", EXIT_NON_CURRENT),
        ("calibration_failed", EXIT_FAILED),
        ("failed", EXIT_FAILED),
    ],
)
def test_exit_codes_carry_state(overall, expected):
    assert status_exit_code({"overall_status": overall}) == expected


def test_status_without_a_calendar_says_so(tmp_path):
    config = _config(tmp_path)
    layout = StoreLayout.from_config(config)
    layout.snapshots_dir.mkdir(parents=True)
    (layout.snapshots_dir / "20260430.json").write_text("{}")

    status = build_status(layout, config, as_of=date(2026, 5, 1))
    assert status["expected_trade_date"] == "20260430"
    assert status["freshness"]["spot_cache_latest"] is None
    assert status["freshness"]["calendar_source"] == "snapshots_on_disk"
    assert status["overall_status"] == "surface_pending"
    assert status["module"] == "quantark.volcalibration"
    assert status["pipeline"] == "mo-test"
    assert status_exit_code(status) == EXIT_NON_CURRENT


def test_status_with_a_calendar_reports_lag(tmp_path):
    spot_csv = tmp_path / "spot.csv"
    spot_csv.write_text(
        "date,spot\n2026-04-29,1.0\n2026-04-30,1.0\n2026-05-04,1.0\n", encoding="utf-8"
    )
    config = _config(tmp_path, spot_csv=spot_csv)
    layout = StoreLayout.from_config(config)
    layout.snapshots_dir.mkdir(parents=True)
    (layout.snapshots_dir / "20260430.json").write_text("{}")

    status = build_status(layout, config, as_of=date(2026, 5, 4))
    assert status["expected_trade_date"] == "20260504"
    assert status["freshness"]["calendar_source"] == str(spot_csv)
    assert status["freshness"]["snapshot_latest"] == "20260430"
    assert status["freshness"]["snapshot_lag_trading_days"] == 1
    assert status["overall_status"] == "snapshot_pending"


def test_an_empty_store_with_no_calendar_fails_closed(tmp_path):
    config = _config(tmp_path)
    layout = StoreLayout.from_config(config)
    with pytest.raises(ValidationError):
        build_status(layout, config, as_of=date(2026, 5, 1))


def _current_excluded_record(layout, config, tag, *, reason):
    """An exclusion record that is current for the snapshot written on disk."""
    import hashlib
    import json as _json

    from quantark.volcalibration.store import builder_fingerprint

    raw = _json.dumps({"source": {"sha256": f"src-{tag}"}}).encode()
    layout.snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.snapshot_path(tag).write_bytes(raw)
    return surface_record(
        tag,
        status="excluded",
        reason=reason,
        n_expiries=1,
        snapshot_sha256=f"src-{tag}",
        snapshot_content_sha256=hashlib.sha256(raw).hexdigest(),
        symbol=config.underlying.symbol,
        price_field=config.underlying.price_field,
        fingerprint=builder_fingerprint(config.surface.fingerprint_payload()),
    )


def test_an_excluded_expected_date_is_non_current_not_failed(tmp_path):
    """A date can be legitimately uncalibratable forever (spec 6.3.2)."""
    config = _config(tmp_path)
    layout = StoreLayout.from_config(config)
    record = _current_excluded_record(
        layout, config, "20240930", reason="insufficient_expiries"
    )
    atomic_write_json(
        layout.surface_manifest, {"schema_version": 1, "records": [record]}
    )
    status = build_status(layout, config, as_of=date(2024, 10, 1))
    assert status["overall_status"] == "surface_excluded"
    assert status_exit_code(status) == EXIT_NON_CURRENT


def test_a_stale_exclusion_is_pending_not_excluded(tmp_path):
    """`surface_excluded` tells an agent to give up; only a current exclusion
    has earned that."""
    config = _config(tmp_path)
    layout = StoreLayout.from_config(config)
    record = _current_excluded_record(
        layout, config, "20240930", reason="static_arbitrage"
    )
    # The snapshot was corrected: a run would retry this date.
    layout.snapshot_path("20240930").write_text(
        '{"source": {"sha256": "corrected"}}', encoding="utf-8"
    )
    atomic_write_json(
        layout.surface_manifest, {"schema_version": 1, "records": [record]}
    )
    status = build_status(layout, config, as_of=date(2024, 10, 1))
    assert status["overall_status"] == "surface_pending"
    assert status_exit_code(status) == EXIT_NON_CURRENT


def test_the_grandfathered_count_is_visible(tmp_path):
    config = _config(tmp_path)
    layout = StoreLayout.from_config(config)
    layout.snapshots_dir.mkdir(parents=True)
    (layout.snapshots_dir / "20260430.json").write_text("{}")
    atomic_write_json(
        layout.surface_manifest,
        {
            "schema_version": 1,
            "records": [
                surface_record(
                    "20260430",
                    status="ok",
                    n_expiries=6,
                    artifact_sha256="a",
                    provenance="grandfathered",
                )
            ],
        },
    )
    status = build_status(layout, config, as_of=date(2026, 4, 30))
    assert status["grandfathered_surface_dates"] == 1


def test_a_missing_spot_csv_fails_closed(tmp_path):
    with pytest.raises(ValidationError):
        load_trading_dates(tmp_path / "absent.csv")
