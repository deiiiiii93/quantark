"""Contract tests for the launchd scheduler installer."""

from __future__ import annotations

import importlib.util
import plistlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "example/mo_volmodels/install_daily_scheduler.py"
SPEC = importlib.util.spec_from_file_location("mo_daily_scheduler", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)


def test_payload_runs_twice_each_weekday_with_safe_worker_default(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    payload = scheduler.scheduler_payload(
        python=Path(sys.executable),
        runtime_dir=runtime,
        hours=[18, 20],
        minute=30,
    )
    assert payload["Label"] == scheduler.LABEL
    assert len(payload["StartCalendarInterval"]) == 10
    assert {item["Weekday"] for item in payload["StartCalendarInterval"]} == {
        1,
        2,
        3,
        4,
        5,
    }
    arguments = payload["ProgramArguments"]
    assert arguments[0] == sys.executable
    assert arguments[2] == "run"
    assert arguments[arguments.index("--surface-workers") + 1] == "1"
    assert payload["StandardOutPath"].startswith(str(runtime))
    assert payload["EnvironmentVariables"]["MPLCONFIGDIR"].startswith(str(runtime))


def test_install_no_load_writes_valid_plist(tmp_path: Path) -> None:
    destination = tmp_path / "LaunchAgents" / f"{scheduler.LABEL}.plist"
    runtime = tmp_path / "runtime"
    code = scheduler.main(
        [
            "install",
            "--destination",
            str(destination),
            "--runtime-dir",
            str(runtime),
            "--python",
            sys.executable,
            "--no-load",
        ]
    )
    assert code == 0
    with destination.open("rb") as handle:
        payload = plistlib.load(handle)
    assert payload["Label"] == scheduler.LABEL
    assert (runtime / "logs").is_dir()


def test_temporal_smoothing_is_an_explicit_scheduler_opt_in(
    tmp_path: Path,
) -> None:
    default_payload = scheduler.scheduler_payload(
        python=Path(sys.executable),
        runtime_dir=tmp_path / "default",
        hours=[18],
        minute=30,
    )
    temporal_payload = scheduler.scheduler_payload(
        python=Path(sys.executable),
        runtime_dir=tmp_path / "temporal",
        hours=[18],
        minute=30,
        temporal_smoothing=True,
    )
    assert "--temporal-smoothing" not in default_payload["ProgramArguments"]
    assert "--temporal-smoothing" in temporal_payload["ProgramArguments"]


def test_virtualenv_interpreter_symlink_is_not_resolved(tmp_path: Path) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable))
    absolute = scheduler.absolute_without_resolving_symlinks(venv_python)
    assert absolute == venv_python
    assert absolute.resolve() == Path(sys.executable).resolve()


def test_invalid_schedule_is_rejected() -> None:
    try:
        scheduler.main(["render", "--hours", "24"])
    except SystemExit as exc:
        assert "--hours" in str(exc)
    else:
        raise AssertionError("invalid schedule should fail")
