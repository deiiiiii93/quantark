"""Install and inspect the macOS launchd job for daily MO calibration.

The generated job runs the operational pipeline at 18:30 and 20:30
Asia/Shanghai, Monday through Friday.  The second invocation is an intentional
source-publication retry; the pipeline lock, manifests, and SHA-keyed caches
make repeated invocations safe and cheap.

Examples::

    .venv/bin/python example/mo_volmodels/install_daily_scheduler.py install
    .venv/bin/python example/mo_volmodels/install_daily_scheduler.py status
    .venv/bin/python example/mo_volmodels/install_daily_scheduler.py uninstall
"""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = Path(__file__).resolve().parent / "14_daily_calibration_pipeline.py"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "output" / "mo_daily_calibration"
LABEL = "com.quantark.mo-daily-calibration"
DEFAULT_DESTINATION = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LAUNCHCTL = Path("/bin/launchctl")


class SchedulerError(RuntimeError):
    """Scheduler installation or launchctl operation failed."""


def calendar_intervals(hours: Sequence[int], minute: int) -> list[dict[str, int]]:
    return [
        {"Weekday": weekday, "Hour": int(hour), "Minute": int(minute)}
        for weekday in range(1, 6)
        for hour in hours
    ]


def absolute_without_resolving_symlinks(path: Path) -> Path:
    """Return an absolute path while preserving a virtualenv interpreter symlink."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def scheduler_payload(
    *,
    python: Path,
    runtime_dir: Path,
    hours: Sequence[int],
    minute: int,
    temporal_smoothing: bool = False,
) -> dict[str, Any]:
    logs = runtime_dir / "logs"
    program_arguments = [
        str(python),
        str(PIPELINE_SCRIPT),
        "run",
        "--runtime-dir",
        str(runtime_dir),
        "--surface-workers",
        "1",
        "--json",
    ]
    if temporal_smoothing:
        program_arguments.append("--temporal-smoothing")
    return {
        "Label": LABEL,
        "ProgramArguments": program_arguments,
        "WorkingDirectory": str(PROJECT_ROOT),
        "StartCalendarInterval": calendar_intervals(hours, minute),
        "RunAtLoad": False,
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "Nice": 5,
        "ThrottleInterval": 60,
        "StandardOutPath": str(logs / "daily.stdout.log"),
        "StandardErrorPath": str(logs / "daily.stderr.log"),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "TZ": "Asia/Shanghai",
            "MPLCONFIGDIR": str(runtime_dir / "matplotlib"),
            "XDG_CACHE_HOME": str(runtime_dir / "cache"),
        },
    }


def atomic_write_plist(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            plistlib.dump(payload, handle, fmt=plistlib.FMT_XML, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def domain() -> str:
    return f"gui/{os.getuid()}"


def service_target() -> str:
    return f"{domain()}/{LABEL}"


def launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(LAUNCHCTL), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise SchedulerError(
            f"launchctl {' '.join(args)} failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed


def install(args: argparse.Namespace) -> int:
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise SchedulerError(f"QuantArk interpreter is unavailable: {args.python}")
    if not PIPELINE_SCRIPT.is_file():
        raise SchedulerError(f"daily pipeline script is missing: {PIPELINE_SCRIPT}")

    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    (args.runtime_dir / "logs").mkdir(parents=True, exist_ok=True)
    payload = scheduler_payload(
        python=absolute_without_resolving_symlinks(args.python),
        runtime_dir=args.runtime_dir.resolve(),
        hours=args.hours,
        minute=args.minute,
        temporal_smoothing=bool(args.temporal_smoothing),
    )
    atomic_write_plist(args.destination, payload)
    print(f"wrote {args.destination}")
    if args.no_load:
        return 0

    # Reinstall is idempotent: bootout may fail when the service is not loaded.
    launchctl("bootout", domain(), str(args.destination), check=False)
    launchctl("bootstrap", domain(), str(args.destination))
    launchctl("enable", service_target())
    print(f"loaded {service_target()}")
    if args.kickstart:
        launchctl("kickstart", "-k", service_target())
        print(f"kickstarted {service_target()}")
    return 0


def scheduler_status(_args: argparse.Namespace) -> int:
    completed = launchctl("print", service_target(), check=False)
    if completed.returncode != 0:
        print(f"{LABEL}: not loaded")
        if completed.stderr:
            print(completed.stderr.strip(), file=sys.stderr)
        return 1
    print(completed.stdout, end="")
    return 0


def uninstall(args: argparse.Namespace) -> int:
    launchctl("bootout", domain(), str(args.destination), check=False)
    try:
        args.destination.unlink()
    except FileNotFoundError:
        pass
    print(f"removed {args.destination}; service unloaded")
    return 0


def render(args: argparse.Namespace) -> int:
    payload = scheduler_payload(
        python=absolute_without_resolving_symlinks(args.python),
        runtime_dir=args.runtime_dir.resolve(),
        hours=args.hours,
        minute=args.minute,
        temporal_smoothing=bool(args.temporal_smoothing),
    )
    sys.stdout.buffer.write(
        plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
    )
    return 0


def add_schedule_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument(
        "--hours",
        type=int,
        nargs="+",
        default=[18, 20],
        help="Shanghai-local launch hours (default: 18 20)",
    )
    parser.add_argument("--minute", type=int, default=30)
    parser.add_argument(
        "--temporal-smoothing",
        action="store_true",
        help="install/render the pipeline with its temporal calibration opt-in",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install")
    add_schedule_args(install_parser)
    install_parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    install_parser.add_argument("--no-load", action="store_true")
    install_parser.add_argument("--kickstart", action="store_true")
    install_parser.set_defaults(handler=install)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(handler=scheduler_status)

    uninstall_parser = subparsers.add_parser("uninstall")
    uninstall_parser.add_argument(
        "--destination", type=Path, default=DEFAULT_DESTINATION
    )
    uninstall_parser.set_defaults(handler=uninstall)

    render_parser = subparsers.add_parser("render")
    add_schedule_args(render_parser)
    render_parser.set_defaults(handler=render)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if hasattr(args, "hours"):
        if not args.hours or any(hour < 0 or hour > 23 for hour in args.hours):
            raise SystemExit("--hours must contain values in [0, 23]")
        if args.minute < 0 or args.minute > 59:
            raise SystemExit("--minute must be in [0, 59]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    try:
        return int(args.handler(args))
    except SchedulerError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
