"""Smoke gates for the execution-framework migration examples (Phase 6)."""
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).parents[2]


def _run_example(name: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p
    )
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "example" / name)],
        capture_output=True, text=True, timeout=600, env=env,
    )


def test_execution_session_demo_runs():
    result = _run_example("execution_session_demo.py")
    assert result.returncode == 0, result.stderr
    assert "session == direct" in result.stdout
    assert "DCN serial == threads == direct" in result.stdout


def test_execution_scenarios_demo_runs():
    result = _run_example("execution_scenarios_demo.py")
    assert result.returncode == 0, result.stderr
    assert "5/5 scenarios" in result.stdout
