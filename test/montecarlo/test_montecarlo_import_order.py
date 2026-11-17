"""``quantark.montecarlo`` must import cleanly as the FIRST quantark import.

``montecarlo`` is asset-agnostic infrastructure that equity engines depend on.
A module-level import from ``quantark.asset`` inside it closes a cycle
(montecarlo -> asset/__init__ -> equity engines -> montecarlo) that only bites
in a fresh interpreter, so each case runs in its own subprocess.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "module",
    [
        "quantark.montecarlo",
        "quantark.montecarlo.conditional_snowball",
    ],
)
def test_montecarlo_imports_first_in_fresh_interpreter(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_montecarlo_import_does_not_load_asset_layer() -> None:
    code = (
        "import sys, quantark.montecarlo\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('quantark.asset'))\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
