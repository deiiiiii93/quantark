"""
Skill runner script for autocallable risk reports (Snowball-first).

This wraps `python -m asset.equity.report.autocallable_risk_report` with safe defaults
for sandboxed environments (e.g., MPLCONFIGDIR).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _find_repo_root(start: Path) -> Path:
    """
    Try to locate the QuantArk repo root from the current working directory.

    This runner can be installed outside the repo (e.g. ~/.codex/skills), so we
    cannot rely on __file__ location.
    """
    p = start.resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "asset" / "equity").exists() and (candidate / "priceenv").exists():
            return candidate
    return start


def main(argv: list[str]) -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    repo_root = _find_repo_root(Path.cwd())
    sys.path.insert(0, str(repo_root))

    from asset.equity.report.autocallable_risk_report import main as report_main

    return int(report_main(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

