#!/usr/bin/env python
"""Stage 16 -- the study progress dashboard.

    .venv/bin/python example/mo_volmodels/16_dashboard.py            # snapshot
    .venv/bin/python example/mo_volmodels/16_dashboard.py --serve    # live

Read-only.  The only thing this writes is the HTML file named by --out.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mo_dashboard.payload import collect  # noqa: E402
from mo_dashboard.render import render  # noqa: E402
from mo_dashboard.serve import serve  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "output/snowball_dashboard_latest.html"
DEFAULT_REGISTRY = PROJECT_ROOT / "example/mo_volmodels/mo_dashboard.yaml"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry", type=Path, default=None,
                        help=f"registry YAML (default {DEFAULT_REGISTRY})")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"snapshot path (default {DEFAULT_OUT})")
    parser.add_argument("--serve", action="store_true", help="run the local server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    registry = (
        args.registry
        if args.registry is not None
        else project_root / "example/mo_volmodels/mo_dashboard.yaml"
    )

    if args.serve:
        if args.open_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}")
        serve(project_root, registry, port=args.port, poll_seconds=args.poll_seconds)
        return 0

    out = (
        args.out
        if args.out is not None
        else project_root / "output/snowball_dashboard_latest.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = collect(project_root, registry)
    out.write_text(render(doc), encoding="utf-8")

    action = (doc.get("chain") or {}).get("next_action") or {}
    fleet = doc.get("fleet") or {}
    counts = fleet.get("counts") or {}
    print(f"[dashboard] {out}")
    print(f"[fleet]     {fleet.get('admitted')}/{fleet.get('expected_cells')} admitted "
          f"({counts.get('fresh', 0)} fresh, {counts.get('stale', 0)} stale, "
          f"{counts.get('void', 0)} void)")
    print(f"[next]      {action.get('node')} — {action.get('why')}")
    for err in doc.get("errors", []):
        print(f"[error]     {err.get('source')}: {err.get('message')}")
    if args.open_browser:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
