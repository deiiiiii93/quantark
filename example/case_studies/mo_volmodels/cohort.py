"""The frozen surface cohort for the 0.4.0 re-baseline gates.

A launchd job (``com.quantark.mo-daily-calibration``) extends
``data/history/`` every weekday, so "the cohort" is a moving target unless it
is pinned.  Crossing 2026-08-01 also admits a 28th snowball inception, which
would re-open Gate G4.  Every gate reads its date list from here.

Admission verdicts come from ``surface_manifest.json``.  The artifacts
themselves carry only the *criteria* used to build them (``min_expiries: 2``),
never a per-surface verdict, so an artifact can never answer "was this
admitted?".
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "example/mo_volmodels/data/history"

# Frozen 2026-08-01 (spec §7A.12).  Raising this is a deliberate re-baseline:
# it changes the G1 count, and past 2026-08-01 it changes the inception fleet
# from 27 to 28 and therefore re-opens G4.
COHORT_ASOF = date(2026, 7, 31)

_STUDY_KEY = "vol_model_backtest"


def _parse(tag: str) -> date:
    return date(int(tag[:4]), int(tag[4:6]), int(tag[6:8]))


def _manifest(history_dir: Optional[Path]) -> Dict[str, Any]:
    path = Path(history_dir or DEFAULT_HISTORY_DIR) / "surface_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _study_exclusions(manifest: Dict[str, Any]) -> set:
    study = (manifest.get("study_admission") or {}).get(_STUDY_KEY) or {}
    return {_parse(tag) for tag in study.get("excluded_dates", [])}


def admitted_dates(history_dir: Optional[Path] = None) -> List[date]:
    """Admitted surface dates at or before ``COHORT_ASOF``, ascending.

    A date is admitted when the manifest record says ``status == "ok"`` AND
    the date is not in ``study_admission.vol_model_backtest.excluded_dates``.
    The second check matters because a history rebuild rewrites per-record
    status from the builder's own ``min_expiries=2`` while preserving the
    top-level study policy — so status alone would re-admit the thin surfaces.
    """
    manifest = _manifest(history_dir)
    excluded = _study_exclusions(manifest)
    out = []
    for record in manifest.get("records", []):
        day = _parse(str(record["date"]))
        if day > COHORT_ASOF or day in excluded:
            continue
        if record.get("status") != "ok":
            continue
        out.append(day)
    return sorted(out)


def excluded_records(history_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Non-admitted records at or before ``COHORT_ASOF``, with their reasons."""
    manifest = _manifest(history_dir)
    excluded = _study_exclusions(manifest)
    out = []
    for record in manifest.get("records", []):
        day = _parse(str(record["date"]))
        if day > COHORT_ASOF:
            continue
        if record.get("status") == "ok" and day not in excluded:
            continue
        out.append(
            {
                "date": day,
                "reason": record.get("reason"),
                "n_expiries": record.get("n_expiries"),
            }
        )
    return sorted(out, key=lambda item: item["date"])
