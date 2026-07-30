"""One-shot golden capture: run the CURRENT engines, freeze every frame.

Run from the repo root (spec §9.1, plan Task 1):

    PYTHONPATH=$PWD <venv>/bin/python -c \\
        "from pathlib import Path; from test.replay_golden.capture import write_goldens; \\
         write_goldens()"
"""
from __future__ import annotations

import json
from pathlib import Path

from . import fixtures


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _run_all():
    from quantark.backtest.otc import (
        AutocallableBacktestEngine,
        BookAutocallableBacktestEngine,
    )

    scalar = AutocallableBacktestEngine(fixtures.make_scalar_bsm_config()).run()
    book = BookAutocallableBacktestEngine(fixtures.make_book_config()).run()
    localvol = AutocallableBacktestEngine(fixtures.make_localvol_config()).run()

    records = fixtures.calibration_records(localvol)
    states = fixtures.result_frames(localvol)["states"]
    priced_days = int(states["alive"].shift(1, fill_value=True).sum())
    assert records, "localvol golden must produce calibration records"
    assert all(r.get("variant") == "localvol" for r in records), records
    dates = [r["date"] for r in records]
    assert dates == sorted(dates), "calibration records out of order"
    assert len(records) <= 3 and len(records) >= 1, (len(records), priced_days)

    return {"scalar_bsm": scalar, "book": book, "localvol": localvol}


def write_goldens(golden_dir: Path = fixtures.GOLDEN_DIR) -> None:
    golden_dir.mkdir(parents=True, exist_ok=True)
    for name, results in _run_all().items():
        for frame_name, df in fixtures.result_frames(results).items():
            df.to_csv(golden_dir / f"{name}_{frame_name}.csv", index=False)
        (golden_dir / f"{name}_summary.json").write_text(
            json.dumps(
                _json_safe(fixtures.result_summary(results)),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        records = fixtures.calibration_records(results)
        if records:
            (golden_dir / f"{name}_calibration_records.json").write_text(
                json.dumps(_json_safe(records), indent=2, sort_keys=True, default=str)
            )
    print(f"goldens written to {golden_dir}")
