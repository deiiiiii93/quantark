"""Golden gate for the backtest replay consolidation (spec §9).

Every deterministic result surface of the three frozen configurations must
reproduce the committed goldens: same columns, same order, same values. Wall-clock
fields are excluded at capture and at read.

Floats compare through ``golden_compare`` rather than bitwise. These goldens were
frozen on ONE machine, and bitwise equality is not an architecture-independent
property: the first CI run of this gate on x86_64 Linux failed nine of these
frames on ARM64-frozen values that agreed to ten significant figures. The
tolerance absorbs that last-ULP noise and nothing more — it is five orders below
any real behavior change, which is what this gate exists to catch. Column names
and order, string and integer fields, and frame shape all still compare exactly.

Regenerating goldens is a deliberate act (see ``replay_golden/capture.py``) —
a diff beyond ULP noise here is a behavior change to fix, never a tolerance to
widen further.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from golden_compare import GOLDEN_ABS_TOL, GOLDEN_REL_TOL, assert_close

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

GOLDEN_DIR = fixtures.GOLDEN_DIR

pytestmark = pytest.mark.skipif(
    not (GOLDEN_DIR / "scalar_bsm_states.csv").is_file(),
    reason="goldens not captured yet (run replay_golden.capture.write_goldens)",
)


@pytest.fixture(scope="module")
def all_results():
    from quantark.backtest.otc import (
        AutocallableBacktestEngine,
        BookAutocallableBacktestEngine,
    )

    return {
        "scalar_bsm": AutocallableBacktestEngine(
            fixtures.make_scalar_bsm_config()
        ).run(),
        "book": BookAutocallableBacktestEngine(fixtures.make_book_config()).run(),
        "localvol": AutocallableBacktestEngine(fixtures.make_localvol_config()).run(),
    }


CASES = [
    (case, frame)
    for case in ("scalar_bsm", "book", "localvol")
    for frame in fixtures.FRAME_NAMES
]


@pytest.mark.parametrize("case,frame_name", CASES, ids=[f"{c}-{f}" for c, f in CASES])
def test_frame_matches_golden(all_results, case, frame_name, tmp_path):
    golden_path = GOLDEN_DIR / f"{case}_{frame_name}.csv"
    assert golden_path.is_file(), f"missing golden {golden_path.name}"

    actual_df = fixtures.result_frames(all_results[case])[frame_name]
    # Round-trip the actual frame through CSV so float formatting is symmetric
    # with the golden file (spec §9.2 canonical comparison contract).
    actual_path = tmp_path / golden_path.name
    actual_df.to_csv(actual_path, index=False)

    if golden_path.read_text().strip() == "":
        # Frame recorded as genuinely empty (e.g. surfaces disabled): the
        # actual frame must be empty too; there is nothing to parse.
        assert actual_df.empty and len(actual_df.columns) == 0
        return
    actual = pd.read_csv(actual_path)
    expected = pd.read_csv(golden_path)
    assert list(actual.columns) == list(expected.columns), (
        f"{case}/{frame_name} column set/order changed"
    )
    pd.testing.assert_frame_equal(
        actual, expected, check_exact=False,
        rtol=GOLDEN_REL_TOL, atol=GOLDEN_ABS_TOL,
    )


@pytest.mark.parametrize("case", ["scalar_bsm", "book", "localvol"])
def test_summary_matches_golden(all_results, case):
    golden_path = GOLDEN_DIR / f"{case}_summary.json"
    expected = json.loads(golden_path.read_text())
    from replay_golden.capture import _json_safe

    actual = json.loads(
        json.dumps(
            _json_safe(fixtures.result_summary(all_results[case])),
            sort_keys=True,
            default=str,
        )
    )
    assert_close(actual, expected, msg=f"{case} summary")


def test_localvol_calibration_records_match_golden(all_results):
    golden_path = GOLDEN_DIR / "localvol_calibration_records.json"
    expected = json.loads(golden_path.read_text())
    from replay_golden.capture import _json_safe

    actual = json.loads(
        json.dumps(
            _json_safe(fixtures.calibration_records(all_results["localvol"])),
            sort_keys=True,
            default=str,
        )
    )
    assert_close(actual, expected, msg="localvol calibration records")
    assert all(r["variant"] == "localvol" for r in actual)
