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

import numpy as np

from golden_compare import GOLDEN_REL_TOL, assert_close

#: Cross-architecture tolerance for these frames, expressed against each
#: COLUMN's scale rather than each element's own magnitude.
#:
#: That is the shape the noise actually has. These columns carry bump
#: derivatives -- gamma is ``(V+ - 2V + V-)/h**2`` -- so the rounding noise is
#: set by the magnitude of the PRICE being differenced, which is roughly uniform
#: down the column. It is therefore roughly constant in ABSOLUTE terms per
#: column, and a per-element relative tolerance has no purchase on the small
#: rows. Measured on the first x86_64 runs against these ARM64 goldens:
#:
#:   gamma            worst drift 4.6e-10, column max    43.0  -> 1.1e-11
#:   gamma_cash_1pct  worst drift 3.8e-08, column max 4,301.2  -> 8.9e-12
#:
#: Same ratio for both, which is why fixing gamma with a flat absolute floor
#: left its scaled sibling failing on the next run. 1e-9 of column scale keeps
#: ~100x over the measured drift, while a real change to these engines moves
#: these columns by percent -- seven orders above it. Integer and non-numeric
#: columns still compare exactly, as do column names, order and frame shape.
COLUMN_SCALE_TOL = 1e-9


def assert_frame_matches(actual, expected, label: str) -> None:
    """Compare a live frame against a frozen golden, column by column."""
    assert list(actual.columns) == list(expected.columns), (
        f"{label} column set/order changed"
    )
    assert len(actual) == len(expected), f"{label} row count changed"
    for column in expected.columns:
        want, got = expected[column], actual[column]
        if not pd.api.types.is_float_dtype(want):
            # Integers, strings, dates and booleans are discrete: exact.
            assert got.equals(want), f"{label} column {column!r} differs"
            continue
        want_values = want.to_numpy(dtype=float)
        got_values = got.to_numpy(dtype=float)
        scale = float(np.max(np.abs(want_values))) if want_values.size else 0.0
        np.testing.assert_allclose(
            got_values, want_values,
            rtol=GOLDEN_REL_TOL, atol=COLUMN_SCALE_TOL * scale,
            err_msg=f"{label} column {column!r}",
        )

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
    assert_frame_matches(
        pd.read_csv(actual_path), pd.read_csv(golden_path), f"{case}/{frame_name}"
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


# --- the tolerance itself, tested ------------------------------------------
#
# The comparison above cannot be validated on the machine the goldens were
# frozen on: everything is bitwise there, so any tolerance passes. What CAN be
# checked locally is that the tolerance admits noise of the size measured
# cross-architecture and still rejects a change of the size a real engine bug
# makes. Without both, a green CI proves nothing.

def _golden_like_frame():
    """A frame with the shape that matters: a large column and a small one
    carrying the same absolute noise, which is what a bump second difference
    does."""
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "count": [1, 2, 3],
            "price": [9504.27292802203, 9611.5, 9118.734185],
            "gamma": [23.74735191146168, 0.037035, -43.011715],
            "gamma_cash_1pct": [2143.1985100094166, -3.1566762373795587, -4301.171488],
        }
    )


def _perturbed(frame, relative_to_column_scale):
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            scale = float(out[column].abs().max())
            out[column] = out[column] + relative_to_column_scale * scale
    return out


def test_the_golden_gate_admits_the_measured_cross_architecture_drift():
    """1.1e-11 of column scale is the worst x86_64-vs-ARM64 drift measured on
    these frames. If this failed, CI could never be green off this machine."""
    expected = _golden_like_frame()
    assert_frame_matches(_perturbed(expected, 1.1e-11), expected, "drift")


def test_the_golden_gate_still_catches_a_real_change():
    """1e-6 of column scale is far below the percent-level move a genuine
    numerics change makes, and must already fail -- otherwise the tolerance has
    swallowed the gate it is supposed to guard."""
    expected = _golden_like_frame()
    with pytest.raises(AssertionError):
        assert_frame_matches(_perturbed(expected, 1e-6), expected, "real change")


def test_the_golden_gate_compares_discrete_columns_exactly():
    """Integers and strings carry no rounding noise, so they get no slack."""
    expected = _golden_like_frame()
    bumped_int = expected.copy()
    bumped_int.loc[0, "count"] = 2
    with pytest.raises(AssertionError):
        assert_frame_matches(bumped_int, expected, "int")

    bumped_str = expected.copy()
    bumped_str.loc[0, "date"] = "2024-01-05"
    with pytest.raises(AssertionError):
        assert_frame_matches(bumped_str, expected, "str")
