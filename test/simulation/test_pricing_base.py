from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.pricing.base import (
    DayStates,
    GateReport,
    StateKey,
    float_key,
    row_keys,
)


def test_row_keys_are_equal_for_equal_bytes_and_differ_otherwise():
    rows = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.000000001]])
    keys = row_keys(rows)
    assert keys.dtype == np.int64 and keys.shape == (3,)
    assert keys[0] == keys[1] != keys[2]


def test_row_keys_normalise_negative_zero():
    assert row_keys(np.array([[0.0, 1.0]]))[0] == row_keys(np.array([[-0.0, 1.0]]))[0]


def test_row_keys_do_not_depend_on_the_hash_seed():
    code = (
        "import numpy as np;"
        "from quantark.backtest.simulation.pricing.base import row_keys;"
        "print(int(row_keys(np.array([[6000.0, 0.22, -0.03]]))[0]))"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}, check=True).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(outs) == 1


def test_float_key_round_trips_a_value_and_separates_neighbours():
    values = np.array([6000.0, 6000.0, np.nextafter(6000.0, 7000.0), -0.0, 0.0])
    keys = float_key(values)
    assert keys[0] == keys[1] != keys[2]
    assert keys[3] == keys[4]


def test_state_key_digest_and_seed_are_deterministic_and_specific():
    a = StateKey("prod", 3, False, 11, 22, 33, "eng")
    b = StateKey("prod", 3, False, 11, 22, 33, "eng")
    c = StateKey("prod", 3, True, 11, 22, 33, "eng")   # differs only in the KI flag
    assert a == b and hash(a) == hash(b)
    assert a.digest() == b.digest() != c.digest()
    assert a.seed() == b.seed() != c.seed()
    assert 0 <= a.seed() < 2**32


def test_day_states_length_and_emptiness():
    day = pd.Timestamp("2024-01-02")
    empty = DayStates(day_index=0, date=day, path_index=np.array([], dtype=int), spot=np.array([]),
                      vol=np.array([]), rate=np.array([]), q_T=np.array([]), div_yield=(),
                      basis_yield=np.array([]), env_key=np.array([], dtype=np.int64),
                      knocked_in=np.array([], dtype=bool))
    assert len(empty) == 0 and empty.empty
    one = DayStates(day_index=1, date=day, path_index=np.array([4]), spot=np.array([6000.0]),
                    vol=np.array([0.2]), rate=np.array([0.02]), q_T=np.array([0.05]), div_yield=(None,),
                    basis_yield=np.array([-0.03]), env_key=np.array([7], dtype=np.int64),
                    knocked_in=np.array([False]))
    assert len(one) == 1 and not one.empty
    assert one.date == day


def test_gate_report_records_the_exact_mode_zero():
    report = GateReport(mode="exact", sampled=0, max_pv_gap_bp=0.0, max_delta_gap_hands=0.0, passed=True)
    assert report.as_dict()["mode"] == "exact" and report.as_dict()["max_pv_gap_bp"] == 0.0
