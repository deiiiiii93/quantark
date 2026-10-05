"""double_no_touch (spec 3C §5.8, §7.2 'no-touch probability' row): against references at the exact double
inputs, inside its own computed bound everywhere, refusing exactly where the bound exceeds Tolerance.ZERO."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import double_no_touch as dnt
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import Tolerance

BANK = json.loads((Path(__file__).parent / "no_touch_reference.json").read_text())["cells"]
ACCURACY = 1e-14             # (i): measured 8.4e-16 on the 637 cells, 5.8e-16 on the equity cells


def _err(cell):
    value, bound = dnt.double_no_touch_with_bound(*cell["args"])
    ref = cell["reference"]
    # the banked reference is the double nearest the 60-digit value: half an ulp of slack
    return abs(value - ref), bound + 0.5 * float(np.spacing(abs(ref)) if ref else 0.0), value, bound


@pytest.mark.parametrize("group", ["z1", "equity", "reviewer", "engine"])
def test_accuracy_on_resolved_cells(group):
    worst = 0.0
    for cell in BANK[group]:
        err, slack, _, bound = _err(cell)
        assert err <= slack, (group, cell["args"], err, bound)
        worst = max(worst, err)
    assert worst <= ACCURACY, (group, worst)


@pytest.mark.parametrize("group", ["stress", "random_reachable", "random_any"])
def test_every_error_inside_its_bound(group):
    ratios = []
    for cell in BANK[group]:
        err, slack, _, bound = _err(cell)
        assert err <= slack, (group, cell["args"], err, bound)
        if bound > 0:
            ratios.append(err / bound)
    assert max(ratios) <= 0.5                                  # measured 0.44 / 0.27 / 0.26


def test_refuses_exactly_where_the_bound_exceeds_zero():
    raised = 0
    for cell in BANK["stress"]:
        S, T, r, q, vol, L, U = cell["args"]
        _, bound = dnt.double_no_touch_with_bound(*cell["args"])
        if bound > Tolerance.ZERO:
            with pytest.raises(NumericalError, match="bound"):
                dnt.double_no_touch(*cell["args"])
            raised += 1
            assert not (vol >= 1e-4 and -0.10 <= r <= 0.50 and abs(q) <= 1.0), cell["args"]
        else:
            p = dnt.double_no_touch(*cell["args"])
            assert 0.0 <= p <= 1.0
    assert raised == 556                                       # §4.12: 556 of the 4,536 lattice cells


def test_long_times_return_the_proven_zero_and_pairs_stay_small():
    for group in ("z1", "stress", "random_reachable", "random_any", "equity"):
        for cell in BANK[group]:
            S, T, r, q, vol, L, U = cell["args"]
            if not L < S < U or T <= 0.0:
                continue
            value, bound, k = dnt._evaluate(*cell["args"])
            # the module's own predicate (log1p distances, (vol sqrt T)^2): the z1 cells at tau = 8 sit on the
            # boundary, where log(U / L) and vol^2 T differ from it by an ulp
            a = math.log1p((U - S) / S) - math.log1p((L - S) / S)
            s = vol * math.sqrt(T)
            if s * s >= 8.0 * a * a:
                assert value == 0.0 and bound == 2e-17
                assert cell["reference"] < 2e-17
            else:
                assert k <= 13, (cell["args"], k)


def test_a_sum_outside_unit_interval_beyond_its_bound_raises(monkeypatch):
    monkeypatch.setattr(dnt, "_evaluate", lambda *a: (1.0 + 1e-6, 1e-12, 1))
    with pytest.raises(NumericalError, match=r"outside \[0, 1\]"):
        dnt.double_no_touch(100.0, 1.0, 0.03, 0.01, 0.2, 80.0, 130.0)
    monkeypatch.setattr(dnt, "_evaluate", lambda *a: (1.0 + 1e-13, 1e-12, 1))
    assert dnt.double_no_touch(100.0, 1.0, 0.03, 0.01, 0.2, 80.0, 130.0) == 1.0


@pytest.mark.parametrize("bad", [dict(spot=float("nan")), dict(vol=0.0), dict(lower=-1.0), dict(upper=70.0),
                                 dict(rate=float("inf"))])
def test_invalid_inputs_raise(bad):
    kw = dict(spot=100.0, maturity=1.0, rate=0.03, div=0.01, vol=0.2, lower=80.0, upper=130.0, **{})
    kw.update(bad)
    with pytest.raises(ValidationError):
        dnt.double_no_touch(**kw)


def test_spot_outside_and_expired():
    assert dnt.double_no_touch(130.0, 1.0, 0.03, 0.01, 0.2, 80.0, 130.0) == 0.0
    assert dnt.double_no_touch(79.0, 1.0, 0.03, 0.01, 0.2, 80.0, 130.0) == 0.0
    assert dnt.double_no_touch(100.0, 0.0, 0.03, 0.01, 0.2, 80.0, 130.0) == 1.0
