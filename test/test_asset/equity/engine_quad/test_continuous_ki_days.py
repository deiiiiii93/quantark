"""The continuous knock-in day split shared by QUAD V1 and QUAD V2 (unified event statistics spec AC13, T22).

One module holds the interval first-passage law, the split of each interval's crossing mass over the trading days it
overlaps and the knock-in table's survival column; both QUAD statistics modules use it (no private cross-imports, no
duplicated block). The outputs are bitwise those of the two former copies (byte-compare in the T22 report); here the
helpers are pinned against closed forms.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from scipy.special import ndtr

from quantark.asset.equity.engine.quad import continuous_ki_days as days_mod
from quantark.asset.equity.engine.quad import v1_statistics
from quantark.asset.equity.engine.quad.v2 import statistics as v2_statistics


def test_both_quad_statistics_modules_use_the_shared_helpers():
    for module in (v1_statistics, v2_statistics):
        source = inspect.getsource(module)
        assert "def _crossed_by" not in source and "def _survival" not in source, module.__name__
        assert "continuous_ki_days" in source, module.__name__
    assert "from .v2.statistics import" not in inspect.getsource(v1_statistics)


def test_crossed_by_is_the_drifted_brownian_first_passage():
    x, barrier, mean, variance = math.log(100.0), math.log(75.0), -0.02, 0.0484
    law = ((np.array([x]), np.array([1.0])), mean, variance, barrier, False)
    fractions = np.array([0.0, 0.25, 0.5, 1.0])
    crossed = days_mod.crossed_by(law, fractions)
    d = barrier - x
    for f, c in zip(fractions, crossed):
        if f == 0.0:
            assert c == 0.0
            continue
        m, v = mean * f, variance * f
        expected = ndtr((d - m) / math.sqrt(v)) + math.exp(2 * m * d / v) * ndtr((d + m) / math.sqrt(v))
        assert c == pytest.approx(expected, rel=1e-14)


def test_mass_already_through_the_barrier_crosses_at_once():
    law = ((np.array([math.log(70.0), math.log(100.0)]), np.array([0.3, 0.7])), 0.0, 0.04, math.log(75.0), False)
    crossed = days_mod.crossed_by(law, np.array([1e-12, 1.0]))
    assert crossed[0] == pytest.approx(0.3, abs=1e-12)


def test_survival_after_rows_drops_knock_outs_at_or_before_each_row():
    times = np.array([0.25, 0.5, 0.75])
    probability = np.array([0.1, 0.05, 0.02])
    survival = days_mod.survival_after_rows(times, probability, np.array([0.5, 0.25]), np.array([0.2, 0.3]))
    np.testing.assert_allclose(survival, [1 - 0.1 - 0.3, 1 - 0.15 - 0.5, 1 - 0.17 - 0.5], rtol=0, atol=1e-15)


def test_split_puts_each_interval_on_its_days_and_decided_mass_on_the_end_day():
    days = np.array([0.1, 0.2, 0.3, 0.4])
    t = np.array([0.2, 0.4])
    calls = []

    def crossed_at(i, inside):
        calls.append((i, inside.tolist()))
        return np.array([0.25, 1.0])            # a quarter of interval 1's crossing by day 0.3

    mass = days_mod.split_interval_mass(days, t, np.array([0.0, 0.4]), np.array([0.0, 0.1]), crossed_at,
                                        engine="test")
    np.testing.assert_allclose(mass, [0.0, 0.0, 0.1, 0.4], rtol=0, atol=1e-15)
    assert calls == [(1, [2])]


def test_a_crossing_interval_without_a_law_is_refused():
    from quantark.util.exceptions import PricingError

    with pytest.raises(PricingError, match="QUAD VX"):
        days_mod.split_interval_mass(np.array([0.1, 0.2, 0.3]), np.array([0.3]), np.array([0.5]), np.array([0.0]),
                                     lambda i, inside: np.zeros(inside.size + 1), engine="QUAD VX")
