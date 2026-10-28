"""Matched-accuracy scoring used by the tuning ladders (spec D13)."""
from __future__ import annotations

import pytest

from example.case_studies.localvol_transition_quad.benchmark import score
from example.case_studies.localvol_transition_quad.engine import OUTPUTS

REF = {"pv": 0.9, "delta": 0.6, "gamma": -3.5, "vega": -1.7e-3, "theta": 6.4e-4, "rho": -2e-6, "carry": -3.5e-5}


def test_deterministic_score_names_the_binding_quantity():
    values = dict(REF, pv=0.9 + 5e-5, theta=6.4e-4 + 8e-6)
    s = score(values, REF)
    assert s["binding"] == "theta" and s["worst"] == pytest.approx(0.8) and s["meets"]
    assert s["ratio"]["pv"] == pytest.approx(0.5)


def test_mc_score_adds_two_standard_errors():
    values = dict(REF)
    stderr = dict.fromkeys(OUTPUTS, 0.0)
    stderr["gamma"] = 0.02                         # 2 SE = 0.04 > budget 0.035
    s = score(values, REF, stderr)
    assert s["binding"] == "gamma" and not s["meets"]
    assert s["ratio"]["gamma"] == pytest.approx(0.04 / 0.035)
