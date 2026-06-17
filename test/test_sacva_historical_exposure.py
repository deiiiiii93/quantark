"""Tests for the SA-CVA non-regulatory HistoricalExposureEngine.

Run (worktree shadow):
    PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
        test/test_sacva_historical_exposure.py -v
"""
import numpy as np
import pandas as pd
import pytest
from types import MappingProxyType
from math import erf, sqrt, log, exp

from quantark.util.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Task 1 — provisional ExposureProfile contract
# ---------------------------------------------------------------------------
from quantark.sacva.exposure._contract_provisional import (
    Measure, ExposureProfile, CONTRACT_VERSION,
)


def test_mc_positional_construction_still_works():
    p = ExposureProfile(np.array([0., 1.]), np.array([5., 3.]), Measure.RISK_NEUTRAL, True)
    assert p.regulatory_eligible and p.epe_discounted is not None
    assert p.ee_undiscounted is None and p.pfe is None


def test_real_world_cannot_be_eligible():
    with pytest.raises(ValidationError):
        ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, True)


def test_eligible_requires_epe_discounted():
    with pytest.raises(ValidationError):
        ExposureProfile(np.array([0., 1.]), None, Measure.RISK_NEUTRAL, True)


def test_real_world_must_not_populate_epe_discounted():
    with pytest.raises(ValidationError):
        ExposureProfile(np.array([0., 1.]), np.array([5., 3.]), Measure.REAL_WORLD, False)


def test_shape_invariants():
    with pytest.raises(ValidationError):
        ExposureProfile(np.array([0., 1., 2.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([2., 1.]))   # wrong length


def test_arrays_and_metadata_are_immutable():
    p = ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([2., 1.]),
                        pfe={9900: np.array([4., 3.])}, metadata={"k": "v"})
    with pytest.raises(ValueError):
        p.ee_undiscounted[0] = 99.0                # read-only array
    assert isinstance(p.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        p.metadata["k"] = "x"


def test_historical_profile_ok_and_version():
    p = ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([2., 1.]),
                        pfe={9900: np.array([4., 3.])}, epe=1.5)
    assert not p.regulatory_eligible and p.pfe[9900][0] == 4.0
    assert p.epe_scalar == 1.5                     # back-compat alias
    assert isinstance(CONTRACT_VERSION, str) and CONTRACT_VERSION


def test_pfe_and_epe_validation():
    with pytest.raises(ValidationError):           # negative PFE
        ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([1., 1.]), pfe={9900: np.array([-1., 2.])})
    with pytest.raises(ValidationError):           # bad bps key
        ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([1., 1.]), pfe={12000: np.array([1., 2.])})
