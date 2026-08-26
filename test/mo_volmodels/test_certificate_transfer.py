"""Tests for example/mo_volmodels/certificate_transfer.py.

The module answers ONE question: do the states this study's replay visits
fall inside the regime span the banked ADI 2-D Greek certificate was designed
to straddle?  It is a coverage audit, not a licence -- the certificate's
admitted verdict is an AGGREGATE mean signed bias over seven archetypes
against a 0.1-contract bound; individually each cell was certified only to
0.5.  Decomposing that into per-date permissions would claim precision the
evidence does not carry, so nothing here gates pricing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "example/mo_volmodels/certificate_transfer.py"

spec = importlib.util.spec_from_file_location("certificate_transfer", MODULE_PATH)
ct = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ct
spec.loader.exec_module(ct)

ORDINARY = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.30, "rho": -0.55}
# The three degenerate fits measured inside the real replay window.
FIT_20230608 = {"v0": 0.034479, "kappa": 1.99, "theta": 0.0001, "sigma": 0.001, "rho": -0.09714}
FIT_20241010 = {"v0": 0.15899, "kappa": 2.068, "theta": 0.018037, "sigma": 0.001459, "rho": -0.00404}
FIT_20250409 = {"v0": 0.14027, "kappa": 3.0, "theta": 0.0030641, "sigma": 0.0031095, "rho": -0.4501}


# ---------------------------------------------------------------------------
# The envelope must not drift from the certification it claims to describe
# ---------------------------------------------------------------------------

def test_the_certified_envelope_matches_stage_16s_own_case_definitions():
    """These constants describe someone else's banked evidence.

    They are transcribed rather than imported (stage 16 is heavy and is
    itself an implementation input to its own certification hash), so this
    test is what stops the transcription from going stale.
    """
    gate_path = ROOT / "example/mo_volmodels/16_adi_greek_certification.py"
    gate_spec = importlib.util.spec_from_file_location("adi_cert_16", gate_path)
    gate = importlib.util.module_from_spec(gate_spec)
    sys.modules[gate_spec.name] = gate
    gate_spec.loader.exec_module(gate)

    cases = {c.name: c for c in gate.certification_cases(quick=False)}
    assert set(cases) == set(ct.CERTIFIED_CELLS)
    for name, case in cases.items():
        cell = ct.CERTIFIED_CELLS[name]
        assert cell.maturity == pytest.approx(case.maturity)
        assert cell.feller_ratio == pytest.approx(
            ct.feller_ratio(
                {
                    "kappa": case.params.kappa,
                    "theta": case.params.theta,
                    "sigma": case.params.sigma,
                }
            )
        )


def test_the_envelope_is_the_measured_extremes_of_those_cells():
    assert ct.CERTIFIED_RATIO_MIN == pytest.approx(0.1920, rel=1e-3)
    assert ct.CERTIFIED_RATIO_MAX == pytest.approx(1898.2434, rel=1e-6)
    assert ct.CERTIFIED_MATURITY_MIN == pytest.approx(0.25)
    assert ct.CERTIFIED_MATURITY_MAX == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# In-span / out-of-span
# ---------------------------------------------------------------------------

def test_an_ordinary_fit_is_in_span_and_names_its_witness():
    v = ct.state_in_span(ORDINARY, remaining_years=2.0)
    assert v.in_span is True
    assert v.bucket == "boundary"
    assert v.witness in ct.CERTIFIED_CELLS
    assert ct.CERTIFIED_CELLS[v.witness].bucket == "boundary"


def test_a_ratio_between_two_certified_cells_is_in_span():
    """2023-06-08 sits at 398 -- degenerate, but under the 1898 archetype.

    Interpolating between certified regimes is what the seven-cell design is
    FOR.  Only going past the extremes is extrapolation.
    """
    v = ct.state_in_span(FIT_20230608, remaining_years=2.5)
    assert ct.feller_ratio(FIT_20230608) == pytest.approx(398.0, rel=1e-3)
    assert v.in_span is True
    assert v.witness == "sigma_collapse"


def test_the_archetypes_own_source_date_is_in_span():
    """sigma_collapse was BUILT from the 2025-04-09 fit, so it must cover it."""
    v = ct.state_in_span(FIT_20250409, remaining_years=3.0)
    assert v.in_span is True
    assert v.witness == "sigma_collapse"


def test_a_ratio_past_the_most_extreme_certified_cell_is_out_of_span():
    """2024-10-10 sits at 35,048 -- 18x beyond the 1,898 archetype."""
    v = ct.state_in_span(FIT_20241010, remaining_years=2.0)
    assert ct.feller_ratio(FIT_20241010) == pytest.approx(35047.9, rel=1e-4)
    assert v.in_span is False
    assert "1898" in v.reason or "1,898" in v.reason


def test_a_ratio_below_the_certified_floor_is_out_of_span():
    below = {"v0": 0.04, "kappa": 0.1, "theta": 0.005, "sigma": 0.7, "rho": -0.3}
    assert ct.feller_ratio(below) < ct.CERTIFIED_RATIO_MIN
    assert ct.state_in_span(below, remaining_years=2.0).in_span is False


@pytest.mark.parametrize("remaining", [3.5, 0.1, 0.0, -1.0])
def test_a_maturity_outside_the_certified_range_is_out_of_span(remaining):
    v = ct.state_in_span(ORDINARY, remaining_years=remaining)
    assert v.in_span is False
    assert "maturit" in v.reason.lower()


@pytest.mark.parametrize(
    "params",
    [
        {"kappa": 2.0, "theta": 0.04},                     # no sigma
        {"kappa": 2.0, "theta": 0.04, "sigma": 0.0},       # sigma outside its bound
        {"kappa": 2.0, "theta": 0.04, "sigma": "n/a"},     # malformed
    ],
)
def test_an_unrankable_state_fails_closed(params):
    v = ct.state_in_span(params, remaining_years=2.0)
    assert v.in_span is False
    assert v.bucket == "unknown"
    assert v.witness is None


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

def _states():
    return [
        ("2023-06-08", FIT_20230608, 2.5),
        ("2024-10-10", FIT_20241010, 2.0),
        ("2025-04-09", FIT_20250409, 3.0),
        ("2025-05-06", ORDINARY, 1.0),
    ]


def test_the_audit_names_every_out_of_span_state():
    report = ct.audit(_states())
    assert report["n_states"] == 4
    assert report["n_out_of_span"] == 1
    assert [row["label"] for row in report["out_of_span"]] == ["2024-10-10"]
    assert report["out_of_span"][0]["feller_ratio"] == pytest.approx(35047.9, rel=1e-4)


def test_the_audit_reports_the_regime_histogram_and_the_ratio_range():
    report = ct.audit(_states())
    assert report["buckets"] == {
        "violated": 0, "boundary": 1, "degenerate": 3, "unknown": 0
    }
    assert report["feller_ratio"]["max"] == pytest.approx(35047.9, rel=1e-4)
    assert report["feller_ratio"]["min"] == pytest.approx(1.7778, rel=1e-3)


def test_the_audit_reports_it_never_raises():
    """Report-only is the whole posture: an out-of-span state is a finding."""
    report = ct.audit(_states() + [("bad", {"kappa": 1.0}, 2.0)])
    assert report["n_out_of_span"] == 2
    assert report["buckets"]["unknown"] == 1
    assert report["covered"] is False


def test_a_fully_covered_fleet_says_so():
    report = ct.audit([("a", ORDINARY, 1.0), ("b", FIT_20250409, 3.0)])
    assert report["covered"] is True
    assert report["n_out_of_span"] == 0
    assert report["out_of_span"] == []


def test_an_empty_audit_does_not_claim_coverage():
    """Nothing measured is not the same as nothing wrong."""
    report = ct.audit([])
    assert report["n_states"] == 0
    assert report["covered"] is None


def test_the_audit_records_the_certificate_it_was_written_against():
    report = ct.audit(_states())
    assert report["study"] == "adi2d-snowball-greeks"
    assert report["certificate"]["ratio_envelope"] == [
        pytest.approx(ct.CERTIFIED_RATIO_MIN),
        pytest.approx(ct.CERTIFIED_RATIO_MAX),
    ]
    assert set(report["certificate"]["cells"]) == set(ct.CERTIFIED_CELLS)


# ---------------------------------------------------------------------------
# The endpoint tolerance is measured, and must not become a loophole
# ---------------------------------------------------------------------------

def test_the_endpoint_tolerance_is_the_archetypes_own_quoted_precision():
    """sigma_collapse is the 2025-04-09 fit quoted to three significant figures.

    Design spec section 5.9 quotes it as sigma=0.00311, kappa=3.0,
    theta=0.00306, and stage 16 hardcodes exactly those.  That quote pins the
    ratio only to within +-0.4849%, so the tolerance is derived from the
    archetype's own specification, not chosen to make a date pass.
    """
    lo = 2 * 3.0 * 0.003055 / 0.003115 ** 2
    hi = 2 * 3.0 * 0.003065 / 0.003105 ** 2
    half_width = (hi - lo) / 2 / ct.CERTIFIED_RATIO_MAX
    assert half_width == pytest.approx(0.004849, rel=1e-3)
    assert ct.ENDPOINT_REL_TOL >= half_width
    assert ct.ENDPOINT_REL_TOL < 0.01, "a wider tolerance stops being precision"


def test_the_tolerance_does_not_admit_a_different_regime():
    """2% past the endpoint is a different state, not a rounding artefact."""
    two_pct = ct.CERTIFIED_RATIO_MAX * 1.02
    params = {"kappa": 3.0, "theta": 0.00306, "sigma": (2 * 3.0 * 0.00306 / two_pct) ** 0.5}
    assert ct.feller_ratio(params) == pytest.approx(two_pct, rel=1e-9)
    assert ct.state_in_span(params, remaining_years=3.0).in_span is False


def test_a_three_year_trade_on_its_inception_day_is_in_span():
    """The archetype's T=3.0 means "a three-year trade", to day-count precision.

    A 3Y snowball struck 2023-05-04 matures 2026-05-06; that is 1098 days,
    or 3.0062 years at ACT/365.25.  Reporting the FIRST DAY of a three-year
    trade as outside a span whose widest cell IS a three-year trade would be
    a day-count artefact reported as a study finding.  A year specified as
    "3.0" is pinned only to within a leap day either way (3*365/365.25 =
    2.9979 .. 3*366/365.25 = 3.0062), which is inside the same endpoint
    tolerance the ratio axis uses.
    """
    assert ct.CERTIFIED_MATURITY_MAX == pytest.approx(3.0)
    assert 3 * 366 / 365.25 == pytest.approx(3.00616, rel=1e-4)
    for remaining in (3.0, 3.0034, 3.0062):
        v = ct.state_in_span(ORDINARY, remaining_years=remaining)
        assert v.in_span is True, f"{remaining} should be in span"


def test_the_maturity_tolerance_does_not_admit_a_longer_trade():
    assert ct.state_in_span(ORDINARY, remaining_years=3.2).in_span is False
    assert ct.state_in_span(ORDINARY, remaining_years=0.2).in_span is False
