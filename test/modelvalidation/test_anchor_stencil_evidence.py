"""The stencil-propagated anchor tolerance, checked against the banked intraday evidence it exists for.

A tolerance cannot be validated on the machine the anchors were frozen on: there the comparison is exact.
So both directions are pinned here from recorded numbers instead of from a CI run -- the drift the
architectures actually produced must be admitted, and the tolerance must stay far below any change that
matters. Neither test prices anything; each resolves the study's cases to read the declared stencils.
"""
import collections
import json
from pathlib import Path

import pytest

from quantark.modelvalidation.anchors import CROSS_ARCH_ABS_TOL, CROSS_ARCH_REL_TOL, anchor_tolerance
from quantark.modelvalidation.yaml_loader import load_study_text

ANCHORS = (Path(__file__).resolve().parents[2]
           / "docs/modelvalidation/certificates/snowball-intraday-daily-ki-bsm/2026-09-19/anchors.json")

#: What x86_64 Linux (GitHub Actions run 36380062818, Python 3.11, 2026-09-28) computed minus the values
#: banked on ARM64 macOS -- the six anchors the flat relative tolerance rejected. Five difference prices
#: over a 1% spot bump or a 1-10 s theta step; the PVs they difference all agreed within 1e-9.
CI_X86_DRIFT = {
    ("equity.snowball.intraday.pde", "near_ki_10s", "desk_theta"): 1.229e-08,
    ("equity.snowball.intraday.pde", "near_ko_on_ko_day_1s", "desk_theta"): 2.175e-07,
    ("equity.snowball.intraday.pde", "near_ko_on_ko_day_1s", "point_gamma"): -9.334e-11,
    ("equity.snowball.intraday.pde", "uniform_profile", "desk_gamma"): 2.365e-10,
    ("equity.snowball.intraday.quad_v2", "near_ki_1s", "desk_theta"): -3.837e-10,
    ("equity.snowball.intraday.quad_v2", "near_ko_on_ko_day_1s", "desk_theta"): 1.663e-10,
}


@pytest.fixture(scope="module")
def tolerances():
    """(candidate, case, quantity) -> (anchored value, cross-architecture tolerance)."""
    anchors = json.loads(ANCHORS.read_text())
    study = load_study_text(anchors["study_source_text"])
    candidates = {c.name(): c for c in study.candidates}
    cases = {c.name: c for c in study.cases}
    out = {}
    for entry in anchors["anchors"]:
        weights = candidates[entry["candidate"]].anchor_noise_weights(cases[entry["case"]])
        pv = entry["values"].get("pv")
        for quantity, value in entry["values"].items():
            out[(entry["candidate"], entry["case"], quantity)] = (value, anchor_tolerance(
                value, pv=pv, weight=weights.get(quantity), rel_tol=CROSS_ARCH_REL_TOL, abs_tol=CROSS_ARCH_ABS_TOL))
    return out


def test_the_drift_ci_measured_is_admitted_with_room_to_spare(tolerances):
    for key, drift in CI_X86_DRIFT.items():
        _, tolerance = tolerances[key]
        assert abs(drift) * 100.0 <= tolerance, f"{key}: drift {drift:.3e} against tolerance {tolerance:.3e}"


def test_the_tolerance_stays_far_below_any_change_that_matters(tolerances):
    """Measured against each quantity's column scale, not the anchored value: a Greek that is ~0 in one case
    has no relative scale of its own. The cubic delta/gamma readout (e0ba6ba7) moved PDE gamma by about 1% of
    scale; anything within four orders of that must still fail off the banking machine."""
    scale, worst = collections.defaultdict(float), collections.defaultdict(float)
    for (_, _, quantity), (value, tolerance) in tolerances.items():
        scale[quantity] = max(scale[quantity], abs(value))
        worst[quantity] = max(worst[quantity], tolerance)
    ceiling = {"pv": 1e-8, "desk_delta": 1e-7, "desk_gamma": 1e-7, "point_delta": 1e-7, "point_gamma": 1e-7,
               "desk_theta": 1e-5}
    for quantity, limit in ceiling.items():
        assert worst[quantity] <= limit * scale[quantity], (
            f"{quantity}: tolerance {worst[quantity]:.3e} is {worst[quantity] / scale[quantity]:.1e} of its scale")
