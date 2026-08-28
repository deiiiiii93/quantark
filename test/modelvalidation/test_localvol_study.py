"""Wiring and soundness tests for the local-vol snowball certification study.

These assert SOUNDNESS, not outcome. At the tiny sampling budget used here the
benchmark cannot meet its standard-error budget, so INCONCLUSIVE is correct;
asserting ADMITTED would be asserting that noise agrees with us. The real
verdict comes from the offline run whose evidence is banked.
"""

from pathlib import Path

import pytest

from quantark.modelvalidation.builders.equity_snowball_localvol import (
    REFERENCE_SPOT,
    build_localvol_market_spec,
    load_surface,
    make_localvol_environment,
)
from quantark.util.exceptions import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "example" / "modelvalidation" / "data"

CRASH = "example/modelvalidation/data/iv_surface_20240208.json"
CALM = "example/modelvalidation/data/iv_surface_20231115.json"

# Pinned from the artifacts as committed. A change here means the surface
# bytes moved, which invalidates every certificate built on them.
CRASH_S0 = 4993.105
CALM_S0 = 6207.268
CRASH_SHA16 = "b0e63653a774b5b3"
CALM_SHA16 = "a7917303394e114f"


def test_both_surface_artifacts_are_present():
    assert (DATA / "iv_surface_20240208.json").is_file()
    assert (DATA / "iv_surface_20231115.json").is_file()


@pytest.mark.parametrize(
    "path, s0, sha16",
    [(CRASH, CRASH_S0, CRASH_SHA16), (CALM, CALM_S0, CALM_SHA16)],
)
def test_artifact_identity_is_pinned(path, s0, sha16):
    """The sha is what pins a certificate to exact surface bytes."""
    surface = load_surface(path, 0.02)
    assert surface.artifact.s0 == pytest.approx(s0, abs=1e-3)
    assert surface.artifact.sha256.startswith(sha16)


def test_surface_is_cached_not_rebuilt():
    """Rebuilding per call would re-run Dupire on every bumped price."""
    assert load_surface(CRASH, 0.02) is load_surface(CRASH, 0.02)


def test_local_vol_is_built_at_the_artifact_spot():
    """Not at a bumped spot -- otherwise delta absorbs a surface-rebuild term."""
    surface = load_surface(CRASH, 0.02)
    bumped = make_localvol_environment(
        {"surface": CRASH, "rate": 0.02}, spot=CRASH_S0 * 1.01
    )
    assert bumped.spot == pytest.approx(CRASH_S0 * 1.01)
    # The surface object handed to the engines is the same one regardless.
    assert load_surface(CRASH, 0.02).local_vol is surface.local_vol


def test_environment_carries_the_artifact_trade_date_and_carry():
    env = make_localvol_environment({"surface": CRASH, "rate": 0.02})
    assert env.valuation_date.date().isoformat() == "2024-02-08"
    assert env.spot == pytest.approx(CRASH_S0, abs=1e-3)


def test_unknown_environment_key_is_refused():
    with pytest.raises(ValidationError, match="localvol_market"):
        build_localvol_market_spec({"surface": CRASH, "rate": 0.02, "vol": 0.2})


def test_missing_surface_path_is_refused():
    with pytest.raises(ValidationError, match="surface"):
        build_localvol_market_spec({"rate": 0.02})
