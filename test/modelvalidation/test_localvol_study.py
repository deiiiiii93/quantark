"""Wiring and soundness tests for the local-vol snowball certification study.

These assert SOUNDNESS, not outcome. At the tiny sampling budget used here the
benchmark cannot meet its standard-error budget, so INCONCLUSIVE is correct;
asserting ADMITTED would be asserting that noise agrees with us. The real
verdict comes from the offline run whose evidence is banked.
"""

import math
from pathlib import Path

import pytest

from quantark.modelvalidation.builders.equity_snowball_localvol import (
    build_localvol_market_spec,
    build_localvol_mc_reference,
    build_localvol_pde_candidate,
    load_surface,
    make_localvol_environment,
)
from quantark.modelvalidation.study import CaseSpec, SamplingPolicy
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


# --------------------------------------------------------------------------
# The PDE candidate
# --------------------------------------------------------------------------

ENV = {"surface": CRASH, "rate": 0.02}
PRODUCT = {
    "strike_moneyness": 1.0,
    "ko_barrier_moneyness": 1.03,
    "ki_barrier_moneyness": 0.85,
    "ko_rate": 0.15,
    "rebate_rate": 0.15,
    "months": 12,
    "maturity": 1.0,
}


def _candidate(**params):
    return build_localvol_pde_candidate(
        environment_params=ENV,
        product_params=PRODUCT,
        quantities=("pv", "delta", "gamma"),
        params=params or {"accuracy": "standard"},
    )


def test_pde_candidate_is_named_for_its_engine():
    assert _candidate().name() == "equity.snowball.localvol_pde"


def test_pde_candidate_records_its_resolved_grid():
    """A profile name is an indirection; the resolved grid is the evidence."""
    params = _candidate().params()
    assert params["engine"] == "LocalVolSnowballPDESolver"
    assert params["grid"]["points"] > 0
    assert params["grid"]["steps_per_day"] > 0


def test_pde_candidate_produces_finite_greeks_with_a_ladder():
    result = _candidate().evaluate(CaseSpec(name="ordinary"))
    assert set(result.values) == {"pv", "delta", "gamma"}
    assert all(math.isfinite(v) for v in result.values.values())
    assert [rung.level for rung in result.ladders] == ["target", "medium"]


def test_pde_candidate_delta_is_stable_across_its_own_ladder():
    """FINDING-2026-08-26: the PDE moved 0.0079 contracts across its whole
    accuracy ladder, 63x tighter than the bound it was failing. If that is no
    longer true, the engine changed and the certification premise with it."""
    result = _candidate().evaluate(CaseSpec(name="ordinary"))
    target, medium = result.ladders
    assert abs(target.values["delta"] - medium.values["delta"]) < 0.05


# --------------------------------------------------------------------------
# The local-vol Monte-Carlo reference
# --------------------------------------------------------------------------

TINY = SamplingPolicy(
    paths_per_batch=1024, min_batches=2, max_batches=3, seed=20260828, bump=0.01
)


def _reference(**params):
    return build_localvol_mc_reference(
        environment_params=ENV,
        product_params=PRODUCT,
        sampling=TINY,
        quantities=("pv", "delta", "gamma"),
        params=params,
    )


def test_reference_declares_the_discretization_it_runs():
    """The FINDING's root cause A was a reference whose declared substeps were
    not the ones it executed. The config must report what run_batch uses."""
    config = _reference().config()
    assert config["substeps_per_interval"] == 8
    assert config["lv_time_sampling"] == "integrated"
    assert config["estimator"] == "plain"
    assert config["engine"] == "LocalVolSnowballMCEngine"


def test_reference_honours_the_seed_contract():
    """_validate_batch requires seed == policy.seed + index; that contract is
    what makes each batch an independent Sobol scramble."""
    batch = _reference().run_batch(CaseSpec(name="ordinary"), 0)
    assert batch.index == 0
    assert batch.seed == TINY.seed


def test_reference_produces_all_three_finite_quantities():
    batch = _reference().run_batch(CaseSpec(name="ordinary"), 1)
    assert set(batch.values) == {"pv", "delta", "gamma"}
    assert all(math.isfinite(v) for v in batch.values.values())


def test_reference_batches_differ():
    """Identical batches would collapse the standard error toward zero and fire
    SE_BUDGET_MET on noise -- a false ADMITTED."""
    ref = _reference()
    a = ref.run_batch(CaseSpec(name="ordinary"), 0)
    b = ref.run_batch(CaseSpec(name="ordinary"), 1)
    assert a.values["pv"] != b.values["pv"]


def test_reference_identity_pins_the_surface_bytes():
    identity = _reference().identity(CaseSpec(name="ordinary"))
    assert identity["surface_sha256"].startswith(CRASH_SHA16)


def test_unsupported_reference_knob_is_refused():
    """A knob that is recorded but never applied would move the identity hash
    while moving no number."""
    with pytest.raises(ValidationError, match="localvol_mc"):
        _reference(martingale_correction=True)
