"""Study builders must express the product variants the release never certified.

Discrete KI, European KI, step-down KO and parachute KO are all *product*
variants, so they belong in the existing studies as cases rather than in new
studies. That only works if the builders can express them -- and if expressing
them leaves every already-certified cell's identity untouched, which is what
lets an amendment carry those cells forward instead of re-pricing them.
"""

from pathlib import Path

import pytest

from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from quantark.modelvalidation.builders.equity_snowball import (
    build_snowball_product_spec,
    make_snowball,
)
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The snowball study's product block, verbatim from snowball_flat_bsm.yaml.
SNOWBALL = {
    "initial_price": 100.0,
    "strike": 100.0,
    "ko_barrier": 103.0,
    "ki_barrier": 85.0,
    "ko_rate": 0.15,
    "rebate_rate": 0.15,
    "months": 12,
    "maturity": 1.0,
    "contract_multiplier": 1.0,
}

#: The phoenix study's product block, verbatim from phoenix_flat_bsm.yaml.
PHOENIX = {
    "initial_price": 100.0,
    "strike": 100.0,
    "ko_barrier": 103.0,
    "ko_rate": 0.0,
    "ki_barrier": 75.0,
    "coupon_barrier": 85.0,
    "coupon_rate": 0.02,
    "num_observations": 12,
    "memory_coupon": False,
    "maturity": 1.0,
    "contract_multiplier": 1.0,
}


def snowball(**overrides):
    return make_snowball(dict(SNOWBALL, **overrides))


def phoenix(**overrides):
    return make_phoenix(dict(PHOENIX, **overrides))


# --- KI monitoring ---------------------------------------------------------


def test_default_ki_monitoring_is_continuous():
    """The certified cells say nothing about KI monitoring and must not move."""
    barriers = snowball().barrier_config
    assert barriers.ki_continuous is True


def test_discrete_ki_is_monitored_on_the_ko_observation_dates():
    """Market convention: a discretely monitored KI shares the KO schedule."""
    barriers = snowball(ki_monitoring="discrete").barrier_config
    assert barriers.ki_continuous is False
    assert barriers.ki_observation_type is ObservationType.DISCRETE
    assert barriers.ki_observation_dates == pytest.approx(
        [(i + 1) / 12 for i in range(12)]
    )


def test_european_ki_is_observed_only_at_maturity():
    barriers = snowball(ki_monitoring="european").barrier_config
    assert barriers.ki_continuous is False
    assert barriers.ki_observation_type is ObservationType.DISCRETE
    assert barriers.ki_observation_dates == [1.0]


def test_unknown_ki_monitoring_is_rejected():
    with pytest.raises(ValidationError, match="ki_monitoring must be one of"):
        build_snowball_product_spec(dict(SNOWBALL, ki_monitoring="sometimes"))


# --- KO barrier shape ------------------------------------------------------


def test_step_down_ko_decrements_the_barrier_once_per_observation():
    """0.5% of initial price per observation: 103.0, 102.5, ... , 97.5."""
    barriers = snowball(ko_stepdown=0.005).barrier_config
    assert barriers.ko_barrier == pytest.approx(
        [103.0 - 0.5 * i for i in range(12)]
    )


def test_parachute_drops_the_final_ko_barrier_onto_the_ki_barrier():
    barriers = snowball(parachute=True).barrier_config
    assert barriers.ko_barrier == [103.0] * 11 + [85.0]
    assert barriers.ko_barrier[-1] == barriers.ki_barrier


def test_step_down_and_parachute_compose():
    barriers = snowball(ko_stepdown=0.005, parachute=True).barrier_config
    assert barriers.ko_barrier == pytest.approx(
        [103.0 - 0.5 * i for i in range(11)] + [85.0]
    )


def test_step_down_through_the_ki_barrier_is_rejected():
    """A KO schedule that walks below KI is a different product, not a step-down."""
    with pytest.raises(ValidationError, match="walks below the ki_barrier"):
        build_snowball_product_spec(dict(SNOWBALL, ko_stepdown=0.05))


def test_flat_ko_barrier_stays_a_scalar():
    """Absent both knobs the barrier must not silently become a 1-element list."""
    assert snowball().barrier_config.ko_barrier == 103.0


# --- Phoenix ---------------------------------------------------------------


def test_phoenix_steps_down_both_the_ko_and_coupon_barriers():
    """A step-down phoenix walks the coupon barrier down with the KO barrier."""
    product = phoenix(ko_stepdown=0.005, coupon_stepdown=0.005)
    assert product.barrier_config.ko_barrier == pytest.approx(
        [103.0 - 0.5 * i for i in range(12)]
    )
    assert product.coupon_config.coupon_barrier == pytest.approx(
        [85.0 - 0.5 * i for i in range(12)]
    )


def test_phoenix_default_barriers_stay_scalar():
    product = phoenix()
    assert product.barrier_config.ko_barrier == 103.0
    assert product.coupon_config.coupon_barrier == 85.0


# --- The property the amendment depends on ---------------------------------
#
# An amendment carries a cell forward only when its candidate identity AND its
# benchmark identity still hash to what the parent banked. If adding these
# variant knobs perturbed either hash, every already-certified cell would
# silently re-price -- hours of benchmark time, and a new claim where a carried
# one was intended. These tests pin that: they must pass before the builder
# change and after it.

STUDY_YAML = {
    "snowball-flat-bsm": "example/modelvalidation/snowball_flat_bsm.yaml",
    "phoenix-flat-bsm": "example/modelvalidation/phoenix_flat_bsm.yaml",
    "ko-reset-flat-bsm": "example/modelvalidation/ko_reset_flat_bsm.yaml",
}

BANKED_CERTIFICATES = sorted(
    (REPO_ROOT / "docs" / "modelvalidation" / "certificates").glob("*/*/certificate.json")
)


def _certificate_id(path):
    return f"{path.parent.parent.name}/{path.parent.name}"


@pytest.mark.parametrize("certificate_path", BANKED_CERTIFICATES, ids=_certificate_id)
def test_banked_cells_keep_their_identity(certificate_path):
    """Every case a certificate banked must still hash identically today."""
    import json

    from quantark.modelvalidation.candidate import candidate_identity
    from quantark.modelvalidation.evidence import identity_hash
    from quantark.modelvalidation.yaml_loader import load_study

    payload = json.loads(certificate_path.read_text())
    study_name = payload["study"]["name"]
    if study_name not in STUDY_YAML:
        pytest.skip(f"no YAML registered for {study_name}")

    study = load_study(REPO_ROOT / STUDY_YAML[study_name])
    cases = {case.name: case for case in study.cases}
    candidates = {c.name(): c for c in study.candidates}

    for name, block in payload["references"].items():
        assert name in cases, f"{study_name} no longer defines case {name!r}"
        assert identity_hash(study.reference.identity(cases[name])) == block["identity_hash"], (
            f"benchmark identity moved for case {name!r}: the amendment would "
            "re-run this reference instead of carrying it"
        )

    for cell in payload["cells"]:
        candidate = candidates[cell["candidate"]]
        current = identity_hash(candidate_identity(candidate, cases[cell["case"]]))
        assert current == cell["identity_hash"], (
            f"candidate identity moved for {cell['candidate']} / {cell['case']}: "
            "the amendment would re-price this cell instead of carrying it"
        )


# --- Case overrides must be validated too ----------------------------------
#
# The study-level validator only ever sees the study's own product block. A
# variant is expressed as a *case override*, which is merged in later by the
# arms -- so without validation at that seam a typo would not raise, it would
# quietly build a different product and certify it under the wrong name.

ENVIRONMENT = {"spot": 100.0, "vol": 0.22, "rate": 0.025, "div_yield": 0.03}


def test_make_snowball_refuses_an_unknown_ki_monitoring():
    """Silently falling back to discrete monitoring would certify a lie."""
    with pytest.raises(ValidationError, match="ki_monitoring must be one of"):
        snowball(ki_monitoring="contineous")


def test_arms_validate_the_merged_case_spec():
    """A step-down that crosses the KI barrier is only catchable by the validator."""
    from quantark.modelvalidation.builders.equity_snowball import SnowballPDECandidate
    from quantark.modelvalidation.study import CaseSpec

    arm = SnowballPDECandidate(
        environment_params=ENVIRONMENT,
        product_params=SNOWBALL,
        quantities=("pv",),
        params={},
    )
    with pytest.raises(ValidationError, match="walks below the ki_barrier"):
        arm.evaluate(CaseSpec(name="crossing", product_params={"ko_stepdown": 0.05}))


def test_phoenix_arms_validate_the_merged_case_spec():
    from quantark.modelvalidation.builders.equity_phoenix import PhoenixQuadCandidate
    from quantark.modelvalidation.study import CaseSpec

    arm = PhoenixQuadCandidate(
        environment_params=ENVIRONMENT,
        product_params=PHOENIX,
        quantities=("pv",),
        params={},
    )
    with pytest.raises(ValidationError, match="walks coupon_barrier below"):
        arm.evaluate(CaseSpec(name="crossing", product_params={"coupon_stepdown": 0.05}))
