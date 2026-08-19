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


# --- KO-reset --------------------------------------------------------------
#
# The KO-reset study carries TWO KO schedules, so a step-down has two barriers
# to walk and the parachute lands on the pre-KI one. Its `discrete_ki` case
# already spells monitoring `ki_continuous: false`, and a banked certificate
# hashes that spelling -- so it stays accepted alongside the newer
# `ki_monitoring`, which is the only way to say "European".

KO_RESET = {
    "initial_price": 100.0,
    "strike": 100.0,
    "maturity_pre": 1.0,
    "maturity_post": 2.0,
    "pre_ko_barrier": 103.0,
    "pre_ko_rate": 0.15,
    "post_ko_barrier": 95.0,
    "post_ko_rate": 0.03,
    "ki_barrier": 80.0,
    "ki_continuous": True,
    "contract_multiplier": 1.0,
}

N_PRE, N_POST = 12, 24


def ko_reset(**overrides):
    from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset

    return make_ko_reset(dict(KO_RESET, **overrides))


def test_ko_reset_default_barriers_stay_scalar():
    product = ko_reset()
    assert product.barrier_config.ko_barrier == 103.0
    assert product.post_barrier_config.ko_barrier == 95.0


def test_ko_reset_legacy_ki_continuous_spelling_still_works():
    """The parent certificate hashes {ki_continuous: false}; it must keep working."""
    from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset

    barriers = make_ko_reset(dict(KO_RESET, ki_continuous=False)).barrier_config
    assert barriers.ki_continuous is False
    assert len(barriers.ki_observation_dates) > 200  # daily over the pre horizon


def test_ko_reset_european_ki_is_observed_once_at_the_pre_maturity():
    barriers = ko_reset(ki_monitoring="european").barrier_config
    assert barriers.ki_continuous is False
    assert barriers.ki_observation_dates == [1.0]


def test_ko_reset_discrete_ki_matches_the_legacy_spelling():
    """Two spellings, one product -- otherwise the vocabulary is a trap."""
    from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset

    legacy = make_ko_reset(dict(KO_RESET, ki_continuous=False)).barrier_config
    modern = ko_reset(ki_monitoring="discrete").barrier_config
    assert modern.ki_continuous == legacy.ki_continuous
    assert modern.ki_observation_dates == legacy.ki_observation_dates


def test_ko_reset_ki_monitoring_overrides_the_legacy_spelling():
    """The study block already sets ki_continuous, so a case that says
    ki_monitoring always merges into a spec carrying BOTH keys. Rejecting that
    combination would make the newer key unusable in this study, so
    ki_monitoring simply wins."""
    barriers = ko_reset(ki_continuous=True, ki_monitoring="european").barrier_config
    assert barriers.ki_continuous is False
    assert barriers.ki_observation_dates == [1.0]


def test_ko_reset_rejects_an_unknown_ki_monitoring():
    with pytest.raises(ValidationError, match="ki_monitoring must be one of"):
        ko_reset(ki_monitoring="at_maturity")


def test_ko_reset_steps_down_both_schedules():
    product = ko_reset(pre_ko_stepdown=0.005, post_ko_stepdown=0.005)
    assert product.barrier_config.ko_barrier == pytest.approx(
        [103.0 - 0.5 * i for i in range(N_PRE)]
    )
    assert product.post_barrier_config.ko_barrier == pytest.approx(
        [95.0 - 0.5 * i for i in range(N_POST)]
    )


def test_ko_reset_parachute_lands_the_last_pre_barrier_on_the_ki_barrier():
    product = ko_reset(parachute=True)
    assert product.barrier_config.ko_barrier == [103.0] * (N_PRE - 1) + [80.0]
    # The post schedule is untouched: the parachute is a pre-KI feature.
    assert product.post_barrier_config.ko_barrier == 95.0


def test_ko_reset_step_down_through_the_ki_barrier_is_rejected():
    from quantark.modelvalidation.builders.equity_ko_reset import (
        build_ko_reset_product_spec,
    )

    with pytest.raises(ValidationError, match="walks below the ki_barrier"):
        build_ko_reset_product_spec(dict(KO_RESET, post_ko_stepdown=0.05))


def test_ko_reset_arms_validate_the_merged_case_spec():
    from quantark.modelvalidation.builders.equity_ko_reset import KOResetQuadCandidate
    from quantark.modelvalidation.study import CaseSpec

    arm = KOResetQuadCandidate(
        environment_params=ENVIRONMENT,
        product_params=KO_RESET,
        quantities=("pv",),
        params={},
    )
    with pytest.raises(ValidationError, match="walks below the ki_barrier"):
        arm.evaluate(CaseSpec(name="crossing", product_params={"post_ko_stepdown": 0.05}))


# --- The rest of the snowball feature surface ------------------------------
#
# Everything the product model offers that the release never priced. Each knob
# is absent by default, so an unset knob leaves the product exactly as the
# original certification built it.

from quantark.util.enum import CouponPayType, OptionType, ProtectionType  # noqa: E402


def test_ki_stepdown_walks_the_knock_in_barrier_down():
    """A KI barrier that declines 0.5% of initial price per observation."""
    barriers = snowball(ki_monitoring="discrete", ki_stepdown=0.005).barrier_config
    assert barriers.ki_barrier == pytest.approx([85.0 - 0.5 * i for i in range(12)])


def test_ki_stepdown_requires_discrete_monitoring():
    """Both deterministic engines refuse a vector KI under continuous
    monitoring, so the study must not be able to ask for one."""
    with pytest.raises(ValidationError, match="ki_stepdown requires"):
        build_snowball_product_spec(dict(SNOWBALL, ki_stepdown=0.005))


def test_ki_stepdown_through_zero_is_rejected():
    with pytest.raises(ValidationError, match="ki_stepdown"):
        build_snowball_product_spec(
            dict(SNOWBALL, ki_monitoring="discrete", ki_stepdown=0.9)
        )


def test_reverse_flips_the_embedded_option():
    product = snowball(is_reverse=True, ko_barrier=97.0, ki_barrier=115.0)
    assert product.is_reverse is True
    assert product.option_type is OptionType.CALL


def test_airbag_reaches_the_airbag_config():
    product = snowball(airbag_barrier=90.0, airbag_participation_rate=0.5)
    assert product.airbag_config.airbag_barrier == 90.0
    assert product.airbag_config.airbag_participation_rate == 0.5


@pytest.mark.parametrize(
    "name,expected",
    [("partial", ProtectionType.PARTIAL), ("full", ProtectionType.FULL)],
)
def test_protection_type_reaches_the_payoff_config(name, expected):
    product = snowball(protection_type=name, protection_rate=0.5)
    assert product.payoff_config.protection_type is expected


def test_participation_rate_reaches_the_payoff_config():
    assert snowball(participation_rate=0.5).payoff_config.participation_rate == 0.5


def test_call_rebate_reaches_the_payoff_config():
    payoff = snowball(call_rebate=True, call_strike=100.0).payoff_config
    assert payoff.call_rebate_enabled is True
    assert payoff.call_strike == 100.0


def test_disable_ko_after_ki_reaches_the_barrier_config():
    assert snowball(disable_ko_after_ki=True).barrier_config.disable_ko_after_ki is True


def test_coupon_pay_type_reaches_the_accrual_config():
    accrual = snowball(coupon_pay_type="expiry").accrual_config
    assert accrual.coupon_pay_type is CouponPayType.EXPIRY


def test_is_annualized_reaches_the_accrual_config():
    assert snowball(is_annualized=False).accrual_config.is_annualized is False


def test_ko_rate_step_makes_the_rate_a_schedule():
    barriers = snowball(ko_rate_step=0.005).barrier_config
    assert barriers.ko_rate == pytest.approx([0.15 + 0.005 * i for i in range(12)])


def test_every_variant_knob_is_absent_by_default():
    """The certified cells set none of these, and must not move because the
    knobs exist."""
    product = snowball()
    assert product.is_reverse is False
    assert product.barrier_config.disable_ko_after_ki is False
    assert product.barrier_config.ko_rate == 0.15
    assert product.barrier_config.ki_barrier == 85.0
    assert product.payoff_config.protection_type is ProtectionType.NONE
    assert product.payoff_config.participation_rate == 1.0
    assert product.payoff_config.call_rebate_enabled is False
    assert product.accrual_config.coupon_pay_type is CouponPayType.INSTANT
    assert product.accrual_config.is_annualized is True
    assert product.airbag_config.airbag_barrier is None


# --- The same feature families on phoenix and KO-reset ---------------------


def test_phoenix_discrete_ki_builds_a_schedule():
    barriers = phoenix(ki_monitoring="discrete").barrier_config
    assert barriers.ki_continuous is False
    assert barriers.ki_observation_schedule is not None


def test_phoenix_ki_stepdown_walks_the_barrier_down():
    barriers = phoenix(ki_monitoring="discrete", ki_stepdown=0.005).barrier_config
    assert barriers.ki_barrier == pytest.approx([75.0 - 0.5 * i for i in range(12)])


def test_phoenix_disable_ko_after_ki_reaches_the_config():
    assert phoenix(disable_ko_after_ki=True).barrier_config.disable_ko_after_ki is True


def test_phoenix_reverse_flips_the_embedded_option():
    product = phoenix(is_reverse=True, ko_barrier=97.0, ki_barrier=125.0,
                      coupon_barrier=115.0)
    assert product.is_reverse is True


def test_phoenix_variant_knobs_are_absent_by_default():
    product = phoenix()
    assert product.is_reverse is False
    assert product.barrier_config.ki_continuous is True
    assert product.barrier_config.disable_ko_after_ki is False


def test_ko_reset_ki_stepdown_uses_monthly_observations():
    """Daily KI monitoring cannot carry a monthly step, so a stepping KI moves
    the schedule to monthly -- twelve observations, twelve levels."""
    barriers = ko_reset(ki_monitoring="discrete", ki_stepdown=0.005).barrier_config
    assert barriers.ki_barrier == pytest.approx([80.0 - 0.5 * i for i in range(12)])
    assert len(barriers.ki_observation_dates) == 12


def test_ko_reset_disable_ko_after_ki_reaches_the_config():
    assert ko_reset(disable_ko_after_ki=True).barrier_config.disable_ko_after_ki is True


def test_ko_reset_variant_knobs_are_absent_by_default():
    product = ko_reset()
    assert product.barrier_config.disable_ko_after_ki is False
    assert product.barrier_config.ki_barrier == 80.0
