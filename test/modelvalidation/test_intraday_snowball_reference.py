"""The intraday RQMC reference: paired bumps on the resolved context, substreams, declared targets, sound identity."""
import dataclasses

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.builders import intraday_snowball
from quantark.modelvalidation.evidence import identity_hash
from quantark.modelvalidation.reference import reference_targets
from quantark.modelvalidation.registry import get_builder
from quantark.modelvalidation.study import CaseSpec, SamplingPolicy, batch_seed
from quantark.modelvalidation.yaml_loader import _ensure_builtin_builders

from test_intraday_common import CONTEXT, ENV, PRODUCT

QUANTITIES = ("pv", "desk_delta", "desk_gamma", "desk_theta", "point_delta", "point_gamma")


def _sampling(paths=1024, seed=20260918):
    return SamplingPolicy(paths_per_batch=paths, min_batches=2, max_batches=2, seed=seed, bump=0.01, seed_scheme="substream")


def _reference(sampling=None, env=ENV):
    _ensure_builtin_builders()
    build = get_builder("equity.snowball.intraday.mc_rqmc", kind="reference")
    return build(environment_params=env, product_params=PRODUCT, sampling=sampling or _sampling(), quantities=QUANTITIES,
                 params={}, context_params=CONTEXT)


def test_builder_requires_a_context_and_refuses_params():
    build = get_builder("equity.snowball.intraday.mc_rqmc", kind="reference")
    with pytest.raises(ValidationError, match="context"):
        build(environment_params=ENV, product_params=PRODUCT, sampling=_sampling(), quantities=("pv",), params={})
    with pytest.raises(ValidationError, match="no params"):
        build(environment_params=ENV, product_params=PRODUCT, sampling=_sampling(), quantities=("pv",),
              params={"steps": 4}, context_params=CONTEXT)
    with pytest.raises(ValidationError, match="does not produce"):
        build(environment_params=ENV, product_params=PRODUCT, sampling=_sampling(), quantities=("pv", "point_vega"),
              params={}, context_params=CONTEXT)
    with pytest.raises(ValidationError, match="substream"):
        build(environment_params=ENV, product_params=PRODUCT, quantities=("pv",), params={}, context_params=CONTEXT,
              sampling=dataclasses.replace(_sampling(), seed_scheme="sequential"))


def test_targets_declare_desk_moves_and_no_point_estimator():
    targets = reference_targets(_reference(), QUANTITIES)
    assert targets["point_delta"] is None and targets["point_gamma"] is None
    assert targets["desk_delta"] == {"estimator": "paired_central_difference", "bump": 0.01}
    assert targets["pv"] == {"estimator": "rqmc_replicate_mean"}
    assert targets["desk_theta"]["estimator"] == "frozen_market_roll_same_seed"


def test_batches_are_finite_deterministic_and_on_their_case_substream():
    reference = _reference()
    case = CaseSpec(name="ordinary")
    first, again, second = reference.run_batch(case, 0), reference.run_batch(case, 0), reference.run_batch(case, 1)
    assert first.seed == batch_seed(reference.sampling, "ordinary", 0) and second.seed == batch_seed(reference.sampling, "ordinary", 1)
    assert set(first.values) == set(QUANTITIES) and first.values == again.values
    assert first.values["pv"] != second.values["pv"]
    assert first.values["point_delta"] == first.values["desk_delta"]   # the recorded proxy is the same finite move


def test_two_cases_never_share_a_scramble(monkeypatch):
    """Review R3: identical economics under two names must not produce identical replicates."""
    reference = _reference()
    a, b = reference.run_batch(CaseSpec(name="case_a"), 0), reference.run_batch(CaseSpec(name="case_b"), 0)
    assert a.seed != b.seed and a.values["pv"] != b.values["pv"]


def test_a_pilot_shares_no_randomization_with_production():
    production, pilot = _sampling(seed=20260918), _sampling(seed=20260917)
    names = ["ordinary", "near_ki_1s", "maturity_day"]
    prod_seeds = {batch_seed(production, n, i) for n in names for i in range(64)}
    pilot_seeds = {batch_seed(pilot, n, i) for n in names for i in range(64)}
    assert not prod_seeds & pilot_seeds


def test_a_bound_reference_runs_the_bound_path_count(monkeypatch):
    """Review R10: the engine gets the effective policy's paths, and the identity records them."""
    seen = []
    monkeypatch.setattr(intraday_snowball, "cell_price", lambda ctx, engine: seen.append(engine.params.num_paths) or 1.0)
    bound = _reference(_sampling(paths=65536)).bind(_sampling(paths=8192))
    bound.run_batch(CaseSpec(name="ordinary"), 0)
    assert set(seen) == {8192}
    assert bound.identity(CaseSpec(name="ordinary"))["sampling"]["paths_per_batch"] == 8192


def test_a_terminated_claim_is_the_pending_ledger_with_zero_spot_greeks():
    reference = _reference()
    case = CaseSpec(name="terminated", product_params={"settlement_lag_days": 30},
                    context_params={"fixings": [{"date": "2026-08-17", "level": 104.0}]})
    batch = reference.run_batch(case, 0)
    assert batch.values["pv"] > 0.0                                    # the KO cash is still pending on 2026-09-10
    assert batch.values["desk_delta"] == 0.0 and batch.values["desk_gamma"] == 0.0
    assert reference.run_batch(case, 1).values["pv"] == batch.values["pv"]   # exact: no sampling


def test_identity_covers_resolved_inputs_targets_and_the_implementation():
    reference = _reference()
    a = reference.identity(CaseSpec(name="a"))
    assert a["builder"] == "equity.snowball.intraday.mc_rqmc" and len(a["implementation"]) == 64
    assert a["targets"]["point_delta"] is None and a["sampling"]["seed_scheme"] == "substream" and "candidates" not in a
    later = reference.identity(CaseSpec(name="a", context_params={"valuation": "2026-09-10T14:59:59+08:00"}))
    assert identity_hash(a) != identity_hash(later)                     # same name, different clock
    other_market = _reference(env={**ENV, "vol": 0.35}).identity(CaseSpec(name="a"))
    assert identity_hash(a) != identity_hash(other_market)              # study-level market change
