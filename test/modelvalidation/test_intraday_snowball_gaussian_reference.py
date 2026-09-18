"""The deterministic intraday reference: a nested ladder of the engine-independent Gaussian solver, a radius that says
what kind it is and covers a closed form, typed undefined quantities, and an identity over everything that moves a
number. The R-numbered tests answer docs/superpowers/reviews/intraday-deterministic-reference-2026-09-18."""
import dataclasses
import math
from math import exp, log, sqrt

import pytest
from scipy.integrate import quad
from scipy.stats import norm

from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind
from quantark.modelvalidation.builders import intraday_gaussian, intraday_gaussian_reference
from quantark.modelvalidation.builders.intraday_common import COMMON_TREES, build_request
from quantark.modelvalidation.evidence import identity_hash
from quantark.modelvalidation.reference import reference_targets, run_reference
from quantark.modelvalidation.registry import get_builder
from quantark.modelvalidation.study import CaseSpec, GateBounds, NormalizedScale, SamplingPolicy, reference_kind
from quantark.modelvalidation.yaml_loader import _ensure_builtin_builders
from quantark.util.exceptions import ValidationError

from test_intraday_common import CONTEXT, ENV

QUANTITIES = ("pv", "desk_delta", "desk_gamma", "desk_theta", "point_delta", "point_gamma")
MONTHLY = {"initial_date": "2026-03-16", "months": 12, "initial_price": 100.0, "strike": 100.0, "ko_barrier": 103.0,
           "ki_barrier": 75.0, "ko_rate": 0.12, "rebate_rate": 0.12, "contract_multiplier": 1.0,
           "ki_observation": "ko_dates"}
LATE = {**CONTEXT, "valuation": "2027-02-17T14:00:00+08:00", "phase": "after"}      # only the maturity observation remains
TWO_LEFT = {**CONTEXT, "valuation": "2027-01-18T14:00:00+08:00", "phase": "after"}  # the last two observations remain
SAMPLING = SamplingPolicy(paths_per_batch=1024, min_batches=2, max_batches=2, seed=20260918, bump=0.01, seed_scheme="substream")
LADDER = {"points": [401, 801, 1601, 3201, 6401], "quick_points": [101, 201, 401, 801, 1601]}
SMALL = {"points": [101, 201, 401, 801, 1601]}


def _reference(params=LADDER, env=ENV, product=MONTHLY, context=LATE, quantities=QUANTITIES):
    _ensure_builtin_builders()
    build = get_builder("equity.snowball.intraday.gaussian", kind="reference")
    return build(environment_params=env, product_params=product, sampling=SAMPLING, quantities=quantities, params=params,
                 context_params=context)


def test_builder_is_deterministic_and_validates_its_ladder():
    assert reference_kind(_reference()) == "deterministic"
    with pytest.raises(ValidationError, match="context"):
        get_builder("equity.snowball.intraday.gaussian", kind="reference")(
            environment_params=ENV, product_params=MONTHLY, sampling=SAMPLING, quantities=("pv",), params=LADDER)
    with pytest.raises(ValidationError, match="points"):
        _reference(params={})
    with pytest.raises(ValidationError, match="at least 5"):
        _reference(params={"points": [801, 1601, 3201, 6401]})
    with pytest.raises(ValidationError, match="nested"):
        _reference(params={"points": [400, 800, 1600, 3200, 6400]})
    with pytest.raises(ValidationError, match="does not take"):
        _reference(params={**LADDER, "paths": 4096})
    with pytest.raises(ValidationError, match="does not produce"):
        _reference(quantities=("pv", "point_vega"))


def test_every_quantity_is_a_declared_target_point_greeks_included():
    targets = reference_targets(_reference(), QUANTITIES)
    assert all(targets[q] is not None for q in QUANTITIES)
    assert targets["desk_delta"] == {"estimator": "central_difference_of_solves", "bump": 0.01}
    assert targets["point_gamma"]["estimator"] == "analytic_derivative_of_the_last_gaussian_expectation"


def _closed_form(ctx, spot):
    """One remaining instant: DF x E[payoff(S_T)], integrated piece by piece between the barrier levels."""
    prod, env = ctx.numerical.product, ctx.pricing_env
    T = ctx.numerical.maturity_tau
    W = float(env.vol_surface.total_variance(100.0, T, spot))
    R, Q = -log(env.get_discount_factor(T)), env.get_div_yield(T) * T
    last = next(e for e in ctx.numerical.remaining_events if e.kind is EventKind.KO)
    mu, sd, B = log(spot) + R - Q - 0.5 * W, sqrt(W), float(last.barrier)
    density = lambda y: norm.pdf((y - mu) / sd) / sd                                     # noqa: E731
    knocked_in = quad(lambda y: prod.get_maturity_payoff_v1(exp(y), env) * density(y), log(5.0), log(75.0), epsabs=1e-13, limit=400)[0]
    alive = quad(lambda y: prod.get_maturity_payoff_v0(exp(y), env) * density(y), log(75.0), log(B), epsabs=1e-13, limit=400)[0]
    return exp(-R) * (float(last.cash) * norm.cdf((mu - log(B)) / sd) + knocked_in + alive)


def _truth(ctx, spot):
    """Price, the finite moves and the point derivatives of the closed form (fourth-order stencils: a second-order
    one at this step carries an eps^2 f(3) / 6 ~ 4e-5 error of its own near the KI jump)."""
    h, eps = 0.01 * spot, 5e-4 * spot
    base, up, down = _closed_form(ctx, spot), _closed_form(ctx, spot + h), _closed_form(ctx, spot - h)
    f = {k: _closed_form(ctx, spot + k * eps) for k in (-2, -1, 1, 2)}
    return {"pv": base, "desk_delta": (up - down) / (2.0 * h), "desk_gamma": (up - 2.0 * base + down) / h ** 2,
            "point_delta": (-f[2] + 8.0 * f[1] - 8.0 * f[-1] + f[-2]) / (12.0 * eps),
            "point_gamma": (-f[2] + 16.0 * f[1] - 30.0 * base + 16.0 * f[-1] - f[-2]) / (12.0 * eps ** 2)}


SLACK = {"pv": 1e-9, "desk_delta": 1e-8, "desk_gamma": 1e-7, "point_delta": 1e-8, "point_gamma": 1e-7}   # quadrature noise through the stencils


# --- review R1: equal refinement values must never create a zero-radius reference ---------------------------------
def test_r1_the_reviewers_ladder_is_refused_because_it_repeats_the_local_grid():
    with pytest.raises(ValidationError, match="at least 101 points"):
        _reference(params={"points": [26, 51, 101, 201, 401]})
    with pytest.raises(ValueError, match="divisible by 4"):
        intraday_gaussian.local_points(203)
    assert [intraday_gaussian.local_points(n) for n in (101, 201, 401, 801)] == [26, 51, 101, 201]     # nested, no floor


@pytest.mark.parametrize("spot", [70.0, 75.0, 78.0, 101.0])
def test_r1_a_single_instant_is_an_identified_analytical_case_and_its_radius_covers_the_closed_form(spot):
    # the reviewer's case: one remaining event, where a repeated local grid once produced radius 0 with an error of
    # 3.6 PV budgets. It is now priced by quadrature, on any admissible ladder, with the quadrature's own error.
    env = {**ENV, "spot": spot}
    ctx = resolve_context(build_request(env, MONTHLY, LATE))
    result = _reference(env=env, params=SMALL).solve(CaseSpec(name="single_event"))
    for quantity, expected in _truth(ctx, spot).items():
        record = result.evidence["quantities"][quantity]
        assert record["rule"] == "exact" and "single_instant_quadrature" in record["basis"], (quantity, record)
        assert result.radius_basis[quantity] == "analytical" and 0.0 < result.radii[quantity] < 1e-6
        assert abs(result.values[quantity] - expected) <= result.radii[quantity] + SLACK[quantity], (
            quantity, result.values[quantity], expected, result.radii[quantity])
    check = result.evidence["grid_solver_cross_check"]
    assert check["points"] == 1601 and 0.0 < abs(check["difference"]) < 1e-3       # the grid solver agrees, less sharply


def test_r1_a_stagnating_ladder_without_a_basis_is_unresolved(monkeypatch):
    # the situation the reviewer found, past the grid check: every level returns the same number, the reported grids
    # look refined, and nothing explains why the value does not move
    real = intraday_gaussian.solve_snowball

    def frozen(ctx, points, width_std, diagnostics=None):
        out = real(ctx, 401, width_std, diagnostics)
        if diagnostics is not None:
            for which in ("global_spacing", "local_spacing"):
                if diagnostics.get(which) is not None:
                    diagnostics[which] *= 400.0 / (points - 1)
        return out

    monkeypatch.setattr(intraday_gaussian, "solve_snowball", frozen)
    result = _reference(context=TWO_LEFT, params=SMALL, env={**ENV, "spot": 78.0}).solve(CaseSpec(name="two_left"))
    assert result.evidence["quantities"]["pv"]["rule"] == "stagnant" and math.isinf(result.radii["pv"])


def test_r1_a_ladder_that_does_not_refine_a_grid_it_uses_is_refused(monkeypatch):
    real = intraday_gaussian.solve_snowball

    def same_local_grid(ctx, points, width_std, diagnostics=None):
        out = real(ctx, points, width_std, diagnostics)
        if diagnostics is not None and diagnostics.get("local_spacing") is not None:
            diagnostics["local_spacing"] = 1e-3                          # what the solve would report had it reused one grid
        return out

    monkeypatch.setattr(intraday_gaussian, "solve_snowball", same_local_grid)
    with pytest.raises(ValidationError, match="does not refine its local_spacing"):
        _reference(context=TWO_LEFT, params=SMALL).solve(CaseSpec(name="two_left"))


@pytest.mark.parametrize("spot", [78.0, 101.0])
def test_a_multi_instant_radius_is_a_calibrated_estimate_with_its_evidence(spot):
    result = _reference(context=TWO_LEFT, env={**ENV, "spot": spot}).solve(CaseSpec(name="two_left"))
    assert result.undefined == {} and result.evidence["levels"] == [401, 801, 1601, 3201, 6401]
    grids = result.evidence["grids"]
    assert [g["points"] for g in grids] == [401, 801, 1601, 3201, 6401] and len({g["instants"] for g in grids}) == 1 and grids[0]["instants"] >= 2
    for which in ("global_spacing", "local_spacing"):
        assert all(a[which] / b[which] == pytest.approx(2.0, rel=1e-12) for a, b in zip(grids, grids[1:]))
    components = result.evidence["residual_components"]
    assert set(components) == {"span", "truncation", "saturation", "roundoff", "total", "first_sd"}
    assert 0.0 < components["total"] < 1e-9 and components["span"] > 50.0
    for quantity in QUANTITIES:
        record = result.evidence["quantities"][quantity]
        assert result.radius_basis[quantity] == "calibrated_estimate" and record["basis"] is None
        assert set(record) == {"values", "extrapolants", "rule", "observed_order", "basis", "ladder_radius", "shared_error", "calibration"}
        if math.isfinite(result.radii[quantity]):
            assert record["calibration"]["covered"] and result.radii[quantity] == record["ladder_radius"] + record["shared_error"]


def test_a_two_instant_value_is_covered_against_a_much_finer_solve():
    # no closed form with two instants: the truth proxy is the extrapolant of a ladder four times finer
    env = {**ENV, "spot": 78.0}
    coarse = _reference(context=TWO_LEFT, env=env, params={"points": [201, 401, 801, 1601, 3201]}).solve(CaseSpec(name="a"))
    fine = _reference(context=TWO_LEFT, env=env, params={"points": [801, 1601, 3201, 6401, 12801]}).solve(CaseSpec(name="a"))
    for quantity in QUANTITIES:
        if math.isfinite(coarse.radii[quantity]) and math.isfinite(fine.radii[quantity]):
            assert abs(coarse.values[quantity] - fine.values[quantity]) <= coarse.radii[quantity] + fine.radii[quantity], quantity


def test_quantities_with_no_value_on_an_event_instant_are_typed_undefined():
    context = {**CONTEXT, "valuation": "2027-02-16T15:00:00+08:00", "phase": "before"}       # on the fixing, before it
    result = _reference(context=context, params=SMALL).solve(CaseSpec(name="on_event"))
    assert set(result.undefined) == {"desk_theta", "point_delta", "point_gamma"}
    assert set(result.values) == {"pv", "desk_delta", "desk_gamma"} == set(result.radii) == set(result.radius_basis)


# --- review R4: the recorded error model is the executed policy ---------------------------------------------------
def test_r4_the_error_model_is_the_policy_object_describing_itself():
    model = _reference().error_model()
    assert model["ladder_policy"] == intraday_gaussian.LADDER_POLICY.describe()
    assert "unextrapolated" in model["ladder_policy"]["branches"] and set(model["exactness_bases"]) == {
        "terminated", "decided_at_valuation", "single_instant_quadrature"}
    assert set(model["residual_components"]) == {"truncation", "saturation", "roundoff", "scaling"}
    assert "validity_domain" in model and model["calibration_controls"]


def test_r4_a_changed_policy_parameter_changes_the_contract_and_the_identity():
    class Cautious(intraday_gaussian_reference.IntradaySnowballGaussianReference):
        policy = dataclasses.replace(intraday_gaussian.LADDER_POLICY, unconfirmed_safety=5.0)

    base = _reference()
    changed = Cautious(sampling=SAMPLING, environment_params=ENV, product_params=MONTHLY, context_params=LATE,
                       quantities=QUANTITIES, params=LADDER)
    assert changed.error_model() != base.error_model()
    assert identity_hash(changed.identity(CaseSpec(name="a"))) != identity_hash(base.identity(CaseSpec(name="a")))


def test_r4_an_amendment_under_a_changed_radius_policy_is_refused(tmp_path):
    from quantark.modelvalidation.amendment import amend
    from quantark.modelvalidation.pipeline import certify
    from test_pipeline_schema2 import Candidate

    class Cautious(intraday_gaussian_reference.IntradaySnowballGaussianReference):
        policy = dataclasses.replace(intraday_gaussian.LADDER_POLICY, stagnation_floor=1e-12)

    from quantark.modelvalidation.study import CertificationStudy, QuantityBounds

    def study(reference_class):
        kwargs = dict(sampling=SAMPLING, environment_params={**ENV, "spot": 78.0}, product_params=MONTHLY,
                      context_params=LATE, quantities=("pv",), params=SMALL)
        return CertificationStudy(
            name="policy", schema=2, quantities=("pv",), quantity_bounds={"pv": QuantityBounds(1e-6)},
            cases=(CaseSpec(name="single_event"),), bounds=GateBounds(cell=1.0, mean_signed_bias=0.2, radius_budget_fraction=0.25),
            scale=NormalizedScale(100.0, 100.0), reference=reference_class(**kwargs), sampling=SAMPLING, source_text="study: policy\n",
            candidates=(_Fixed(),))

    class _Fixed(Candidate):
        def _values(self, case):
            return {"pv": 2.4315528230}, {}

        def resolved_inputs(self, case):
            return {"quantities": ["pv"]}

    parent = certify(study(intraday_gaussian_reference.IntradaySnowballGaussianReference), out_dir=tmp_path / "parent")
    assert parent.payload["decisions"] == {"fake.cand": "ADMITTED"}
    with pytest.raises(ValidationError, match="error model"):
        amend(study(Cautious), parent.path, tmp_path / "amended", reason="a looser stagnation floor")


# --- identity ------------------------------------------------------------------------------------------------------
def test_quick_binds_the_wiring_ladder_and_every_number_moving_input_is_in_the_identity():
    reference, case = _reference(), CaseSpec(name="a")
    quick = reference.bind(SAMPLING, quick=True)
    assert quick.levels == (101, 201, 401, 801, 1601) and reference.bind(SAMPLING).levels == (401, 801, 1601, 3201, 6401)
    base = identity_hash(reference.identity(case))
    assert identity_hash(quick.identity(case)) != base
    assert identity_hash(_reference(params={**LADDER, "width_std": 9.0}).identity(case)) != base
    assert identity_hash(_reference(env={**ENV, "vol": 0.25}).identity(case)) != base
    assert identity_hash(reference.identity(CaseSpec(name="a", context_params={"valuation": "2027-02-18T14:00:00+08:00"}))) != base
    assert identity_hash(_reference().identity(case)) == base


def test_the_reference_and_the_candidates_do_not_invalidate_each_other():
    trees = intraday_gaussian_reference.REFERENCE_TREES
    assert "quantark/modelvalidation/builders/intraday_snowball.py" not in trees          # the candidates' file
    assert set(intraday_gaussian_reference.GAUSSIAN_TREES) <= set(trees) and not set(intraday_gaussian_reference.GAUSSIAN_TREES) & set(COMMON_TREES)
    assert "quantark/intraday" in trees and "quantark/modelvalidation/builders/intraday_common.py" in trees


def test_run_reference_banks_the_solve_as_a_typed_deterministic_estimate(tmp_path):
    from quantark.modelvalidation.evidence import CheckpointStore

    reference, case = _reference(params=SMALL), CaseSpec(name="single_event")
    store = CheckpointStore(tmp_path / "checkpoints")
    kwargs = dict(builder=reference, case=case, quantities=QUANTITIES, scale=NormalizedScale(100.0, 100.0),
                  bounds=GateBounds(cell=1.0, mean_signed_bias=0.2, radius_budget_fraction=0.25), policy=SAMPLING, store=store)
    first = run_reference(**kwargs)
    assert first.kind == "deterministic" and first.std_errors == {} and first.batches == 0 and set(first.radii) == set(QUANTITIES)
    assert set(first.radius_basis.values()) == {"analytical"}
    again = run_reference(**kwargs, resume=True)
    assert again.values == first.values and again.radii == first.radii and again.evidence == first.evidence
    assert again.radius_basis == first.radius_basis
