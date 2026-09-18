"""The deterministic intraday reference: a nested ladder of the engine-independent Gaussian solver, a declared
radius that covers a closed form, typed undefined quantities, and an identity over everything that moves a number."""
import math
from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind
from quantark.modelvalidation.builders.intraday_common import build_request
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
SAMPLING = SamplingPolicy(paths_per_batch=1024, min_batches=2, max_batches=2, seed=20260918, bump=0.01, seed_scheme="substream")
LADDER = {"points": [501, 1001, 2001, 4001], "quick_points": [126, 251, 501, 1001]}


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
    with pytest.raises(ValidationError, match="at least 4"):
        _reference(params={"points": [501, 1001, 2001]})
    with pytest.raises(ValidationError, match="nested"):
        _reference(params={"points": [500, 1000, 2000, 4000]})
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


@pytest.mark.parametrize("spot", [78.0, 101.0])
def test_the_declared_radius_covers_the_closed_form_for_price_and_both_kinds_of_greek(spot):
    env = {**ENV, "spot": spot}
    reference = _reference(env=env)
    ctx = resolve_context(build_request(env, MONTHLY, LATE))
    result = reference.solve(CaseSpec(name="single_event"))
    h = 0.01 * spot
    base, up, down = _closed_form(ctx, spot), _closed_form(ctx, spot + h), _closed_form(ctx, spot - h)
    # fourth-order stencils: a second-order one at this step carries an eps^2 f(3) / 6 ~ 4e-5 error of its own near the KI jump
    eps = 5e-4 * spot
    f = {k: _closed_form(ctx, spot + k * eps) for k in (-2, -1, 1, 2)}
    truth = {"pv": base, "desk_delta": (up - down) / (2.0 * h), "desk_gamma": (up - 2.0 * base + down) / h ** 2,
             "point_delta": (-f[2] + 8.0 * f[1] - 8.0 * f[-1] + f[-2]) / (12.0 * eps),
             "point_gamma": (-f[2] + 16.0 * f[1] - 30.0 * base + 16.0 * f[-1] - f[-2]) / (12.0 * eps ** 2)}
    slack = {"pv": 1e-9, "desk_delta": 1e-8, "desk_gamma": 1e-7, "point_delta": 1e-8, "point_gamma": 1e-7}    # quadrature noise through the stencils
    for quantity, expected in truth.items():
        assert math.isfinite(result.radii[quantity]), (quantity, result.evidence["quantities"][quantity])
        assert abs(result.values[quantity] - expected) <= result.radii[quantity] + slack[quantity], (
            quantity, result.values[quantity], expected, result.radii[quantity])
    assert result.undefined == {} and result.evidence["levels"] == [501, 1001, 2001, 4001]
    assert set(result.evidence["quantities"]["pv"]) == {"values", "extrapolants", "rule", "observed_order"}


def test_quantities_with_no_value_on_an_event_instant_are_typed_undefined():
    context = {**CONTEXT, "valuation": "2027-02-16T15:00:00+08:00", "phase": "before"}       # on the fixing, before it
    result = _reference(context=context, params={"points": [126, 251, 501, 1001]}).solve(CaseSpec(name="on_event"))
    assert set(result.undefined) == {"desk_theta", "point_delta", "point_gamma"}
    assert set(result.values) == {"pv", "desk_delta", "desk_gamma"} and set(result.radii) == set(result.values)


def test_quick_binds_the_wiring_ladder_and_every_number_moving_input_is_in_the_identity():
    reference, case = _reference(), CaseSpec(name="a")
    quick = reference.bind(SAMPLING, quick=True)
    assert quick.levels == (126, 251, 501, 1001) and reference.bind(SAMPLING).levels == (501, 1001, 2001, 4001)
    base = identity_hash(reference.identity(case))
    assert identity_hash(quick.identity(case)) != base
    assert identity_hash(_reference(params={**LADDER, "width_std": 9.0}).identity(case)) != base
    assert identity_hash(_reference(env={**ENV, "vol": 0.25}).identity(case)) != base
    assert identity_hash(reference.identity(CaseSpec(name="a", context_params={"valuation": "2027-02-18T14:00:00+08:00"}))) != base
    assert identity_hash(_reference().identity(case)) == base


def test_run_reference_banks_the_solve_as_a_typed_deterministic_estimate(tmp_path):
    from quantark.modelvalidation.evidence import CheckpointStore

    reference, case = _reference(params={"points": [126, 251, 501, 1001]}), CaseSpec(name="single_event")
    store = CheckpointStore(tmp_path / "checkpoints")
    kwargs = dict(builder=reference, case=case, quantities=QUANTITIES, scale=NormalizedScale(100.0, 100.0),
                  bounds=GateBounds(cell=1.0, mean_signed_bias=0.2, radius_budget_fraction=0.25), policy=SAMPLING, store=store)
    first = run_reference(**kwargs)
    assert first.kind == "deterministic" and first.std_errors == {} and first.batches == 0 and set(first.radii) == set(QUANTITIES)
    again = run_reference(**kwargs, resume=True)
    assert again.values == first.values and again.radii == first.radii and again.evidence == first.evidence
