"""Anti-circularity: the audit must fail on a wrong Greek that the algebra hides.

The linear book ``V = a S + sum_i c_i F_i`` is used throughout because its
buckets, pinned delta and nodal rhoq are exact -- a central difference on a
function linear in each ``F_i`` has no truncation error -- so any residual
these tests see is the audit noticing something, not the bump.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from bucket_hedge_fixtures import (
    AnalyticQuote,
    linear_book_pricer,
    linear_book_risk,
)
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.backtest.futures_risk import (
    CarryRiskSettings,
    FuturesBookRisk,
    held_book_risk,
)
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import (
    AUDIT_STATUSES,
    CarryAuditResult,
    audit_held_book,
    buckets_of,
    direct_frozen_curve_book_delta,
    direct_nodal_rhoq,
    direct_parallel_rhoq,
    direct_pinned_delta,
    not_measured_audit,
    refinement_ladder,
    sample_buckets,
    scenario_book_value,
    stabilised,
    status_from_interval,
)
from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError

RATE = 0.0
SPOT = 100.0
TENORS = (0.25, 0.50)
PRICES = (100.0, 100.0)
CONTRACTS = ("IF2503", "IF2506")
SPOT_COEFFICIENT = 1.0
COEFFICIENTS = (1.0, -0.5)
POINTS = 1.0

SETTINGS = CarryRiskSettings(
    reference_notional=1_000.0,
    reference_multiplier=1.0,
    audit_spot_bump_rel=0.005,
    audit_yield_bump=1e-4,
    delta_tolerance_hands=0.01,
    rhoq_tolerance_bp=0.01,
)


def quotes():
    return tuple(
        IndexFuturesQuote(
            contract=c, maturity=t, price=p, multiplier=1.0
        )
        for c, t, p in zip(CONTRACTS, TENORS, PRICES)
    )


def context(extrapolation: str = "flat_q"):
    return CarryCurveContext(
        quotes=quotes(),
        spot=SPOT,
        rate_curve=FlatRateCurve(rate=RATE),
        extrapolation=extrapolation,
        underlying="index",
        valuation_date="2025-03-03",
    )


def linear_pricer(coefficients=COEFFICIENTS, spot_coefficient=SPOT_COEFFICIENT):
    """``V = a S + sum_i c_i F_i``, from the shared analytic fixtures."""
    return linear_book_pricer(
        tenors=TENORS,
        futures_coefficients=coefficients,
        spot_coefficient=spot_coefficient,
        rate=RATE,
    )


def measured_risk(ctx=None, coefficients=COEFFICIENTS):
    """The book risk a correct sampler produces for the linear book."""
    ctx = ctx or context()
    price_at = linear_pricer(coefficients)
    buckets = buckets_of(ctx, sample_buckets(price_at, ctx, POINTS))
    reference = linear_book_risk(
        spot=SPOT,
        quotes=tuple(
            AnalyticQuote(c, t, p, 1.0) for c, t, p in zip(CONTRACTS, TENORS, PRICES)
        ),
        futures_coefficients=coefficients,
    )
    return FuturesBookRisk(spot=SPOT, delta_q=reference.delta_q, buckets=buckets)


def nodes_holdings(risk: FuturesBookRisk):
    return {b.contract: -b.bucket_currency / b.multiplier for b in risk.buckets}


def spot_parallel_holdings(risk: FuturesBookRisk):
    """The design's exact primary-policy targets for this two-node book."""
    a, b = risk.buckets
    df = risk.delta_f_derived
    gap = b.tenor_years - a.tenor_years
    return {
        a.contract: -a.bucket_currency / a.multiplier
        - df * b.tenor_years / gap * risk.spot / (a.multiplier * a.price),
        b.contract: -b.bucket_currency / b.multiplier
        + df * a.tenor_years / gap * risk.spot / (b.multiplier * b.price),
    }


# ---------------------------------------------------------------------------
# The measured directions are exact for this book
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extrapolation", ["flat_q", "flat_forward_carry"])
def test_direct_directions_reproduce_the_analytic_book(extrapolation):
    ctx = context(extrapolation)
    price_at = linear_pricer()
    step = SETTINGS.audit_spot_bump_rel * SPOT
    assert direct_pinned_delta(price_at, ctx, step) == pytest.approx(
        SPOT_COEFFICIENT, abs=1e-9
    )
    assert direct_frozen_curve_book_delta(price_at, ctx, {}, step) == pytest.approx(
        1.5, abs=1e-9
    )
    rho = direct_nodal_rhoq(price_at, ctx, {}, SETTINGS.audit_yield_bump)
    assert rho[CONTRACTS[0]] == pytest.approx(-25.0, rel=1e-7)
    assert rho[CONTRACTS[1]] == pytest.approx(25.0, rel=1e-7)
    assert direct_parallel_rhoq(
        price_at, ctx, {}, SETTINGS.audit_yield_bump
    ) == pytest.approx(0.0, abs=1e-7)


def test_the_held_hedge_enters_each_scenario_at_fixed_quantities():
    ctx = context()
    price_at = linear_pricer()
    holdings = {CONTRACTS[0]: -3.0, CONTRACTS[1]: 1.5}
    scenario = ctx.bump_node_yield(CONTRACTS[0], 1e-4)
    value = scenario_book_value(price_at, ctx, scenario, holdings)
    product = price_at(scenario.spot, scenario.dividend())
    hedge = sum(
        holdings[q.contract] * q.multiplier * (q.price - ctx.quote(q.contract).price)
        for q in scenario.quotes
    )
    assert value == pytest.approx(product + hedge)


# ---------------------------------------------------------------------------
# A perturbed bucket: the algebra cancels, the audit does not
# ---------------------------------------------------------------------------


def perturbed(risk: FuturesBookRisk, delta: float, index: int = 0):
    buckets = list(risk.buckets)
    buckets[index] = replace(
        buckets[index], bucket_currency=buckets[index].bucket_currency + delta
    )
    return FuturesBookRisk(spot=risk.spot, delta_q=risk.delta_q, buckets=tuple(buckets))


def test_a_wrong_bucket_still_cancels_in_the_derived_identity():
    risk = perturbed(measured_risk(), 0.05)
    _, rho = held_book_risk(risk, nodes_holdings(risk))
    # Sized from the wrong bucket, the ALGEBRA reports a perfectly hedged node.
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in rho.values())


def test_a_wrong_bucket_fails_the_direct_audit():
    ctx = context()
    price_at = linear_pricer()
    risk = perturbed(measured_risk(ctx), 0.05)
    result = audit_held_book(
        price_at, ctx, risk, nodes_holdings(risk), settings=SETTINGS
    )
    assert result.status == "fail"
    assert CONTRACTS[0] in result.reason or "identity" in result.reason
    # The direct nodal measurement sees the residual the algebra erased.
    assert result.direct_nodal_rhoq[CONTRACTS[0]] != pytest.approx(0.0, abs=1e-6)
    assert result.mapped_nodal_rhoq[CONTRACTS[0]] == pytest.approx(0.0, abs=1e-12)
    # And the derived D_F disagrees with the repriced one by the perturbation.
    assert result.delta_f_direct == pytest.approx(SPOT_COEFFICIENT, abs=1e-9)
    assert result.identity_residual_hands == pytest.approx(-0.05, abs=1e-6)


def test_a_correct_book_passes_the_same_audit():
    ctx = context()
    risk = measured_risk(ctx)
    result = audit_held_book(
        linear_pricer(), ctx, risk, nodes_holdings(risk), settings=SETTINGS
    )
    assert result.status == "pass"
    assert result.identity_residual_hands == pytest.approx(0.0, abs=1e-9)
    assert result.worst_rhoq_error_bp < SETTINGS.rhoq_tolerance_bp


# ---------------------------------------------------------------------------
# Numerical validity and objective neutrality are different questions
# ---------------------------------------------------------------------------


def test_a_correct_nodes_audit_passes_while_leaving_spot_delta():
    ctx = context()
    risk = measured_risk(ctx)
    holdings = nodes_holdings(risk)
    result = audit_held_book(
        linear_pricer(),
        ctx,
        risk,
        holdings,
        settings=SETTINGS,
        ideal_net_delta=risk.delta_f_derived,
        ideal_net_parallel_rhoq=0.0,
    )
    assert result.status == "pass"
    assert result.objective_achieved is True
    # ... and the book is deliberately NOT spot neutral: D_F remains.
    assert result.direct_net_delta == pytest.approx(1.0, abs=1e-9)
    assert result.direct_net_parallel_rhoq == pytest.approx(0.0, abs=1e-6)


def test_the_primary_policy_zeroes_spot_and_parallel_but_not_the_nodes():
    ctx = context()
    risk = measured_risk(ctx)
    holdings = spot_parallel_holdings(risk)
    assert holdings[CONTRACTS[0]] == pytest.approx(-3.0, rel=1e-9)
    assert holdings[CONTRACTS[1]] == pytest.approx(1.5, rel=1e-9)
    result = audit_held_book(
        linear_pricer(),
        ctx,
        risk,
        holdings,
        settings=SETTINGS,
        ideal_net_delta=0.0,
        ideal_net_parallel_rhoq=0.0,
    )
    assert result.status == "pass"
    assert result.objective_achieved is True
    assert result.direct_net_delta == pytest.approx(0.0, abs=1e-9)
    assert result.direct_net_parallel_rhoq == pytest.approx(0.0, abs=1e-6)
    # K = D_F S T_a T_b / (T_b - T_a) = 50; the shape risk is concentrated.
    assert result.direct_nodal_rhoq[CONTRACTS[0]] == pytest.approx(50.0, rel=1e-6)
    assert result.direct_nodal_rhoq[CONTRACTS[1]] == pytest.approx(-50.0, rel=1e-6)


def test_wrong_holdings_keep_numerical_validity_and_lose_the_objective():
    ctx = context()
    risk = measured_risk(ctx)
    holdings = {c: h + 1.0 for c, h in spot_parallel_holdings(risk).items()}
    result = audit_held_book(
        linear_pricer(),
        ctx,
        risk,
        holdings,
        settings=SETTINGS,
        ideal_net_delta=0.0,
        ideal_net_parallel_rhoq=0.0,
    )
    # The Greeks are right, so the direct measurement agrees with the algebra.
    assert result.status == "pass"
    # The book is not where the policy wanted it to be, and the audit says so.
    assert result.objective_achieved is False
    assert result.direct_net_delta != pytest.approx(0.0, abs=1e-3)


def test_auditing_the_target_instead_of_the_actual_book_hides_the_error():
    ctx = context()
    risk = measured_risk(ctx)
    ideal = spot_parallel_holdings(risk)
    actual = {c: float(round(h)) for c, h in ideal.items()}
    actual[CONTRACTS[1]] = actual[CONTRACTS[1]] + 2.0
    on_actual = audit_held_book(
        linear_pricer(), ctx, risk, actual, settings=SETTINGS,
        ideal_net_delta=0.0, ideal_net_parallel_rhoq=0.0,
    )
    on_ideal = audit_held_book(
        linear_pricer(), ctx, risk, ideal, settings=SETTINGS,
        holdings_kind="ideal",
        ideal_net_delta=0.0, ideal_net_parallel_rhoq=0.0,
    )
    assert on_ideal.objective_achieved is True
    assert on_actual.objective_achieved is False
    assert on_actual.holdings_kind == "actual"
    assert on_ideal.holdings_kind == "ideal"
    assert on_actual.direct_net_delta != pytest.approx(
        on_ideal.direct_net_delta, abs=1e-6
    )


def test_rebalancing_inside_a_scenario_manufactures_a_false_pass():
    """Re-sizing at each bump reports zero risk for ANY starting book."""
    ctx = context()
    price_at = linear_pricer()
    risk = measured_risk(ctx)
    wrong = {c: 0.0 for c in risk.contracts}
    step = SETTINGS.audit_yield_bump

    def rebalanced_value(scenario):
        scenario_risk = FuturesBookRisk(
            spot=scenario.spot,
            delta_q=risk.delta_q,
            buckets=buckets_of(scenario, sample_buckets(price_at, scenario, POINTS)),
        )
        return scenario_book_value(
            price_at, ctx, scenario, nodes_holdings(scenario_risk)
        )

    up, down = (
        ctx.bump_node_yield(CONTRACTS[0], step),
        ctx.bump_node_yield(CONTRACTS[0], -step),
    )
    cheating = (rebalanced_value(up) - rebalanced_value(down)) / (2 * step)
    honest = (
        scenario_book_value(price_at, ctx, up, wrong)
        - scenario_book_value(price_at, ctx, down, wrong)
    ) / (2 * step)
    assert honest == pytest.approx(-25.0, rel=1e-6)
    assert cheating != pytest.approx(honest, rel=1e-3)
    # The fixed-quantity audit is the honest one.
    measured = direct_nodal_rhoq(price_at, ctx, wrong, step)
    assert measured[CONTRACTS[0]] == pytest.approx(honest, rel=1e-9)


# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------


def test_an_unrun_audit_is_not_measured_and_never_zero():
    result = not_measured_audit(
        holdings_kind="actual", reason="not a scheduled audit date", settings=SETTINGS
    )
    assert result.status == "not_measured"
    assert not result.measured
    assert not result.passed
    assert math.isnan(result.direct_net_delta)
    assert math.isnan(result.net_delta_audit_error_hands)
    assert math.isnan(result.identity_residual_hands)
    assert result.mapped_nodal_rhoq == {}


def test_every_status_is_declared():
    assert AUDIT_STATUSES == ("pass", "fail", "not_measured", "inconclusive")
    with pytest.raises(ValidationError):
        CarryAuditResult(
            status="ok",
            reason="",
            holdings_kind="actual",
            spot_step=1.0,
            yield_step=1e-4,
            reference_multiplier=1.0,
            reference_notional=1.0,
        )
    inconclusive = CarryAuditResult(
        status="inconclusive",
        reason="confidence interval wider than the exposure budget",
        holdings_kind="actual",
        spot_step=1.0,
        yield_step=1e-4,
        reference_multiplier=1.0,
        reference_notional=1.0,
    )
    assert inconclusive.measured
    assert not inconclusive.passed


def test_an_unresolved_settings_object_cannot_run_an_audit():
    with pytest.raises(ValidationError):
        audit_held_book(
            linear_pricer(),
            context(),
            measured_risk(),
            {},
            settings=CarryRiskSettings(),
        )
    with pytest.raises(ValidationError):
        not_measured_audit(
            holdings_kind="actual", reason="x", settings=CarryRiskSettings()
        )


def test_an_unknown_holdings_kind_is_rejected():
    with pytest.raises(ValidationError):
        audit_held_book(
            linear_pricer(),
            context(),
            measured_risk(),
            {},
            settings=SETTINGS,
            holdings_kind="target",
        )


def test_a_failed_reprice_is_an_error_not_a_zero_residual():
    ctx = context()

    def price(spot, dividend):
        return float("inf")

    with pytest.raises(ValidationError):
        audit_held_book(price, ctx, measured_risk(ctx), {}, settings=SETTINGS)


# ---------------------------------------------------------------------------
# Refinement ladders
# ---------------------------------------------------------------------------


def test_a_ladder_runs_at_least_three_levels_and_halves_the_step():
    ctx = context()
    price_at = linear_pricer()
    samples = refinement_ladder(
        lambda step: direct_pinned_delta(price_at, ctx, step),
        initial_step=0.5,
        levels=3,
    )
    assert len(samples) >= 3
    assert [s for s, _ in samples][:3] == [0.5, 0.25, 0.125]
    assert all(value == pytest.approx(SPOT_COEFFICIENT, abs=1e-9) for _, value in samples)
    assert stabilised(samples, tolerance=1e-9)


def test_a_ladder_extends_while_the_measurement_still_moves():
    calls = []

    def measure(step):
        calls.append(step)
        # A first-order estimator: the error only halves per level.
        return 1.0 + step

    samples = refinement_ladder(
        measure, initial_step=1.0, levels=3, max_levels=6, tolerance=1e-3
    )
    assert len(samples) == 6
    assert not stabilised(samples, tolerance=1e-3)


def test_a_ladder_stops_once_the_measurement_has_settled():
    samples = refinement_ladder(
        lambda step: 7.0, initial_step=1.0, levels=3, max_levels=8, tolerance=0.0
    )
    assert len(samples) == 3
    assert stabilised(samples, tolerance=0.0)


def test_a_ladder_rejects_too_few_levels_or_a_bad_step():
    with pytest.raises(ValidationError):
        refinement_ladder(lambda step: 1.0, initial_step=1.0, levels=2)
    with pytest.raises(ValidationError):
        refinement_ladder(lambda step: 1.0, initial_step=0.0, levels=3)
    with pytest.raises(ValidationError):
        refinement_ladder(lambda step: 1.0, initial_step=1.0, levels=3, max_levels=2)


def test_the_audit_records_its_effective_bumps_and_price_count():
    ctx = context()
    risk = measured_risk(ctx)
    result = audit_held_book(
        linear_pricer(), ctx, risk, nodes_holdings(risk), settings=SETTINGS
    )
    assert result.spot_step == pytest.approx(0.5)
    assert result.yield_step == 1e-4
    assert result.reference_notional == 1_000.0
    assert result.reference_multiplier == 1.0
    # Two pinned spot prices, two frozen spot prices, two per node, two
    # parallel and two product-only parallel.
    assert result.price_calls == 4 + 2 * len(ctx.quotes) + 4 + 4 * 3
    assert len(result.identity_samples) == 3
    assert result.identity_samples[-1].spot_bump_rel == 0.00025


def test_nonlinear_identity_refines_both_deltas_without_changing_hedge_delta():
    """W(S,F)=0.1*S*F1**2 has a mixed cubic term along fixed-q spot bumps.

    Its 1% delta has a known 0.1-hand secant error. Buckets and pinned
    spot delta are exact central differences, so an exact chain check must
    not mix that 1% delta with the local bucket derivative.
    """
    ctx = context()
    settings = replace(SETTINGS, audit_spot_bump_rel=0.01)

    def price_at(s, div):
        f = s * math.exp((RATE - div.get_yield(TENORS[0])) * TENORS[0])
        return 0.1 * s * f * f

    pricing_delta = direct_frozen_curve_book_delta(price_at, ctx, {}, 1.0)
    risk = FuturesBookRisk(SPOT, pricing_delta, buckets_of(ctx, sample_buckets(price_at, ctx, 1.0)))
    result = audit_held_book(price_at, ctx, risk, {}, settings=settings)
    assert result.status == "pass"
    assert result.identity_status == "pass"
    assert result.finite_bump_identity_residual_hands == pytest.approx(0.1, abs=1e-8)
    assert result.pricing_delta_local_gap_hands == pytest.approx(0.0999375, abs=1e-8)
    assert [s.residual_hands for s in result.identity_samples] == pytest.approx(
        [0.001, 0.00025, 0.0000625], abs=1e-8
    )
    assert risk.delta_q == pytest.approx(3000.1, abs=1e-8)
    assert abs(result.net_delta_audit_error_hands) < 1e-9

    # A wrong reported hedge Greek still fails its own reproduction check,
    # even though the separately measured local chain identity closes.
    bad = audit_held_book(price_at, ctx, replace(risk, delta_q=pricing_delta + 0.2),
                          {}, settings=settings)
    assert bad.status == "fail"
    assert bad.identity_status == "pass"
    assert bad.net_delta_audit_error_hands == pytest.approx(-0.2, abs=1e-8)


def test_unstable_matched_identity_is_inconclusive_even_when_last_residual_is_zero(monkeypatch):
    from quantark.backtest.replay import carry_risk

    samples = tuple(carry_risk.IdentitySample(h, 1.5, 1., r) for h, r in
                    zip(SETTINGS.identity_spot_bumps_rel, (0.05, 0.03, 0.0)))
    monkeypatch.setattr(carry_risk, "sample_chain_identity", lambda *args: samples)
    result = audit_held_book(linear_pricer(), context(), measured_risk(), {}, settings=SETTINGS)
    assert result.identity_residual_hands == 0.0
    assert result.identity_status == "inconclusive"
    assert result.status == "inconclusive"


def test_unavailable_large_pinned_bump_does_not_hide_valid_local_measurements():
    base = linear_pricer()
    ctx = context()
    settings = replace(SETTINGS, audit_spot_bump_rel=.01)

    def price_at(s, div):
        if abs(s - SPOT) > .5 and abs(div.get_yield(TENORS[0]) - RATE) > .01:
            raise ValidationError("large pinned spot scenario outside supported yield range")
        return base(s, div)

    result = audit_held_book(price_at, ctx, measured_risk(), {}, settings=settings)
    assert result.status == "pass"
    assert result.identity_status == "pass"
    assert result.finite_bump_identity_status == "inconclusive"
    assert math.isnan(result.finite_bump_identity_residual_hands)
    assert "supported yield range" in result.finite_bump_identity_reason


def test_a_wide_confidence_interval_is_inconclusive_not_a_pass():
    # Contains zero, but cannot resolve the exposure budget.
    status, reason = status_from_interval(0.0, half_width=0.5, tolerance=0.01)
    assert status == "inconclusive"
    assert "budget" in reason
    # Resolved and inside the budget.
    assert status_from_interval(0.002, half_width=0.001, tolerance=0.01)[0] == "pass"
    # Resolved and outside it, even allowing for the interval.
    assert status_from_interval(0.05, half_width=0.001, tolerance=0.01)[0] == "fail"
    # Not finite at all.
    assert status_from_interval(float("nan"), 0.0, 0.01)[0] == "not_measured"
    with pytest.raises(ValidationError):
        status_from_interval(0.0, -1.0, 0.01)


def test_a_deterministic_engine_needs_no_interval():
    assert status_from_interval(0.0, half_width=0.0, tolerance=0.0)[0] == "pass"
    assert status_from_interval(1.0, half_width=0.0, tolerance=0.0)[0] == "fail"


def test_the_hedge_gap_is_opt_in_and_costs_nothing_undeclared():
    """Undeclared, it must not price: a rebalance policy is a desk convention."""
    result = audit_held_book(linear_pricer(), context(), measured_risk(), {}, settings=SETTINGS)
    assert math.isnan(result.pricing_delta_hedge_gap_hands)
    assert result.hedge_gap_status == "not_measured"
    assert result.price_calls == 4 + 2 * len(context().quotes) + 4 + 4 * 3


def test_the_hedge_gap_measures_the_pricing_delta_against_the_desks_own_secant():
    """W = 0.1*S*F1**2 again: its 1% delta has a known secant error.

    The hedge gap must be measured against the secant across the DESK's
    band, not against a local derivative, so it differs from the local gap
    by exactly the difference of the two references.
    """
    ctx = context()
    settings = replace(SETTINGS, audit_spot_bump_rel=0.01, hedge_resolution_rel=0.0025)

    def price_at(s, div):
        f = s * math.exp((RATE - div.get_yield(TENORS[0])) * TENORS[0])
        return 0.1 * s * f * f

    pricing_delta = direct_frozen_curve_book_delta(price_at, ctx, {}, 1.0)
    risk = FuturesBookRisk(SPOT, pricing_delta, buckets_of(ctx, sample_buckets(price_at, ctx, 1.0)))
    result = audit_held_book(price_at, ctx, risk, {}, settings=settings)

    hedge_delta = direct_frozen_curve_book_delta(price_at, ctx, {}, 0.0025 * ctx.spot)
    assert result.hedge_gap_status == "measured"
    assert result.pricing_delta_hedge_gap_hands == pytest.approx(
        (pricing_delta - hedge_delta) / settings.reference_multiplier, abs=1e-12
    )
    # Two extra price calls, and the local gap is a DIFFERENT number.
    assert result.price_calls == 4 + 2 * len(ctx.quotes) + 4 + 4 * 3 + 2
    assert result.pricing_delta_hedge_gap_hands != pytest.approx(
        result.pricing_delta_local_gap_hands, abs=1e-9
    )


def test_an_out_of_range_hedge_resolution_is_rejected():
    for bad in (0.0, -0.0025, 1.0, 2.0, float("nan")):
        with pytest.raises(ValidationError):
            replace(SETTINGS, hedge_resolution_rel=bad)
