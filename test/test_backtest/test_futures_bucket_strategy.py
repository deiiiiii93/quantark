"""The three hedge objectives: exact holdings, intended residuals, refusals.

Every residual claim is checked by the INDEPENDENT repricing audit as well as
by the algebra, because ``held_book_risk`` called with the same targets that
produced it can only agree with itself.
"""

from __future__ import annotations

import math

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
    FuturesBucket,
    held_book_risk,
)
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import audit_held_book
from quantark.backtest.strategy import (
    HEDGE_OBJECTIVES,
    FuturesBucketHedgeStrategy,
    FuturesHedgeTargets,
    ideal_targets,
)
from quantark.backtest.strategy.base_strategy import (
    AssetClass,
    BaseStrategy,
    HedgingTarget,
)
from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError

RATE = 0.0
SPOT = 100.0
TENORS = (0.25, 0.50)
PRICES = (100.0, 100.0)
CONTRACTS = ("IF2503", "IF2506")
COEFFICIENTS = (1.0, -0.5)

SETTINGS = CarryRiskSettings(
    reference_notional=1_000.0,
    reference_multiplier=1.0,
    audit_spot_bump_rel=0.005,
    audit_yield_bump=1e-4,
    delta_tolerance_hands=0.01,
    rhoq_tolerance_bp=0.01,
)


def make_risk(
    *,
    multipliers=(1.0, 1.0),
    coefficients=COEFFICIENTS,
    quantity: float = 1.0,
    tenors=TENORS,
    prices=PRICES,
    contracts=CONTRACTS,
    spot: float = SPOT,
) -> FuturesBookRisk:
    """The design's section 3.3 book, optionally re-scaled or re-signed."""
    reference = linear_book_risk(
        spot=spot,
        quotes=tuple(
            AnalyticQuote(c, t, p, m)
            for c, t, p, m in zip(contracts, tenors, prices, multipliers)
        ),
        futures_coefficients=coefficients,
    )
    buckets = tuple(
        FuturesBucket(
            contract=c,
            expiry_date=None,
            tenor_years=t,
            price=p,
            multiplier=m,
            bucket_currency=quantity * b,
        )
        for c, t, p, m, b in zip(
            contracts, tenors, prices, multipliers, reference.buckets
        )
    )
    return FuturesBookRisk(
        spot=spot, delta_q=quantity * reference.delta_q, buckets=buckets
    )


def audit(holdings, *, risk=None, coefficients=COEFFICIENTS, **kwargs):
    """Reprice the held book independently of the sizing algebra."""
    risk = risk if risk is not None else make_risk(coefficients=coefficients)
    ctx = CarryCurveContext(
        quotes=tuple(
            IndexFuturesQuote(contract=c, maturity=t, price=p, multiplier=m)
            for c, t, p, m in zip(
                risk.contracts,
                (b.tenor_years for b in risk.buckets),
                (b.price for b in risk.buckets),
                (b.multiplier for b in risk.buckets),
            )
        ),
        spot=risk.spot,
        rate_curve=FlatRateCurve(rate=RATE),
        extrapolation="flat_q",
        underlying="index",
        valuation_date="2025-03-03",
    )
    price_at = linear_book_pricer(
        tenors=tuple(b.tenor_years for b in risk.buckets),
        futures_coefficients=coefficients,
        rate=RATE,
    )
    return audit_held_book(
        price_at, ctx, risk, holdings, settings=SETTINGS, **kwargs
    )


# ---------------------------------------------------------------------------
# Exact holdings for each objective
# ---------------------------------------------------------------------------


IDEAL = {
    "nodes": {CONTRACTS[0]: -1.0, CONTRACTS[1]: 0.5},
    "spot_far": {CONTRACTS[0]: -1.0, CONTRACTS[1]: -0.5},
    "spot_parallel": {CONTRACTS[0]: -3.0, CONTRACTS[1]: 1.5},
}


@pytest.mark.parametrize("objective", HEDGE_OBJECTIVES)
def test_ideal_holdings_match_the_design_table(objective):
    targets, _, _ = ideal_targets(make_risk(), objective)
    for contract, expected in IDEAL[objective].items():
        assert targets[contract] == pytest.approx(expected, rel=1e-12)


def test_the_nodes_objective_leaves_the_pinned_spot_delta():
    result = audit(IDEAL["nodes"])
    assert result.status == "pass"
    assert result.direct_net_delta == pytest.approx(1.0, abs=1e-9)
    assert result.direct_net_parallel_rhoq == pytest.approx(0.0, abs=1e-6)
    assert all(
        v == pytest.approx(0.0, abs=1e-6) for v in result.direct_nodal_rhoq.values()
    )


def test_the_spot_far_objective_leaves_the_far_node():
    result = audit(IDEAL["spot_far"])
    assert result.status == "pass"
    assert result.direct_net_delta == pytest.approx(0.0, abs=1e-9)
    # D_F * S * T_n = 1 * 100 * 0.5
    assert result.direct_nodal_rhoq[CONTRACTS[0]] == pytest.approx(0.0, abs=1e-6)
    assert result.direct_nodal_rhoq[CONTRACTS[1]] == pytest.approx(50.0, rel=1e-6)
    assert result.direct_net_parallel_rhoq == pytest.approx(50.0, rel=1e-6)


def test_the_primary_objective_leaves_a_plus_k_minus_k_pair():
    result = audit(IDEAL["spot_parallel"])
    assert result.status == "pass"
    assert result.direct_net_delta == pytest.approx(0.0, abs=1e-9)
    assert result.direct_net_parallel_rhoq == pytest.approx(0.0, abs=1e-6)
    assert result.direct_nodal_rhoq[CONTRACTS[0]] == pytest.approx(50.0, rel=1e-6)
    assert result.direct_nodal_rhoq[CONTRACTS[1]] == pytest.approx(-50.0, rel=1e-6)


def test_zero_parallel_rhoq_does_not_mean_the_carry_risk_is_gone():
    """The unhedged book already has zero parallel rhoq; the fold ADDS it."""
    risk = make_risk()
    assert risk.parallel_rhoq == pytest.approx(0.0, abs=1e-12)
    unhedged = audit({})
    assert unhedged.direct_net_parallel_rhoq == pytest.approx(0.0, abs=1e-6)
    far_fold = audit(IDEAL["spot_far"])
    assert far_fold.direct_net_parallel_rhoq == pytest.approx(50.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Corrections, multipliers, signs
# ---------------------------------------------------------------------------


def test_the_correction_is_reported_separately_from_the_bucket_legs():
    targets, correction, chosen = ideal_targets(make_risk(), "spot_parallel")
    assert chosen == CONTRACTS
    assert correction[CONTRACTS[0]] == pytest.approx(-2.0)
    assert correction[CONTRACTS[1]] == pytest.approx(1.0)
    # buckets + correction = target
    assert targets[CONTRACTS[0]] == pytest.approx(-1.0 + correction[CONTRACTS[0]])
    assert targets[CONTRACTS[1]] == pytest.approx(0.5 + correction[CONTRACTS[1]])


def test_the_default_pair_is_the_earliest_and_latest_eligible_tenor():
    risk = make_risk(
        contracts=("A", "B", "C"),
        tenors=(0.25, 0.5, 1.0),
        prices=(100.0, 100.0, 100.0),
        multipliers=(1.0, 1.0, 1.0),
        coefficients=(1.0, -0.5, 0.25),
    )
    _, correction, chosen = ideal_targets(risk, "spot_parallel")
    assert chosen == ("A", "C")
    assert correction["B"] == 0.0


def test_an_explicit_pair_is_honoured_and_changes_the_shape_risk():
    risk = make_risk(
        contracts=("A", "B", "C"),
        tenors=(0.25, 0.5, 1.0),
        prices=(100.0, 100.0, 100.0),
        multipliers=(1.0, 1.0, 1.0),
        coefficients=(1.0, -0.5, 0.25),
    )
    default_targets, _, _ = ideal_targets(risk, "spot_parallel")
    near_targets, _, chosen = ideal_targets(risk, "spot_parallel", ("A", "B"))
    assert chosen == ("A", "B")
    _, default_rho = held_book_risk(risk, default_targets)
    _, near_rho = held_book_risk(risk, near_targets)
    # Both are parallel neutral, but a near pair amplifies |K|.
    assert sum(default_rho.values()) == pytest.approx(0.0, abs=1e-9)
    assert sum(near_rho.values()) == pytest.approx(0.0, abs=1e-9)
    assert max(abs(v) for v in near_rho.values()) > max(
        abs(v) for v in default_rho.values()
    )


def test_distinct_multipliers_divide_each_leg_by_its_own_size():
    risk = make_risk(multipliers=(200.0, 300.0))
    targets, _, _ = ideal_targets(risk, "nodes")
    assert targets[CONTRACTS[0]] == pytest.approx(-1.0 / 200.0)
    assert targets[CONTRACTS[1]] == pytest.approx(0.5 / 300.0)
    result = audit(targets, risk=risk)
    assert result.status == "pass"
    assert all(
        v == pytest.approx(0.0, abs=1e-6) for v in result.direct_nodal_rhoq.values()
    )


@pytest.mark.parametrize("quantity", [1.0, -1.0, 2.0])
def test_a_signed_book_quantity_flips_or_scales_every_leg(quantity):
    risk = make_risk(quantity=quantity)
    targets, _, _ = ideal_targets(risk, "spot_parallel")
    for contract, unit in IDEAL["spot_parallel"].items():
        assert targets[contract] == pytest.approx(quantity * unit, rel=1e-12)


def test_the_strategy_has_no_product_quantity_argument():
    import inspect

    signature = inspect.signature(FuturesBucketHedgeStrategy().target_legs)
    assert "product_quantity" not in signature.parameters
    assert set(signature.parameters) == {
        "buckets",
        "delta_q",
        "spot",
        "held_contracts",
    }


# ---------------------------------------------------------------------------
# Ratio, rounding and the no-trade band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.0, 1.0, 0.4, 1.2])
def test_the_hedge_ratio_scales_every_leg_including_oversized(ratio):
    strategy = FuturesBucketHedgeStrategy(
        objective="spot_parallel", hedge_ratio=ratio, round_contracts=False
    )
    risk = make_risk()
    plan = strategy.target_legs(
        buckets=risk.buckets, delta_q=risk.delta_q, spot=risk.spot
    )
    for contract, ideal in IDEAL["spot_parallel"].items():
        assert plan.ideal[contract] == pytest.approx(ideal)
        assert plan.scaled[contract] == pytest.approx(ratio * ideal)
        assert plan.rounded[contract] == pytest.approx(ratio * ideal)
    assert plan.hedge_ratio == ratio


def test_a_partial_ratio_leaves_the_predicted_residual():
    ratio = 0.4
    risk = make_risk()
    strategy = FuturesBucketHedgeStrategy(
        objective="spot_parallel", hedge_ratio=ratio, round_contracts=False
    )
    plan = strategy.target_legs(
        buckets=risk.buckets, delta_q=risk.delta_q, spot=risk.spot
    )
    result = audit(plan.scaled)
    # D_actual = (1 - eta) D + eta D* with D* = 0.
    assert result.status == "pass"
    assert result.direct_net_delta == pytest.approx(
        (1.0 - ratio) * risk.delta_q, abs=1e-9
    )


def test_rounding_follows_the_existing_python_convention():
    risk = make_risk(coefficients=(0.5, -0.25))
    strategy = FuturesBucketHedgeStrategy(objective="nodes", round_contracts=True)
    plan = strategy.target_legs(
        buckets=risk.buckets, delta_q=risk.delta_q, spot=risk.spot
    )
    # -0.5 and +0.25 exactly: banker's rounding gives -0.0 and 0.
    assert plan.scaled[CONTRACTS[0]] == pytest.approx(-0.5)
    assert plan.rounded[CONTRACTS[0]] == float(round(-0.5))
    assert plan.rounded[CONTRACTS[0]] == 0.0
    assert plan.rounded[CONTRACTS[1]] == float(round(0.25))
    assert plan.rounding_error()[CONTRACTS[0]] == pytest.approx(0.5)


def test_the_band_is_strict_and_lives_outside_the_target_kernel():
    strategy = FuturesBucketHedgeStrategy(delta_threshold=1.0)
    assert not strategy.should_rebalance(0.0, 1.0)
    assert strategy.should_rebalance(0.0, 1.0000001)
    assert not strategy.should_rebalance(3.0, 2.0)
    assert strategy.should_rebalance(3.0, 1.5)
    # The plan itself is unaffected by the band.
    risk = make_risk()
    plan = strategy.target_legs(
        buckets=risk.buckets, delta_q=risk.delta_q, spot=risk.spot
    )
    assert plan.rounded[CONTRACTS[0]] == -3.0


def test_the_plan_records_the_residuals_the_policy_intended():
    risk = make_risk()
    strategy = FuturesBucketHedgeStrategy(objective="spot_far")
    plan = strategy.target_legs(
        buckets=risk.buckets, delta_q=risk.delta_q, spot=risk.spot
    )
    assert plan.ideal_net_delta == pytest.approx(0.0, abs=1e-12)
    assert plan.ideal_net_parallel_rhoq == pytest.approx(50.0, rel=1e-12)
    assert plan.ideal_gross_nodal_rhoq == pytest.approx(50.0, rel=1e-12)
    assert plan.objective == "spot_far"


def test_retired_contracts_get_an_explicit_zero_target():
    risk = make_risk()
    strategy = FuturesBucketHedgeStrategy()
    plan = strategy.target_legs(
        buckets=risk.buckets,
        delta_q=risk.delta_q,
        spot=risk.spot,
        held_contracts={"IF2412": 4.0, CONTRACTS[0]: -2.0},
    )
    assert plan.retired_contracts == ("IF2412",)
    assert plan.rounded["IF2412"] == 0.0
    assert plan.scaled["IF2412"] == 0.0
    assert plan.ideal["IF2412"] == 0.0


def test_targets_snapshot_their_mappings():
    ideal = {CONTRACTS[0]: 1.0}
    plan = FuturesHedgeTargets(
        objective="nodes",
        ideal=ideal,
        scaled=dict(ideal),
        rounded=dict(ideal),
        correction={CONTRACTS[0]: 0.0},
        correction_pair=None,
        retired_contracts=[],
        ideal_net_delta=0.0,
        ideal_net_rhoq={CONTRACTS[0]: 0.0},
        hedge_ratio=1.0,
        rounded_requested=True,
    )
    ideal[CONTRACTS[0]] = 99.0
    assert plan.ideal[CONTRACTS[0]] == 1.0
    assert plan.retired_contracts == ()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_the_primary_policy_refuses_a_single_node_even_when_df_is_negligible():
    risk = make_risk(
        contracts=("A",), tenors=(0.25,), prices=(100.0,), multipliers=(1.0,),
        coefficients=(1.0,),
    )
    negligible = FuturesBookRisk(
        spot=risk.spot,
        delta_q=sum(b.price / risk.spot * b.bucket_currency for b in risk.buckets),
        buckets=risk.buckets,
    )
    assert negligible.delta_f_derived == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValidationError, match="two distinct"):
        ideal_targets(negligible, "spot_parallel")
    # The comparison policies still work with one node.
    assert ideal_targets(negligible, "nodes")[0]["A"] == pytest.approx(
        -negligible.buckets[0].bucket_currency
    )
    assert "A" in ideal_targets(negligible, "spot_far")[0]


def test_an_unavailable_or_reversed_explicit_pair_fails_without_fallback():
    risk = make_risk()
    with pytest.raises(ValidationError, match="eligible contracts"):
        ideal_targets(risk, "spot_parallel", (CONTRACTS[0], "IF2512"))
    with pytest.raises(ValidationError, match="eligible contracts"):
        ideal_targets(risk, "spot_parallel", (CONTRACTS[0], CONTRACTS[0]))
    with pytest.raises(ValidationError, match="increasing tenor"):
        ideal_targets(risk, "spot_parallel", (CONTRACTS[1], CONTRACTS[0]))


def test_a_pair_is_meaningless_for_the_other_objectives():
    risk = make_risk()
    for objective in ("nodes", "spot_far"):
        with pytest.raises(ValidationError, match="spot_parallel"):
            ideal_targets(risk, objective, CONTRACTS)
        with pytest.raises(ValidationError, match="spot_parallel"):
            FuturesBucketHedgeStrategy(
                objective=objective, correction_pair=CONTRACTS
            )


def test_unknown_objectives_and_non_finite_settings_are_rejected():
    risk = make_risk()
    with pytest.raises(ValidationError):
        ideal_targets(risk, "spot_near")
    with pytest.raises(ValidationError):
        FuturesBucketHedgeStrategy(objective="delta_only")
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValidationError):
            FuturesBucketHedgeStrategy(hedge_ratio=bad)
        with pytest.raises(ValidationError):
            FuturesBucketHedgeStrategy(delta_threshold=bad)
    with pytest.raises(ValidationError):
        FuturesBucketHedgeStrategy(correction_pair=(CONTRACTS[0],))


def test_non_finite_book_greeks_are_rejected_before_sizing():
    risk = make_risk()
    strategy = FuturesBucketHedgeStrategy()
    with pytest.raises(ValidationError):
        strategy.target_legs(
            buckets=risk.buckets, delta_q=float("nan"), spot=risk.spot
        )
    with pytest.raises(ValidationError):
        strategy.target_legs(buckets=(), delta_q=1.0, spot=risk.spot)


# ---------------------------------------------------------------------------
# The strategy is multi-leg, and says so
# ---------------------------------------------------------------------------


def test_it_is_a_base_strategy_but_refuses_the_scalar_protocol():
    strategy = FuturesBucketHedgeStrategy()
    assert isinstance(strategy, BaseStrategy)
    assert strategy.asset_class is AssetClass.EQUITY
    assert strategy.hedging_target is HedgingTarget.DELTA
    assert strategy.hedge_instrument == "futures"
    with pytest.raises(ValidationError, match="multi-leg"):
        strategy.calculate_hedge_size(None, {"delta": 1.0}, {})
    with pytest.raises(ValidationError, match="per leg"):
        strategy.should_hedge(None, {"delta": 1.0}, {})
    assert not hasattr(strategy, "target_contracts")


def test_get_parameters_reports_the_whole_policy():
    strategy = FuturesBucketHedgeStrategy(
        objective="spot_parallel",
        correction_pair=CONTRACTS,
        delta_threshold=0.5,
        round_contracts=False,
        hedge_ratio=0.8,
    )
    assert strategy.get_parameters() == {
        "objective": "spot_parallel",
        "correction_pair": CONTRACTS,
        "delta_threshold": 0.5,
        "round_contracts": False,
        "hedge_ratio": 0.8,
    }
    assert not math.isnan(strategy.hedge_ratio)
