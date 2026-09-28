"""Sampled futures buckets against analytic payoffs, and signed aggregation.

Every reference here comes from ``bucket_hedge_fixtures``, which never
imports the sampler it validates.  The pricing callbacks below depend on
nothing but ``(spot, dividend)``, which is exactly the boundary the replay
adapter enforces.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from bucket_hedge_fixtures import (
    AnalyticQuote,
    EarlyDigitalCase,
    early_digital_forward_derivative,
    early_digital_price,
    flat_forward_carry_forward,
    flat_q_forward,
    forward_claim_risk,
    log_forward_elasticities,
    pinned_forward_elasticity,
)
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.backtest.futures_risk import held_book_risk
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import (
    aggregate_book_risk,
    buckets_of,
    measure_product_carry_risk,
    sample_buckets,
)
from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError

RATE = 0.02
SPOT = 4700.0
VALUATION = datetime(2025, 3, 3)
POINTS = 1.0

CHAIN = (
    ("IF2503", 0.25, 4680.0),
    ("IF2506", 0.50, 4655.0),
    ("IF2509", 0.75, 4610.0),
)


def quotes(count: int = 3):
    return tuple(
        IndexFuturesQuote(
            contract=c,
            maturity=t,
            price=p,
            multiplier=200.0,
            expiry_date=datetime(2025, 3, 3) + __import__("datetime").timedelta(
                days=int(round(t * 365))
            ),
        )
        for c, t, p in CHAIN[:count]
    )


def analytic(count: int = 3):
    return tuple(AnalyticQuote(c, t, p, 200.0) for c, t, p in CHAIN[:count])


def context(count: int = 3, *, extrapolation: str = "flat_q", spot: float = SPOT):
    return CarryCurveContext(
        quotes=quotes(count),
        spot=spot,
        rate_curve=FlatRateCurve(rate=RATE),
        extrapolation=extrapolation,
        underlying="index",
        valuation_date=VALUATION,
    )


class CountingPricer:
    """A fixed-state pricing callback that records every scenario it saw."""

    def __init__(self, price):
        self._price = price
        self.calls: list[tuple[float, float]] = []

    def __call__(self, spot, dividend):
        value = self._price(float(spot), dividend)
        self.calls.append((float(spot), value))
        return value

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def spots(self) -> set:
        return {spot for spot, _ in self.calls}


def forward_claim_pricer(maturity: float, notional: float = 1.0):
    """PV of ``notional * S_T``: a claim whose carry exposure is exact."""

    def price(spot, dividend):
        q = float(dividend.get_yield(maturity))
        return (
            notional
            * math.exp(-RATE * maturity)
            * spot
            * math.exp((RATE - q) * maturity)
        )

    return CountingPricer(price)


# ---------------------------------------------------------------------------
# Buckets against the analytic forward claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extrapolation", ["flat_q", "flat_forward_carry"])
@pytest.mark.parametrize("maturity", [0.1, 0.375, 0.5, 0.9, 1.4])
def test_forward_claim_buckets_match_the_analytic_derivatives(
    extrapolation, maturity
):
    ctx = context(extrapolation=extrapolation)
    pricer = forward_claim_pricer(maturity, notional=1_000.0)
    buckets = buckets_of(ctx, sample_buckets(pricer, ctx, POINTS))
    reference = forward_claim_risk(
        spot=SPOT,
        quotes=analytic(),
        maturity=maturity,
        rate=RATE,
        convention=extrapolation,
        notional=1_000.0,
    )
    for bucket, expected in zip(buckets, reference.buckets):
        # A one-point central difference carries O(h^2) truncation; the
        # far-tail node's large log elasticity makes it visible at 1e-7.
        # test_the_bucket_error_falls_like_the_square_of_the_bump pins that
        # this residual is the bump ladder and not a modelling error.
        assert bucket.bucket_currency == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_a_claim_inside_the_first_interval_only_loads_the_first_node():
    ctx = context(extrapolation="flat_forward_carry")
    buckets = buckets_of(ctx, sample_buckets(forward_claim_pricer(0.1), ctx, POINTS))
    assert buckets[0].bucket_currency != pytest.approx(0.0, abs=1e-9)
    assert buckets[1].bucket_currency == pytest.approx(0.0, abs=1e-12)
    assert buckets[2].bucket_currency == pytest.approx(0.0, abs=1e-12)


def test_a_flat_q_claim_inside_an_interval_loads_both_bracketing_nodes():
    ctx = context(extrapolation="flat_q")
    buckets = buckets_of(ctx, sample_buckets(forward_claim_pricer(0.375), ctx, POINTS))
    assert buckets[0].bucket_currency != pytest.approx(0.0, abs=1e-9)
    assert buckets[1].bucket_currency != pytest.approx(0.0, abs=1e-9)
    assert buckets[2].bucket_currency == pytest.approx(0.0, abs=1e-12)


def test_the_flat_forward_tail_derivative_is_not_the_log_elasticity():
    maturity = 1.4
    ctx = context(extrapolation="flat_forward_carry")
    pricer = forward_claim_pricer(maturity)
    buckets = buckets_of(ctx, sample_buckets(pricer, ctx, POINTS))
    value = pricer.calls[0][1]  # any base-order price is the right scale
    elasticities = log_forward_elasticities(
        analytic(), maturity, convention="flat_forward_carry"
    )
    alpha = (maturity - 0.75) / (0.75 - 0.50)
    assert elasticities[-1] == pytest.approx(1.0 + alpha)
    assert elasticities[-2] == pytest.approx(-alpha)
    # dV/dF_n divides by F_n; using 1 + alpha directly is off by that factor.
    price = math.exp(-RATE * maturity) * flat_forward_carry_forward(
        SPOT, analytic(), maturity
    )
    assert buckets[-1].bucket_currency == pytest.approx(
        price * (1.0 + alpha) / CHAIN[2][2], rel=1e-6
    )
    assert buckets[-1].bucket_currency != pytest.approx(1.0 + alpha, rel=1e-3)
    assert buckets[-2].bucket_currency == pytest.approx(
        -price * alpha / CHAIN[1][2], rel=1e-6
    )
    assert buckets[0].bucket_currency == pytest.approx(0.0, abs=1e-12)
    assert value > 0.0


def test_the_bucket_error_falls_like_the_square_of_the_bump():
    """The far-tail residual is the central difference, not a wrong Greek.

    Halving the bump must quarter the error.  A residual that did not follow
    that law would be a modelling problem, and widening the tolerance would
    have hidden it.
    """
    maturity = 1.4
    ctx = context(extrapolation="flat_forward_carry")
    reference = forward_claim_risk(
        spot=SPOT,
        quotes=analytic(),
        maturity=maturity,
        rate=RATE,
        convention="flat_forward_carry",
        notional=1_000.0,
    ).buckets[-2]
    errors = []
    for points in (2.0, 1.0, 0.5):
        sampled = buckets_of(
            ctx,
            sample_buckets(
                forward_claim_pricer(maturity, notional=1_000.0), ctx, points
            ),
        )
        errors.append(abs(sampled[-2].bucket_currency - reference))
    assert errors[0] > errors[1] > errors[2] > 0.0
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.05)


def test_the_one_node_limit_is_the_shared_first_segment():
    for extrapolation in ("flat_q", "flat_forward_carry"):
        ctx = context(1, extrapolation=extrapolation)
        buckets = buckets_of(
            ctx, sample_buckets(forward_claim_pricer(0.4), ctx, POINTS)
        )
        assert len(buckets) == 1
        reference = forward_claim_risk(
            spot=SPOT,
            quotes=analytic(1),
            maturity=0.4,
            rate=RATE,
            convention=extrapolation,
        )
        assert buckets[0].bucket_currency == pytest.approx(
            reference.buckets[0], rel=1e-7
        )


# ---------------------------------------------------------------------------
# The early-monitoring digital: 29/30 of the spot delta is unspanned
# ---------------------------------------------------------------------------


def digital_case_context(extrapolation: str = "flat_forward_carry"):
    case = EarlyDigitalCase()
    chain = (
        AnalyticQuote("IF-A", case.first_tenor, 4695.0, 200.0),
        AnalyticQuote("IF-B", 90.0 / 365.0, 4685.0, 200.0),
    )
    ctx = CarryCurveContext(
        quotes=tuple(
            IndexFuturesQuote(
                contract=q.contract,
                maturity=q.tenor_years,
                price=q.price,
                multiplier=q.multiplier,
            )
            for q in chain
        ),
        spot=case.spot,
        rate_curve=FlatRateCurve(rate=case.rate),
        extrapolation=extrapolation,
        underlying="index",
        valuation_date=VALUATION,
    )
    return case, chain, ctx


def digital_pricer(case: EarlyDigitalCase):
    def price(spot, dividend):
        t = case.observation
        q = float(dividend.get_yield(t))
        forward = spot * math.exp((case.rate - q) * t)
        return early_digital_price(
            forward, case.barrier, case.vol, t, case.discount, case.notional
        )

    return CountingPricer(price)


@pytest.mark.parametrize("extrapolation", ["flat_q", "flat_forward_carry"])
def test_early_digital_buckets_and_the_29_30_residual(extrapolation):
    case, chain, ctx = digital_case_context(extrapolation)
    pricer = digital_pricer(case)
    risk = measure_product_carry_risk(
        pricer,
        ctx,
        delta_q=case.frozen_curve_delta(chain, extrapolation),
        points=POINTS,
    )
    forward = case.forward(chain, extrapolation)
    dv_df = early_digital_forward_derivative(
        forward, case.barrier, case.vol, case.observation, case.discount, case.notional
    )
    elasticities = log_forward_elasticities(
        chain, case.observation, convention=extrapolation
    )
    for bucket, elasticity, quote in zip(risk.buckets, elasticities, chain):
        expected = dv_df * forward * elasticity / quote.price
        assert bucket.bucket_currency == pytest.approx(expected, rel=2e-6, abs=1e-6)

    from quantark.backtest.futures_risk import FuturesBookRisk

    book = FuturesBookRisk(
        spot=case.spot, delta_q=risk.delta_q, buckets=risk.buckets
    )
    assert book.delta_f_derived / book.delta_q == pytest.approx(
        case.pinned_fraction, rel=1e-6
    )
    assert case.pinned_fraction == pytest.approx(29.0 / 30.0)
    assert pinned_forward_elasticity(
        chain, case.observation, convention=extrapolation
    ) == pytest.approx(29.0 / 30.0)


# ---------------------------------------------------------------------------
# Quantity weighting happens exactly once, at aggregation
# ---------------------------------------------------------------------------


def unit_risk(maturity=0.9, extrapolation="flat_q", spot=SPOT, count=3):
    ctx = context(count, extrapolation=extrapolation, spot=spot)
    pricer = forward_claim_pricer(maturity, notional=1_000.0)
    return measure_product_carry_risk(
        pricer,
        ctx,
        delta_q=forward_claim_risk(
            spot=spot,
            quotes=analytic(count),
            maturity=maturity,
            rate=RATE,
            convention=extrapolation,
            notional=1_000.0,
        ).delta_q,
        points=POINTS,
    )


@pytest.mark.parametrize("quantity", [1.0, -1.0, 2.0])
def test_aggregation_scales_a_single_product_exactly_once(quantity):
    risk = unit_risk()
    book = aggregate_book_risk([(quantity, risk)])
    assert book.delta_q == pytest.approx(quantity * risk.delta_q)
    for aggregated, unit in zip(book.buckets, risk.buckets):
        assert aggregated.bucket_currency == pytest.approx(
            quantity * unit.bucket_currency
        )


def test_a_mixed_signed_book_adds_in_product_order():
    first = unit_risk(maturity=0.9)
    second = unit_risk(maturity=1.4)
    book = aggregate_book_risk([(1.0, first), (-2.0, second)])
    assert book.delta_q == pytest.approx(first.delta_q - 2.0 * second.delta_q)
    for aggregated, a, b in zip(book.buckets, first.buckets, second.buckets):
        assert aggregated.bucket_currency == pytest.approx(
            a.bucket_currency - 2.0 * b.bucket_currency
        )


def test_offsetting_positions_cancel_to_a_flat_book():
    risk = unit_risk()
    book = aggregate_book_risk([(1.0, risk), (-1.0, risk)])
    assert book.delta_q == pytest.approx(0.0, abs=1e-12)
    assert book.gross_nodal_rhoq == pytest.approx(0.0, abs=1e-9)
    delta, rho = held_book_risk(book, {})
    assert delta == pytest.approx(0.0, abs=1e-12)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in rho.values())


def test_per_product_samples_stay_unit_position():
    risk = unit_risk()
    aggregate_book_risk([(7.0, risk)])
    reference = unit_risk()
    assert risk.delta_q == pytest.approx(reference.delta_q)
    for a, b in zip(risk.buckets, reference.buckets):
        assert a.bucket_currency == pytest.approx(b.bucket_currency)


# ---------------------------------------------------------------------------
# The bump set is one fixed state
# ---------------------------------------------------------------------------


def test_sampling_costs_exactly_two_prices_per_node_at_a_pinned_spot():
    ctx = context()
    pricer = forward_claim_pricer(0.9)
    sample_buckets(pricer, ctx, POINTS)
    assert pricer.count == 2 * len(ctx.quotes)
    assert pricer.spots == {SPOT}


def test_measure_records_its_own_provenance():
    ctx = context()
    # A tenor inside the first interval, so the first node actually moves the
    # price; under flat_q a 0.9y claim has an exactly zero front bucket.
    pricer = forward_claim_pricer(0.1)
    risk = measure_product_carry_risk(pricer, ctx, delta_q=1.0, points=0.5)
    assert risk.price_calls == 1 + 2 * len(ctx.quotes)
    assert risk.effective_points == 0.5
    assert risk.base_price == pytest.approx(pricer.calls[0][1])
    assert risk.elapsed_seconds >= 0.0
    provenance = risk.provenance()
    assert provenance["effective_points"] == 0.5
    assert provenance["extrapolation"] == "flat_q"
    assert len(provenance["samples"]) == len(ctx.quotes)
    assert provenance["samples"][0]["price_up"] != provenance["samples"][0]["price_down"]


def test_a_supplied_base_price_is_not_re_priced():
    ctx = context()
    pricer = forward_claim_pricer(0.9)
    risk = measure_product_carry_risk(
        pricer, ctx, delta_q=1.0, points=POINTS, base_price=123.0
    )
    assert risk.base_price == 123.0
    assert risk.price_calls == 2 * len(ctx.quotes)


def test_every_scenario_reaches_the_pricer_through_the_shared_builder():
    ctx = context(extrapolation="flat_forward_carry")
    seen = []

    def price(spot, dividend):
        seen.append(float(dividend.get_yield(0.3)))
        return 1.0

    base = float(ctx.dividend().get_yield(0.3))
    sample_buckets(CountingPricer(price), ctx, POINTS)
    # Six curves, two per node, all rebuilt by the same builder.  At t = 0.3
    # only the two bracketing nodes move the yield; the third node's pair
    # rebuilds to exactly the base curve, which is itself the statement that
    # a 0.3y claim has no far-node bucket.
    assert len(seen) == 6
    assert len({round(v, 15) for v in seen}) == 5
    assert seen[4] == pytest.approx(base, abs=1e-15)
    assert seen[5] == pytest.approx(base, abs=1e-15)
    assert len({round(v, 15) for v in seen[:4]}) == 4


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


def test_a_non_finite_price_is_an_error_not_a_zero_bucket():
    ctx = context()

    def price(spot, dividend):
        return float("nan")

    with pytest.raises(ValidationError):
        sample_buckets(price, ctx, POINTS)


def test_a_raising_pricer_propagates():
    ctx = context()

    class Boom(RuntimeError):
        pass

    def price(spot, dividend):
        raise Boom("engine failed")

    with pytest.raises(Boom):
        sample_buckets(price, ctx, POINTS)


def test_a_non_positive_bump_is_rejected():
    ctx = context()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            sample_buckets(forward_claim_pricer(0.9), ctx, bad)


def test_a_non_finite_delta_is_rejected():
    ctx = context()
    with pytest.raises(ValidationError):
        measure_product_carry_risk(
            forward_claim_pricer(0.9), ctx, delta_q=float("nan"), points=POINTS
        )


def test_aggregation_rejects_different_coordinates():
    base = unit_risk()
    moved_spot = unit_risk(spot=SPOT + 1.0)
    fewer_nodes = unit_risk(count=2)
    other_convention = unit_risk(extrapolation="flat_forward_carry")
    for other in (moved_spot, fewer_nodes, other_convention):
        with pytest.raises(ValidationError):
            aggregate_book_risk([(1.0, base), (1.0, other)])


def test_aggregation_rejects_a_non_finite_quantity_and_an_empty_book():
    risk = unit_risk()
    with pytest.raises(ValidationError):
        aggregate_book_risk([(float("nan"), risk)])
    with pytest.raises(ValidationError):
        aggregate_book_risk([])


def test_sample_order_must_follow_the_curve():
    ctx = context()
    samples = sample_buckets(forward_claim_pricer(0.9), ctx, POINTS)
    with pytest.raises(ValidationError):
        buckets_of(ctx, tuple(reversed(samples)))
    with pytest.raises(ValidationError):
        buckets_of(ctx, samples[:2])


# ---------------------------------------------------------------------------
# The replay adapter's boundary
# ---------------------------------------------------------------------------


def test_the_replay_callback_shares_one_fixed_state():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from replay_golden import fixtures

    from quantark.backtest.replay import AutocallableEngineConfig
    from quantark.backtest.replay.product_replay import ProductReplay
    from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
    from quantark.param import FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment

    replay = ProductReplay(
        product=fixtures._snowball_product(),
        product_quantity=-1.0,
        has_lifecycle=True,
        lifecycle=AutocallableLifecycleState(),
        surface_engine=None,
        event_stats_engine=None,
        engine_config=AutocallableEngineConfig(),
        market_data=fixtures._market_data(),
        start_date=None,
        underlying="CSI500",
        actions_sink=[],
        event_prob_sink=[],
        daily_event_sink=[],
        surfaces_sink=[],
    )
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT, asset_name="CSI500"),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=context().dividend(),
        valuation_date=VALUATION,
    )
    seen = []

    class RecordingEngine:
        def price(self, product, environment):
            seen.append(environment)
            return float(environment.spot) * 0.5

    price_at = replay.carry_price_callback(
        fixtures._snowball_product(), env, engine=RecordingEngine()
    )
    ctx = context()
    sample_buckets(price_at, ctx, POINTS)
    assert len(seen) == 2 * len(ctx.quotes)
    for environment in seen:
        assert environment.vol_surface is env.vol_surface
        assert environment.rate_curve is env.rate_curve
        assert environment.valuation_date == env.valuation_date
        assert float(environment.spot) == pytest.approx(SPOT)
    # Every scenario carries its own dividend object; none is the base one.
    assert len({id(e.div_yield) for e in seen}) == len(seen)
    assert all(e.div_yield is not env.div_yield for e in seen)
