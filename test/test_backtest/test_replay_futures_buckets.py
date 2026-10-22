"""End-to-end multi-leg hedging in the unified replay.

The synthetic chain retires its front contract mid-run, which is where the
interesting behaviour lives: a retired leg must close through the validated
plan rather than the legacy roll, and the automatic correction pair must move
with eligibility instead of pointing at a contract that no longer trades.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import bucket_hedge_fixtures as fixtures
from quantark.backtest.futures_risk import held_book_risk
from quantark.backtest.replay import (
    FuturesBucketHedgeStrategy,
    ReplayBacktestConfig,
    ReplayBacktestEngine,
    ReplayProduct,
)
from quantark.backtest.transaction_costs import TransactionCostModel
from quantark.util.exceptions import ValidationError

RETIREMENT_DATE = pd.Timestamp("2024-01-06")


def make_engine(
    *,
    objective: str = "spot_parallel",
    quantities=(-1.0,),
    round_contracts: bool = False,
    delta_threshold: float = 0.0,
    correction_pair=None,
    extrapolation: str = "flat_q",
    dataset=None,
    cost_model=None,
    product_kwargs=None,
    **config_kwargs,
) -> ReplayBacktestEngine:
    products = [
        ReplayProduct(
            product=fixtures.market_product(**(product_kwargs or {})),
            quantity=q,
            position_id=index,
            has_lifecycle=True,
        )
        for index, q in enumerate(quantities)
    ]
    options = dict(
        products=products,
        market_data=dataset if dataset is not None else fixtures.market_dataset(),
        engine_config=fixtures.market_engine_config(extrapolation),
        strategy=FuturesBucketHedgeStrategy(
            objective=objective,
            correction_pair=correction_pair,
            round_contracts=round_contracts,
            delta_threshold=delta_threshold,
        ),
        calculate_surfaces=False,
        calculate_event_probabilities=False,
    )
    if cost_model is not None:
        options["transaction_cost_model"] = cost_model
    options.update(config_kwargs)
    return ReplayBacktestEngine(ReplayBacktestConfig(**options))


def run_capturing_risk(engine):
    """Run, keeping every day's aggregated book risk and leg plan."""
    risks = {}
    targets = {}
    original_risk = engine._measure_book_carry_risk
    original_targets = engine._bucket_targets

    def measure(date, env, context, alive_specs):
        risk = original_risk(date, env, context, alive_specs)
        risks[pd.Timestamp(date)] = risk
        return risk

    def plan(date, risk, carried, any_alive):
        result = original_targets(date, risk, carried, any_alive)
        targets[pd.Timestamp(date)] = result[0]
        return result

    engine._measure_book_carry_risk = measure
    engine._bucket_targets = plan
    results = engine.run()
    return results, risks, targets


# ---------------------------------------------------------------------------
# The book trades every leg
# ---------------------------------------------------------------------------


def test_a_bucket_run_holds_every_eligible_node():
    engine = make_engine()
    results, risks, _ = run_capturing_risk(engine)
    trades = results.trades_df()
    assert set(trades["contract"]) == {"IM2401", "IM2402", "IM2403"}
    first = pd.Timestamp("2024-01-02")
    assert risks[first].contracts == ("IM2401", "IM2402", "IM2403")
    assert set(engine.hedge_book.contracts()) == {"IM2402", "IM2403"}


def test_the_ideal_targets_meet_the_policy_residual_table():
    engine = make_engine()
    _, risks, targets = run_capturing_risk(engine)
    date = pd.Timestamp("2024-01-02")
    risk, plan = risks[date], targets[date]
    delta, rho = held_book_risk(risk, plan.ideal)
    assert delta == pytest.approx(0.0, abs=1e-6)
    assert sum(rho.values()) == pytest.approx(0.0, abs=1e-6)
    # Shape risk survives, concentrated on the chosen pair.
    assert plan.correction_pair == ("IM2401", "IM2403")
    assert abs(rho["IM2401"]) > 0.0
    assert rho["IM2402"] == pytest.approx(0.0, abs=1e-6)
    assert rho["IM2401"] == pytest.approx(-rho["IM2403"], rel=1e-6)


@pytest.mark.parametrize(
    "objective, zero_delta, zero_parallel",
    [("nodes", False, True), ("spot_far", True, False), ("spot_parallel", True, True)],
)
def test_each_objective_leaves_exactly_its_declared_residual(
    objective, zero_delta, zero_parallel
):
    engine = make_engine(objective=objective)
    _, risks, targets = run_capturing_risk(engine)
    date = pd.Timestamp("2024-01-02")
    delta, rho = held_book_risk(risks[date], targets[date].ideal)
    assert (abs(delta) < 1e-6) is zero_delta
    assert (abs(sum(rho.values())) < 1e-6) is zero_parallel


# ---------------------------------------------------------------------------
# Several products
# ---------------------------------------------------------------------------


def test_a_two_product_book_aggregates_its_risk_linearly():
    single = make_engine(quantities=(1.0,))
    _, single_risks, _ = run_capturing_risk(single)
    book = make_engine(quantities=(1.0, -2.0))
    _, book_risks, _ = run_capturing_risk(book)
    date = pd.Timestamp("2024-01-02")
    unit = single_risks[date]
    aggregated = book_risks[date]
    # Identical products at +1 and -2 net to -1 unit of the same risk.
    assert aggregated.delta_q == pytest.approx(-unit.delta_q, rel=1e-9)
    for combined, one in zip(aggregated.buckets, unit.buckets):
        assert combined.bucket_currency == pytest.approx(
            -one.bucket_currency, rel=1e-9
        )


def test_every_product_gets_buckets_not_only_the_one_that_built_the_env():
    """The shared env is built through replay zero alone.

    A second product whose buckets came from its own ``last_carry_context``
    would find nothing there, so the engine passes ONE context explicitly.
    """
    engine = make_engine(quantities=(1.0, -2.0))
    _, risks, _ = run_capturing_risk(engine)
    assert engine._replays[0].last_carry_context is not None
    assert engine._replays[1].last_carry_context is None
    date = pd.Timestamp("2024-01-02")
    assert len(risks[date].buckets) == 3
    assert all(
        b.bucket_currency != 0.0 for b in risks[date].buckets
    )


def test_the_book_prices_every_product_with_its_own_engine():
    engine = make_engine(quantities=(1.0, -2.0))
    seen = []
    original = engine._measure_book_carry_risk

    def measure(date, env, context, alive_specs):
        seen.append([id(spec[2]) for spec in alive_specs])
        return original(date, env, context, alive_specs)

    engine._measure_book_carry_risk = measure
    engine.run()
    assert seen
    for day in seen:
        assert len(day) == 2
        assert day == [id(e) for e in engine._pricing_engines]


# ---------------------------------------------------------------------------
# Retirement
# ---------------------------------------------------------------------------


def test_a_retired_leg_is_closed_and_the_pair_moves_with_eligibility():
    engine = make_engine()
    results, risks, targets = run_capturing_risk(engine)
    trades = results.trades_df()
    closes = trades[trades["reason"] == "bucket_leg_retired"]
    assert list(closes["contract"]) == ["IM2401"]
    assert pd.Timestamp(closes["date"].iloc[0]) == RETIREMENT_DATE
    assert closes["trade_type"].iloc[0] == "hedge_close"
    # Before retirement the pair spans the whole curve; afterwards it spans
    # what is left, which is a different calendar spread.
    assert targets[pd.Timestamp("2024-01-05")].correction_pair == ("IM2401", "IM2403")
    assert targets[RETIREMENT_DATE].correction_pair == ("IM2402", "IM2403")
    assert risks[RETIREMENT_DATE].contracts == ("IM2402", "IM2403")


def test_a_retired_leg_closes_even_inside_a_wide_no_trade_band():
    engine = make_engine(delta_threshold=1_000.0)
    # Seed a leg the band would otherwise never let the engine touch.
    engine.hedge_book.trade("IM2401", 2.0, 4699.0, 200.0)
    results, _, _ = run_capturing_risk(engine)
    trades = results.trades_df()
    # The band suppressed every ordinary rebalance ...
    assert set(trades["reason"]) <= {"bucket_leg_retired", "product_terminated"}
    # ... but the contract that left the curve was still closed.
    assert list(trades["contract"]) == ["IM2401"]
    assert trades["quantity"].iloc[0] == pytest.approx(-2.0)
    assert pd.Timestamp(trades["date"].iloc[0]) == RETIREMENT_DATE
    assert engine.hedge_book.quantity("IM2401") == 0.0


def test_the_legacy_roll_never_runs_for_a_bucket_hedge():
    engine = make_engine()
    calls = []
    engine._roll_contract = lambda *a, **k: calls.append(a)
    results, _, _ = run_capturing_risk(engine)
    assert calls == []
    assert not any(
        str(t).startswith("roll_") for t in results.trades_df()["trade_type"]
    )


def test_an_explicit_pair_that_leaves_the_curve_fails_without_a_fallback():
    engine = make_engine(correction_pair=("IM2401", "IM2403"))
    with pytest.raises(ValidationError, match="infeasible"):
        engine.run()
    assert engine._hedge_infeasibility is not None
    assert engine._hedge_infeasibility["objective"] == "spot_parallel"
    assert "IM2401" not in engine._hedge_infeasibility["eligible_contracts"]
    assert "eligible contracts" in engine._hedge_infeasibility["cause"]


def test_a_single_eligible_node_makes_the_primary_policy_infeasible():
    # A window that ends before the single contract drops below the minimum
    # tenor, so the curve is genuinely one node rather than none.
    dataset = fixtures.market_dataset(
        chain=fixtures.MARKET_CHAIN[:1], dates=fixtures.MARKET_DATES[:4]
    )
    engine = make_engine(dataset=dataset)
    with pytest.raises(ValidationError, match="two distinct futures tenors"):
        engine.run()
    assert engine._hedge_infeasibility["eligible_contracts"] == ["IM2401"]
    # The comparison policies are feasible on the same one-node curve.
    for objective in ("nodes", "spot_far"):
        results = make_engine(objective=objective, dataset=dataset).run()
        assert not results.trades_df().empty


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


class FailingOnSecondLeg(TransactionCostModel):
    """Raises the second time it is asked for a cost on a given day."""

    def __init__(self):
        self.calls = 0

    def calculate_cost(self, **kwargs) -> float:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("cost model exploded")
        return 0.0


def test_a_cost_model_that_fails_mid_plan_leaves_no_partial_rebalance():
    engine = make_engine(cost_model=FailingOnSecondLeg())
    with pytest.raises(RuntimeError, match="exploded"):
        engine.run()
    assert engine.hedge_book.contracts() == ()
    assert engine.hedge_book.realized_pnl == 0.0
    assert engine._trades == []
    assert engine._transaction_costs == 0.0


def test_a_missing_mark_for_a_held_leg_stops_the_day_before_any_trade():
    engine = make_engine()
    run_capturing_risk(engine)
    # Drop a contract the book still holds and ask for its mark.
    with pytest.raises(ValidationError, match="stale price"):
        engine._hedge_marks(None, {"IM2403": 4600.0})


def test_a_planned_trade_with_an_invalid_price_is_refused():
    engine = make_engine()
    bad_rows = {
        "IM2402": {"futures_price": 0.0, "multiplier": 200.0,
                   "expiry_date": pd.Timestamp("2024-02-23")},
    }
    with pytest.raises(ValidationError, match="invalid mark"):
        engine._rebalance_buckets(
            date=pd.Timestamp("2024-01-02"),
            selected=None,
            chain_rows=bad_rows,
            risk=None,
            any_alive=False,
            carried={"IM2402": 3.0},
        )
    assert engine._trades == []
    assert engine.hedge_book.contracts() == ()


def test_a_planned_trade_with_no_mark_at_all_is_refused():
    engine = make_engine()
    with pytest.raises(ValidationError, match="no tradable mark"):
        engine._rebalance_buckets(
            date=pd.Timestamp("2024-01-02"),
            selected=None,
            chain_rows={},
            risk=None,
            any_alive=False,
            carried={"IM2402": 3.0},
        )
    assert engine._trades == []


def test_the_commit_is_all_or_nothing():
    from quantark.backtest.replay.engine import _PlannedTrade

    engine = make_engine()
    good = _PlannedTrade(
        date=pd.Timestamp("2024-01-02"), contract="IM2402", quantity_delta=1.0,
        price=4675.0, multiplier=200.0, notional=935_000.0, cost=7.0,
        trade_type="hedge_rebalance", reason="delta_rebalance",
    )
    broken = _PlannedTrade(
        date=pd.Timestamp("2024-01-02"), contract="IM2403", quantity_delta=1.0,
        price=-1.0, multiplier=200.0, notional=1.0, cost=3.0,
        trade_type="hedge_rebalance", reason="delta_rebalance",
    )
    with pytest.raises(ValidationError):
        engine._commit_bucket_trades([good, broken])
    assert engine.hedge_book.contracts() == ()
    assert engine._transaction_costs == 0.0
    assert engine._trades == []
    # The same plan without the broken leg commits cleanly.
    engine._commit_bucket_trades([good])
    assert engine.hedge_book.quantity("IM2402") == 1.0
    assert engine._transaction_costs == 7.0
    assert len(engine._trades) == 1


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


def knocking_out_dataset():
    """Spot jumps above the KO barrier on the fourth day."""
    spots = [
        fixtures.MARKET_SPOT,
        fixtures.MARKET_SPOT,
        fixtures.MARKET_SPOT,
        fixtures.MARKET_SPOT * 1.12,
        fixtures.MARKET_SPOT * 1.12,
        fixtures.MARKET_SPOT * 1.12,
        fixtures.MARKET_SPOT * 1.12,
    ]
    return fixtures.market_dataset(spots=spots)


def test_a_terminated_book_closes_every_leg_and_bypasses_the_band():
    engine = make_engine(
        dataset=knocking_out_dataset(),
        product_kwargs={
            "ko_barrier": fixtures.MARKET_SPOT * 1.05,
            # The window is a week long, so the KO observation has to be in it.
            "ko_observation_days": (4.0, 50.0),
        },
    )
    held_before = []
    original = engine._rebalance_buckets

    def spy(*, date, selected, chain_rows, risk, any_alive, carried):
        if not any_alive:
            held_before.append(dict(carried))
        return original(
            date=date, selected=selected, chain_rows=chain_rows, risk=risk,
            any_alive=any_alive, carried=carried,
        )

    engine._rebalance_buckets = spy
    results = engine.run()
    assert not any(r.lifecycle.alive for r in engine._replays)
    assert held_before and held_before[0]
    trades = results.trades_df()
    closed = trades[trades["reason"] == "product_terminated"]
    assert set(closed["trade_type"]) == {"hedge_close"}
    assert engine.hedge_book.contracts() == ()


def test_a_dead_book_needs_no_feasibility_check_and_no_pricer():
    engine = make_engine()
    engine.hedge_book.trade("IM2402", 3.0, 4675.0, 200.0)
    rows = {
        "IM2402": {"futures_price": 4680.0, "multiplier": 200.0,
                   "expiry_date": pd.Timestamp("2024-02-23")},
    }
    targets = engine._rebalance_buckets(
        date=pd.Timestamp("2024-01-09"),
        selected=None,
        chain_rows=rows,
        risk=None,
        any_alive=False,
        carried={"IM2402": 3.0},
    )
    assert targets is None
    assert engine.hedge_book.contracts() == ()
    assert engine._trades[-1]["reason"] == "product_terminated"


def test_the_plan_records_a_decision_row_for_every_coordinate():
    engine = make_engine(delta_threshold=1_000.0)
    results, _, _ = run_capturing_risk(engine)
    rebalances = results.rebalances_df()
    first = rebalances[rebalances["date"] == pd.Timestamp("2024-01-03")]
    assert set(first["active_contract"]) == {"IM2401", "IM2402", "IM2403"}
    # Skipped trades keep the target the policy planned, so the no-trade
    # error stays measurable against it.
    assert (first["trade_contracts"] == 0.0).all()
    assert (~first["should_rebalance"]).all()
    assert first["target_contracts"].abs().sum() > 0.0


def test_carried_holdings_are_captured_before_the_rebalance():
    engine = make_engine()
    captured = []
    original = engine._rebalance_buckets

    def spy(*, date, selected, chain_rows, risk, any_alive, carried):
        captured.append((pd.Timestamp(date), dict(carried)))
        return original(
            date=date, selected=selected, chain_rows=chain_rows, risk=risk,
            any_alive=any_alive, carried=carried,
        )

    engine._rebalance_buckets = spy
    results = engine.run()
    assert captured[0][1] == {}
    trades = results.trades_df()
    day_one = trades[trades["date"] == pd.Timestamp("2024-01-02")]
    expected = {
        row["contract"]: row["quantity"] for _, row in day_one.iterrows()
    }
    assert captured[1][1] == pytest.approx(expected)


def test_rounding_leaves_the_execution_error_the_formula_predicts():
    engine = make_engine(round_contracts=True, quantities=(-50.0,))
    _, risks, targets = run_capturing_risk(engine)
    date = pd.Timestamp("2024-01-02")
    plan = targets[date]
    for contract, rounded in plan.rounded.items():
        assert rounded == float(round(plan.scaled[contract]))
        assert abs(plan.rounding_error()[contract]) <= 0.5
    ideal_delta, _ = held_book_risk(risks[date], plan.ideal)
    actual_delta, _ = held_book_risk(risks[date], plan.rounded)
    predicted = sum(
        b.multiplier * b.price * (plan.rounded[b.contract] - plan.ideal[b.contract])
        / risks[date].spot
        for b in risks[date].buckets
    )
    assert actual_delta - ideal_delta == pytest.approx(predicted, rel=1e-9)


def test_the_state_row_keeps_selected_contract_semantics():
    engine = make_engine()
    results, _, _ = run_capturing_risk(engine)
    states = results.states_df()
    date = pd.Timestamp("2024-01-03")
    row = states[states["date"] == date].iloc[0]
    selected = row["active_contract"]
    rebalances = results.rebalances_df()
    same_day = rebalances[
        (rebalances["date"] == date)
        & (rebalances["active_contract"] == selected)
    ]
    # The scalar column is the SELECTED leg's post-trade count, not the hedge.
    assert row["futures_contracts"] == pytest.approx(
        same_day["target_contracts"].iloc[0]
    )
    assert row["futures_contracts"] != pytest.approx(
        engine.hedge_book.gross(), rel=1e-6
    )
    greeks = results.greeks_df()
    greek_row = greeks[greeks["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    # The bucket-mode hedge delta is the WHOLE book's sum h_i m_i F_i / S; it
    # cannot be reconstructed from one multiplier and one contract count.
    assert greek_row["post_hedge_futures_delta"] != pytest.approx(
        greek_row["post_hedge_contracts"] * greek_row["futures_multiplier"],
        rel=1e-6,
    )
    assert math.isfinite(greek_row["post_hedge_futures_delta"])


def test_the_one_leg_view_refuses_to_answer_for_a_multi_leg_book():
    engine = make_engine()
    run_capturing_risk(engine)
    assert len(engine.hedge_book.contracts()) > 1
    with pytest.raises(ValidationError, match="multi-leg"):
        _ = engine.hedge_position
