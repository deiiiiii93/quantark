"""Pruning a stream drops its report only (spec 3B §5.1 "streams", Review Focus 2; DESIGN §4.9b): the termination
outcomes, the coupon cash and the ledger stay, bit for bit (unified event statistics T12)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.cashleg.event_distribution import EventType
from pde_statistics.cases import CASES

KEPT = ("observations.probability", "observations.survival", "observations.discounted_cash",
        "observations.cash_to_pay", "ledger.determination_times", "ledger.payment_times", "ledger.discounted_cash",
        "ledger.cash_to_pay")


def _assert_outcomes_kept(full, lean):
    for path in KEPT:
        assert np.array_equal(full.require(path), lean.require(path)), path
    assert full.pv == lean.pv and full.reconciliation_error == lean.reconciliation_error
    assert full.probability_residual == lean.probability_residual
    assert full.maturity == lean.maturity and full.ki_settlement == lean.ki_settlement
    assert full.knock_in.expected_loss == lean.knock_in.expected_loss


@pytest.mark.parametrize("streams", [frozenset({EventType.KO}), frozenset(), frozenset({EventType.KI})],
                         ids=["ko", "none", "ki"])
def test_snowball_pruning_is_reporting_only(streams):
    solver, product, e = SnowballPDESolver(PDEParams()), CASES["sb_cont"][0](), CASES["sb_cont"][1]()
    full, lean = solver.calculate_event_stats(product, e), solver.calculate_event_stats(product, e, streams=streams)
    _assert_outcomes_kept(full, lean)
    reason = lean.unavailable["knock_in.probability"]
    assert reason == (full.unavailable["knock_in.probability"] if EventType.KI in streams else "not requested")


@pytest.mark.parametrize("case", ["ph_cont", "ph_memory"])
@pytest.mark.parametrize("streams", [frozenset({EventType.KO}), frozenset(), frozenset({EventType.KI})],
                         ids=["ko", "none", "ki"])
def test_phoenix_pruning_is_reporting_only(case, streams):
    solver, product, e = PhoenixPDESolver(PDEParams()), CASES[case][0](), CASES[case][1]()
    full, lean = solver.calculate_event_stats(product, e), solver.calculate_event_stats(product, e, streams=streams)
    _assert_outcomes_kept(full, lean)
    # the coupon probabilities are pruned, the coupon cash is not (decision 3)
    assert np.array_equal(full.coupons.discounted_cash, lean.coupons.discounted_cash)
    assert np.array_equal(full.coupons.cash_to_pay, lean.coupons.cash_to_pay)
    if EventType.COUPON not in streams:
        assert lean.coupons.probability is None and lean.unavailable["coupons.probability"] == "not requested"
