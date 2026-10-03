"""Memory-coupon cash against QUAD V2 (spec 3B §5.3, §7.2)."""
import dataclasses

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.v2.engine import PhoenixQuadEngineV2
from quantark.asset.equity.param import PDEParams
from pde_statistics.cases import CASES

PER_DATE, TOTAL = 2e-4, 5e-4


@pytest.mark.slow
def test_study_product_memory_coupon_cash_against_v2(value_mode):
    value_mode("phoenix")
    product, env = CASES["ph_memory"][0](), CASES["ph_memory"][1]()
    pde = PhoenixPDESolver(PDEParams()).calculate_event_stats(product, env)
    v2 = PhoenixQuadEngineV2().calculate_event_stats(product, env)
    assert np.max(np.abs(pde.expected_discounted_coupon_cashflow - v2.expected_discounted_coupon_cashflow)) <= PER_DATE
    assert abs(pde.expected_discounted_coupon_cashflow.sum() - v2.expected_discounted_coupon_cashflow.sum()) <= TOTAL
    assert abs(pde.reconciliation_error) <= 1e-10


def test_trigger_probability_is_memory_free(value_mode):
    value_mode("phoenix")
    mem, env = CASES["ph_memory"][0](), CASES["ph_memory"][1]()
    plain = CASES["ph_cont"][0]()
    a = PhoenixPDESolver(PDEParams()).calculate_event_stats(mem, env).coupon_probability
    b = PhoenixPDESolver(PDEParams()).calculate_event_stats(plain, env).coupon_probability
    assert np.max(np.abs(a - b)) <= 1e-12


def _expiry(memory):
    from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
    from pde_3a.products import PHOENIX
    return make_phoenix(dict(PHOENIX, memory_coupon=memory, coupon_pay_type="expiry"))


@pytest.mark.parametrize("memory", [False, True])
def test_expiry_buckets_hold_the_per_earning_cash(value_mode, memory):
    """EXPIRY (spec 3B §5.3, decision 5): the ledger's n + 1 payment buckets carry exactly the coupon cash
    reported per earning date, and the coupon payment is path dependent."""
    from pde_3a.products import env
    value_mode("phoenix")
    pde = PhoenixPDESolver(PDEParams()).calculate_event_stats(_expiry(memory), env())
    assert pde.coupon_payment_is_path_dependent is True
    n = pde.ko_times.size
    buckets = pde.expected_discounted_cashflows[n:2 * n + 1]
    assert abs(buckets.sum() - pde.expected_discounted_coupon_cashflow.sum()) <= 1e-10
    assert abs(pde.reconciliation_error) <= 1e-10


def test_nodal_termination_units_are_the_knock_out_columns(value_mode):
    """Plan F4's control: without the coupon-barrier smoothing (NODAL) a termination unit T_j is the
    knock-out column KO_j, node for node, on the not-knocked-in block at every interior record. (At the
    maturity record the unit is U's; on the knocked-in block W's upper row is the always-knock-out row
    where the value's is the maturity row.)"""
    from quantark.asset.equity.engine.pde.core.march import march
    from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
    from pde_3a.products import env
    value_mode("phoenix")
    solver = PhoenixPDESolver(PDEParams(event_projection="nodal"))
    p = prepare(solver, _expiry(False), env(), OutputRequest(Operation.EVENT_STATS, event_stats=True))
    L = p.statistics.layout
    alive = march(p).states["stats_v0"]
    interior = slice(0, L.n_ko - 1)
    assert np.max(np.abs(alive[:, L.term_unit[interior]] - alive[:, L.ko[interior]])) <= 1e-15


@pytest.mark.slow
@pytest.mark.parametrize("memory", [False, True])
def test_expiry_buckets_match_v2(value_mode, memory):
    """V2's ledger (quad/v2/engine.py ``_event_stats_from_result``): n knock-out rows, n coupon rows at
    the knock-out settlements, then the terminal rows with the terminal coupon merged into the maturity
    cash -- so the knock-out buckets compare per payment time before maturity, and the coupon cash per
    earning date in total. At the terminal payment time the two split one date differently: V2 books the
    coupon earned at the maturity observation on that record's row, this ledger books it in the terminal
    bucket unless a knock-out pays it (measured: 0.0016 vs V2's 0.017 on that row, plain)."""
    from pde_3a.products import env
    value_mode("phoenix")
    product, e = _expiry(memory), env()
    pde = PhoenixPDESolver(PDEParams()).calculate_event_stats(product, e)
    v2 = PhoenixQuadEngineV2().calculate_event_stats(product, e)
    n, n2 = pde.ko_times.size, v2.ko_times.size
    mine_t, mine = pde.payment_times[n:2 * n], pde.expected_discounted_cashflows[n:2 * n]
    theirs_t, theirs = v2.payment_times[n2:2 * n2], v2.expected_discounted_cashflows[n2:2 * n2]
    terminal_payment = float(pde.payment_times[2 * n])
    for t in np.unique(mine_t[~np.isclose(mine_t, terminal_payment)]):
        got = mine[np.isclose(mine_t, t)].sum()
        want = theirs[np.isclose(theirs_t, t)].sum()
        assert abs(got - want) <= TOTAL, t
    assert abs(pde.expected_discounted_coupon_cashflow.sum() - v2.expected_discounted_coupon_cashflow.sum()) <= TOTAL


def test_carried_arrears_add_less_than_their_discounted_face(value_mode):
    """Review Focus 1: arrears carried in are paid at most once, with the first coupon that triggers."""
    value_mode("phoenix")
    env = CASES["ph_memory"][1]()
    base = CASES["ph_memory"][0]()
    carried = CASES["ph_memory"][0]()
    carried.coupon_config = dataclasses.replace(carried.coupon_config, initial_coupon_arrears=2.0)
    a = PhoenixPDESolver(PDEParams()).calculate_event_stats(base, env)
    b = PhoenixPDESolver(PDEParams()).calculate_event_stats(carried, env)
    extra = float(b.expected_discounted_coupon_cashflow.sum() - a.expected_discounted_coupon_cashflow.sum())
    first_pay = float(env.get_discount_factor(float(a.ko_times[0])))
    assert 0.0 < extra < 2.0 * first_pay
    assert abs(b.reconciliation_error) <= 1e-10
