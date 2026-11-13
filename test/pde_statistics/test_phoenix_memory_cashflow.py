"""Memory-coupon cash against QUAD V2 (spec 3B §5.3, §7.2)."""
import dataclasses

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.v2.engine import PhoenixQuadEngineV2
from quantark.asset.equity.param import PDEParams
from pde_statistics.cases import CASES
from pde_statistics.v2_frozen import v2_event_stats

PER_DATE, TOTAL = 2e-4, 5e-4


def test_study_product_memory_coupon_cash_against_v2():
    product, env = CASES["ph_memory"][0](), CASES["ph_memory"][1]()
    pde = PhoenixPDESolver(PDEParams()).calculate_event_stats(product, env)
    v2 = v2_event_stats("phoenix_ph_memory", lambda: PhoenixQuadEngineV2().calculate_event_stats(product, env))
    assert np.max(np.abs(pde.coupons.discounted_cash - v2.coupons.discounted_cash)) <= PER_DATE
    assert abs(pde.coupons.discounted_cash.sum() - v2.coupons.discounted_cash.sum()) <= TOTAL
    assert abs(pde.reconciliation_error) <= 1e-10


def test_trigger_probability_is_memory_free():
    mem, env = CASES["ph_memory"][0](), CASES["ph_memory"][1]()
    plain = CASES["ph_cont"][0]()
    a = PhoenixPDESolver(PDEParams()).calculate_event_stats(mem, env).coupons.probability
    b = PhoenixPDESolver(PDEParams()).calculate_event_stats(plain, env).coupons.probability
    assert np.max(np.abs(a - b)) <= 1e-12


def _expiry(memory):
    from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
    from pde_3a.products import PHOENIX
    return make_phoenix(dict(PHOENIX, memory_coupon=memory, coupon_pay_type="expiry"))


@pytest.mark.parametrize("memory", [False, True])
def test_expiry_buckets_hold_the_per_earning_cash(memory):
    """EXPIRY (spec 3B §5.3, decision 5): the ledger's n + 1 payment buckets carry exactly the coupon cash
    reported per earning date, and the coupon payment is path dependent."""
    from pde_3a.products import env
    pde = PhoenixPDESolver(PDEParams()).calculate_event_stats(_expiry(memory), env())
    assert bool(np.all(pde.coupons.payment_is_path_dependent)) is True
    n = pde.observations.size
    buckets = pde.ledger.discounted_cash[n:2 * n + 1]
    assert abs(buckets.sum() - pde.coupons.discounted_cash.sum()) <= 1e-10
    assert abs(pde.reconciliation_error) <= 1e-10


def test_nodal_termination_units_are_the_knock_out_columns():
    """Plan F4's control: without the coupon-barrier smoothing (NODAL) a termination unit T_j is the
    knock-out column KO_j, node for node, on the not-knocked-in block at every interior record. (At the
    maturity record the unit is U's; on the knocked-in block W's upper row is the always-knock-out row
    where the value's is the maturity row.)"""
    from quantark.asset.equity.engine.pde.core.march import march
    from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
    from pde_3a.products import env
    solver = PhoenixPDESolver(PDEParams(event_projection="nodal"))
    p = prepare(solver, _expiry(False), env(), OutputRequest(Operation.EVENT_STATS, event_stats=True))
    L = p.statistics.layout
    alive = march(p).states["stats_v0"]
    interior = slice(0, L.n_ko - 1)
    assert np.max(np.abs(alive[:, L.term_unit[interior]] - alive[:, L.ko[interior]])) <= 1e-15


@pytest.mark.parametrize("memory", [False, True])
def test_expiry_buckets_match_v2(memory):
    """Both ledgers (``EventStatistics.ledger``, row order KO, path-dependent coupon cash, terminal rows): n
    knock-out rows, then n coupon rows at the knock-out settlements, then the terminal rows -- so the coupon
    buckets compare per payment time before maturity, and the coupon cash per earning date in total. At the
    terminal payment time the two split one date differently (V2 books the coupon earned at the maturity
    observation on that record's row: 0.017 vs this ledger's 0.0016, plain), so it is left out."""
    from pde_3a.products import env
    product, e = _expiry(memory), env()
    pde = PhoenixPDESolver(PDEParams()).calculate_event_stats(product, e)
    v2 = v2_event_stats(f"phoenix_expiry_memory_{str(memory).lower()}",
                        lambda: PhoenixQuadEngineV2().calculate_event_stats(product, e))
    n, n2 = pde.observations.size, v2.observations.size
    mine_t, mine = pde.ledger.payment_times[n:2 * n], pde.ledger.discounted_cash[n:2 * n]
    theirs_t, theirs = v2.ledger.payment_times[n2:2 * n2], v2.ledger.discounted_cash[n2:2 * n2]
    terminal_payment = float(pde.ledger.payment_times[2 * n])
    for t in np.unique(mine_t[~np.isclose(mine_t, terminal_payment)]):
        got = mine[np.isclose(mine_t, t)].sum()
        want = theirs[np.isclose(theirs_t, t)].sum()
        assert abs(got - want) <= TOTAL, t
    assert abs(pde.coupons.discounted_cash.sum() - v2.coupons.discounted_cash.sum()) <= TOTAL


def _c105_at_start():
    """The coupon barrier (105) above the knock-out barrier (103): a knock-out that misses the coupon releases
    the arrears (spec 3B decision 4, the release rule)."""
    from datetime import datetime
    from quantark.util.calendar import CalendarType, create_calendar
    from intraday.conftest import dated_phoenix, flat_env
    t0 = datetime(2026, 3, 16)
    product = dated_phoenix(create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028)), t0, coupon_barrier=105.0)
    return product, flat_env(t0)


def test_release_rule_product_against_v2():
    product, env = _c105_at_start()
    pde = PhoenixPDESolver(PDEParams()).calculate_event_stats(product, env)
    v2 = v2_event_stats("phoenix_release_rule", lambda: PhoenixQuadEngineV2().calculate_event_stats(product, env))
    assert np.max(np.abs(pde.coupons.discounted_cash - v2.coupons.discounted_cash)) <= PER_DATE
    assert abs(pde.coupons.discounted_cash.sum() - v2.coupons.discounted_cash.sum()) <= TOTAL
    assert abs(pde.reconciliation_error) <= 1e-10


@pytest.mark.slow
def test_release_rule_product_price_against_monte_carlo():
    """Eight independent 25,000-path runs (200,000 paths): the PDE price within three batch standard errors."""
    from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
    from quantark.asset.equity.param import MCParams
    product, env = _c105_at_start()
    pde = PhoenixPDESolver(PDEParams()).price(product, env)
    runs = np.array([PhoenixMCEngine(params=MCParams(num_paths=25_000, seed=seed)).price(product, env)
                     for seed in range(1, 9)])
    se = runs.std(ddof=1) / np.sqrt(runs.size)
    assert abs(pde - runs.mean()) <= 3.0 * se, (pde, runs.mean(), se)


def test_carried_arrears_add_less_than_their_discounted_face():
    """Review Focus 1: arrears carried in are paid at most once, with the first coupon that triggers."""
    env = CASES["ph_memory"][1]()
    base = CASES["ph_memory"][0]()
    carried = CASES["ph_memory"][0]()
    carried.coupon_config = dataclasses.replace(carried.coupon_config, initial_coupon_arrears=2.0)
    a = PhoenixPDESolver(PDEParams()).calculate_event_stats(base, env)
    b = PhoenixPDESolver(PDEParams()).calculate_event_stats(carried, env)
    extra = float(b.coupons.discounted_cash.sum() - a.coupons.discounted_cash.sum())
    first_pay = float(env.get_discount_factor(float(a.observations.times[0])))
    assert 0.0 < extra < 2.0 * first_pay
    assert abs(b.reconciliation_error) <= 1e-10
