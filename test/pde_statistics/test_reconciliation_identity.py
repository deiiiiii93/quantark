"""The reconciliation identity (spec 3B §5.2, §7.2): |pv - Σ ledger| <= 1e-10 on every native 1D route."""
import itertools

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.cashleg.event_distribution import EventType
from quantark.execution import OutputKind, PricingRequest, PricingSession
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from pde_3a.products import KO_RESET, PHOENIX, SNOWBALL, env
from pde_statistics.cases import CASES, trading_clock_env

BOUND = 1e-10


def _flagged(p):
    setattr(p, "_otc_lifecycle_knocked_in", True)
    return p


SNOWBALL_CASES = {
    "cont": (lambda: CASES["sb_cont"][0](), {}),
    "cont_86": (lambda: CASES["sb_cont"][0](), {"spot": 86.0}),
    "cont_102_5": (lambda: CASES["sb_cont"][0](), {"spot": 102.5}),
    "cont_3m": (lambda: CASES["sb_cont_3m"][0](), {}),
    "cont_knocked_in_by_spot": (lambda: CASES["sb_cont"][0](), {"spot": 82.45}),
    "cont_lifecycle_flag": (lambda: _flagged(CASES["sb_cont"][0]()), {}),
    "discrete": (lambda: CASES["sb_discrete"][0](), {}),
    "european": (lambda: CASES["sb_european"][0](), {}),
    "daily": (lambda: CASES["sb_daily"][0](), {}),
    "dkak_cont": (lambda: make_snowball(dict(SNOWBALL, disable_ko_after_ki=True)), {}),
    "dkak_disc": (lambda: make_snowball(dict(SNOWBALL, disable_ko_after_ki=True, ki_monitoring="discrete")), {}),
    "reverse": (lambda: make_snowball(dict(SNOWBALL, is_reverse=True, ko_barrier=97.0, ki_barrier=115.0)), {}),
    "airbag": (lambda: make_snowball(dict(SNOWBALL, airbag_barrier=90.0, airbag_participation_rate=0.5)), {}),
}
ENVS = {"flat": lambda **kw: env(**kw), "trading_clock": lambda **kw: trading_clock_env(**kw)[0]}
STREAMS = {"all": None, "ko_only": frozenset({EventType.KO})}


def _entry(entry, solver, product, e, streams):
    if entry == "direct":
        return solver.calculate_event_stats(product, e, streams=streams)
    if entry == "pwe":
        return solver._session_outputs(product, e, want_events=True, streams=streams).event_stats
    with PricingSession() as session:
        return session.execute(solver, PricingRequest(
            product=product, pricing_env=e, outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}))).value.event_stats


def _ids(*keys):
    return "-".join(map(str, keys))


def _check(stats):
    assert abs(stats.reconciliation_error) <= BOUND, stats.reconciliation_error
    ledger = float(np.sum(stats.expected_discounted_cashflows))
    assert abs(float(stats.pv) - ledger - stats.reconciliation_error) <= 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("case,env_name,accuracy,projection,streams,entry", [
    p for p in itertools.product(sorted(SNOWBALL_CASES), sorted(ENVS), ("standard", "high"),
                                 (None, "nodal"), sorted(STREAMS), ("direct", "pwe", "session"))
    if p[5] == "direct" or (p[2] == "standard" and p[4] == "all")], ids=_ids)
def test_snowball_identity(value_mode, case, env_name, accuracy, projection, streams, entry):
    value_mode("snowball")
    make, over = SNOWBALL_CASES[case]
    solver = SnowballPDESolver(PDEParams(accuracy=accuracy, event_projection=projection))
    _check(_entry(entry, solver, make(), ENVS[env_name](**over), STREAMS[streams]))


@pytest.mark.parametrize("case", ["cont", "discrete", "reverse"])
def test_snowball_identity_fast(value_mode, case):
    """The matrix's quick slice: one standard flat-env direct run per KI class (the slow matrix has the rest)."""
    value_mode("snowball")
    make, over = SNOWBALL_CASES[case]
    _check(SnowballPDESolver(PDEParams()).calculate_event_stats(make(), env(**over)))


def test_local_vol_snowball_identity(value_mode):
    """Review Focus 5: the local-vol route marches the same value-mode statistics."""
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
    from pde_bitwise.capture_book import env_grid_vol, snowball

    value_mode("snowball")
    stats = LocalVolSnowballPDESolver(params=PDEParams()).calculate_event_stats(snowball("continuous"), env_grid_vol())
    assert stats.provenance.statistics_model == "local_vol_1d"
    _check(stats)


# --- Phoenix, INSTANT coupons (Task 13; spec 3B §5.3) ------------------------------------------------

def _arrears(p, arrears=2.0):
    import dataclasses
    p.coupon_config = dataclasses.replace(p.coupon_config, initial_coupon_arrears=arrears)
    return p


PHOENIX_CASES = {
    "plain": (lambda: CASES["ph_cont"][0](), {}),
    "memory": (lambda: CASES["ph_memory"][0](), {}),
    "near_coupon": (lambda: CASES["ph_memory"][0](), {"spot": 86.0}),
    "c105": (lambda: make_phoenix(dict(PHOENIX, memory_coupon=True, coupon_barrier=105.0)), {}),
    "coincident": (lambda: make_phoenix(dict(PHOENIX, coupon_barrier=103.0)), {}),
    "arrears": (lambda: _arrears(CASES["ph_memory"][0]()), {}),
    "expiry_plain": (lambda: make_phoenix(dict(PHOENIX, coupon_pay_type="expiry")), {}),
    "expiry_memory": (lambda: make_phoenix(dict(PHOENIX, memory_coupon=True, coupon_pay_type="expiry")), {}),
}


@pytest.mark.slow
@pytest.mark.parametrize("case,env_name,accuracy,projection,streams,entry", [
    p for p in itertools.product(sorted(PHOENIX_CASES), sorted(ENVS), ("standard", "high"),
                                 (None, "nodal"), sorted(STREAMS), ("direct", "pwe", "session"))
    if p[5] == "direct" or (p[2] == "standard" and p[4] == "all")], ids=_ids)
def test_phoenix_identity(value_mode, case, env_name, accuracy, projection, streams, entry):
    value_mode("phoenix")
    make, over = PHOENIX_CASES[case]
    solver = PhoenixPDESolver(PDEParams(accuracy=accuracy, event_projection=projection))
    _check(_entry(entry, solver, make(), ENVS[env_name](**over), STREAMS[streams]))


@pytest.mark.parametrize("case", ["plain", "memory", "coincident"])
def test_phoenix_identity_fast(value_mode, case):
    value_mode("phoenix")
    make, over = PHOENIX_CASES[case]
    _check(PhoenixPDESolver(PDEParams()).calculate_event_stats(make(), env(**over)))


@pytest.mark.slow
@pytest.mark.parametrize("projection", [None, "nodal"])
def test_local_vol_memory_phoenix_identity(value_mode, projection):
    """Review Focus 5: a daily-KI memory Phoenix on a crash local-vol surface."""
    from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import LocalVolPhoenixPDESolver
    from pde_3a.products import localvol_phoenix_case

    value_mode("phoenix")
    product, e, lv = localvol_phoenix_case("crash", 1.0, "daily_ki_memory")
    solver = LocalVolPhoenixPDESolver(params=PDEParams(event_projection=projection), local_vol_surface=lv)
    _check(solver.calculate_event_stats(product, e))


# --- KO-reset, both schedules and the staged pre-maturity (Task 15; spec 3B §5.4) ---------------------

KO_RESET_CASES = {
    "cont": (lambda: make_ko_reset(KO_RESET), {}),
    "cont_knocked_in_by_spot": (lambda: make_ko_reset(KO_RESET), {"spot": 78.0}),
    "discrete": (lambda: make_ko_reset(dict(KO_RESET, ki_continuous=False)), {}),
}


@pytest.mark.slow
@pytest.mark.parametrize("case,env_name,accuracy,projection,streams,entry", [
    p for p in itertools.product(sorted(KO_RESET_CASES), sorted(ENVS), ("standard", "high"),
                                 (None, "nodal"), sorted(STREAMS), ("direct", "pwe", "session"))
    if p[5] == "direct" or (p[2] == "standard" and p[4] == "all")], ids=_ids)
def test_ko_reset_identity(value_mode, case, env_name, accuracy, projection, streams, entry):
    value_mode("ko_reset")
    make, over = KO_RESET_CASES[case]
    solver = KOResetSnowballPDESolver(PDEParams(accuracy=accuracy, event_projection=projection))
    _check(_entry(entry, solver, make(), ENVS[env_name](**over), STREAMS[streams]))


@pytest.mark.parametrize("case", sorted(KO_RESET_CASES))
def test_ko_reset_identity_fast(value_mode, case):
    value_mode("ko_reset")
    make, over = KO_RESET_CASES[case]
    _check(KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(make(), env(**over)))


@pytest.mark.parametrize("projection", [None, "nodal"])
@pytest.mark.parametrize("memory", [False, True], ids=["plain", "memory"])
def test_expiry_phoenix_identity(value_mode, memory, projection):
    """EXPIRY (Task 14): coupons paid at termination through the termination-unit columns (plan F4)."""
    value_mode("phoenix")
    product = make_phoenix(dict(PHOENIX, memory_coupon=memory, coupon_pay_type="expiry"))
    _check(PhoenixPDESolver(PDEParams(event_projection=projection)).calculate_event_stats(product, env()))


@pytest.mark.parametrize("pay_type", ["instant", "expiry"])
@pytest.mark.parametrize("projection", [None, "nodal"])
@pytest.mark.parametrize("spot", [78.0, 95.0], ids=["miss", "pay"])
def test_dated_memory_phoenix_on_its_first_observation_date(value_mode, spot, projection, pay_type):
    """F16: a memory coupon decided at valuation -- the miss reads the one-missed state, the pay adds today's
    coupon (EXPIRY: through the termination value at the spot)."""
    from pde_bitwise.capture_book import _dated, _dated_env, _first_obs, _phoenix_expiry

    value_mode("phoenix")
    # a fixed coupon fraction: 0.8.0 accrues the floating first period to zero at its own date, and a
    # zero missed coupon would leave the one-missed state equal to state 0
    p, _ = _dated("phoenix", fixed_fraction=1.0 / 12.0)
    if pay_type == "expiry":
        p = _phoenix_expiry(p)
    stats = PhoenixPDESolver(PDEParams(event_projection=projection)).calculate_event_stats(
        p, _dated_env(p, _first_obs(p), spot))
    _check(stats)          # a miss read from state 0 would miss the arrears the value carries in state 1
    if projection is None:  # the cell-average route decides today's coupon at the spot (nodal interpolates)
        trigger = stats.coupon_probability[0]
        assert abs(trigger - 1.0) <= 1e-12 if spot == 95.0 else trigger == 0.0
        paid = stats.expected_discounted_coupon_cashflow[0]
        assert paid > 0.0 if spot == 95.0 else paid == 0.0
