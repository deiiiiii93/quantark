"""A prepared session's rich outputs carry the lifecycle state (spec 3C decision 29; §7.2 'session lifecycle')."""
from datetime import timedelta

import pytest

from pde_routes.cases import env_const
from pde_routes.sessions import GRID, PV, session
from quantark.asset.equity.engine.pde import BarrierPDESolver, EuropeanPDESolver, SnowballPDESolver
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.execution.errors import CapabilityError
from quantark.util.enum import OptionType


# copied from the 3C evidence `z4_session_lifecycle.py:28-33`: a realized, unpaid cashflow of 7 due in 91 days
def state(terminal):
    from pde_routes.cases import VAL_DATE as _VAL_DATE
    from quantark.asset.equity.lifecycle import (BarrierLifecycleState, LifecycleCashflowLedger, LifecycleEventType,
                                                 RealizedCashflow, ValuationPoint)
    cash = RealizedCashflow(cashflow_id="claim:expiry", event_type=LifecycleEventType.EXPIRY, amount=7.0,
                            determination_date=_VAL_DATE - timedelta(days=1), payment_date=_VAL_DATE + timedelta(days=91))
    return BarrierLifecycleState(alive=not terminal, expired=terminal, valuation_point=ValuationPoint(date=_VAL_DATE),
                                 ledger=LifecycleCashflowLedger([cash]))

VAN = lambda: EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)   # noqa: E731


@pytest.mark.parametrize("terminal", [True, False])
def test_european_session_pv_and_rows_follow_the_state(terminal):
    env = env_const(100.0)
    st = state(terminal)
    direct = EuropeanPDESolver().price(VAN(), env, lifecycle_state=st)
    assert session(EuropeanPDESolver(), VAN(), env, {PV}, lifecycle_state=st) == direct
    if terminal:
        with pytest.raises(CapabilityError):                      # a terminated claim has no grid (today 9.60)
            session(EuropeanPDESolver(), VAN(), env, {PV, GRID}, lifecycle_state=st)
        return
    out = session(EuropeanPDESolver(), VAN(), env, {PV, GRID}, lifecycle_state=st, levels=[90.0, 110.0])
    assert out.pv == direct                                       # 16.549649; today 9.601810
    for row in out.grid:
        g = EuropeanPDESolver().calculate_greeks(VAN(), env_const(row["spot"]), lifecycle_state=st)
        assert abs(row["price"] - g["price"]) <= 1e-3 and abs(row["delta"] - g["delta"]) <= 2e-4


@pytest.mark.parametrize("engine,product", [(BarrierPDESolver, None), (SnowballPDESolver, "snowball")])
def test_engines_without_lifecycle_support_refuse_rich_outputs(engine, product):
    from pde_bitwise.capture_book import snowball
    from pde_routes.cases import barrier
    p = snowball() if product else barrier("UO", True, 130.0)
    with pytest.raises(CapabilityError):
        session(engine(), p, env_const(100.0), {PV, GRID}, lifecycle_state=state(False))     # today 2.67 / 97.16


def test_stateless_request_is_unchanged():
    env = env_const(100.0)
    out = session(EuropeanPDESolver(), VAN(), env, {PV, GRID}, levels=[100.0])
    assert out.pv == EuropeanPDESolver().price(VAN(), env)


def test_snowball_event_stats_with_a_state_refuses():
    from pde_bitwise.capture_book import snowball
    from quantark.execution.contracts import OutputKind
    with pytest.raises(CapabilityError):
        session(SnowballPDESolver(), snowball(), env_const(100.0), {PV, OutputKind.EVENT_STATS},
                lifecycle_state=state(False))
