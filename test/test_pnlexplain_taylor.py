"""Spec tests 1 (Taylor part), 4 (route/units/cash), 7 (clocks)."""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.param import BumpConfig, EngineParams
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod, Factor, RowKind
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

FRI = datetime(2026, 6, 26)
SAT = datetime(2026, 6, 27)
MON = datetime(2026, 6, 29)
Q = 5.0
K = 105.0                                                       # off the d2 = 0 point: vanna/volga != 0
CLOSED_FORM = ["delta", "gamma", "vega", "theta", "rho"]      # every name auto-routes analytical
FD = dict(rel=2e-2, abs=1e-6)                                   # spec §12 "FD tolerance"


def _env(spot, vol, rate, div, date, calendar=None):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate), div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=date, calendar=calendar)


def _snaps(ds=0.5, dvol=0.002, dr=0.0005, dq=0.0, d1=SAT, calendar=None, engine=None):
    days = (d1 - FRI).days
    e0 = _env(100.0, 0.20, 0.03, 0.01, FRI, calendar)
    e1 = _env(100.0 + ds, 0.20 + dvol, 0.03 + dr, 0.01 + dq, d1, calendar)
    eng = engine or BlackScholesEngine()
    s0 = ValuationSnapshot(EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=1.0),
                           eng, e0, date=FRI, quantity=Q)
    s1 = ValuationSnapshot(EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=1.0 - days / 365),
                           eng, e1, date=d1, quantity=Q)
    return s0, s1


def _rows(res, method=ExplainMethod.TAYLOR):
    return {r.term: r for r in res.rows if r.method is method}


def _analytical(s0, names):
    return GreeksCalculator().calculate_analytical_greeks(s0.product, s0.pricing_env, greeks=names)


def test_default_stencil_routes_numerical_and_matches_closed_form_to_fd_tolerance():
    s0, s1 = _snaps()
    g = _analytical(s0, ["delta", "gamma", "vega", "vanna", "volga", "rho", "dividend_rho"])
    res = explain(s0, s1)
    rows = _rows(res)
    assert res.metadata["route"] == "numerical"        # vanna/volga + theta sub-rows are numerical-only
    assert rows["delta"].pnl == pytest.approx(Q * g["delta"] * 0.5, **FD)
    assert rows["gamma"].pnl == pytest.approx(Q * 0.5 * g["gamma"] * 0.5 ** 2, **FD)
    assert rows["vega"].pnl == pytest.approx(Q * (g["vega"] / 0.01) * 0.002, **FD)
    assert rows["vanna"].pnl == pytest.approx(Q * g["vanna"] * 0.5 * 0.002, **FD)
    assert rows["volga"].pnl == pytest.approx(Q * 0.5 * g["volga"] * 0.002 ** 2, **FD)
    assert rows["rho"].pnl == pytest.approx(Q * (g["rho"] / 0.01) * 0.0005, **FD)
    assert rows["theta"].pnl == pytest.approx(res.metadata["time_pure"], abs=1e-12)
    assert rows["theta"].kind is RowKind.COMPONENT and rows["r_theta"].kind is RowKind.INFORMATIONAL
    assert rows["convexity_theta"].pnl == pytest.approx(
        rows["theta_contract"].pnl - rows["r_theta"].pnl - rows["q_theta"].pnl, abs=1e-12)
    assert rows["ledger_carry"].pnl == 0.0
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)
    components = [r.pnl for r in res.rows_for(ExplainMethod.TAYLOR, kind=RowKind.COMPONENT)
                  if r.method is ExplainMethod.TAYLOR and r.factor is not Factor.UNEXPLAINED]
    assert res.unexplained == pytest.approx((res.pv_alive_t1.total - res.pv_t0.total) - sum(components), abs=1e-12)
    # The extended stencil adds the time-cross and third-order terms in (S, sigma, t); no stencil
    # carries a delta-rate cross term, so the comparison is made without a rate move.
    s0r, s1r = _snaps(dr=0.0)
    std_r = explain(s0r, s1r)
    ext = explain(s0r, s1r, config=PnLExplainConfig(stencil="extended"))
    assert abs(ext.unexplained) < abs(std_r.unexplained)
    assert "gamma_theta" in _rows(ext) and "speed" in _rows(ext)
    assert rows["vanna"].factor is Factor.VOL and rows["delta"].factor is Factor.SPOT
    assert rows["vanna"].moves == {"spot_return": pytest.approx(0.005), "vol_pts": pytest.approx(0.2)}


def test_closed_form_stencil_routes_analytical_and_is_exact():
    s0, s1 = _snaps()
    res = explain(s0, s1, config=PnLExplainConfig(stencil=CLOSED_FORM))
    rows = _rows(res)
    assert res.metadata["route"] == "analytical" and res.metadata["vega_scale"] == pytest.approx(0.01)
    g = _analytical(s0, ["delta", "gamma", "vega", "theta", "rho"])
    assert rows["delta"].pnl == pytest.approx(Q * g["delta"] * 0.5, rel=1e-12)
    assert rows["gamma"].pnl == pytest.approx(Q * 0.5 * g["gamma"] * 0.5 ** 2, rel=1e-12)
    assert rows["vega"].pnl == pytest.approx(Q * (g["vega"] / 0.01) * 0.002, rel=1e-12)
    assert rows["rho"].pnl == pytest.approx(Q * (g["rho"] / 0.01) * 0.0005, rel=1e-12)
    assert rows["theta"].pnl == pytest.approx(res.metadata["time_pure"], abs=1e-12)
    assert "r_theta" not in rows                      # sub-rows only when requested
    assert rows["theta_contract"].pnl == pytest.approx(Q * g["theta"] * 1, rel=1e-12)   # per day x 1-day gap
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)


def test_numerical_route_with_custom_vol_bump_agrees_with_dv_dsigma():
    s0, s1 = _snaps()
    params = EngineParams(bump_config=BumpConfig(vol_bump=0.02))
    res = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical", params=params))
    assert res.metadata["route"] == "numerical" and res.metadata["vega_scale"] == pytest.approx(0.02)
    g = _analytical(s0, ["vega"])
    vega_row = _rows(res)["vega"]
    assert vega_row.pnl == pytest.approx(Q * (g["vega"] / 0.01) * 0.002, **FD)
    assert vega_row.greek == pytest.approx(Q * g["vega"] / 0.01, **FD)        # position-level dV/dsigma
    assert vega_row.cash_greek == pytest.approx(vega_row.greek * 0.01, rel=1e-12)
    assert vega_row.moves == {"vol_pts": pytest.approx(0.2)}


def test_cash_greek_columns_follow_the_desk_table():
    s0, s1 = _snaps()
    res = explain(s0, s1, config=PnLExplainConfig(stencil=CLOSED_FORM))
    rows = _rows(res)
    g = _analytical(s0, ["delta", "gamma", "vega", "rho"])
    S = 100.0
    assert rows["delta"].cash_greek == pytest.approx(Q * g["delta"] * S, rel=1e-12)
    assert rows["gamma"].cash_greek == pytest.approx(Q * g["gamma"] * S * S / 100.0, rel=1e-12)
    assert rows["vega"].cash_greek == pytest.approx(Q * g["vega"], rel=1e-12)            # per 1 vol pt
    assert rows["rho"].cash_greek == pytest.approx(Q * g["rho"], rel=1e-12)              # per 1%
    assert rows["theta"].cash_greek == pytest.approx(res.metadata["time_pure"] / 1.0)   # 1-day gap
    assert rows["delta"].moves == {"spot_return": pytest.approx(0.005)}
    vanna = _rows(explain(s0, s1))["vanna"]
    assert vanna.cash_greek == pytest.approx(Q * _analytical(s0, ["vanna"])["vanna"] * S * 0.01, **FD)


def test_clocks_exact_gap_versus_per_step():
    cal = create_calendar(CalendarType.CHINA_SSE)
    s0, s1 = _snaps(d1=MON, calendar=cal)
    exact = explain(s0, s1)
    r = _rows(exact)
    assert r["theta"].pnl == pytest.approx(exact.metadata["time_pure"], abs=1e-12)
    assert r["theta"].moves == {"days": 3.0, "trading_days": 1.0}
    assert r["theta"].greek == pytest.approx(exact.metadata["time_pure"] / 3.0)
    assert exact.metadata["n_steps"] == 1 and exact.metadata["gap_scale"] == 1.0

    calc = GreeksCalculator()
    theta_1d = calc.calculate_numerical_greeks(s0.product, s0.pricing_env, s0.engine, greeks=["theta_1d"])["theta_1d"]
    theta_1td = calc.calculate_numerical_greeks(s0.product, s0.pricing_env, s0.engine, greeks=["theta_1td"])["theta_1td"]
    per_1d = explain(s0, s1, config=PnLExplainConfig(time_term="per_step", clock="1d"))
    assert _rows(per_1d)["theta"].pnl == pytest.approx(Q * theta_1d * 3, rel=1e-9)
    assert per_1d.metadata["n_steps"] == 3
    per_1td = explain(s0, s1, config=PnLExplainConfig(time_term="per_step", clock="1td"))
    assert _rows(per_1td)["theta"].pnl == pytest.approx(Q * theta_1td * 1, rel=1e-9)
    assert per_1td.metadata["n_steps"] == 1
    # the calculator rolls a float-maturity product by one day per step (product.time_shift),
    # so three one-day steps are not the exact three-day revaluation: per_step misexplains the weekend
    assert _rows(per_1d)["theta"].pnl != pytest.approx(r["theta"].pnl, abs=1e-9)

    s0n, s1n = _snaps(d1=MON)                     # no calendar
    with pytest.raises(ValidationError):
        explain(s0n, s1n, config=PnLExplainConfig(time_term="per_step", clock="1td"))


def test_terminal_position_has_only_the_time_row():
    from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
    s0, s1 = _snaps(d1=MON)
    st = AutocallableLifecycleState()
    st.mark_ko(FRI - timedelta(days=10), cashflow=50.0, settlement_date=MON + timedelta(days=5))
    t0 = ValuationSnapshot(s0.product, s0.engine, s0.pricing_env, date=FRI, quantity=Q, lifecycle_state=st)
    t1 = ValuationSnapshot(s1.product, s1.engine, s1.pricing_env, date=MON, quantity=Q, lifecycle_state=st)
    res = explain(t0, t1)
    rows = _rows(res)
    assert res.pv_t0.contingent_mtm == 0.0
    assert rows["theta"].pnl == pytest.approx(res.metadata["time_pure"], abs=1e-12)
    assert all(r.pnl == 0.0 for k, r in rows.items() if k not in ("theta", "unexplained", "ledger_carry"))
    assert rows["delta"].greek is None
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)
