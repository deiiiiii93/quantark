"""Desk-convention benchmark: docs/pnl-explainer/otc_option_onl_explainer.xlsx.

That workbook is a trading desk's one-day PnL attribution for a short snowball on 000905.SH.
It stores desk-quoted greeks and attributes with

    delta_pnl = delta_cash * dS/S0                   delta_cash = dV/dS * S
    gamma_pnl = 0.5 * gamma_cash * (dS/S0)^2 * 100   gamma_cash = d2V/dS2 * S^2/100
    vega_pnl  = vega_1pct * dvol * 100               vega_1pct  = dV/dvol * 0.01
    theta_pnl = theta_1td                            one TRADING day
    rho_pnl   = rho_1pct * drate * 100               rho_1pct   = dV/dr * 0.01
    rhoq_pnl  = rhoq_1pct * ddiv * 100               rhoq_1pct  = dV/dq * 0.01

These tests pin the module to those conventions, and to the general rule that covers every
term of the extended stencil: for coefficient k and exponents (a,b,c,d,e) over
(spot, vol, rate, div, time),

    pnl = k * cash_greek * sr^a * 100^max(a-1,0) * vp^b * rp^c * qp^d * n^e

with sr = dS/S0, vp = dvol*100, rp = drate*100, qp = ddiv*100, n = steps.
The workbook's own market data is used so the numbers stay recognisable next to it.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import ExplainMethod, PnLExplainConfig, RowKind, ValuationSnapshot, explain
from quantark.pnlexplain.equity.taylor import TERM_SPEC
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import OptionType

T0, T1 = datetime(2026, 9, 2), datetime(2026, 9, 3)      # Wed -> Thu: 1 calendar, 1 trading day
FRI, MON = datetime(2026, 9, 4), datetime(2026, 9, 7)    # 3 calendar days, 1 trading day
S0, S1 = 7729.2332, 7755.3612
V0, V1 = 0.25261, 0.25271
Q0, Q1 = 0.11837689, 0.11810775
R0, R1 = 0.014, 0.0155                                   # the workbook's rate is flat; move it here
QTY = -5750.9803                                         # SELLER, the workbook's side
CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2029))
REL = 1e-9

SR = (S1 - S0) / S0
VP = (V1 - V0) * 100.0
RP = (R1 - R0) * 100.0
QP = (Q1 - Q0) * 100.0


def _env(spot, vol, rate, div, date):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate), div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=date, calendar=CAL, bus_days_in_year=244)


def _rows(stencil="standard", time_term="exact_gap", clock=None, t0=T0, t1=T1,
          spot1=S1, vol1=V1, rate1=R1, div1=Q1, quantity=QTY):
    p = EuropeanVanillaOption(strike=7800.0, option_type=OptionType.CALL,
                              exercise_date=datetime(2028, 4, 24))
    eng = BlackScholesEngine()
    res = explain(
        ValuationSnapshot(p, eng, _env(S0, V0, R0, Q0, t0), date=t0, quantity=quantity),
        ValuationSnapshot(p, eng, _env(spot1, vol1, rate1, div1, t1), date=t1, quantity=quantity),
        config=PnLExplainConfig(methods=(ExplainMethod.TAYLOR,), stencil=stencil,
                                greeks_method="numerical", time_term=time_term, clock=clock))
    return {r.term: r for r in res.rows if r.method is ExplainMethod.TAYLOR}, res


def test_cash_greeks_follow_the_desk_quote_conventions():
    """The workbook's greek COLUMNS, re-derived from the module's raw derivatives."""
    rows, _ = _rows()
    assert rows["delta"].cash_greek == pytest.approx(rows["delta"].greek * S0, rel=REL)
    assert rows["gamma"].cash_greek == pytest.approx(rows["gamma"].greek * S0 * S0 / 100, rel=REL)
    assert rows["vega"].cash_greek == pytest.approx(rows["vega"].greek * 0.01, rel=REL)
    assert rows["rho"].cash_greek == pytest.approx(rows["rho"].greek * 0.01, rel=REL)
    assert rows["dividend_rho"].cash_greek == pytest.approx(
        rows["dividend_rho"].greek * 0.01, rel=REL)


def test_the_workbook_six_term_block_reproduces_the_module_rows():
    """The desk's own formulas, fed with the module's greeks, must give the module's rows."""
    rows, _ = _rows()
    assert rows["delta"].pnl == pytest.approx(rows["delta"].cash_greek * SR, rel=REL)
    assert rows["gamma"].pnl == pytest.approx(
        0.5 * rows["gamma"].cash_greek * SR ** 2 * 100, rel=REL)
    assert rows["vega"].pnl == pytest.approx(rows["vega"].cash_greek * VP, rel=REL)
    assert rows["rho"].pnl == pytest.approx(rows["rho"].cash_greek * RP, rel=REL)
    assert rows["dividend_rho"].pnl == pytest.approx(rows["dividend_rho"].cash_greek * QP, rel=REL)
    # and in raw-derivative form, which is what the algebra actually says
    assert rows["delta"].pnl == pytest.approx(rows["delta"].greek * (S1 - S0), rel=REL)
    assert rows["gamma"].pnl == pytest.approx(0.5 * rows["gamma"].greek * (S1 - S0) ** 2, rel=REL)
    assert rows["vega"].pnl == pytest.approx(rows["vega"].greek * (V1 - V0), rel=REL)


@pytest.mark.parametrize("term", [
    "delta", "gamma", "speed", "vega", "volga", "vanna", "zomma",
    "theta", "charm", "color", "vega_theta", "rho", "dividend_rho",
    "dividend_volga", "delta_q",
])
def test_every_extended_term_matches_the_desk_spreadsheet_formula(term):
    """One closed form covers the whole stencil; the workbook's six rows are its special cases."""
    rows, _ = _rows(stencil="extended", time_term="per_step")
    n = rows["theta"].moves["days"]
    k, (a, b, c, d, e) = TERM_SPEC[term]
    expected = (k * rows[term].cash_greek * SR ** a * 100.0 ** max(a - 1, 0)
                * VP ** b * RP ** c * QP ** d * n ** e)
    assert rows[term].pnl == pytest.approx(expected, rel=REL, abs=1e-12)


def test_components_sum_to_the_total_pnl():
    """The residual row is what makes the attribution close; nothing else may leak."""
    for stencil in ("first_order", "standard", "extended"):
        rows, res = _rows(stencil=stencil)
        comp = sum(r.pnl for r in res.rows
                   if r.method is ExplainMethod.TAYLOR and r.kind is RowKind.COMPONENT)
        assert comp == pytest.approx(res.total_pnl, rel=REL, abs=1e-9), stencil


def test_trading_day_theta_is_the_desk_convention_over_a_weekend():
    """theta_1td, not theta_1d: a Fri->Mon step decays ONE trading day, not three.

    The module's default (exact_gap) closes the time gap exactly; per_step on the trading
    clock agrees with it, while the calendar clock applies a daily theta three times and
    overstates the decay.
    """
    quiet = dict(spot1=S0, vol1=V0, rate1=R0, div1=Q0)     # freeze the market: pure time decay
    gap, res_gap = _rows(t0=FRI, t1=MON, **quiet)
    td, res_td = _rows(t0=FRI, t1=MON, time_term="per_step", clock="1td", **quiet)
    cal, res_cal = _rows(t0=FRI, t1=MON, time_term="per_step", clock="1d", **quiet)

    assert gap["theta"].pnl == pytest.approx(res_gap.total_pnl, rel=REL)
    assert td["theta"].pnl == pytest.approx(res_td.total_pnl, rel=REL)
    assert cal["theta"].pnl != pytest.approx(res_cal.total_pnl, rel=1e-4)
    assert abs(cal["theta"].pnl) > abs(td["theta"].pnl)     # three calendar days overstate decay

    # over a single trading day the two clocks agree, which is why the workbook's step is safe
    one_gap, _ = _rows(**quiet)
    one_td, _ = _rows(time_term="per_step", clock="1td", **quiet)
    one_cal, _ = _rows(time_term="per_step", clock="1d", **quiet)
    assert one_td["theta"].pnl == pytest.approx(one_gap["theta"].pnl, rel=REL)
    assert one_cal["theta"].pnl == pytest.approx(one_gap["theta"].pnl, rel=REL)


def test_rows_are_linear_in_the_position_sign():
    """A desk books the seller side by sign; every row must flip, none may be absolute."""
    long_rows, long_res = _rows(stencil="extended", quantity=-QTY)
    short_rows, short_res = _rows(stencil="extended", quantity=QTY)
    assert short_res.total_pnl == pytest.approx(-long_res.total_pnl, rel=REL)
    for term, row in long_rows.items():
        assert short_rows[term].pnl == pytest.approx(-row.pnl, rel=REL, abs=1e-12), term
