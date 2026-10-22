"""Spec tests 1 (waterfall part), 2 (orders, sticky), 3 (time-step equivalence, contract identity)."""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.vol.vol_surface import BlackImpliedVolSurface
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod, Factor, MARKET_FACTORS, RowKind
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.greek_conventions import GreekConvention
from quantark.util.exceptions import ValidationError

FRI = datetime(2026, 6, 26)
MON = datetime(2026, 6, 29)
WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))
ENG = BlackScholesEngine()      # one engine object on both sides: MODEL is identity-detected


class SkewSurface(BlackImpliedVolSurface):
    """Linear skew in absolute strike: exposes sticky-strike vs sticky-moneyness."""
    is_smile = True

    def __init__(self, atm, slope):
        self.atm, self.slope = atm, slope

    def get_vol(self, strike, time_to_maturity, spot=None):
        return self.atm + self.slope * (float(strike) - 100.0)


def _env(spot, vol, rate, div, date):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div), valuation_date=date)


def _call(m=None, exercise_date=None):
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=m,
                                 exercise_date=exercise_date)


def _pair(vol0=FlatVolSurface(0.20), vol1=FlatVolSurface(0.22), spot1=103.0, rate1=0.032, div1=0.012):
    e0 = _env(100.0, vol0, 0.03, 0.01, FRI)
    e1 = _env(spot1, vol1, rate1, div1, MON)
    s0 = ValuationSnapshot(_call(1.0), ENG, e0, date=FRI, quantity=2.0)
    s1 = ValuationSnapshot(_call(1.0 - 3 / 365), ENG, e1, date=MON, quantity=2.0)
    return s0, s1


def _tol(total):
    return 1e-10 * max(1.0, abs(total))


def test_waterfall_is_exact_and_time_row_is_time_pure():
    s0, s1 = _pair()
    res = explain(s0, s1, config=WF)
    eng = BlackScholesEngine()
    assert res.pv_t0.total == pytest.approx(2.0 * eng.price(s0.product, s0.pricing_env))
    assert res.pv_t1.total == pytest.approx(2.0 * eng.price(s1.product, s1.pricing_env))
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    rows = res.rows_for(ExplainMethod.WATERFALL, kind=RowKind.COMPONENT)
    assert [r.factor for r in rows[:7]] == list(MARKET_FACTORS)
    assert [r.step for r in rows[:7]] == list(range(1, 8))
    time_row = rows[0]
    expected_time = 2.0 * (eng.price(s1.product, s0.pricing_env) - eng.price(s0.product, s0.pricing_env))
    assert time_row.pnl == pytest.approx(expected_time, abs=1e-12)
    assert res.metadata["time_pure"] == pytest.approx(expected_time, abs=1e-12)
    assert time_row.moves == {"days": 3.0}
    basis_row = [r for r in rows if r.factor is Factor.BASIS][0]
    model_row = [r for r in rows if r.factor is Factor.MODEL][0]
    assert basis_row.pnl == 0.0 and model_row.pnl == 0.0
    assert basis_row.metadata["changed"] is False
    event = [r for r in res.rows if r.factor is Factor.LIFECYCLE_EVENT][0]
    assert event.method is ExplainMethod.SHARED and event.pnl == 0.0


def test_row_metadata_separates_market_change_from_instrument_applicability():
    """`changed` reports the market, `applicable` the instrument: a vol move on a spot position is
    changed=True, applicable=False, pnl 0 (review finding)."""
    from quantark.asset.equity.engine.analytical import DeltaOneEngine
    from quantark.asset.equity.product.deltaone import SpotInstrument
    from quantark.util.enum.deltaone_enums import DeltaOneType
    spot = SpotInstrument(underlying="X", deltaone_type=DeltaOneType.STOCK)
    d1 = DeltaOneEngine()
    e0 = PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(0.20),
                            rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                            valuation_date=FRI)
    e1 = PricingEnvironment(spot_quote=SpotQuote(spot=101.0), vol_surface=FlatVolSurface(0.25),
                            rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                            valuation_date=MON)
    res = explain(ValuationSnapshot(spot, d1, e0, date=FRI), ValuationSnapshot(spot, d1, e1, date=MON), config=WF)
    rows = {r.factor: r for r in res.rows_for(ExplainMethod.WATERFALL, kind=RowKind.COMPONENT)}
    assert rows[Factor.SPOT].metadata == {"changed": True, "applicable": True}
    assert rows[Factor.SPOT].pnl == pytest.approx(1.0)
    # the vol surface has no tenor to be read at on a spot position: it is not a move of THIS
    # instrument's coordinate, so it is neither applicable nor (at this coordinate) changed
    assert rows[Factor.VOL].metadata == {"changed": False, "applicable": False} and rows[Factor.VOL].pnl == 0.0
    # an inapplicable factor that DID move at the coordinate is reported as changed but unpriced
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1e-8)
    res2 = explain(ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                                           maturity=3 / 365), ENG, e0, date=FRI),
                   ValuationSnapshot(call, ENG, e1, date=MON), config=WF)
    vol = [r for r in res2.rows_for(ExplainMethod.WATERFALL, kind=RowKind.COMPONENT) if r.factor is Factor.VOL][0]
    assert vol.metadata["applicable"] is False and vol.pnl == 0.0
    total = [r for r in res.rows if r.kind is RowKind.SUMMARY]
    assert len(total) == 1 and total[0].pnl == pytest.approx(res.total_pnl)
    assert res.unexplained is None


def test_reordered_waterfall_sums_exactly_but_time_row_differs():
    s0, s1 = _pair()
    order = (Factor.SPOT, Factor.VOL, Factor.RATE, Factor.DIVIDEND, Factor.BASIS, Factor.MODEL, Factor.TIME)
    res = explain(s0, s1, config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), waterfall_order=order))
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    time_row = [r for r in res.rows if r.factor is Factor.TIME and r.method is ExplainMethod.WATERFALL][0]
    assert time_row.step == 7
    assert time_row.pnl != pytest.approx(res.metadata["time_pure"], abs=1e-9)


@pytest.mark.parametrize("order", [
    MARKET_FACTORS,
    (Factor.TIME, Factor.VOL, Factor.SPOT, Factor.RATE, Factor.DIVIDEND, Factor.BASIS, Factor.MODEL),
])
def test_sticky_moneyness_is_order_independent(order):
    s0, s1 = _pair(vol0=SkewSurface(0.20, 0.002), vol1=SkewSurface(0.21, 0.002))
    cfg = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), waterfall_order=order,
                           spot_convention=GreekConvention.STICKY_MONEYNESS)
    res = explain(s0, s1, config=cfg)
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    res_strike = explain(s0, s1, config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), waterfall_order=order))
    assert res.total_pnl == pytest.approx(res_strike.total_pnl)
    spot_m = res.by_factor(ExplainMethod.WATERFALL)["spot"]
    spot_k = res_strike.by_factor(ExplainMethod.WATERFALL)["spot"]
    if order.index(Factor.SPOT) < order.index(Factor.VOL):
        # the convention decides how the t0 surface travels with spot: SPOT and VOL split differently
        assert spot_m != pytest.approx(spot_k, abs=1e-9)
    else:
        # once the t1 surface is applied, spot moves under it: the convention is moot by design
        assert spot_m == pytest.approx(spot_k, abs=1e-12)


def test_date_based_and_float_rolls_give_the_same_time_row():
    s0, s1 = _pair()
    dated0 = ValuationSnapshot(_call(exercise_date=FRI + timedelta(days=365)), ENG,
                               s0.pricing_env, date=FRI, quantity=2.0)
    dated1 = ValuationSnapshot(dated0.product, ENG, s1.pricing_env, date=MON, quantity=2.0)
    a = explain(s0, s1, config=WF).by_factor(ExplainMethod.WATERFALL)
    b = explain(dated0, dated1, config=WF).by_factor(ExplainMethod.WATERFALL)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_contract_replacement_and_unrolled_float_are_rejected():
    s0, s1 = _pair()
    bad = ValuationSnapshot(EuropeanVanillaOption(strike=105.0, option_type=OptionType.CALL, maturity=1.0 - 3 / 365),
                            ENG, s1.pricing_env, date=MON, quantity=2.0)
    with pytest.raises(ValidationError, match="contract replacement"):
        explain(s0, bad, config=WF)
    unrolled = ValuationSnapshot(_call(1.0), ENG, s1.pricing_env, date=MON, quantity=2.0)
    with pytest.raises(ValidationError):
        explain(s0, unrolled, config=WF)


def test_unchanged_factors_are_not_priced(monkeypatch):
    s0, s1 = _pair(vol1=FlatVolSurface(0.20), rate1=0.03, div1=0.01)   # only time + spot move
    calls = []
    real = BlackScholesEngine.price

    def counting(self, product, env, **kw):
        calls.append(env.valuation_date)
        return real(self, product, env, **kw)

    monkeypatch.setattr(BlackScholesEngine, "price", counting)
    res = explain(s0, s1, config=WF)
    # base, time, spot, and the t1 endpoint: four pricings, no more
    assert len(calls) == 4
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))


import itertools  # noqa: E402


def test_shapley_matches_brute_force_average_and_sums_to_total():
    from quantark.pnlexplain.equity.coordinate import resolve_coordinate
    from quantark.pnlexplain.equity.factor_diff import build_factor_moves
    from quantark.pnlexplain.equity.lifecycle import resolve_transition
    from quantark.pnlexplain.equity.scenario import ScenarioCache
    from quantark.pnlexplain.equity.waterfall import sequential_rows

    s0, s1 = _pair(rate1=0.03, div1=0.01)          # TIME, SPOT, VOL change (3 effective factors)
    cfg = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), interaction="shapley")
    res = explain(s0, s1, config=cfg)
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    sh = {r.factor: r.pnl for r in res.rows_for(ExplainMethod.WATERFALL, kind=RowKind.COMPONENT)
          if r.method is ExplainMethod.WATERFALL}
    assert all(r.step is None for r in res.rows if r.method is ExplainMethod.WATERFALL)

    tr = resolve_transition(s0, s1, None, calendar_days=3)
    coord = resolve_coordinate(s0.product, 100.0, tr.product_alive_t1, s1.pricing_env)
    mv = build_factor_moves(s0, s1, coord, engine_alive_t1=tr.engine_alive_t1, lifecycle_changed=False)
    cache = ScenarioCache(s0, s1, tr, mv, cfg)
    eff = [f for f in MARKET_FACTORS if f in cache.effective]
    assert len(eff) == 3
    acc = {f: 0.0 for f in eff}
    perms = list(itertools.permutations(eff))
    for perm in perms:
        order = tuple(perm) + tuple(f for f in MARKET_FACTORS if f not in eff)
        for row in sequential_rows(cache, order):
            if row.factor in acc:
                acc[row.factor] += row.pnl / len(perms)
    for f in eff:
        assert sh[f] == pytest.approx(acc[f], abs=1e-9)


def test_unrolled_contract_must_be_declared_and_then_has_no_contract_theta():
    """The equity BacktestEngine reprices an untracked float-maturity product with a constant
    tenor. Explaining that requires an explicit contract_roll_days=0; nothing is guessed."""
    from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
    s0, s1 = _pair()
    same = ValuationSnapshot(s0.product, ENG, s1.pricing_env, date=MON, quantity=2.0)   # T unchanged
    with pytest.raises(ValidationError):                                             # undeclared: rejected
        explain(s0, same)
    declared = LifecycleTransition(product_alive_t1=s0.product, engine_alive_t1=ENG,
                                   state_before=None, state_after=None, contract_roll_days=0)
    res = explain(s0, same, transition=declared)
    assert res.metadata["contract_roll_days"] == 0 and res.metadata["contract_rolled"] is False
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    # a flat environment and a constant tenor: the valuation date alone moves nothing
    assert res.metadata["time_pure"] == pytest.approx(0.0, abs=1e-12)
    taylor = {r.term: r for r in res.rows if r.method is ExplainMethod.TAYLOR}
    assert taylor["theta"].pnl == pytest.approx(0.0, abs=1e-12)
    assert taylor["theta_contract"].pnl == 0.0 and taylor["r_theta"].pnl == 0.0
    with pytest.raises(ValidationError):                                             # more than the step
        explain(s0, same, transition=LifecycleTransition(s0.product, ENG, None, None, contract_roll_days=4))
