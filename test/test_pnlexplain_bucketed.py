"""Spec test 9: tenor-vega bucket rows replace the scalar vega row and sum to it within FD
tolerance; key-rate rows are an informational (carry-invariant) view beneath the scalar rho."""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote, TermStructureVolSurface
from quantark.param.div import ContinuousDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.pnlexplain import ExplainMethod, Factor, PnLExplainConfig, RowKind, ValuationSnapshot, explain
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

T0, T1 = datetime(2026, 6, 26), datetime(2026, 6, 29)
ENG = BlackScholesEngine()


def _env(spot, date, vols, rates, times=(0.5, 1.0, 2.0)):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=TermStructureVolSurface(times=list(times), vols=list(vols)),
        rate_curve=LinearRateCurve(pillars=list(zip([0.5, 1.0, 2.0], rates))),
        div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=date)


def _snaps(env1=None):
    s0 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
                           ENG, _env(100.0, T0, (0.20, 0.22, 0.24), (0.030, 0.032, 0.034)), date=T0)
    e1 = env1 if env1 is not None else _env(101.0, T1, (0.21, 0.225, 0.24), (0.031, 0.032, 0.035))
    s1 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0 - 3 / 365),
                           ENG, e1, date=T1)
    return s0, s1


def test_bucket_rows_replace_scalar_vega_and_reconcile():
    s0, s1 = _snaps()
    scalar = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical"))
    bucketed = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical", bucketed=True))
    terms = [r.term for r in bucketed.rows if r.method is ExplainMethod.TAYLOR]
    assert "vega" not in terms and "rho" in terms                       # vega replaced, rho kept
    assert bucketed.metadata["bucketed_factors"] == ("vol",)
    vega_rows = [r for r in bucketed.rows if r.term.startswith("vega.") and r.kind is RowKind.COMPONENT]
    assert [r.term for r in vega_rows] == ["vega.0.5", "vega.1", "vega.2"]        # each pillar exactly once
    assert len(terms) == len(set(terms))
    assert all("tenor" in r.moves for r in vega_rows)
    scalar_vega = [r for r in scalar.rows if r.term == "vega"][0].pnl
    assert sum(r.pnl for r in vega_rows) == pytest.approx(scalar_vega, rel=5e-2, abs=1e-6)
    assert bucketed.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)
    # bucket rows sit where the scalar vega row would have been (right after delta)
    assert terms.index("vega.0.5") == terms.index("delta") + 1
    # the scalar rho is the component; the key-rate rows are informational and sit beneath it
    rho = [r for r in bucketed.rows if r.term == "rho"][0]
    assert rho.kind is RowKind.COMPONENT
    kr = [r for r in bucketed.rows if r.term.startswith("rate_keyrate.")]
    assert [r.term for r in kr] == ["rate_keyrate.0.5", "rate_keyrate.1", "rate_keyrate.2", "rate_keyrate.parallel"]
    assert all(r.kind is RowKind.INFORMATIONAL and r.metadata["convention"] == "carry_invariant" for r in kr)
    assert terms.index("rate_keyrate.0.5") == terms.index("rho") + 1
    par = kr[-1]
    assert "sum_of_buckets" in par.metadata and "reconciles" in par.metadata
    # the parallel row's move IS the scalar rate move (same coordinate), never a fabricated zero
    assert par.moves["rate_pct"] == pytest.approx(rho.moves["rate_pct"], rel=1e-12)
    assert par.moves["rate_pct"] != 0.0
    # the scalar Taylor rows are unchanged by the opt-in
    for term in ("delta", "gamma", "rho", "theta"):
        a = [r for r in scalar.rows if r.term == term][0].pnl
        b = [r for r in bucketed.rows if r.term == term][0].pnl
        assert a == pytest.approx(b, abs=1e-12)


def test_parallel_keyrate_row_requires_the_scalar_rate_move():
    """No tenor at the coordinate => no defined parallel move => ValidationError, not a zero row."""
    import dataclasses
    from types import SimpleNamespace
    from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
    from quantark.pnlexplain.equity.bucketed import bucketed_rows
    from quantark.pnlexplain.equity.coordinate import resolve_coordinate
    from quantark.pnlexplain.equity.factor_diff import build_factor_moves
    s0, s1 = _snaps()
    coord = resolve_coordinate(s0.product, 100.0, s1.product, s1.pricing_env)
    moves = build_factor_moves(s0, s1, coord, engine_alive_t1=ENG, lifecycle_changed=False)
    from quantark.asset.equity.param import EngineParams
    bump = EngineParams().get_effective_bump_config()
    cache = SimpleNamespace(snap0=s0, snap1=s1, moves=moves, bump_engine_t0=ENG)
    rows, covered = bucketed_rows(cache, GreeksCalculator(), bump, "instrument")
    assert covered == frozenset({Factor.VOL}) and Factor.RATE in rows
    broken = dataclasses.replace(moves, d_rate=None,
                                 coordinate=dataclasses.replace(coord, tenor_t1=None))
    cache_broken = SimpleNamespace(snap0=s0, snap1=s1, moves=broken, bump_engine_t0=ENG)
    with pytest.raises(ValidationError, match="parallel key-rate"):
        bucketed_rows(cache_broken, GreeksCalculator(), bump, "instrument")


def test_bucketed_honours_configured_bumps():
    from quantark.asset.equity.param import BumpConfig, EngineParams
    s0, s1 = _snaps()
    params = EngineParams(bump_config=BumpConfig(vol_bump=0.02, rate_bump=0.0005))
    res = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical", bucketed=True, params=params))
    vega_rows = [r for r in res.rows if r.term.startswith("vega.")]
    kr_rows = [r for r in res.rows if r.term.startswith("rate_keyrate.")]
    assert vega_rows and kr_rows
    assert all(r.metadata["bump_size"] == pytest.approx(0.02) for r in vega_rows)
    assert all(r.metadata["bump_size"] == pytest.approx(0.0005) for r in kr_rows)


def test_bucketed_samples_t1_on_t0_pillars():
    """A differing t1 grid is allowed: the t1 surface is read at the t0 pillars (spec §7.6)."""
    finer = _env(101.0, T1, (0.205, 0.21, 0.225, 0.24, 0.25), (0.031, 0.032, 0.035),
                 times=(0.25, 0.5, 1.0, 2.0, 3.0))
    s0, s1 = _snaps(env1=finer)
    res = explain(s0, s1, config=PnLExplainConfig(greeks_method="numerical", bucketed=True))
    vega_rows = [r for r in res.rows if r.term.startswith("vega.")]
    assert [r.term for r in vega_rows] == ["vega.0.5", "vega.1", "vega.2"]          # t0 pillars
    for row in vega_rows:
        tau = row.metadata["pillar"]
        expected = (finer.get_vol(100.0, tau) - s0.pricing_env.get_vol(100.0, tau)) * 100.0
        assert row.moves["vol_pts"] == pytest.approx(expected)
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)


def test_bucketed_rejects_mismatched_structures():
    s0, s1 = _snaps()
    flat = ValuationSnapshot(s1.product, ENG, PricingEnvironment(
        spot_quote=SpotQuote(spot=101.0), vol_surface=FlatVolSurface(0.22),
        rate_curve=FlatRateCurve(rate=0.032), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=T1), date=T1)
    with pytest.raises(ValidationError):
        explain(s0, flat, config=PnLExplainConfig(bucketed=True))
    # flat objects on BOTH sides keep the scalar rows (nothing to bucket)
    flat0 = ValuationSnapshot(s0.product, ENG, PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(0.20),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=T0), date=T0)
    res = explain(flat0, flat, config=PnLExplainConfig(bucketed=True))
    terms = [r.term for r in res.rows if r.method is ExplainMethod.TAYLOR]
    assert "vega" in terms and "rho" in terms and res.metadata["bucketed_factors"] == ()
