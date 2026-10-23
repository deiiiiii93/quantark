"""SIMM honours ``EquityPortfolio.trade_value`` through the portfolio adapter (ship spec 2026-10-23
portfolio-trade-value D8; AC1, AC7).

Flag on, the adapter passes ``trade_value=True`` to positions whose ``get_simm_sensitivities`` accepts it and the
equity sensitivity engine reads delta / vega from the trade risk (bump Greeks, cash legs included). Flag off is
today's product path, bitwise; a direct position call without the keyword stays product-only.
"""
import copy

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.simm import SIMMConfig
from quantark.simm.engines.classification.bucket_mapper import BucketMapper
from quantark.simm.engines.portfolio_adapter import SIMMPortfolioAdapter
from quantark.simm.sensitivity import vol_weighted_vega_equity
from test_portfolio.test_equity_portfolio_trade_value import _book

CONFIG = SIMMConfig(calculation_currency="USD", calculate_delta=True, calculate_vega=True)


def _amounts(collection):
    """{(kind, trade_id): amount} for the equity delta / vega sensitivities."""
    out = {}
    for s in collection.sensitivities:
        kind = type(s).__name__
        if kind in ("EquityDeltaSensitivity", "EquityVegaSensitivity"):
            out[(kind, s.trade_id)] = s.amount
    return out


def _expected(pf, trade):
    """Amounts from the engine's formulas on product Greeks (get_greeks) or trade-risk totals."""
    ref = copy.deepcopy(pf)
    gc = GreeksCalculator()
    out = {}
    for p in ref.positions.values():
        env = ref.pricing_environments[p.underlying]
        if trade and hasattr(p, "get_trade_risk"):
            greeks = copy.deepcopy(p).get_trade_risk(env, gc, greeks=["delta"]).total
            greeks["vega"] = copy.deepcopy(p).get_trade_risk(env, gc, greeks=["vega"]).total["vega"]
        else:
            greeks = p.get_greeks(env, gc)
        bucket = BucketMapper().classify_equity_bucket(p.underlying)
        out[("EquityDeltaSensitivity", p.position_id)] = 0.01 * env.spot * greeks.get("delta", 0.0)
        if greeks.get("vega", 0.0) != 0.0:
            out[("EquityVegaSensitivity", p.position_id)] = vol_weighted_vega_equity(greeks["vega"], bucket)
    return out


def test_adapter_flag_off_is_the_product_path_bitwise():
    pf = _book()
    got = _amounts(SIMMPortfolioAdapter(CONFIG).portfolio_to_sensitivities(pf))
    assert got == _expected(pf, trade=False)


def test_adapter_flag_on_takes_delta_and_vega_from_trade_risk():
    pf = _book(trade_value=True)
    got = _amounts(SIMMPortfolioAdapter(CONFIG).portfolio_to_sensitivities(pf))
    want = _expected(pf, trade=True)
    assert got == want
    # the trade numbers equal the portfolio switch's Greeks, and differ from the product ones
    ref = copy.deepcopy(pf)
    env = ref.pricing_environments["X"]
    switch = ref.position_greeks(ref.positions["note"], GreeksCalculator())
    assert got[("EquityDeltaSensitivity", "note")] == 0.01 * env.spot * switch["delta"]
    product = _expected(pf, trade=False)
    for pos_id in ("vanilla", "note"):
        assert got[("EquityDeltaSensitivity", pos_id)] != product[("EquityDeltaSensitivity", pos_id)]
    assert got[("EquityDeltaSensitivity", "trs")] == product[("EquityDeltaSensitivity", "trs")]


def test_direct_position_call_stays_product_only():
    pf = _book(trade_value=True)
    note = pf.positions["note"]
    envs = pf.pricing_environments
    product = _expected(pf, trade=False)
    trade = _expected(pf, trade=True)
    direct = _amounts(copy.deepcopy(note).get_simm_sensitivities(CONFIG, envs))
    assert direct == {k: v for k, v in product.items() if k[1] == "note"}
    keyword = _amounts(copy.deepcopy(note).get_simm_sensitivities(CONFIG, envs, trade_value=True))
    assert keyword == {k: v for k, v in trade.items() if k[1] == "note"}
