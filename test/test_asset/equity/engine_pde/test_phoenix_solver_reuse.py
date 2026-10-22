"""A reused Phoenix solver prices as a fresh one: no termination surface W leaks from one solve into the next.

The terminal coupon stage discounts with W when one exists, and W is rebuilt only after the terminal
events -- so a solver (or a PDEEngine, which caches one per product type) read the previous price's W at
maturity: a shorter note after a longer one priced wrong, a longer one after a shorter one raised.
"""
import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from pde_3a.products import PHOENIX, env


def _phoenix(pay="expiry", years=1):
    return make_phoenix(dict(PHOENIX, coupon_pay_type=pay, maturity=float(years), num_observations=12 * years))


@pytest.mark.parametrize("first,then", [
    (("expiry", 2), ("expiry", 1)),      # 0.8.0: read an interior column of the 2y W (-3.1e-4)
    (("expiry", 1), ("expiry", 2)),      # 0.8.0: IndexError, the 1y W is too short
    (("expiry", 2), ("instant", 1)),     # 0.8.0: the stale W outranked the INSTANT discount
], ids=["long_then_short", "short_then_long", "expiry_then_instant"])
def test_a_book_on_one_engine_prices_each_trade_as_alone(first, then):
    alone = PDEEngine(PDEParams()).price(_phoenix(*then), env())
    engine = PDEEngine(PDEParams())
    engine.price(_phoenix(*first), env())
    assert engine.price(_phoenix(*then), env()) == alone


def test_a_rate_bump_on_one_solver_with_a_terminal_lag_prices_as_fresh():
    product = _phoenix()
    product.settlement_convention = SettlementConvention(lag=0.2, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    fresh = PhoenixPDESolver(PDEParams()).price(product, env(rate=0.035))
    solver = PhoenixPDESolver(PDEParams())
    solver.price(product, env())
    assert solver.price(product, env(rate=0.035)) == fresh     # 0.8.0: +3.3e-5
