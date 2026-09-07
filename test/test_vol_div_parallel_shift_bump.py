"""Vol and dividend bumps must preserve the SHAPE of the market object.

``build_vol_bumped_env`` / ``build_div_bumped_env`` replaced any surface that
was not flat or ATM-term-structure with ``FlatVolSurface(sigma(K,T) + bump)``
(and any yield that was not continuous or term-structure with a constant), so
a smile lost its skew, a grid lost the GridVolSurface type the local-vol
engines require, and a carry-implied yield lost its term shape. The one
primitive is ``parallel_shifted``: a flat object stays flat (legacy floats,
bitwise), a term structure shifts every node, a grid shifts every cell and
stays a GridVolSurface, anything else is wrapped.
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.riskmeasures.greeks import bump_envs
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div.dividend_yield import (
    DividendYield, NoDividend, TermStructureDividendYield,
)
from quantark.param.vol.vol_surface import (
    BlackImpliedVolSurface, GridVolSurface, TermStructureVolSurface,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError

BUMP = 0.01
VD = datetime(2024, 1, 15)
STRIKES = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
MATS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def _grid(level=0.20):
    nT, nK = len(MATS), len(STRIKES)
    skew = np.array([0.06 * (nK - 1 - j) / (nK - 1) for j in range(nK)])   # falls with strike
    return GridVolSurface(STRIKES, MATS, np.tile(level + skew, (nT, 1)))


def _env(vol_surface=None, div_yield=None):
    return PricingEnvironment(
        spot_quote=SpotQuote(100.0), vol_surface=vol_surface or _grid(),
        rate_curve=FlatRateCurve(0.03),
        div_yield=div_yield if div_yield is not None else ContinuousDividendYield(0.01),
        valuation_date=VD,
    )


# --- vol primitive ------------------------------------------------------------------

def test_vol_parallel_shifted_flat_stays_flat_bitwise():
    s = FlatVolSurface(0.2)
    up = s.parallel_shifted(BUMP)
    assert type(up) is FlatVolSurface and up.volatility == 0.2 + BUMP
    assert s.volatility == 0.2


def test_vol_parallel_shifted_term_structure_shifts_every_node_bitwise():
    s = TermStructureVolSurface(times=[0.5, 1.0, 2.0], vols=[0.18, 0.2, 0.22])
    up = s.parallel_shifted(-BUMP)
    assert type(up) is TermStructureVolSurface and list(up.times) == [0.5, 1.0, 2.0]
    assert list(up.vols) == [float(v) + (-BUMP) for v in s.vols]


def test_vol_parallel_shifted_grid_stays_a_grid_with_every_cell_shifted():
    s = _grid()
    up = s.parallel_shifted(BUMP)
    assert type(up) is GridVolSurface                    # local-vol engines isinstance-gate on this
    assert up.strikes == s.strikes and up.maturities == s.maturities
    assert np.array_equal(up.iv_grid, s.iv_grid + BUMP)
    assert up.get_vol(85.0, 0.7, 100.0) == pytest.approx(s.get_vol(85.0, 0.7, 100.0) + BUMP, abs=1e-12)


class _MysterySmile(BlackImpliedVolSurface):
    is_smile = True

    def get_vol(self, strike, t, spot):
        return 0.2 + 0.001 * (spot - strike) / 10.0 + 0.01 * t


def test_vol_parallel_shifted_unknown_family_is_wrapped_and_keeps_its_smile():
    s = _MysterySmile()
    up = s.parallel_shifted(BUMP)
    for k, t in ((80.0, 0.3), (100.0, 1.0), (125.0, 2.5)):
        assert up.get_vol(k, t, 100.0) == pytest.approx(s.get_vol(k, t, 100.0) + BUMP, abs=1e-12)
    assert up.is_smile is True


def test_vol_parallel_shifted_trading_clock_surface_fails_closed_on_the_unit():
    from quantark.param.vol.trading_clock_surface import TradingClockVolSurface

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), datetime(2026, 2, 9), datetime(2028, 2, 9))
    wrapped = TradingClockVolSurface(FlatVolSurface(0.2), m)
    with pytest.raises(ValidationError, match="unit"):
        wrapped.parallel_shifted(BUMP)


# --- div primitive ------------------------------------------------------------------

def test_div_parallel_shifted_continuous_stays_continuous_bitwise():
    q = ContinuousDividendYield(0.015)
    up = q.parallel_shifted(BUMP)
    assert type(up) is ContinuousDividendYield and up.div_yield == 0.015 + BUMP


def test_div_parallel_shifted_term_structure_shifts_every_node_bitwise():
    q = TermStructureDividendYield(times=[0.5, 1.0, 2.0], yields=[0.01, 0.015, 0.02])
    up = q.parallel_shifted(-BUMP)
    assert type(up) is TermStructureDividendYield and list(up.times) == [0.5, 1.0, 2.0]
    assert list(up.yields) == [float(y) + (-BUMP) for y in q.yields]


def test_div_parallel_shifted_no_dividend_is_the_legacy_constant():
    up = NoDividend().parallel_shifted(BUMP)
    assert type(up) is ContinuousDividendYield and up.div_yield == 0.0 + BUMP


class _MysteryYield(DividendYield):
    def get_yield(self, t):
        return 0.01 + 0.005 * t


def test_div_parallel_shifted_unknown_family_is_wrapped():
    q = _MysteryYield()
    up = q.parallel_shifted(BUMP)
    for t in (0.1, 1.0, 2.5):
        assert up.get_yield(t) == pytest.approx(q.get_yield(t) + BUMP, abs=1e-12)


def test_div_parallel_shifted_trading_clock_shifts_the_calendar_quoted_inner():
    from quantark.param.div.trading_clock_yield import TradingClockDividendYield

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), datetime(2026, 2, 9), datetime(2028, 2, 9))
    wrapped = TradingClockDividendYield(_MysteryYield(), m)
    up = wrapped.parallel_shifted(BUMP)
    assert type(up) is TradingClockDividendYield and up.time_map is m
    for u in (0.1, 0.5, 1.0, 1.7):
        c = float(m.to_calendar(u))
        assert up.get_yield(u) == pytest.approx((0.01 + 0.005 * c + BUMP) * c / u, rel=1e-12)


# --- the helpers and their consumers ----------------------------------------------------

def test_build_vol_bumped_env_keeps_a_grid_a_grid():
    env = _env()
    up = bump_envs.build_vol_bumped_env(env, None, env.get_vol(100.0, 1.0), BUMP, direction=1.0)
    assert type(up.vol_surface) is GridVolSurface
    assert np.array_equal(up.vol_surface.iv_grid, env.vol_surface.iv_grid + BUMP)
    assert type(env.vol_surface) is GridVolSurface                      # input untouched


def test_build_vol_bumped_env_still_refuses_a_non_positive_atm_vol():
    env = _env(FlatVolSurface(0.005))
    with pytest.raises(ValidationError, match="must be positive"):
        bump_envs.build_vol_bumped_env(env, None, 0.005, BUMP, direction=-1.0)


def test_build_div_bumped_env_keeps_the_yield_term_shape():
    env = _env(div_yield=_MysteryYield())
    up = bump_envs.build_div_bumped_env(env, None, env.get_div_yield(1.0), BUMP, direction=1.0)
    for t in (0.1, 1.0, 2.5):
        assert up.get_div_yield(t) == pytest.approx(env.get_div_yield(t) + BUMP, abs=1e-12)


def _digital():
    from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
    from quantark.util.enum import OptionType

    return CashOrNothingDigitalOption(strike=100.0, payout=1.0, option_type=OptionType.CALL, maturity=1.0)


def test_numerical_vega_on_a_smile_keeps_the_skew():
    """A digital under a smile prices off the replicating call spread; flattening
    the surface at sigma(K, T) drops the skew term from the bumped price."""
    from quantark.asset.equity.engine.analytical.digital_option_engine import DigitalOptionAnalyticalEngine

    env, product, engine = _env(), _digital(), DigitalOptionAnalyticalEngine()
    base = engine.price(product, env)
    up = engine.price(product, _env(_grid().parallel_shifted(BUMP)))
    # one-sided calculator vega = P&L per configured bump (here one vol point)
    vega = GreeksCalculator().calculate_numerical_vega(product, env, engine, vol_bump=BUMP)
    assert vega == pytest.approx(up - base, rel=1e-12)


def test_numerical_vega_on_a_local_vol_engine_shifts_the_grid():
    from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
    from quantark.asset.equity.param import PDEParams
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType

    env = _env()
    product = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    engine = LocalVolPDESolver(PDEParams())
    base = engine.price(product, env)
    up = engine.price(product, _env(_grid().parallel_shifted(BUMP)))
    vega = GreeksCalculator().calculate_numerical_vega(product, env, engine, vol_bump=BUMP)
    assert vega == pytest.approx(up - base, rel=1e-9)             # one-sided: P&L per bump


def test_position_trade_greeks_vega_on_a_grid_is_a_parallel_shift():
    from quantark.asset.equity.engine.analytical.digital_option_engine import DigitalOptionAnalyticalEngine
    from quantark.portfolio.equity.position import EquityPosition

    env = _env()
    pos = EquityPosition(product=_digital(), quantity=1.0, entry_price=0.0, underlying="TEST",
                         engine=DigitalOptionAnalyticalEngine(), entry_timestamp=VD, cash_legs=[])
    b = pos.engine.params.get_effective_bump_config().vol_bump
    expected = (pos.get_trade_value(_env(_grid().parallel_shifted(b)))
                - pos.get_trade_value(_env(_grid().parallel_shifted(-b)))) / (2.0 * b)
    assert pos.get_trade_greeks(env, GreeksCalculator())["vega"] == pytest.approx(expected, rel=1e-12)
