"""The 3B evidence's statistics cases, for the tests (spec 3B §5.7, §7.2).

Copied verbatim from the evidence directories:

  ``docs/superpowers/specs/2026-10-03-pde-1d-event-statistics-evidence/lab3b.py`` lines 29-45   CASES
  ``docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-evidence/lab.py`` lines 63-81      daily_ki_snowball
  ``docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-evidence/lab.py`` lines 323-344    trading_clock_env
                                                (every ``lab.`` prefix dropped)

``ENV``, ``SNOWBALL``, ``PHOENIX``, ``KO_RESET`` and ``env`` are the 3A lab's, banked in
``test/pde_3a/products.py``. Nothing of the evidence's patching machinery is copied here.
"""
from __future__ import annotations

from pde_3a.products import ENV, KO_RESET, PHOENIX, SNOWBALL, env  # noqa: F401

from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from quantark.util.enum import ObservationType


def daily_ki_snowball():
    """The byte book's daily discrete KI snowball (252 observations)."""
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption

    ref = make_snowball(dict(SNOWBALL, ki_monitoring="discrete"))
    bc = ref.barrier_config
    return SnowballOption(
        initial_price=ref.initial_price, strike=ref.strike,
        barrier_config=BarrierConfig(
            ko_barrier=bc.ko_barrier, ko_rate=bc.ko_rate,
            ko_observation_type=bc.ko_observation_type,
            ko_observation_dates=list(bc.ko_observation_dates),
            ki_barrier=bc.ki_barrier, ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=[k / 252.0 for k in range(1, 253)],
        ),
        payoff_config=ref.payoff_config, contract_multiplier=1.0, maturity=1.0, is_reverse=False,
    )


def trading_clock_env(vol=0.22, rate=0.025, div_yield=0.03, spot=100.0, days_per_year=244):
    """Flat vol on the SSE trading clock: variance accrues on trading days only."""
    from datetime import datetime

    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
    from quantark.priceenv import PricingEnvironment
    from quantark.util.calendar import CalendarType, create_calendar
    from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock

    anchor = datetime(2026, 2, 9)
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2029))
    tmap = BusinessTimeMap(TradingClock(cal, days_per_year), anchor, datetime(2028, 6, 30))
    e = PricingEnvironment(
        rate_curve=FlatRateCurve(rate), valuation_date=anchor, spot_quote=SpotQuote(spot),
        vol_surface=TradingClockVolSurface(FlatVolSurface(vol), tmap),
        div_yield=ContinuousDividendYield(div_yield), bus_days_in_year=days_per_year,
    )
    is_td = tmap._is_td
    breaks = [i / 365.0 for i in range(1, len(is_td)) if is_td[i] != is_td[i - 1]]
    return e, breaks


#: name -> (product factory, environment factory): the statistics cases
CASES = {
    "sb_cont": (lambda: make_snowball(SNOWBALL), env),
    "sb_cont_near_ki": (lambda: make_snowball(SNOWBALL), lambda: env(spot=86.0)),
    "sb_cont_near_ko": (lambda: make_snowball(SNOWBALL), lambda: env(spot=102.5)),
    "sb_cont_3m": (lambda: make_snowball(dict(SNOWBALL, months=3, maturity=0.25)), env),
    "sb_cont_vol12": (lambda: make_snowball(SNOWBALL), lambda: env(vol=0.12)),
    "sb_discrete": (lambda: make_snowball(dict(SNOWBALL, ki_monitoring="discrete")), env),
    "sb_discrete_near_ki": (lambda: make_snowball(dict(SNOWBALL, ki_monitoring="discrete")), lambda: env(spot=86.0)),
    "sb_european": (lambda: make_snowball(dict(SNOWBALL, ki_monitoring="european")), env),
    "sb_daily": (daily_ki_snowball, env),
    "ph_cont": (lambda: make_phoenix(PHOENIX), env),
    "ph_cont_near_coupon": (lambda: make_phoenix(PHOENIX), lambda: env(spot=86.0)),
    "ph_memory": (lambda: make_phoenix(dict(PHOENIX, memory_coupon=True)), env),
    "kr_cont": (lambda: make_ko_reset(KO_RESET), env),
}

#: the S1 cases the exact reference covers (spec 3B §4.1); S2's Phoenix cases are ph_cont and ph_cont_near_coupon
REFERENCE_CASES = ("sb_cont", "sb_cont_near_ki", "sb_cont_near_ko", "sb_cont_3m", "sb_cont_vol12", "sb_discrete",
                   "sb_discrete_near_ki", "sb_european", "sb_daily", "ph_cont", "ph_cont_near_coupon")

#: the reference's panel widths where they are not the defaults (evidence ``s1_fix_exact.py`` line 14)
REFERENCE_KWARGS = {"sb_daily": dict(width=0.008, kie_width=0.012)}
