"""Fixtures for the zero-diffusion PDE step tests (spec §4.5)."""
from datetime import datetime

import numpy as np
import scipy.linalg

from quantark.asset.equity.engine.pde import EuropeanPDESolver
from quantark.param import SpotQuote
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock

ANCHOR = datetime(2026, 2, 9)
R, Q = 0.02, 0.01


def build_coeffs_with_zero_step(mu_sign=+1):
    """StepCoefficients on a CNY-straddling grid with holiday sigma_step == 0.

    Returns (StepCoefficients, dx, r, q). mu_sign flips r/q to test both
    upwind directions (mu = r - q at sigma == 0).
    """
    r, q = (R, Q) if mu_sign >= 0 else (Q, R)
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), ANCHOR, datetime(2027, 2, 9))
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=ANCHOR,
        spot_quote=SpotQuote(100.0),
        vol_surface=TradingClockVolSurface(FlatVolSurface(0.20), m),
        div_yield=ContinuousDividendYield(q),
    )
    solver = EuropeanPDESolver()
    t_vec = np.linspace(0.0, 40.0 / 365.0, 40 * 4 + 1)
    num_x = 51
    dx = 0.004
    dx_vec = np.full(num_x - 1, dx)
    sc = solver._build_step_coefficients(env, 100.0, t_vec, dx_vec, num_x)
    return sc, dx, r, q


def advance_zero_diffusion_step(v0, x, r, q, dt):
    """One theta=1 upwind pure-advection step: (I - dt*A) v1 = v0."""
    n = x.size
    dx = float(x[1] - x[0])
    mu = r - q
    l = np.zeros(n)
    c = np.zeros(n)
    u = np.zeros(n)
    if mu >= 0.0:
        u[:] = mu / dx
        c[:] = -mu / dx - r
    else:
        l[:] = -mu / dx
        c[:] = mu / dx - r
    # banded (I - dt*A): ab rows are (upper, diag, lower)
    ab = np.zeros((3, n))
    ab[0, 1:] = -dt * u[:-1]
    ab[1, :] = 1.0 - dt * c
    ab[2, :-1] = -dt * l[1:]
    return scipy.linalg.solve_banded((1, 1), ab, np.asarray(v0, dtype=float))
