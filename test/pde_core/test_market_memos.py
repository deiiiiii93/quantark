"""Market memos key on values, not object identity (spec 2 decisions 14, 18, 19)."""
from datetime import datetime

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.modelvalidation.builders.equity_snowball import make_snowball

SB = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
          rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)


def _env(rate=0.03):
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
                              rate_curve=FlatRateCurve(rate=rate), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=datetime(2026, 1, 5))


def test_mutating_the_rate_after_a_solve_resamples():
    p = make_snowball(SB)
    env = _env(0.03)
    s = SnowballPDESolver(PDEParams())
    s.price(p, env)
    env.rate_curve.rate = 0.07
    live = s.price(p, env)
    fresh = SnowballPDESolver(PDEParams()).price(p, _env(0.07))
    assert live == fresh


def _coefficients(env):
    """The step-coefficient builder on a fixed geometry (the memo's own seam).

    A flat environment's price does not show a stale memo: with one coefficient
    set, ``_flat_exact_step_coefficients`` re-reads r, q and sigma live.
    """
    import numpy as np
    num_x = 20
    sc = SnowballPDESolver(PDEParams())._build_step_coefficients(
        env, 100.0, np.linspace(0.0, 1.0, 13), np.full(num_x - 1, 0.05), num_x)
    return [np.concatenate(s) for s in sc.lcu_sets]


def _same(a, b):
    import numpy as np
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_mutating_the_rate_after_a_build_resamples_the_coefficients():
    env = _env(0.03)
    _coefficients(env)
    env.rate_curve.rate = 0.07
    assert _same(_coefficients(env), _coefficients(_env(0.07)))


def test_a_spot_change_resamples_the_coefficients_of_a_spot_reading_surface():
    env = _env()
    env.vol_surface = _SpotReadingVol(volatility=0.2)
    _coefficients(env)
    env.spot_quote.spot = 101.0
    fresh = _env()
    fresh.vol_surface = _SpotReadingVol(volatility=0.2)
    fresh.spot_quote.spot = 101.0
    assert _same(_coefficients(env), _coefficients(fresh))


def test_discount_memo_follows_the_live_rate():
    s = SnowballPDESolver(PDEParams())
    env = _env(0.03)
    s._df_between_times(env, 0.0, 1.0)
    env.rate_curve.rate = 0.07
    assert s._df_between_times(env, 0.0, 1.0) == SnowballPDESolver(PDEParams())._df_between_times(_env(0.07), 0.0, 1.0)


class _Opaque:
    """A market object with no value fingerprint: never memoized."""
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_unfingerprintable_market_object_is_sampled_fresh_every_time():
    p = make_snowball(SB)
    env = _env()
    env.vol_surface = _Opaque(FlatVolSurface(volatility=0.2))
    s = SnowballPDESolver(PDEParams())
    a = s.price(p, env)
    env.vol_surface._inner = FlatVolSurface(volatility=0.3)
    b = s.price(p, env)
    assert a != b


class _SpotReadingVol(FlatVolSurface):
    """A surface whose vol reads the spot ``TermCoefficients.from_env`` passes to ``get_vol``."""

    def get_vol(self, strike, time_to_maturity, spot=None):
        return 0.2 + 1e-3 * (spot or 0) / 100


def test_a_spot_change_with_a_spot_reading_surface_resamples():
    p = make_snowball(SB)
    env = _env()
    env.vol_surface = _SpotReadingVol(volatility=0.2)
    s = SnowballPDESolver(PDEParams())
    s.price(p, env)
    env.spot_quote.spot = 101.0
    live = s.price(p, env)
    fresh_env = _env()
    fresh_env.vol_surface = _SpotReadingVol(volatility=0.2)
    fresh_env.spot_quote.spot = 101.0
    assert live == SnowballPDESolver(PDEParams()).price(p, fresh_env)


def test_a_holiday_added_to_the_environment_calendar_re_resolves_settlement():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    env = _env()
    env.calendar = cal
    p = make_snowball(SB)
    s = SnowballPDESolver(PDEParams())
    k1 = s._observation_cache_key(env, p, "ko")
    cal.holidays.add(datetime(2026, 2, 5))
    assert s._observation_cache_key(env, p, "ko") != k1


def test_calendar_identity_names_holidays_and_weekend():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2027))
    ident = cal.identity()
    cal.holidays.add(datetime(2026, 2, 5))
    assert cal.identity() != ident


def test_value_equal_curves_do_not_share_a_discount_memo():
    # The discount memo rounds (t, T) to 12 decimals and the first build wins,
    # so its sharing domain decides last ULPs: it stays one curve object, as
    # every banked number was computed. A process-wide value-keyed store let
    # one solve's times seed another's (byte book: a phoenix vega bump moved
    # in its last digit depending on the rows priced before it).
    from quantark.asset.equity.engine.pde import base_pde_solver
    a, b = _env(0.03), _env(0.03)
    s = SnowballPDESolver(PDEParams())
    s._df_between_times(a, 0.0, 1.0)
    s._df_between_times(b, 0.0, 1.0)
    memo_a = base_pde_solver._per_object_memo(a.rate_curve, base_pde_solver._ENV_DF_MEMO)
    memo_b = base_pde_solver._per_object_memo(b.rate_curve, base_pde_solver._ENV_DF_MEMO)
    assert memo_a is not memo_b and memo_a and memo_b
