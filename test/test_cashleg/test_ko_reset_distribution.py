"""The KO-reset event distribution (spec 3B §5.4, plan F5, F6; Review Focus 3)."""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.numerical import Tolerance
from intraday.conftest import dated_ko_reset, flat_env
from pde_3a.products import KO_RESET, env
from pde_statistics.v2_frozen import v2_event_stats


def test_both_schedules_and_both_maturities():
    stats = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(make_ko_reset(KO_RESET), env())
    d = EventDistribution.from_autocallable_stats(stats)
    n = stats.pre_ko_times.size + stats.post_ko_times.size
    assert d.event_times.size == n and np.all(np.diff(d.event_times) >= 0)
    assert d.probabilities[EventType.MATURITY_NO_KO] == pytest.approx(stats.pre_maturity_probability, abs=1e-12)
    assert d.determination_times[EventType.MATURITY_NO_KO] == stats.pre_maturity_time
    assert d.payment_times_for(EventType.MATURITY_NO_KO) == stats.pre_maturity_payment_time
    assert d.payment_times_for(EventType.MATURITY_WITH_KI) == stats.post_maturity_payment_time
    # F5: the pre-maturity mass leaves the survival at its determination, the last node keeps today's convention
    i_pre = int(np.searchsorted(d.event_times, stats.pre_maturity_time, side="right"))
    ko_cum = np.cumsum(d.probabilities[EventType.KO])
    assert d.survival_probability[i_pre] == pytest.approx(
        1.0 - ko_cum[i_pre - 1] - stats.pre_maturity_probability, abs=1e-12)
    assert d.survival_probability[-1] == pytest.approx(
        1.0 - ko_cum[-1] - stats.pre_maturity_probability, abs=1e-12)
    assert d.provenance == stats.provenance


def test_coincident_pre_and_post_dates_validate():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    product = dated_ko_reset(cal, datetime(2026, 3, 16))
    stats = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(product, flat_env(datetime(2026, 4, 1)))
    d = EventDistribution.from_autocallable_stats(stats)
    total = d.probabilities[EventType.KO].sum() + d.probabilities[EventType.MATURITY_NO_KO] \
        + d.probabilities[EventType.MATURITY_WITH_KI]
    assert abs(total - 1.0) <= Tolerance.PROBABILITY


def test_legacy_quad_keeps_one_maturity_outcome_but_both_schedules():
    stats = KOResetSnowballQuadEngine(QuadParams(grid_points=301)).calculate_event_stats(make_ko_reset(KO_RESET), env())
    d = EventDistribution.from_autocallable_stats(stats)
    assert d.event_times.size == stats.pre_ko_times.size + stats.post_ko_times.size
    assert d.determination_times == {}


def test_rebased_offsets_are_refused():
    import dataclasses
    stats = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(make_ko_reset(KO_RESET), env())
    with pytest.raises(NotImplementedError, match="offsets"):
        EventDistribution.from_autocallable_stats(dataclasses.replace(stats, post_ko_times_are_offsets=True))


def _mc_stats(post_ko_mode):
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.asset.equity.product.option import create_ko_reset_snowball
    from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    mc_env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
                                rate_curve=FlatRateCurve(rate=0.05), div_yield=ContinuousDividendYield(div_yield=0.02),
                                valuation_date=datetime(2024, 1, 1))
    product = create_ko_reset_snowball(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=2.0,
                                       post_ko_mode=post_ko_mode, ki_continuous=False)
    return SnowballMCEngine(params=MCParams(num_paths=1000, time_steps=252)).calculate_event_stats(product, mc_env)


def test_mc_absolute_converts_with_both_schedules():
    from quantark.util.enum import PostKOScheduleMode
    stats = _mc_stats(PostKOScheduleMode.ABSOLUTE)
    assert stats.pre_ko_payment_times is not None and stats.post_ko_payment_times is not None
    d = EventDistribution.from_autocallable_stats(stats)
    assert d.event_times.size == stats.pre_ko_times.size + stats.post_ko_times.size


def test_mc_rebased_is_refused():
    from quantark.util.enum import PostKOScheduleMode
    stats = _mc_stats(PostKOScheduleMode.REBASED)
    assert stats.post_ko_times_are_offsets
    with pytest.raises(NotImplementedError, match="offsets"):
        EventDistribution.from_autocallable_stats(stats)


def test_quad_v2_reports_both_maturity_outcomes():
    from quantark.asset.equity.engine.quad.v2.engine import KOResetSnowballQuadEngineV2
    stats = v2_event_stats(
        "ko_reset_kr_cont",
        lambda: KOResetSnowballQuadEngineV2().calculate_event_stats(make_ko_reset(KO_RESET), env()))
    d = EventDistribution.from_autocallable_stats(stats)
    assert d.event_times.size == stats.pre_ko_times.size + stats.post_ko_times.size
    assert d.determination_times == {EventType.MATURITY_NO_KO: stats.pre_maturity_time,
                                     EventType.MATURITY_WITH_KI: stats.post_maturity_time}
    assert d.probabilities[EventType.MATURITY_WITH_KI] == pytest.approx(stats.post_maturity_probability, abs=1e-12)
