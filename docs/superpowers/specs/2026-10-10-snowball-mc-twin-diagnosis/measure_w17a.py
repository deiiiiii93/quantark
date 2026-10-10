"""W17a moved values: every MC value the new tests pin (test/test_mc_time_node_merge.py), plus the BS snowball /
Phoenix twins on EVERY valuation day 1..364 and the REBASED KO-reset event-stats cell the probe flagged. Runs on any
tree (it imports nothing W17a adds); compare a c02296cb export with the branch:

    env PYTHONPATH=<tree>:<branch>/test:<this dir> <venv>/bin/python measure_w17a.py > <out>.txt

Rows: ``cell | form | value`` (repr floats, or the refusal)."""
import warnings
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

warnings.simplefilter("ignore")

import quantark  # noqa: E402
from test_dated_expiry_settles import HESTON, SLV_LEVERAGE, env as _env  # noqa: E402

from quantark.asset.equity.engine.mc import PhoenixMCEngine, SnowballMCEngine  # noqa: E402
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (  # noqa: E402
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
)
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (  # noqa: E402
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
)
from quantark.asset.equity.param import MCParams  # noqa: E402
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption  # noqa: E402
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule  # noqa
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig  # noqa
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402
from quantark.util.enum.option_enums import PostKOScheduleMode  # noqa: E402

INITIAL = datetime(2025, 1, 5)
DAYS = [round(k / 12 * 365) for k in range(1, 13)]
PRE, POST = DAYS[:6], DAYS[6:]
COLLISION_DAYS = (14, 41, 53, 61, 92, 170, 209)
KO_RESET_COLLISION_DAYS = (26, 65, 78, 82)


def on(day):
    return INITIAL + timedelta(days=day)


def env(spot, day, grid=False):
    return _env(spot, on(day), grid=grid)


def _schedule(form, days, level):
    if form == "dated":
        return ObservationSchedule(records=[ObservationRecord(observation_date=on(d), barrier=level) for d in days])
    return ObservationSchedule(records=[ObservationRecord(observation_time=d / 365.0, barrier=level) for d in days])


def _expiry(p, form):
    p.initial_date = INITIAL
    if form == "dated":
        p.maturity = None
        p.exercise_date = on(365)


def _age(p, form, day):
    if form == "numeric":
        p.time_shift(day / 365.0, on(day), env(100.0, day), keep_landing=True)
    return p


def snowball(form, day):
    p = create_standard_snowball(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.15,
                                 is_annualized=False)
    _expiry(p, form)
    p.barrier_config = replace(p.barrier_config, ko_observation_dates=None,
                               ko_observation_schedule=_schedule(form, DAYS, 103.0))
    p.validate()
    return _age(p, form, day)


def phoenix(form, day):
    p = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                coupon_rate=0.01, num_observations=12, memory_coupon=False, is_annualized=False)
    _expiry(p, form)
    p.barrier_config = replace(p.barrier_config, ko_observation_dates=None,
                               ko_observation_schedule=_schedule(form, DAYS, 103.0))
    p.validate()
    return _age(p, form, day)


def ko_reset(form, day):
    dated = form == "dated"
    p = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=None if dated else 1.0, initial_date=INITIAL,
        exercise_date=on(365) if dated else None,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(form, PRE, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(form, POST, 95.0)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE)
    p.validate()
    return _age(p, form, day)


def _mc(seed=7, paths=256):
    return MCParams(seed=seed, num_paths=paths)


ROUTES = {
    "bs_snowball": (snowball, lambda: SnowballMCEngine(params=_mc()), False),
    "bs_phoenix": (phoenix, lambda: PhoenixMCEngine(params=_mc()), False),
    "lv_snowball": (snowball, lambda: LocalVolSnowballMCEngine(params=_mc()), True),
    "heston_snowball": (snowball, lambda: HestonSnowballMCEngine(HESTON, params=_mc()), False),
    "slv_snowball": (snowball, lambda: HestonSLVSnowballMCEngine(HESTON, params=_mc(), leverage_surface=SLV_LEVERAGE),
                     True),
    "lv_phoenix": (phoenix, lambda: LocalVolPhoenixMCEngine(params=_mc()), True),
    "heston_phoenix": (phoenix, lambda: HestonPhoenixMCEngine(HESTON, params=_mc()), False),
    "slv_phoenix": (phoenix, lambda: HestonSLVPhoenixMCEngine(HESTON, params=_mc(), leverage_surface=SLV_LEVERAGE),
                    True),
}


def attempt(fn):
    try:
        return repr(float(fn()))
    except Exception as exc:  # noqa: BLE001
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:100]}"


def row(cell, form, value):
    print(f"{cell} | {form} | {value}", flush=True)


if __name__ == "__main__":
    print(f"# quantark from {quantark.__file__}")
    # 1. the pinned twins (test_dated_twin_equals_numeric_on_collision_days / test_ko_reset_continuous_ki_twin...)
    for route, (build, factory, grid) in ROUTES.items():
        for day in COLLISION_DAYS:
            for form in ("dated", "numeric"):
                row(f"twin:{route}-d{day}", form,
                    attempt(lambda: factory().price(build(form, day), env(100.0, day, grid=grid))))
    for day in KO_RESET_COLLISION_DAYS:
        for form in ("dated", "numeric"):
            row(f"twin:bs_ko_reset_continuous_ki-d{day}", form,
                attempt(lambda: SnowballMCEngine(params=_mc()).price(ko_reset(form, day), env(100.0, day))))
    # 2. every valuation day, BS snowball / Phoenix (256 x seed 7) and KO-reset (to day 181)
    for name, build, engine, last in (("snowball", snowball, SnowballMCEngine, 364),
                                      ("phoenix", phoenix, PhoenixMCEngine, 364),
                                      ("ko_reset_continuous_ki", ko_reset, SnowballMCEngine, 181)):
        for day in range(1, last + 1):
            for form in ("dated", "numeric"):
                row(f"allday:bs_{name}-d{day}", form,
                    attempt(lambda: engine(params=_mc()).price(build(form, day), env(100.0, day))))
    # 3. the REBASED KO-reset grid the probe flagged (test_ko_reset_snowball_mc_engine::...event_stats_shape[Rebased])
    import test_ko_reset_snowball_mc_engine as t

    def rebased_stats():
        product = t.create_ko_reset_snowball(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=2.0,
                                             post_ko_mode=PostKOScheduleMode.REBASED, ki_continuous=False)
        return SnowballMCEngine(params=MCParams(num_paths=1000, time_steps=252)).calculate_event_stats(
            product, t.create_pricing_env())
    row("event_stats:ko_reset_rebased(test_ko_reset_event_stats_shape[Rebased])", "pv",
        attempt(lambda: rebased_stats().pv))
    row("event_stats:ko_reset_rebased(test_ko_reset_event_stats_shape[Rebased])", "sum_post_ko_probability",
        attempt(lambda: float(np.sum(rebased_stats().post_ko_probability))))
