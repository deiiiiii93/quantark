"""The simulated-path backtest prices the snapped contract and keys the initial phase (spec 2026-10-08 4a)."""
from dataclasses import replace

import numpy as np
import pytest

from quantark.asset.equity.lifecycle.snapping import snap_contract
from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.simulation.config import CacheConfig
from quantark.backtest.simulation.paths.market_path import trading_calendar
from quantark.backtest.simulation.pricing.base import DayStates, StateKey, float_key
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer, product_fingerprint
from quantark.backtest.simulation.pricing.surface import LifeSurfacePricer

from simulation.conftest import KO_BARRIER, RATE, SPOT, pde_engine_config, short_snowball, surface_pricing

from datetime import date

DATES = trading_calendar(date(2024, 1, 2), 8)


def _off_day_snowball():
    """The review's control: a knock-out at day 1.4, observed on day 1's close."""
    return short_snowball(maturity_days=6, ko_days=(1.4, 5), ki_days=(1, 3))


def _pricers(product):
    pricing = surface_pricing()
    cache = StateCache(CacheConfig(memory_bytes=8_000_000))
    exact = RepricingPricer(product, engine_config=pde_engine_config(), start_date=DATES[0], underlying="CSI1000",
                            cache=cache)
    surface = LifeSurfacePricer(product, engine_config=pde_engine_config(), start_date=DATES[0], dates=DATES,
                                underlying="CSI1000", vol_step=pricing.vol_step, q_step=pricing.q_step,
                                surface_cache_bytes=pricing.surface_cache_bytes, gate=pricing.gate, cache=cache)
    return exact, surface


def _row(day_index, spot, **extra):
    return DayStates(day_index=day_index, date=DATES[day_index], path_index=np.array([0]), spot=np.array([spot]),
                     vol=np.array([0.22]), rate=np.array([RATE]), q_T=np.array([0.05]),
                     div_yield=(SignedDividendYield(0.05),), basis_yield=np.array([-0.03]),
                     env_key=float_key(np.array([RATE])), knocked_in=np.array([False]), **extra)


def test_the_pricers_fingerprint_the_snapped_contract():
    product = _off_day_snowball()
    exact, surface = _pricers(product)
    assert exact.product_fingerprint == product_fingerprint(snap_contract(product))
    assert exact.product_fingerprint != product_fingerprint(product)
    assert surface.product_fingerprint == exact.product_fingerprint


def test_the_surface_and_the_exact_copy_agree_after_the_observation():
    """The surface priced the day-1.4 knock-out as still ahead on day 1's close; the exact copy had dropped it."""
    exact, surface = _pricers(_off_day_snowball())
    row = _row(1, KO_BARRIER * 0.995)            # alive, just under the knock-out level
    surface_pv = float(surface.price_day(row)[0][0])
    exact_pv = float(exact.price_day(row)[0][0])
    assert abs(surface_pv - exact_pv) <= 5e-4 * SPOT, (surface_pv, exact_pv)


def test_default_phase_and_knock_in_day_leave_the_key_digest_unchanged():
    base = dict(product_fingerprint="p", day_index=3, knocked_in=False, spot_key=1, vol_key=2, env_key=3,
                engine_fingerprint="e")
    assert StateKey(**base).digest() == StateKey(**base, initial=False, knock_in_day=-1).digest()
    assert StateKey(**base, initial=True).digest() != StateKey(**base).digest()
    assert StateKey(**base, knock_in_day=4).digest() != StateKey(**base).digest()


def test_the_initial_copy_keeps_a_record_landing_on_day_zero():
    product = short_snowball(ko_days=(0, 5))
    exact, _ = _pricers(product)
    initial = exact.aged_product(DATES[0], knocked_in=False, initial=True)
    post = exact.aged_product(DATES[0], knocked_in=False)
    assert (len(initial.barrier_config.ko_observation_schedule.records)
            == len(post.barrier_config.ko_observation_schedule.records) + 1)


def test_an_initial_state_without_a_landing_record_shares_the_post_close_key():
    exact, _ = _pricers(short_snowball())
    assert exact.state_keys(_row(0, SPOT, initial=True)) == exact.state_keys(_row(0, SPOT))
