"""Profile representative 1D MC pricings to rank migration candidates by measured share.

Counterpart of docs/adi2d-greek-perf/perf-headroom/prof_mc.py for the 1D engines.
Run from the repo root:  PYTHONPATH=$PWD <venv>/bin/python docs/mc1d-perf/prof_baseline.py
"""

import cProfile
import io
import pstats
import time
from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.param import MCParams
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, OptionType, ProtectionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.localvol.mc_kernel import price_barrier_lv_mc, price_european_lv_mc


def flat_env(vol=0.20, rate=0.05, div=0.02):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def realistic_lv_surface():
    """A smiled LV grid large enough that lookup costs are representative."""
    t_grid = np.linspace(0.0, 2.0, 25)
    k_grid = np.exp(np.linspace(np.log(50.0), np.log(200.0), 61))
    logm = np.log(k_grid / 100.0)
    smile = 0.20 + 0.15 * logm**2 - 0.05 * logm
    term = 1.0 + 0.1 * np.sqrt(np.maximum(t_grid, 0.0))[:, None]
    grid = np.clip(smile[None, :] * term, 0.05, 1.5)
    return LocalVolSurface(k_grid, t_grid, grid)


def run_profiled(label, fn):
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    out = fn()
    prof.disable()
    wall = time.perf_counter() - t0
    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream).sort_stats("tottime")
    stats.print_stats(14)
    text = stream.getvalue()
    print(f"\n{'=' * 72}\n{label}: result={out!r}, wall={wall:.3f}s\n{'=' * 72}")
    # keep only the table body
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("ncalls"))
    print("\n".join(lines[start : start + 16]))
    return wall


def euro_case():
    env = flat_env()
    product = EuropeanVanillaOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    engine = EuropeanMCEngine(
        params=MCParams(num_paths=200_000, time_steps=252, seed=42),
        method=MonteCarloMethod.PSEUDO,
    )
    return engine.price(product, env)


def snowball_case():
    env = flat_env()
    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    payoff_config = PayoffConfig(
        rebate_rate=0.15,
        call_rebate_enabled=False,
        call_strike=None,
        call_participation_rate=1.0,
        include_principal=False,
        participation_rate=1.0,
        protection_type=ProtectionType.NONE,
        protection_rate=0.0,
    )
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )
    engine = SnowballMCEngine(
        params=MCParams(num_paths=100_000, time_steps=252, seed=42),
        method=MonteCarloMethod.PSEUDO,
    )
    return engine.price(product, env)


def lv_euro_case(lv):
    n = 252
    return price_european_lv_mc(
        s0=100.0, strike=100.0, is_call=True, lv_surface=lv,
        step_dt=np.full(n, 1.0 / n), r_fwd=np.full(n, 0.05),
        carry_fwd=np.full(n, 0.02), disc_factor=float(np.exp(-0.05)),
        num_paths=100_000, seed=42,
    )


def lv_barrier_case(lv):
    n = 252
    return price_barrier_lv_mc(
        s0=100.0, strike=100.0, is_call=True, lv_surface=lv,
        step_dt=np.full(n, 1.0 / n), r_fwd=np.full(n, 0.05),
        carry_fwd=np.full(n, 0.02), disc_factor=float(np.exp(-0.05)),
        barrier=130.0, is_up=True, is_out=True, rebate=1.0,
        continuous=True, num_paths=100_000, seed=42,
    )


if __name__ == "__main__":
    lv = realistic_lv_surface()
    run_profiled("euro MC pseudo 200k x 252", euro_case)
    run_profiled("snowball MC pseudo 100k x 252 (continuous-KI bridge)", snowball_case)
    run_profiled("LV euro kernel 100k x 252 (25x61 grid)", lambda: lv_euro_case(lv))
    run_profiled("LV barrier kernel 100k x 252 (continuous)", lambda: lv_barrier_case(lv))
