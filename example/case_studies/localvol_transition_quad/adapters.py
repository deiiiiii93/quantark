"""Production baselines for the local-vol QUAD study: products, environment, PDE/MC full risk.

Every number here comes from the shipped engines through their public API
(spec D10). Bumped engines are built explicitly: the production
``GreeksCalculator`` returns zero vega for a prebuilt local-vol surface.
Units are those of ``LocalVolQuad.full_risk`` (spec Contracts): PV per unit
notional S0; delta = dPV/dS*S0; gamma = d2PV/dS2*S0**2; vega per +0.01 LV;
theta per calendar day; rho and carry per +1bp.
"""
from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter

import numpy as np

from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import LocalVolSnowballMCEngine
from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver
from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures.greeks.bump_envs import advance_theta_bump
from quantark.asset.equity.riskmeasures.greeks.theta_shift import shift_for_theta
from quantark.param import ContinuousDividendYield, FlatRateCurve, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierType, CouponPayType, ObservationType, OptionType, ProtectionType
from quantark.util.enum.engine_enums import MonteCarloMethod

from .contract import Contract
from .market import Market

VALUATION_DATE = datetime(2024, 2, 8)
THETA_DAYS = 1
MC_SPOT_BUMP = 0.01
MAX_BATCH_PATHS = 2 ** 16
RSS_LIMIT_BYTES = 16 * 1024 ** 3
OUTPUTS = ("pv", "delta", "gamma", "vega", "theta", "rho", "carry")


def production_snowball(c: Contract) -> SnowballOption:
    """Production snowball with daily discrete KI and the builders' pinned accrual flags (spec D7)."""
    barrier = BarrierConfig(
        ko_barrier=c.ko, ko_rate=c.ko_rate, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(c.ko_times), ki_barrier=c.ki, disable_ko_after_ki=False,
        ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=list(c.ki_times))
    return SnowballOption(
        initial_price=c.s0, strike=c.strike, barrier_config=barrier,
        payoff_config=PayoffConfig(rebate_rate=c.rebate_rate, protection_type=ProtectionType.NONE),
        accrual_config=AccrualConfig(coupon_pay_type=CouponPayType.INSTANT, is_annualized=True,
                                     is_annualized_rebate=False),
        contract_multiplier=1.0, maturity=c.maturity)


def production_vanilla(c: Contract) -> EuropeanVanillaOption:
    kind = OptionType.CALL if c.option == "call" else OptionType.PUT
    return EuropeanVanillaOption(strike=c.strike, option_type=kind, maturity=c.maturity)


def production_up_out(c: Contract) -> BarrierOption:
    return BarrierOption(strike=c.strike, option_type=OptionType.CALL, barrier=c.ko,
                         barrier_type=BarrierType.UP_OUT, maturity=c.maturity,
                         observation_type=ObservationType.DISCRETE, observation_dates=list(c.ko_times))


def production_product(c: Contract):
    return {"snowball": production_snowball, "vanilla": production_vanilla,
            "up_out": production_up_out}[c.kind](c)


def environment(market: Market, spot: float | None = None) -> PricingEnvironment:
    """ACT/365 calendar-day environment, no business calendar: theta steps exactly 1/365 (spec D8)."""
    return PricingEnvironment(
        rate_curve=FlatRateCurve(market.r), valuation_date=VALUATION_DATE,
        spot_quote=SpotQuote(float(market.spot if spot is None else spot)),
        vol_surface=market.vol_quote(), div_yield=ContinuousDividendYield(div_yield=market.q))


def theta_shift(product, env: PricingEnvironment):
    """Production theta shift (``shift_for_theta``) after asserting the 1/365 step."""
    bumped_date, time_bump, _ = advance_theta_bump(env, THETA_DAYS, "auto")
    if time_bump != THETA_DAYS / 365:
        raise AssertionError(f"theta step {time_bump} is not {THETA_DAYS}/365")
    return shift_for_theta(product, env, time_bump, bumped_date)


def _pde_solver(c: Contract, params: PDEParams, surface):
    cls = {"snowball": LocalVolSnowballPDESolver, "vanilla": LocalVolPDESolver,
           "up_out": LocalVolBarrierPDESolver}[c.kind]
    return cls(params=params, local_vol_surface=surface)


def _timed(timings: dict, name: str, fn):
    start = perf_counter()
    value = fn()
    timings[name] = perf_counter() - start
    return value


def pde_full_risk(c: Contract, market: Market, params: PDEParams) -> dict:
    """Eight valuations, each on a FRESH production solver (spec D9)."""
    timings: dict = {}
    s0 = market.spot
    product = production_product(c)
    base = _timed(timings, "base", lambda: _pde_solver(c, params, market.surface).calculate_greeks(
        product, environment(market)))
    out = {"pv": base["price"] / s0, "delta": base["delta"], "gamma": base["gamma"] * s0}

    def price(m: Market, name: str) -> float:
        return _timed(timings, name, lambda: _pde_solver(c, params, m.surface).price(
            production_product(c), environment(m)))

    for kind in ("vega", "rho", "carry"):
        up, down = price(market.bumped(kind, 1), f"{kind}+"), price(market.bumped(kind, -1), f"{kind}-")
        out[kind] = (up - down) / 2 / s0
    shift = theta_shift(product, environment(market))
    shifted = _timed(timings, "theta", lambda: _pde_solver(c, params, market.surface).price(
        shift.product, shift.env))
    out["theta"] = (shifted + shift.fixed_pv - base["price"]) / s0
    return dict(values=out, timings=timings)


@dataclass(frozen=True)
class MCConfig:
    total_paths: int
    batches: int
    substeps: int = 8
    seed: int = 20260828
    time_sampling: str = "integrated"

    def __post_init__(self):
        if self.total_paths % self.batches or self.total_paths // self.batches > MAX_BATCH_PATHS:
            raise ValueError("total_paths must split evenly into batches of at most 2**16 paths")
        if self.batches < 2:
            raise ValueError("at least two batches are needed for a standard error")


def _peak_rss() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def mc_full_risk(c: Contract, market: Market, config: MCConfig) -> dict:
    """Ten valuations per RQMC batch (spot +/- for delta/gamma), common random numbers across bumps.

    Each batch is a fresh engine with its own scrambling seed; quantities are
    formed per batch, so the reported standard errors are those of the
    differences, not of the individual prices.
    """
    if c.kind != "snowball":
        raise ValueError("MC baseline is run on the snowball only")
    s0, n = market.spot, config.total_paths // config.batches
    timings = {name: 0.0 for name in ("base", "spot+", "spot-", "vega+", "vega-", "theta",
                                      "rho+", "rho-", "carry+", "carry-")}
    rows = []
    for b in range(config.batches):
        def price(m: Market, product, env, name):
            engine = LocalVolSnowballMCEngine(
                local_vol_surface=m.surface,
                params=MCParams(seed=config.seed + b, num_paths=n, use_qmc=True, rqmc_min_batches=1,
                                rqmc_max_batches=1, rqmc_paths_mode="per_batch"),
                method=MonteCarloMethod.RANDOMIZED_QUASI, substeps_per_interval=config.substeps,
                lv_time_sampling=config.time_sampling, estimator="plain")
            start = perf_counter()
            value = engine.price(product, env)
            timings[name] += perf_counter() - start
            if _peak_rss() > RSS_LIMIT_BYTES:
                raise MemoryError("MC baseline exceeded the 16 GB RSS guard")
            return value

        product = production_snowball(c)
        h = MC_SPOT_BUMP * s0
        p0 = price(market, product, environment(market), "base")
        pu = price(market, product, environment(market, s0 + h), "spot+")
        pd = price(market, product, environment(market, s0 - h), "spot-")
        row = {"pv": p0 / s0, "delta": (pu - pd) / (2 * h), "gamma": (pu - 2 * p0 + pd) / h ** 2 * s0}
        for kind in ("vega", "rho", "carry"):
            up = price(market.bumped(kind, 1), product, environment(market.bumped(kind, 1)), f"{kind}+")
            down = price(market.bumped(kind, -1), product, environment(market.bumped(kind, -1)), f"{kind}-")
            row[kind] = (up - down) / 2 / s0
        shift = theta_shift(product, environment(market))
        row["theta"] = (price(market, shift.product, shift.env, "theta") + shift.fixed_pv - p0) / s0
        rows.append(row)
    values = {k: float(np.mean([r[k] for r in rows])) for k in OUTPUTS}
    stderr = {k: float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(config.batches)) for k in OUTPUTS}
    return dict(values=values, stderr=stderr, timings=timings, peak_rss=_peak_rss())
