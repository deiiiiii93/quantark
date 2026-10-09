"""Shared fixtures for the W7 2D Heston V0 evidence scripts (certain-V0 control and continuous-KI contracts).

Run any script from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <script>
"""
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import HestonSnowballPDESolver  # noqa: E402
from quantark.asset.equity.product.option.observation_schedule import (  # noqa: E402
    ObservationRecord, ObservationSchedule)
from quantark.asset.equity.product.option.snowball_config import (  # noqa: E402
    AccrualConfig, BarrierConfig, PayoffConfig)
from quantark.asset.equity.product.option.snowball_option import SnowballOption  # noqa: E402
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402
from quantark.volmodels.heston import HestonParams  # noqa: E402

S0, VOL, T = 100.0, 0.2, 2.0
HP = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)   # W3's TestHeston2DRows params


def env(r=0.03, q=0.0, vol=VOL):
    kw = {"div_yield": ContinuousDividendYield(div_yield=q)} if q else {}
    return PricingEnvironment(spot_quote=SpotQuote(spot=S0, asset_name="X"), vol_surface=FlatVolSurface(volatility=vol),
                              rate_curve=FlatRateCurve(rate=r), valuation_date=datetime(2024, 1, 2), **kw)


def barrier(ko, ki, continuous, n_obs=24, ko_rate=0.15):
    obs = [(i + 1) / n_obs * T for i in range(n_obs)]
    return BarrierConfig(
        ko_barrier=ko, ko_rate=ko_rate,
        ko_observation_schedule=ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=ko) for t in obs]),
        ki_barrier=ki,
        ki_observation_type=ObservationType.CONTINUOUS if continuous else ObservationType.DISCRETE,
        ki_continuous=continuous,
        ki_observation_schedule=None if continuous else ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=ki) for t in obs]))


def snowball(ko, ki, continuous=False, rebate_rate=0.10, annualized=False, reverse=False, n_obs=24, ko_rate=0.15):
    return SnowballOption(
        initial_price=S0, strike=S0, barrier_config=barrier(ko, ki, continuous, n_obs, ko_rate),
        payoff_config=PayoffConfig(rebate_rate=rebate_rate, include_principal=True),
        accrual_config=AccrualConfig(is_annualized_rebate=annualized), maturity=T, tenor=T, is_reverse=reverse)


def certain_v0(rebate_rate=0.10, annualized=False, reverse=False, continuous=False, ko=None, ki=None, n_obs=24,
               ko_rate=0.15):
    """W3's control: KO 5x beyond the spot and KI 100x beyond it, each on its own side."""
    ko0, ki0 = (20.0, 10000.0) if reverse else (500.0, 1.0)
    return snowball(ko if ko is not None else ko0, ki if ki is not None else ki0, continuous, rebate_rate, annualized,
                    reverse, n_obs, ko_rate)


def closed_form(product, e, rebate_rate, annualized, r=0.03):
    tenor = product.get_contract_tenor(e) if annualized else 1.0
    return (S0 + rebate_rate * S0 * tenor) * math.exp(-r * T)


def solver(hp=HP, n_x=96, n_v=24, n_t=48, **kw):
    return HestonSnowballPDESolver(model_params=hp, n_x=n_x, n_v=n_v, n_t=n_t, **kw)
