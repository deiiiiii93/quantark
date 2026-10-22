"""Historical PFE/EE demo (non-regulatory).

Prices a real European call against historical (real-world) bootstrap paths and
reports EE / PFE. The profile is REAL_WORLD / not regulatory-eligible, so it can
never feed SA-CVA capital (MAR50.34(1)).

Run:
    PYTHONPATH=$PWD .venv/bin/python example/sacva_historical_pfe_demo.py
"""
from datetime import datetime

import numpy as np
import pandas as pd

from quantark.param import (
    ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote,
)
from quantark.priceenv.pricing_environment import PricingEnvironment
from quantark.asset.equity.product.option.european_vanilla_option import (
    EuropeanVanillaOption,
)
from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.util.enum import OptionType
from quantark.util.calendar import DayCountConvention

from quantark.sacva.models.enums import CreditQuality
from quantark.sacva.portfolio.trade import CVATrade
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.exposure.historical.calibration import (
    HistoricalCalibration, HistoricalMarketDataSet,
)
from quantark.sacva.exposure.historical.engine import (
    HistoricalExposureEngine, HistoricalExposureConfig,
)

VAL, EXP = datetime(2026, 6, 17), datetime(2027, 6, 17)
ASSET, S0 = "ACME", 100.0


class FlatCreditCurve:
    def get_survival_probability(self, t):
        return float(np.exp(-0.02 * t))

    recovery_rate = 0.4


# historical level series ending exactly at today's spot
rng = np.random.default_rng(0)
rel = np.exp(np.cumsum(rng.normal(0.0, 0.012, 1500)))
levels = S0 * rel / rel[-1]
series = pd.Series(levels, index=pd.bdate_range(end=VAL, periods=1500))
cal = HistoricalCalibration(HistoricalMarketDataSet({ASSET: series}))

env = PricingEnvironment(
    rate_curve=FlatRateCurve(0.02), valuation_date=VAL,
    spot_quote=SpotQuote(spot=S0, asset_name=ASSET),
    vol_surface=FlatVolSurface(0.2), div_yield=ContinuousDividendYield(0.0),
    day_count_convention=DayCountConvention.CALENDAR_DAYS)
call = CVATrade(trade_id="call", engine=BlackScholesEngine(),
                product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                              exercise_date=EXP),
                env=env, quantity=1.0, trade_currency="USD")
cp = Counterparty(name="CP", netting_sets=[NettingSet("n1", [call])],
                  credit_curve=FlatCreditCurve(), bucket=2, credit_quality=CreditQuality.IG)

cfg = HistoricalExposureConfig(
    path_mode="BOOTSTRAP", scheme="BLOCK_FHS", block_length=10, n_paths=8000, seed=1,
    n_steps=8, confidences_bps=(9500, 9900), drift_modes={ASSET: "EMPIRICAL_MEAN"})
prof = HistoricalExposureEngine(cal, cfg).compute(cp)

print("times :", np.round(prof.times, 4))
print("EE    :", np.round(prof.ee_undiscounted, 4))
print("PFE95 :", np.round(prof.pfe[9500], 4))
print("PFE99 :", np.round(prof.pfe[9900], 4))
print("epe   :", round(prof.epe, 4))
print("eligible:", prof.regulatory_eligible, " measure:", prof.measure.value)
