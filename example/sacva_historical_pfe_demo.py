"""Historical PFE/EE demo (non-regulatory).

Run:
    PYTHONPATH=$PWD .venv/bin/python example/sacva_historical_pfe_demo.py
"""
import numpy as np
import pandas as pd

from quantark.sacva.exposure.historical.calibration import (
    HistoricalCalibration, HistoricalMarketDataSet,
)
from quantark.sacva.exposure.historical.engine import (
    HistoricalExposureEngine, HistoricalExposureConfig,
)
from quantark.sacva.exposure._contract_provisional import (
    AnalyticValueSurface, CVATrade, NettingSet, Counterparty,
)

rng = np.random.default_rng(0)
idx = pd.bdate_range("2018-01-01", periods=800)
fx = pd.Series(1.1 * np.exp(np.cumsum(rng.normal(0, 0.008, 800))), index=idx)
cal = HistoricalCalibration(HistoricalMarketDataSet({"FX_B": fx}))

# A reporting-currency-settled FX forward: value(S) = S - K.
surf = AnalyticValueSurface(lambda S, t, ds: S - 1.1)
cp = Counterparty("CP", [NettingSet("ns", [CVATrade("fwd", surf, "FX_B")], True)])

cfg = HistoricalExposureConfig(
    path_mode="BOOTSTRAP", scheme="BLOCK_FHS", block_length=10, n_paths=8000, seed=1,
    grid_times=(0., 0.25, 0.5, 0.75, 1.0), confidences_bps=(9500, 9900),
    factor_keys=("FX_B",), today_levels={"FX_B": float(fx.iloc[-1])},
    drift_modes={"FX_B": "EMPIRICAL_MEAN"})

prof = HistoricalExposureEngine(cal, cfg).compute(cp)

print("times :", prof.times)
print("EE    :", np.round(prof.ee_undiscounted, 4))
print("PFE95 :", np.round(prof.pfe[9500], 4))
print("PFE99 :", np.round(prof.pfe[9900], 4))
print("epe   :", round(prof.epe, 4))
print("eligible:", prof.regulatory_eligible, " measure:", prof.measure.value)
