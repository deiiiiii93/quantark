"""Tests for the SA-CVA non-regulatory HistoricalExposureEngine.

Run (worktree shadow):
    PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
        test/test_sacva_historical_exposure.py -v
"""
import numpy as np
import pandas as pd
import pytest
from types import MappingProxyType
from math import erf, sqrt, log, exp

from quantark.util.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Task 1 — provisional ExposureProfile contract
# ---------------------------------------------------------------------------
from quantark.sacva.exposure._contract_provisional import (
    Measure, ExposureProfile, CONTRACT_VERSION,
)


def test_mc_positional_construction_still_works():
    p = ExposureProfile(np.array([0., 1.]), np.array([5., 3.]), Measure.RISK_NEUTRAL, True)
    assert p.regulatory_eligible and p.epe_discounted is not None
    assert p.ee_undiscounted is None and p.pfe is None


def test_real_world_cannot_be_eligible():
    with pytest.raises(ValidationError):
        ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, True)


def test_eligible_requires_epe_discounted():
    with pytest.raises(ValidationError):
        ExposureProfile(np.array([0., 1.]), None, Measure.RISK_NEUTRAL, True)


def test_real_world_must_not_populate_epe_discounted():
    with pytest.raises(ValidationError):
        ExposureProfile(np.array([0., 1.]), np.array([5., 3.]), Measure.REAL_WORLD, False)


def test_shape_invariants():
    with pytest.raises(ValidationError):
        ExposureProfile(np.array([0., 1., 2.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([2., 1.]))   # wrong length


def test_arrays_and_metadata_are_immutable():
    p = ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([2., 1.]),
                        pfe={9900: np.array([4., 3.])}, metadata={"k": "v"})
    with pytest.raises(ValueError):
        p.ee_undiscounted[0] = 99.0                # read-only array
    assert isinstance(p.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        p.metadata["k"] = "x"


def test_historical_profile_ok_and_version():
    p = ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([2., 1.]),
                        pfe={9900: np.array([4., 3.])}, epe=1.5)
    assert not p.regulatory_eligible and p.pfe[9900][0] == 4.0
    assert p.epe_scalar == 1.5                     # back-compat alias
    assert isinstance(CONTRACT_VERSION, str) and CONTRACT_VERSION


def test_pfe_and_epe_validation():
    with pytest.raises(ValidationError):           # negative PFE
        ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([1., 1.]), pfe={9900: np.array([-1., 2.])})
    with pytest.raises(ValidationError):           # bad bps key
        ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([1., 1.]), pfe={12000: np.array([1., 2.])})


# ---------------------------------------------------------------------------
# Task 2 — provisional repricing scaffold
# ---------------------------------------------------------------------------
from quantark.sacva.exposure._contract_provisional import (
    AnalyticValueSurface, BoundedAnalyticValueSurface, aggregate_epe, CVATrade,
    NettingSet, Counterparty,
)


def test_analytic_forward_surface_linear():
    surf = AnalyticValueSurface(lambda S, t, ds: S - 100.0)
    v = surf.value_at(np.array([90., 110.]), 0.5, None)
    assert np.allclose(v, [-10., 10.])


def test_bounded_surface_raises_out_of_bounds():
    surf = BoundedAnalyticValueSurface(lambda S, t, ds: S - 100.0, low=50., high=150.)
    with pytest.raises(ValidationError):
        surf.value_at(np.array([200.]), 0.5, None)


def test_aggregate_epe_exact_values():
    a = np.array([[5., -3.], [-2., 4.]]); b = np.array([[-4., 1.], [6., -1.]])
    enf = aggregate_epe([a, b], True, np.array([1., 1.]))
    gross = aggregate_epe([a, b], False, np.array([1., 1.]))
    assert np.allclose(enf, [2.5, 1.5])      # hand-computed netted EPE
    assert np.allclose(gross, [5.5, 2.5])    # hand-computed gross EPE
    assert np.all(gross >= enf)


def test_trade_capability_flags_default_supported():
    tr = CVATrade("t", AnalyticValueSurface(lambda S, t, ds: S), "EQ_A")
    assert not tr.requires_continuous_barrier and not tr.requires_fx_conversion


# ---------------------------------------------------------------------------
# Task 3 — HistoricalMarketDataSet
# ---------------------------------------------------------------------------
from quantark.sacva.exposure.historical.calibration import HistoricalMarketDataSet


def _series(n=300, start=100.0, seed=0, end="2021-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_alignment_intersects_and_returns_len():
    a, b = _series(300, 100, 1), _series(300, 1.1, 2)
    ds = HistoricalMarketDataSet({"EQ_A": a, "FX_B": b})
    r = ds.log_returns()
    assert set(r.columns) == {"EQ_A", "FX_B"} and len(r) == 299


def test_insufficient_history_raises():
    with pytest.raises(ValidationError):
        HistoricalMarketDataSet({"EQ_A": _series(50, 100, 3)}, min_raw_obs=250).log_returns()


def test_today_level_uses_common_valuation_date():
    a = _series(300, 100, 4, end="2021-01-01")
    b = _series(310, 1.1, 5, end="2021-01-08")          # b extends past a
    ds = HistoricalMarketDataSet({"EQ_A": a, "FX_B": b})
    vd = ds.valuation_date()
    assert ds.today_level("FX_B") == float(b.loc[vd])   # common aligned date
    assert ds.today_level("FX_B") != float(b.iloc[-1])


# ---------------------------------------------------------------------------
# Task 4 — HistoricalCalibration
# ---------------------------------------------------------------------------
from quantark.sacva.exposure.historical.calibration import HistoricalCalibration, DriftMode


def _cal(n=400, seed=5):
    return HistoricalCalibration(HistoricalMarketDataSet(
        {"EQ_A": _series(n, 100, seed), "FX_B": _series(n, 1.1, seed + 1)}))


def test_drift_modes_and_subset():
    cal = _cal()
    r = cal.adjusted_log_returns(["EQ_A"], {"EQ_A": DriftMode.ZERO_LOG_MEAN})
    assert abs(r["EQ_A"].mean()) < 1e-12 and list(r.columns) == ["EQ_A"]


def test_ewma_seed_sample_variance_and_today_vol():
    cal = _cal()
    sig = cal.ewma_sigma("EQ_A", 0.94)
    assert np.isclose(sig[0], max(np.sqrt(np.var(cal._r["EQ_A"], ddof=1)), cal.vol_floor))
    assert cal.ewma_sigma_today("EQ_A", 0.94) > 0


def test_correlation_diagnostic_psd_and_nan_guard():
    cal = _cal()
    C = cal.correlation_diagnostic()
    assert C.shape == (2, 2) and np.min(np.linalg.eigvalsh(C)) > -1e-10
    flat = pd.Series(np.full(400, 7.0), index=_series(400).index)        # constant
    cal2 = HistoricalCalibration(HistoricalMarketDataSet({"EQ_A": _series(400), "C": flat}))
    with pytest.raises(ValidationError):
        cal2.correlation_diagnostic()


# ---------------------------------------------------------------------------
# Task 5 — Resampler
# ---------------------------------------------------------------------------
from quantark.sacva.exposure.historical.resampling import Resampler, ResamplingScheme


def _excess_kurtosis(x):
    x = x.ravel(); m = x.mean(); v = x.var()
    return float(((x - m) ** 4).mean() / v ** 2 - 3.0)


def _resid(seed=11):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(4, size=(400, 2))
    z[:, 1] = 0.8 * z[:, 0] + 0.2 * z[:, 1]
    return z


def test_iid_preserves_comovement():
    z = _resid()
    samp = Resampler(ResamplingScheme.IID_RAW, seed=1).sample(z, 2000, 50)
    c_src = np.corrcoef(z, rowvar=False)[0, 1]
    c_samp = np.corrcoef(samp.reshape(-1, 2), rowvar=False)[0, 1]
    assert abs(c_src - c_samp) < 0.05


def test_block_preserves_fat_tails():
    z = _resid()
    samp = Resampler(ResamplingScheme.BLOCK_FHS, block_length=10, seed=2).sample(z, 3000, 60)
    assert _excess_kurtosis(samp[..., 0]) > 1.0


def test_stationary_positive_path():
    z = _resid()
    r = Resampler(ResamplingScheme.STATIONARY_BLOCK, expected_block_length=8, seed=1)
    samp = r.sample(z, 100, 20)
    assert samp.shape == (100, 20, 2)


def test_block_requires_length_and_validates_inputs():
    z = _resid()
    with pytest.raises(ValidationError):
        Resampler(ResamplingScheme.BLOCK_FHS).sample(z, 10, 5)          # no block_length
    with pytest.raises(ValidationError):
        Resampler(ResamplingScheme.STATIONARY_BLOCK).sample(z, 10, 5)   # no expected len
    with pytest.raises(ValidationError):
        Resampler(ResamplingScheme.IID_RAW, seed=1).sample(z[:20], 10, 5, min_raw_obs=250)
    with pytest.raises(ValidationError):
        Resampler(ResamplingScheme.IID_RAW, seed=1).sample(z.ravel(), 10, 5)   # 1D
    with pytest.raises(ValidationError):
        Resampler(ResamplingScheme.STATIONARY_BLOCK, expected_block_length=8, overlap=True)


# ---------------------------------------------------------------------------
# Task 6 — HistoricalPathGenerator
# ---------------------------------------------------------------------------
from quantark.sacva.exposure.historical.path_generator import HistoricalPathGenerator, PathMode


def _const_cal(r=0.001, n=300):
    idx = pd.bdate_range("2019-01-01", periods=n)
    lvl = 100.0 * np.exp(r * np.arange(n))
    cal = HistoricalCalibration(HistoricalMarketDataSet({"EQ_A": pd.Series(lvl, index=idx)}))
    return cal, float(lvl[-1])


def test_no_sqrt_t_exact_reconstruction():
    r = 0.001
    cal, S0 = _const_cal(r)
    g = HistoricalPathGenerator(cal, ("EQ_A",), {"EQ_A": S0}, min_replay_windows=5)
    s = g.generate(PathMode.REPLAY_RAW, np.array([0., 2 / 252, 5 / 252]))
    assert np.allclose(s[:, 1, 0], S0 * np.exp(2 * r))   # 2 daily log-returns compounded
    assert np.allclose(s[:, 2, 0], S0 * np.exp(5 * r))   # 5 total — NOT sqrt-scaled
    assert np.allclose(s[:, 0, 0], S0)


def test_replay_deterministic_and_bootstrap_seeded():
    a_s = _series(400, 100, 9)
    b_s = _series(400, 1.1, 10)
    cal2 = HistoricalCalibration(HistoricalMarketDataSet({"EQ_A": a_s, "FX_B": b_s}))
    g = HistoricalPathGenerator(cal2, ("EQ_A", "FX_B"),
                                {"EQ_A": float(a_s.iloc[-1]), "FX_B": float(b_s.iloc[-1])})
    grid = np.array([0., 0.5, 1.0])
    modes = {"EQ_A": DriftMode.ZERO_LOG_MEAN, "FX_B": DriftMode.ZERO_LOG_MEAN}
    a = g.generate(PathMode.BOOTSTRAP, grid, scheme=ResamplingScheme.BLOCK_FHS,
                   block_length=10, n_paths=500, seed=7, drift_modes=modes)
    b = g.generate(PathMode.BOOTSTRAP, grid, scheme=ResamplingScheme.BLOCK_FHS,
                   block_length=10, n_paths=500, seed=7, drift_modes=modes)
    c = g.generate(PathMode.BOOTSTRAP, grid, scheme=ResamplingScheme.BLOCK_FHS,
                   block_length=10, n_paths=500, seed=99, drift_modes=modes)
    assert np.array_equal(a, b) and a.shape == (500, 3, 2)        # seed reproducible
    assert not np.array_equal(a, c)                               # different seed differs
    assert np.allclose(a[:, 0, :], [g.today_levels["EQ_A"], g.today_levels["FX_B"]])
    assert a[:, -1, 0].std() > 0                                  # terminal not constant


def test_replay_insufficient_windows_raises():
    cal, S0 = _const_cal(n=300)
    g = HistoricalPathGenerator(cal, ("EQ_A",), {"EQ_A": S0}, min_replay_windows=10_000)
    with pytest.raises(ValidationError):
        g.generate(PathMode.REPLAY_RAW, np.array([0., 0.5]))


def test_grid_start_must_be_zero():
    cal, S0 = _const_cal()
    g = HistoricalPathGenerator(cal, ("EQ_A",), {"EQ_A": S0}, min_replay_windows=5)
    with pytest.raises(ValidationError):
        g.generate(PathMode.REPLAY_RAW, np.array([0.1, 0.5]))
