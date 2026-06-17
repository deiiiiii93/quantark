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


# ---------------------------------------------------------------------------
# Task 7 — PFEProfileAssembler + Kupiec
# ---------------------------------------------------------------------------
from quantark.sacva.exposure.historical.pfe import PFEProfileAssembler, kupiec_pof


def test_pfe_nonneg_monotone_max_and_epe():
    rng = np.random.default_rng(31)
    E = np.abs(rng.standard_normal((5000, 4)))
    asm = PFEProfileAssembler(confidences_bps=(9500, 9900, 10000), m_tail_min=10)
    res = asm.assemble(E, np.array([0., 1., 2., 3.]))
    assert np.all(res["ee_undiscounted"] >= 0)
    assert np.all(res["pfe"][9900] >= res["pfe"][9500] - 1e-9)
    assert np.allclose(res["pfe"][10000], E.max(axis=0))
    asm2 = PFEProfileAssembler(confidences_bps=(9900,), m_tail_min=1)
    assert np.isclose(asm2.assemble(np.ones((100, 3)), np.array([0., 1., 2.]))["epe"], 2.0)


def test_pfe_validation():
    asm = PFEProfileAssembler(confidences_bps=(9900,), m_tail_min=10)
    with pytest.raises(ValidationError):
        asm.assemble(np.abs(np.random.default_rng(3).standard_normal((50, 3))),
                     np.array([0., .5, 1.]))                      # tail too thin
    with pytest.raises(ValidationError):
        asm.assemble(np.ones((100, 3)), np.array([0., 1.]))       # times length mismatch
    with pytest.raises(ValidationError):
        PFEProfileAssembler(confidences_bps=(12000,)).assemble(np.ones((100, 2)),
                                                               np.array([0., 1.]))
    with pytest.raises(ValidationError):
        asm.assemble(np.ones((100, 3)), np.array([0., 1., 0.5]))  # non-increasing times


def test_quantile_method_order_statistic():
    E = np.arange(10, dtype=float)[:, None]
    lin = PFEProfileAssembler((9500,), quantile_method="linear", m_tail_min=0).assemble(
        E, np.array([0.]))["pfe"][9500][0]
    cons = PFEProfileAssembler((9500,), quantile_method="inverted_cdf", m_tail_min=0).assemble(
        E, np.array([0.]))["pfe"][9500][0]
    assert cons == 9.0
    assert lin <= cons


def test_kupiec_deterministic():
    assert not kupiec_pof(10, 1000, 0.99)[1]      # ~expected -> accept
    assert kupiec_pof(100, 1000, 0.99)[1]         # 10% at 99% -> reject
    with pytest.raises(ValidationError):
        kupiec_pof(2000, 1000, 0.99)              # x > n


# ---------------------------------------------------------------------------
# Task 8 — HistoricalExposureEngine
# ---------------------------------------------------------------------------
from quantark.sacva.exposure.historical.engine import (
    HistoricalExposureEngine, HistoricalExposureConfig,
)


def _Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal_return_cal(sigma_d=0.01, n=2000, seed=3):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, sigma_d, n)
    lvl = 1.0 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2010-01-01", periods=n + 1)
    s = pd.Series(np.concatenate([[1.0], lvl]), index=idx)
    return HistoricalCalibration(HistoricalMarketDataSet({"FX_B": s})), float(s.iloc[-1])


def test_engine_forward_EE_matches_lognormal_closed_form():
    # IID_RAW (not FHS) so the terminal law is ~lognormal with deterministic variance.
    sigma_d = 0.01
    cal, S0 = _normal_return_cal(sigma_d)
    K = S0
    surf = AnalyticValueSurface(lambda S, t, ds: S - K)
    cp = Counterparty("CP", [NettingSet("ns", [CVATrade("fwd", surf, "FX_B")], True)])
    cfg = HistoricalExposureConfig(
        path_mode="BOOTSTRAP", scheme="IID_RAW", n_paths=60000, seed=5,
        grid_times=(0.0, 1.0), confidences_bps=(9900,), factor_keys=("FX_B",),
        today_levels={"FX_B": S0}, drift_modes={"FX_B": "ZERO_LOG_MEAN"})
    prof = HistoricalExposureEngine(cal, cfg).compute(cp)
    n_days = 252
    v = n_days * float(np.var(cal._r["FX_B"].to_numpy(), ddof=1))
    a = log(S0)
    d1 = (a - log(K) + v) / sqrt(v)
    d2 = d1 - sqrt(v)
    ee_cf = exp(a + 0.5 * v) * _Phi(d1) - K * _Phi(d2)
    assert abs(prof.ee_undiscounted[-1] - ee_cf) / ee_cf < 0.05
    assert prof.measure is Measure.REAL_WORLD and not prof.regulatory_eligible
    assert prof.epe_discounted is None and prof.metadata["path_mode"] == "BOOTSTRAP"


def test_engine_rejects_today_level_mismatch():
    cal, S0 = _normal_return_cal()
    surf = AnalyticValueSurface(lambda S, t, ds: S - S0)
    cp = Counterparty("CP", [NettingSet("ns", [CVATrade("f", surf, "FX_B")], True)])
    cfg = HistoricalExposureConfig(
        path_mode="BOOTSTRAP", scheme="IID_RAW", n_paths=500, seed=1,
        grid_times=(0., 1.), factor_keys=("FX_B",),
        today_levels={"FX_B": S0 * 1.10}, drift_modes={"FX_B": "ZERO_LOG_MEAN"})
    with pytest.raises(ValidationError):
        HistoricalExposureEngine(cal, cfg).compute(cp)


def test_engine_rejects_out_of_scope_and_bad_config():
    cal, S0 = _normal_return_cal()
    surf = AnalyticValueSurface(lambda S, t, ds: S - 1.0)
    bad = CVATrade("x", surf, "FX_B", requires_continuous_barrier=True)
    cp = Counterparty("CP", [NettingSet("ns", [bad], True)])
    cfg = HistoricalExposureConfig(
        path_mode="BOOTSTRAP", scheme="BLOCK_FHS", block_length=5, n_paths=100, seed=1,
        grid_times=(0., 1.), factor_keys=("FX_B",), today_levels={"FX_B": S0},
        drift_modes={"FX_B": "ZERO_LOG_MEAN"})
    with pytest.raises(ValidationError):
        HistoricalExposureEngine(cal, cfg).compute(cp)
    with pytest.raises(ValidationError):                       # bad scheme -> ValidationError
        HistoricalExposureConfig(
            path_mode="BOOTSTRAP", scheme="NOPE", block_length=5, n_paths=100,
            grid_times=(0., 1.), factor_keys=("FX_B",), today_levels={"FX_B": S0},
            drift_modes={"FX_B": "ZERO_LOG_MEAN"})
    with pytest.raises(ValidationError):                       # empty netting set
        HistoricalExposureEngine(cal, cfg).compute(
            Counterparty("E", [NettingSet("n", [], True)]))


# ---------------------------------------------------------------------------
# Task 9 — Regulatory guard + merge gate (the keystone)
# ---------------------------------------------------------------------------
from quantark.sacva.exposure.historical.regulatory_guard import (
    assert_regulatory_eligible, ProvisionalRegulatoryCVAStub,
)


def test_historical_profile_rejected_by_capital_path():
    cal, S0 = _normal_return_cal()
    surf = AnalyticValueSurface(lambda S, t, ds: S - 1.0)
    cp = Counterparty("CP", [NettingSet("ns", [CVATrade("f", surf, "FX_B")], True)])
    cfg = HistoricalExposureConfig(
        path_mode="BOOTSTRAP", scheme="BLOCK_FHS", block_length=5, n_paths=2000, seed=1,
        grid_times=(0., 0.5, 1.0), confidences_bps=(9900,), factor_keys=("FX_B",),
        today_levels={"FX_B": S0}, drift_modes={"FX_B": "ZERO_LOG_MEAN"})
    prof = HistoricalExposureEngine(cal, cfg).compute(cp)
    with pytest.raises(ValidationError):
        assert_regulatory_eligible(prof)
    with pytest.raises(ValidationError):
        ProvisionalRegulatoryCVAStub().compute(cp, prof)


def test_guard_accepts_risk_neutral_eligible():
    rn = ExposureProfile(np.array([0., 1.]), np.array([5., 3.]), Measure.RISK_NEUTRAL, True)
    assert_regulatory_eligible(rn)


def test_merge_gate_provisional_removed_once_canonical_exists():
    import importlib, importlib.util, inspect, pkgutil
    if importlib.util.find_spec("quantark.sacva.exposure.engine") is None:
        pytest.skip("canonical contract not yet landed by MC session")
    import quantark.sacva.exposure.historical as hist
    for mi in pkgutil.walk_packages(hist.__path__, hist.__name__ + "."):
        src = inspect.getsource(importlib.import_module(mi.name))
        assert "_contract_provisional" not in src, \
            f"{mi.name} still imports provisional contract"


# ---------------------------------------------------------------------------
# ZenMux review fixes (iteration 1)
# ---------------------------------------------------------------------------
def test_non_finite_level_rejected():
    s = _series(300, 100, 1).copy()
    s.iloc[-1] = np.inf
    with pytest.raises(ValidationError):
        HistoricalMarketDataSet({"EQ_A": s})


def test_bootstrap_accepts_exact_minimum_history():
    # min_raw_obs levels -> min_raw_obs-1 returns; bootstrap must not reject the
    # exact advertised minimum (return-vector count mismatch fix).
    cal = HistoricalCalibration(HistoricalMarketDataSet(
        {"FX_B": _series(250, 1.1, 7)}, min_raw_obs=250))
    g = HistoricalPathGenerator(cal, ("FX_B",), {"FX_B": float(_series(250, 1.1, 7).iloc[-1])})
    s = g.generate(PathMode.BOOTSTRAP, np.array([0., 0.5, 1.0]),
                   scheme=ResamplingScheme.IID_RAW, n_paths=100, seed=1,
                   drift_modes={"FX_B": DriftMode.ZERO_LOG_MEAN})
    assert s.shape == (100, 3, 1)


def test_float_block_length_rejected():
    z = _resid()
    with pytest.raises(ValidationError):
        Resampler(ResamplingScheme.BLOCK_FHS, block_length=10.0).sample(z, 10, 5)
    with pytest.raises(ValidationError):
        Resampler(ResamplingScheme.STATIONARY_BLOCK,
                  expected_block_length=float("nan")).sample(z, 10, 5)


# ---------------------------------------------------------------------------
# ZenMux review fixes (iteration 2)
# ---------------------------------------------------------------------------
def test_grid_refinement_keeps_terminal_horizon():
    # same final maturity must use the same total day count regardless of how many
    # intermediate report nodes are inserted (cumulative-boundary rounding).
    r = 0.001
    cal, S0 = _const_cal(r)
    g = HistoricalPathGenerator(cal, ("EQ_A",), {"EQ_A": S0}, min_replay_windows=5)
    coarse = g.generate(PathMode.REPLAY_RAW, np.array([0.0, 1.0]))
    fine = g.generate(PathMode.REPLAY_RAW, np.linspace(0.0, 1.0, 11))
    assert np.allclose(coarse[:, -1, 0], fine[:, -1, 0])     # terminal grid-invariant


def test_tail_adequacy_exact_threshold_not_rejected():
    E = np.abs(np.random.default_rng(5).standard_normal((100000, 2)))
    asm = PFEProfileAssembler(confidences_bps=(9999,), m_tail_min=10)   # exactly 10 tail paths
    res = asm.assemble(E, np.array([0., 1.]))
    assert 9999 in res["pfe"]
