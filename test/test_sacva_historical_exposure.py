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


from datetime import datetime

# real quant-ark objects for engine tests
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

# canonical exposure contract (post-merge) + portfolio model
from quantark.sacva.exposure.engine import Measure, ExposureProfile
from quantark.sacva.portfolio.trade import CVATrade
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.models.enums import CreditQuality

VAL = datetime(2026, 6, 17)
EXP_1Y = datetime(2027, 6, 17)


# ---------------------------------------------------------------------------
# Canonical ExposureProfile — additive historical fields + invariants (spec §5)
# ---------------------------------------------------------------------------


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


def test_real_world_profile_is_never_eligible():
    # a REAL_WORLD profile may carry epe_discounted (MC rejected-fixture pattern) but
    # can never be regulatory_eligible, so it can never feed the capital path.
    p = ExposureProfile(np.array([0., 1.]), np.array([5., 3.]), Measure.REAL_WORLD, False)
    assert not p.regulatory_eligible


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


def test_historical_profile_ok():
    p = ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([2., 1.]),
                        pfe={9900: np.array([4., 3.])}, epe=1.5)
    assert not p.regulatory_eligible and p.pfe[9900][0] == 4.0 and p.epe == 1.5


def test_pfe_and_epe_validation():
    with pytest.raises(ValidationError):           # negative PFE
        ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([1., 1.]), pfe={9900: np.array([-1., 2.])})
    with pytest.raises(ValidationError):           # bad bps key
        ExposureProfile(np.array([0., 1.]), None, Measure.REAL_WORLD, False,
                        ee_undiscounted=np.array([1., 1.]), pfe={12000: np.array([1., 2.])})


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
# Task 8 — HistoricalExposureEngine (real quant-ark objects)
# ---------------------------------------------------------------------------
from quantark.sacva.exposure.historical.engine import (
    HistoricalExposureEngine, HistoricalExposureConfig,
)


class _FlatCreditCurve:
    def __init__(self, hazard=0.02, recovery=0.4):
        self.h, self.R = hazard, recovery

    def get_survival_probability(self, t):
        return float(np.exp(-self.h * t))

    @property
    def recovery_rate(self):
        return self.R


def _env(spot=100.0, vol=0.2, rate=0.0, asset="ACME"):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate), valuation_date=VAL,
        spot_quote=SpotQuote(spot=spot, asset_name=asset),
        vol_surface=FlatVolSurface(vol), div_yield=ContinuousDividendYield(0.0),
        day_count_convention=DayCountConvention.CALENDAR_DAYS)


def _call_trade(trade_id="t1", strike=100.0, qty=1.0, env=None):
    env = env if env is not None else _env()
    opt = EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL,
                                exercise_date=EXP_1Y)
    return CVATrade(trade_id=trade_id, product=opt, engine=BlackScholesEngine(),
                    env=env, quantity=qty, trade_currency="USD")


def _counterparty(netting_sets):
    return Counterparty(name="CP", netting_sets=netting_sets,
                        credit_curve=_FlatCreditCurve(), bucket=2,
                        credit_quality=CreditQuality.IG)


def _hist_cal(asset="ACME", spot=100.0, sigma_d=0.01, n=2000, seed=3):
    """Historical calibration for ``asset`` whose last level == ``spot`` exactly (so
    today-level reconciliation passes) and whose log-returns are ~N(0, sigma_d**2)."""
    rng = np.random.default_rng(seed)
    rel = np.exp(np.cumsum(rng.normal(0.0, sigma_d, n)))
    lvl = spot * rel / rel[-1]                      # scale to last==spot; log-returns unchanged
    s = pd.Series(lvl, index=pd.bdate_range(end="2026-06-17", periods=n))
    return HistoricalCalibration(HistoricalMarketDataSet({asset: s}))


def _cfg(**kw):
    base = dict(path_mode="BOOTSTRAP", scheme="IID_RAW", n_paths=4000, seed=5,
                n_steps=4, drift_modes={"ACME": "ZERO_LOG_MEAN"}, confidences_bps=(9900,))
    base.update(kw)
    return HistoricalExposureConfig(**base)


def _Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def test_engine_call_terminal_EE_matches_lognormal():
    # terminal node prices the contractual payoff max(S_T-K,0); under IID_RAW the
    # terminal law is ~lognormal, so undiscounted terminal EE matches the closed form.
    sigma_d, S0, K = 0.01, 100.0, 100.0
    cal = _hist_cal("ACME", S0, sigma_d, n=2000, seed=3)
    trade = _call_trade("c1", strike=K, env=_env(spot=S0, rate=0.0, asset="ACME"))
    prof = HistoricalExposureEngine(cal, _cfg(n_paths=20000)).compute(
        _counterparty([NettingSet("n1", [trade])]))
    v = 252 * float(np.var(cal._r["ACME"].to_numpy(), ddof=1))
    a = log(S0); d1 = (a - log(K) + v) / sqrt(v); d2 = d1 - sqrt(v)
    ee_cf = exp(a + 0.5 * v) * _Phi(d1) - K * _Phi(d2)
    assert abs(prof.ee_undiscounted[-1] - ee_cf) / ee_cf < 0.06
    assert prof.measure is Measure.REAL_WORLD and not prof.regulatory_eligible
    assert prof.epe_discounted is None and prof.metadata["path_mode"] == "BOOTSTRAP"
    assert np.all(prof.pfe[9900] >= 0)


def test_engine_deterministic():
    cal = _hist_cal("ACME", 100.0, seed=3)
    cp = _counterparty([NettingSet("n1", [_call_trade("c1", env=_env(asset="ACME"))])])
    p1 = HistoricalExposureEngine(cal, _cfg(n_paths=3000)).compute(cp)
    p2 = HistoricalExposureEngine(cal, _cfg(n_paths=3000)).compute(cp)
    assert np.allclose(p1.ee_undiscounted, p2.ee_undiscounted)        # seed determinism
    assert p1.ee_undiscounted[-1] > 0                                 # long call has positive EE


def test_engine_netting_le_gross():
    cal = _hist_cal("ACME", 100.0, seed=3)
    env = _env(spot=100.0, asset="ACME")
    trades = [_call_trade("L", 100.0, 1.0, env), _call_trade("S", 100.0, -1.0, env)]
    eng = HistoricalExposureEngine(cal, _cfg(n_paths=3000))
    pn = eng.compute(_counterparty([NettingSet("n1", trades, netting_enforceable=True)]))
    pg = eng.compute(_counterparty([NettingSet("n1", trades, netting_enforceable=False)]))
    assert np.all(pn.ee_undiscounted <= pg.ee_undiscounted + 1e-9)
    assert np.allclose(pn.ee_undiscounted, 0.0)                       # perfect offset nets to 0


def test_engine_rejects_today_level_mismatch():
    cal = _hist_cal("ACME", 100.0, seed=3)
    cp = _counterparty([NettingSet("n1", [_call_trade("c1", env=_env(spot=110.0, asset="ACME"))])])
    with pytest.raises(ValidationError):                             # env spot 110 != calib 100
        HistoricalExposureEngine(cal, _cfg()).compute(cp)


def test_engine_rejects_missing_factor_and_bad_config():
    cal = _hist_cal("ACME", 100.0, seed=3)
    cp = _counterparty([NettingSet("n1", [_call_trade("c1", env=_env(spot=100.0, asset="OTHER"))])])
    with pytest.raises(ValidationError):                             # factor not in calibration
        HistoricalExposureEngine(cal, _cfg(drift_modes={"OTHER": "ZERO_LOG_MEAN"})).compute(cp)
    with pytest.raises(ValidationError):                             # bad scheme
        HistoricalExposureConfig(path_mode="BOOTSTRAP", scheme="NOPE", n_paths=100,
                                 drift_modes={"ACME": "ZERO_LOG_MEAN"})


# ---------------------------------------------------------------------------
# Task 9 — Regulatory guard + merge gate (the keystone)
# ---------------------------------------------------------------------------
from quantark.sacva.cva.engine import RegulatoryCVAEngine


def test_historical_profile_rejected_by_capital_path():
    # the keystone: the real MC-owned RegulatoryCVAEngine must reject a real-world
    # (historical) profile so it can never fund SA-CVA capital (MAR50.34(1)).
    cal = _hist_cal("ACME", 100.0, seed=3)
    cp = _counterparty([NettingSet("n1", [_call_trade("c1", env=_env(asset="ACME"))])])
    prof = HistoricalExposureEngine(cal, _cfg(n_paths=2000)).compute(cp)
    assert prof.measure is Measure.REAL_WORLD and not prof.regulatory_eligible
    with pytest.raises(ValidationError):
        RegulatoryCVAEngine().compute(_FlatCreditCurve(), prof)


def test_capital_path_accepts_risk_neutral_eligible():
    rn = ExposureProfile(np.array([0., 1.]), np.array([5., 3.]), Measure.RISK_NEUTRAL, True)
    # a risk-neutral, regulatory-eligible profile is consumed without error
    assert RegulatoryCVAEngine().compute(_FlatCreditCurve(), rn) >= 0.0


def test_merge_gate_no_provisional_artifacts():
    # the provisional contract AND the provisional regulatory-guard stub are gone.
    import importlib, inspect, pkgutil
    import quantark.sacva.exposure.historical as hist
    names = []
    for mi in pkgutil.walk_packages(hist.__path__, hist.__name__ + "."):
        names.append(mi.name)
        src = inspect.getsource(importlib.import_module(mi.name))
        assert "_contract_provisional" not in src, \
            f"{mi.name} still imports the provisional contract"
    assert not any(n.endswith(".regulatory_guard") for n in names), \
        "provisional regulatory_guard stub must be deleted at merge"


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
