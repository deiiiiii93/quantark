"""
Tests for the per-day IV-surface history channel (vol_history) and the
surface-based pricing-environment modes in ``ProductReplay.build_env``.

Covers:
- ``IvSurfaceArtifact`` schema validation (fail-closed) and typed accessors.
- ``VolSurfaceHistory`` carry-forward over manifest-excluded dates.
- ``AutocallableMarketDataSet.surface_history`` backward compatibility.
- ``build_env`` vol modes: flat_atm_remaining / term_structure / full_grid,
  the parity-forward dividend term structure, and surface provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.backtest.otc import (
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
)
from quantark.backtest.otc._replay import ProductReplay
from quantark.backtest.otc.market import SignedDividendYield
from quantark.backtest.otc.state import AutocallableLifecycleState
from quantark.backtest.otc.vol_history import IvSurfaceArtifact, VolSurfaceHistory
from quantark.param import (
    FlatVolSurface,
    GridVolSurface,
    TermStructureDividendYield,
    TermStructureVolSurface,
)
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close


RATE = 0.02
STRIKES = [90.0, 100.0, 110.0]
MATURITIES = [0.25, 0.5, 1.0]
ATM_VOLS_A = [0.18, 0.19, 0.21]
ATM_VOLS_B = [0.22, 0.23, 0.25]
Q_PILLARS = [0.01, 0.015, 0.02]

DATE_A = date(2024, 1, 2)
DATE_EXCLUDED = date(2024, 1, 3)
DATE_B = date(2024, 1, 4)

REAL_HISTORY_DIR = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "mo_volmodels"
    / "data"
    / "history"
)


# ---------------------------------------------------------------------------
# Synthetic artifact / history builders
# ---------------------------------------------------------------------------
def _artifact_payload(trade_date: date, *, s0: float, atm_vols) -> dict:
    """Small deterministic artifact: linear skew iv(T, K) = atm(T) + 0.05*(K/s0 - 1)."""
    iv_grid = [
        [atm + 0.05 * (k / s0 - 1.0) for k in STRIKES] for atm in atm_vols
    ]
    atm_pillars = [
        {
            "T": t,
            "expiry_date": (trade_date + timedelta(days=round(t * 365))).isoformat(),
            "atm_vol": atm,
        }
        for t, atm in zip(MATURITIES, atm_vols)
    ]
    per_expiry = [
        {
            "T": t,
            "expiry_date": (trade_date + timedelta(days=round(t * 365))).isoformat(),
            "r": RATE,
            "q": q,
            "forward": s0 * math.exp((RATE - q) * t),
            "df": math.exp(-RATE * t),
        }
        for t, q in zip(MATURITIES, Q_PILLARS)
    ]
    return {
        "trade_date": trade_date.isoformat(),
        "s0": s0,
        "strikes": list(STRIKES),
        "maturities": list(MATURITIES),
        "iv_grid": iv_grid,
        "atm_pillars": atm_pillars,
        "per_expiry": per_expiry,
        "extrapolation_policy": {
            "beyond_last_listed_expiry": "flat_total_variance",
            "max_listed_T": max(MATURITIES),
        },
        "admission": {"status": "ok"},
    }


def _write_history(
    root: Path,
    artifacts: dict,
    excluded=(),
) -> Path:
    """Write a history dir; an artifact payload of None means an 'ok' record
    whose artifact file is missing (fail-closed test case)."""
    history_dir = root / "history"
    surface_dir = history_dir / "iv_surface"
    surface_dir.mkdir(parents=True)
    records = []
    for d, payload in artifacts.items():
        sha = None
        if payload is not None:
            raw = json.dumps(payload).encode()
            (surface_dir / f"mo_iv_surface_{d:%Y%m%d}.json").write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
        records.append(
            {
                "date": f"{d:%Y%m%d}",
                "status": "ok",
                "artifact_sha256": sha,
                "reason": None,
                "detail": None,
            }
        )
    for d in excluded:
        records.append(
            {
                "date": f"{d:%Y%m%d}",
                "status": "excluded",
                "artifact_sha256": None,
                "reason": "static_arbitrage",
                "detail": "synthetic exclusion",
            }
        )
    records.sort(key=lambda rec: rec["date"])
    manifest = {
        "schema_version": 1,
        "gap_policy": "consumers carry forward previous admitted surface",
        "records": records,
    }
    (history_dir / "surface_manifest.json").write_text(json.dumps(manifest))
    return history_dir


@pytest.fixture()
def history_dir(tmp_path) -> Path:
    return _write_history(
        tmp_path,
        {
            DATE_A: _artifact_payload(DATE_A, s0=100.0, atm_vols=ATM_VOLS_A),
            DATE_B: _artifact_payload(DATE_B, s0=101.0, atm_vols=ATM_VOLS_B),
        },
        excluded=(DATE_EXCLUDED,),
    )


def _artifact_sha(history_dir: Path, d: date) -> str:
    raw = (history_dir / "iv_surface" / f"mo_iv_surface_{d:%Y%m%d}.json").read_bytes()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# VolSurfaceHistory loader
# ---------------------------------------------------------------------------
class TestVolSurfaceHistory:
    def test_admitted_dates_exclude_excluded_records(self, history_dir):
        history = VolSurfaceHistory(history_dir)
        assert history.admitted_dates == [DATE_A, DATE_B]

    def test_carry_forward_picks_previous_admitted_date(self, history_dir):
        history = VolSurfaceHistory(history_dir)
        carried = history.surface_for(DATE_EXCLUDED)
        assert carried.trade_date == DATE_A
        assert is_close(carried.s0, 100.0)
        # Later dates carry the most recent admitted surface forward.
        assert history.surface_for(date(2024, 1, 10)).trade_date == DATE_B
        # Exact admitted dates return their own artifact.
        assert history.surface_for(DATE_B).trade_date == DATE_B
        # pandas Timestamps are accepted (engine passes them).
        assert history.surface_for(pd.Timestamp(DATE_A)).trade_date == DATE_A

    def test_error_before_first_admitted_date(self, history_dir):
        history = VolSurfaceHistory(history_dir)
        with pytest.raises(ValidationError, match="No admitted IV surface"):
            history.surface_for(date(2024, 1, 1))

    def test_excluded_records_never_surface(self, history_dir):
        history = VolSurfaceHistory(history_dir)
        for d in pd.date_range(DATE_A, periods=10, freq="D"):
            assert history.surface_for(d).trade_date != DATE_EXCLUDED

    def test_sha_for_matches_file_bytes(self, history_dir):
        history = VolSurfaceHistory(history_dir)
        assert history.sha_for(DATE_A) == _artifact_sha(history_dir, DATE_A)
        assert history.sha_for(DATE_EXCLUDED) == _artifact_sha(history_dir, DATE_A)
        assert history.surface_for(DATE_A).sha256 == _artifact_sha(history_dir, DATE_A)

    def test_missing_artifact_for_ok_record_fails_closed(self, tmp_path):
        history_dir = _write_history(tmp_path, {DATE_A: None})
        history = VolSurfaceHistory(history_dir)
        with pytest.raises(ValidationError, match="artifact"):
            history.surface_for(DATE_A)

    def test_broken_artifact_schema_fails_closed(self, tmp_path):
        broken = _artifact_payload(DATE_A, s0=100.0, atm_vols=ATM_VOLS_A)
        broken["iv_grid"] = [[0.2] * len(STRIKES)]  # wrong shape: 1 x 3 vs 3 x 3
        history_dir = _write_history(tmp_path, {DATE_A: broken})
        history = VolSurfaceHistory(history_dir)
        with pytest.raises(ValidationError, match="iv_grid"):
            history.surface_for(DATE_A)

    def test_missing_required_key_fails_closed(self, tmp_path):
        broken = _artifact_payload(DATE_A, s0=100.0, atm_vols=ATM_VOLS_A)
        del broken["atm_pillars"]
        history_dir = _write_history(tmp_path, {DATE_A: broken})
        history = VolSurfaceHistory(history_dir)
        with pytest.raises(ValidationError, match="atm_pillars"):
            history.surface_for(DATE_A)

    def test_artifact_accessors(self, history_dir):
        artifact = VolSurfaceHistory(history_dir).surface_for(DATE_A)
        assert artifact.trade_date == DATE_A
        assert list(artifact.strikes) == STRIKES
        assert list(artifact.maturities) == MATURITIES
        expected_grid = _artifact_payload(DATE_A, s0=100.0, atm_vols=ATM_VOLS_A)[
            "iv_grid"
        ]
        for actual_row, expected_row in zip(artifact.iv_grid, expected_grid):
            for actual, expected in zip(actual_row, expected_row):
                assert is_close(actual, expected)
        assert [p["T"] for p in artifact.atm_pillars] == MATURITIES
        assert [p["atm_vol"] for p in artifact.atm_pillars] == ATM_VOLS_A
        assert [p["T"] for p in artifact.per_expiry] == MATURITIES
        assert is_close(artifact.max_listed_T, max(MATURITIES))
        assert (
            artifact.extrapolation_policy["beyond_last_listed_expiry"]
            == "flat_total_variance"
        )


# ---------------------------------------------------------------------------
# Market dataset integration
# ---------------------------------------------------------------------------
def _market_frames(dates):
    spot_data = pd.DataFrame({"date": dates, "spot": [100.0] * len(dates)})
    vol_data = pd.DataFrame({"date": dates, "volatility": [0.20] * len(dates)})
    rate_data = pd.DataFrame({"date": dates, "rate": [RATE] * len(dates)})
    futures_rows = [
        {
            "date": d,
            "contract": "IF2406",
            "futures_price": 100.0 * 1.001,
            "expiry_date": pd.Timestamp("2024-06-21"),
            "multiplier": 300.0,
        }
        for d in dates
    ]
    return spot_data, vol_data, rate_data, pd.DataFrame(futures_rows)


def _market_data(dates, surface_history=None) -> AutocallableMarketDataSet:
    spot_data, vol_data, rate_data, futures_data = _market_frames(dates)
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=spot_data,
        vol_data=vol_data,
        rate_data=rate_data,
        futures_data=futures_data,
        surface_history=surface_history,
    )


class TestMarketDataSetSurfaceHistory:
    def test_backward_compatible_without_surface_history(self):
        dates = pd.date_range("2024-01-02", periods=5, freq="D")
        dataset = _market_data(dates)
        assert dataset.surface_history is None
        assert len(dataset.dates) == 5

    def test_surface_history_attaches_without_changing_dates(self, history_dir):
        dates = pd.date_range("2024-01-02", periods=5, freq="D")
        history = VolSurfaceHistory(history_dir)
        with_history = _market_data(dates, surface_history=history)
        without_history = _market_data(dates)
        assert with_history.surface_history is history
        assert with_history.dates.equals(without_history.dates)


# ---------------------------------------------------------------------------
# build_env surface modes
# ---------------------------------------------------------------------------
def _product(maturity: float):
    return create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=maturity,
        contract_multiplier=100.0,
        ko_barrier=103.0,
        ki_barrier=97.0,
        ko_rate=0.02,
        num_observations=2,
        ko_observation_dates=[maturity / 2.0, maturity],
        ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[maturity / 2.0],
        include_principal=True,
    )


def _make_replay(market_data, engine_config, product, start_date) -> ProductReplay:
    return ProductReplay(
        product=product,
        product_quantity=-1.0,
        has_lifecycle=True,
        lifecycle=AutocallableLifecycleState(),
        # pricing engines now flow explicitly from the engine (Task 10);
        # ProductReplay no longer holds one.
        surface_engine=None,
        event_stats_engine=None,
        engine_config=engine_config,
        market_data=market_data,
        start_date=pd.Timestamp(start_date),
        underlying="CSI500",
        actions_sink=[],
        event_prob_sink=[],
        daily_event_sink=[],
        surfaces_sink=[],
    )


def _build_env(replay: ProductReplay, dataset: AutocallableMarketDataSet, d: date):
    ts = pd.Timestamp(d)
    market = dataset.get_market_row(ts)
    selected = dataset.get_futures_slice(ts).sort_values("expiry_date").iloc[0]
    return replay.build_env(ts, market, selected)


def _surface_engine_config(mode: str) -> AutocallableEngineConfig:
    return AutocallableEngineConfig(vol_source="surface", surface_vol_mode=mode)


class TestBuildEnvSurfaceModes:
    def test_scalar_mode_is_unchanged_default(self, history_dir):
        dates = pd.date_range(DATE_A, periods=3, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        replay = _make_replay(dataset, AutocallableEngineConfig(), _product(0.75), DATE_A)

        env, basis_yield, implied_q, futures_ttm = _build_env(replay, dataset, DATE_A)

        assert isinstance(env.vol_surface, FlatVolSurface)
        assert is_close(env.vol_surface.get_vol(90.0, 0.5, 100.0), 0.20)
        assert isinstance(env.div_yield, SignedDividendYield)
        assert replay.last_surface_provenance is None

    def test_surface_mode_requires_surface_history(self):
        dates = pd.date_range(DATE_A, periods=3, freq="D")
        dataset = _market_data(dates)  # no surface_history attached
        replay = _make_replay(
            dataset, _surface_engine_config("term_structure"), _product(0.75), DATE_A
        )
        with pytest.raises(ValidationError, match="surface_history"):
            _build_env(replay, dataset, DATE_A)

    def test_flat_atm_remaining_interpolates_between_pillars(self, history_dir):
        dates = pd.date_range(DATE_A, periods=3, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        config = _surface_engine_config("flat_atm_remaining")
        replay = _make_replay(dataset, config, _product(0.75), DATE_A)

        env, *_ = _build_env(replay, dataset, DATE_A)

        # Remaining maturity is exactly 0.75y on the start date: between the
        # 0.5y and 1.0y ATM pillars, so total-variance interpolation applies.
        assert isinstance(env.vol_surface, FlatVolSurface)
        total_variances = [v * v * t for v, t in zip(ATM_VOLS_A, MATURITIES)]
        expected = math.sqrt(np.interp(0.75, MATURITIES, total_variances) / 0.75)
        sampled = env.vol_surface.get_vol(90.0, 3.0, 100.0)
        assert is_close(sampled, expected)
        assert min(ATM_VOLS_A[1:]) < sampled < max(ATM_VOLS_A[1:])

    def test_flat_atm_remaining_hits_pillar_exactly(self, history_dir):
        dates = pd.date_range(DATE_A, periods=3, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        config = _surface_engine_config("flat_atm_remaining")
        replay = _make_replay(dataset, config, _product(0.5), DATE_A)

        env, *_ = _build_env(replay, dataset, DATE_A)

        assert is_close(env.vol_surface.get_vol(90.0, 0.1, 100.0), ATM_VOLS_A[1])

    def test_flat_atm_remaining_refreshes_daily(self, history_dir):
        dates = pd.date_range(DATE_A, periods=3, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        config = _surface_engine_config("flat_atm_remaining")
        replay = _make_replay(dataset, config, _product(0.75), DATE_A)

        env_b, *_ = _build_env(replay, dataset, DATE_B)

        # Two days elapsed: remaining = 0.75 - 2/365, sampled on artifact B.
        remaining = 0.75 - 2.0 / 365.0
        total_variances = [v * v * t for v, t in zip(ATM_VOLS_B, MATURITIES)]
        expected = math.sqrt(np.interp(remaining, MATURITIES, total_variances) / remaining)
        assert is_close(env_b.vol_surface.get_vol(90.0, 3.0, 100.0), expected)

    def test_term_structure_mode_samples_pillar_vols(self, history_dir):
        dates = pd.date_range(DATE_A, periods=3, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        replay = _make_replay(
            dataset, _surface_engine_config("term_structure"), _product(0.75), DATE_A
        )

        env, *_ = _build_env(replay, dataset, DATE_A)

        assert isinstance(env.vol_surface, TermStructureVolSurface)
        for t, atm in zip(MATURITIES, ATM_VOLS_A):
            # Strike is ignored by the term-structure surface.
            assert is_close(env.vol_surface.get_vol(90.0, t, 100.0), atm)
            assert is_close(env.vol_surface.get_vol(110.0, t, 100.0), atm)

    def test_full_grid_mode_reproduces_grid_nodes(self, history_dir):
        dates = pd.date_range(DATE_A, periods=3, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        replay = _make_replay(
            dataset, _surface_engine_config("full_grid"), _product(0.75), DATE_A
        )

        env, *_ = _build_env(replay, dataset, DATE_A)

        assert isinstance(env.vol_surface, GridVolSurface)
        payload = _artifact_payload(DATE_A, s0=100.0, atm_vols=ATM_VOLS_A)
        for i, t in enumerate(MATURITIES):
            for j, k in enumerate(STRIKES):
                assert is_close(
                    env.vol_surface.get_vol(k, t, 100.0), payload["iv_grid"][i][j]
                )

    def test_dividend_curve_from_parity_forwards(self, history_dir):
        dates = pd.date_range(DATE_A, periods=3, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        replay = _make_replay(
            dataset, _surface_engine_config("term_structure"), _product(0.75), DATE_A
        )

        env, basis_yield, implied_q, _ = _build_env(replay, dataset, DATE_A)

        assert isinstance(env.div_yield, TermStructureDividendYield)
        payload = _artifact_payload(DATE_A, s0=100.0, atm_vols=ATM_VOLS_A)
        for row in payload["per_expiry"]:
            t = row["T"]
            expected_q = RATE - math.log(row["forward"] / payload["s0"]) / t
            assert is_close(env.div_yield.get_yield(t), expected_q)
            assert is_close(env.div_yield.get_yield(t), row["q"])
        # TermStructureDividendYield flat-extrapolates beyond the last pillar.
        assert is_close(env.div_yield.get_yield(5.0), Q_PILLARS[-1])
        # Rate stays on the existing flat channel; basis still futures-implied.
        assert is_close(env.rate_curve.get_rate(0.7), RATE)
        assert is_close(env.basis_yield.get_basis_yield(1.0), basis_yield)
        assert is_close(implied_q, max(0.0, RATE - basis_yield))

    def test_surface_provenance_records_artifact_used(self, history_dir):
        dates = pd.date_range(DATE_A, periods=3, freq="D")
        history = VolSurfaceHistory(history_dir)
        dataset = _market_data(dates, surface_history=history)
        replay = _make_replay(
            dataset, _surface_engine_config("term_structure"), _product(0.75), DATE_A
        )

        _build_env(replay, dataset, DATE_EXCLUDED)  # excluded -> carries DATE_A

        provenance = replay.last_surface_provenance
        assert provenance is not None
        assert provenance["surface_date"] == DATE_A.isoformat()
        assert provenance["surface_sha"] == _artifact_sha(history_dir, DATE_A)
        assert provenance["surface_extrapolation"] == "flat_total_variance"
        assert is_close(provenance["surface_max_listed_T"], max(MATURITIES))

    def test_engine_config_validates_vol_source_and_mode(self):
        with pytest.raises(ValidationError, match="vol_source"):
            AutocallableEngineConfig(vol_source="bogus")
        with pytest.raises(ValidationError, match="surface_vol_mode"):
            AutocallableEngineConfig(
                vol_source="surface", surface_vol_mode="bogus"
            )
        config = AutocallableEngineConfig()
        assert config.vol_source == "scalar"
        assert config.surface_vol_mode == "flat_atm_remaining"


# ---------------------------------------------------------------------------
# Engine-level smoke: provenance columns appear only in surface mode
# ---------------------------------------------------------------------------
def _run_engine(dataset, engine_config) -> "object":
    config = AutocallableBacktestConfig(
        product=_product(0.75),
        market_data=dataset,
        engine_config=engine_config,
        calculate_surfaces=False,
        calculate_event_probabilities=False,
        product_quantity=-1.0,
        underlying="CSI500",
    )
    return AutocallableBacktestEngine(config).run()


class TestEngineSurfaceSmoke:
    def _mc_config(self, **overrides) -> AutocallableEngineConfig:
        return AutocallableEngineConfig(
            pricing_engine_type=EngineType.MONTE_CARLO,
            mc_params=MCParams(num_paths=64, time_steps=8, seed=7),
            quad_params=QuadParams(grid_points=101, num_std_devs=4.0),
            **overrides,
        )

    def test_surface_mode_records_provenance_columns(self, history_dir):
        dates = pd.date_range(DATE_A, periods=4, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        results = _run_engine(
            dataset, self._mc_config(vol_source="surface")
        )
        states = results.states_df
        assert {"surface_date", "surface_sha"}.issubset(states.columns)
        assert list(states["surface_date"]) == [
            DATE_A.isoformat(),
            DATE_A.isoformat(),  # excluded date carries forward
            DATE_B.isoformat(),
            DATE_B.isoformat(),
        ]

    def test_scalar_mode_has_no_provenance_columns(self, history_dir):
        dates = pd.date_range(DATE_A, periods=4, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        results = _run_engine(dataset, self._mc_config())
        assert "surface_date" not in results.states_df.columns
        assert "surface_sha" not in results.states_df.columns

    def test_surface_mode_changes_daily_vol_input(self, history_dir):
        dates = pd.date_range(DATE_A, periods=4, freq="D")
        dataset = _market_data(dates, surface_history=VolSurfaceHistory(history_dir))
        scalar = _run_engine(dataset, self._mc_config())
        surface = _run_engine(dataset, self._mc_config(vol_source="surface"))
        # ATM-at-remaining vol (~0.2035) differs from the 0.20 scalar channel,
        # so surface-mode prices must move off the scalar path.
        assert not is_close(
            float(surface.greeks_df["price"].iloc[0]),
            float(scalar.greeks_df["price"].iloc[0]),
        )


# ---------------------------------------------------------------------------
# Real-artifact smoke case (one admitted date from the built 3y history)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (REAL_HISTORY_DIR / "surface_manifest.json").exists(),
    reason="real IV surface history not present",
)
class TestRealArtifactSmoke:
    def test_real_artifact_loads_and_matches_manifest(self):
        history = VolSurfaceHistory(REAL_HISTORY_DIR)
        artifact = history.surface_for(date(2023, 5, 4))

        assert artifact.trade_date == date(2023, 5, 4)
        manifest = json.loads(
            (REAL_HISTORY_DIR / "surface_manifest.json").read_text()
        )
        record = next(r for r in manifest["records"] if r["date"] == "20230504")
        assert artifact.sha256 == record["artifact_sha256"]

        ts = artifact.term_structure_vol_surface()
        for pillar in artifact.atm_pillars:
            assert is_close(
                ts.get_vol(0.0, pillar["T"], 0.0), pillar["atm_vol"]
            )
        grid = artifact.grid_vol_surface()
        assert is_close(
            grid.get_vol(
                artifact.strikes[3], artifact.maturities[2], artifact.s0
            ),
            artifact.iv_grid[2][3],
        )
        assert is_close(artifact.max_listed_T, max(artifact.maturities))

    def test_real_carry_forward_over_excluded_date(self):
        history = VolSurfaceHistory(REAL_HISTORY_DIR)
        # 20231227 is excluded (static arbitrage) in the real manifest.
        carried = history.surface_for(date(2023, 12, 27))
        assert carried.trade_date < date(2023, 12, 27)
        assert carried.trade_date == max(
            d for d in history.admitted_dates if d <= date(2023, 12, 27)
        )
