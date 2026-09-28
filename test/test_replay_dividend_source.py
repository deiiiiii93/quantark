"""
Tests for the opt-in ``dividend_source`` channel of the replay backtest.

The replay engine has always priced with ONE flat dividend yield implied from
the single active hedge contract (simple compounding, floored at zero).  The
``dividend_source`` knob on ``AutocallableEngineConfig`` lets a run price off
the whole listed futures chain instead (``"futures_curve"``: continuous,
signed, term-structured) or off the option-implied forwards of the admitted
IV-surface artifact (``"surface_forwards"``) while the vol channel stays
scalar.  The default (``None``) is byte-for-byte the historical behaviour.
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

from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.backtest.replay import (
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
    FuturesRollPolicy,
    ReplayBacktestConfig,
    ReplayBacktestEngine,
    ReplayProduct,
)
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.backtest.replay.dividend_source import term_dividend_yield
from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.replay.product_replay import ProductReplay
from quantark.param import FlatRateCurve, FlatVolSurface
from quantark.param.vol.surface_history import VolSurfaceHistory
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError


RATE = 0.02
SPOT = 100.0
VOL = 0.22
# Three listed contracts with distinct implied carry so the term structure is
# unmistakably non-flat: q_i = 3%, 6%, 9% (continuous compounding).
CHAIN = (
    ("IM2401", pd.Timestamp("2024-01-19"), 0.03),
    ("IM2402", pd.Timestamp("2024-02-23"), 0.06),
    ("IM2403", pd.Timestamp("2024-03-15"), 0.09),
)
DATES = pd.date_range("2024-01-02", periods=5, freq="D")
# End-to-end chain: the front contract expires 3 days into the run, inside the
# roll policy's 5-day window, so the ACTIVE hedge contract is IM2402 (q = 6%)
# while the 6-day product sits under the first node (q = 3%).  That is
# exactly the gap between "flat q from the hedge contract" and "q(T)".
E2E_CHAIN = (
    ("IM2401", pd.Timestamp("2024-01-05"), 0.03),
    ("IM2402", pd.Timestamp("2024-02-23"), 0.06),
    ("IM2403", pd.Timestamp("2024-03-15"), 0.09),
)

# Synthetic IV-surface artifact (same shape the vol-history tests use).
STRIKES = [90.0, 100.0, 110.0]
MATURITIES = [0.25, 0.5, 1.0]
ATM_VOLS = [0.18, 0.19, 0.21]
Q_PILLARS = [0.01, 0.015, 0.02]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _ttm(expiry: pd.Timestamp, d: pd.Timestamp) -> float:
    return (expiry - d).days / 365.0


def _futures_rows(dates, chain=CHAIN, spot=SPOT):
    rows = []
    for d in dates:
        for contract, expiry, q in chain:
            t = _ttm(expiry, d)
            rows.append(
                {
                    "date": d,
                    "contract": contract,
                    "futures_price": spot * math.exp((RATE - q) * t),
                    "expiry_date": expiry,
                    "multiplier": 200.0,
                }
            )
    return rows


def _market_data(dates=DATES, chain=CHAIN, surface_history=None):
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": [SPOT] * len(dates)}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [VOL] * len(dates)}),
        rate_data=pd.DataFrame({"date": dates, "rate": [RATE] * len(dates)}),
        futures_data=pd.DataFrame(_futures_rows(dates, chain)),
        surface_history=surface_history,
    )


def _product():
    return create_standard_snowball(
        initial_price=SPOT,
        strike=SPOT,
        maturity=6.0 / 365.0,
        contract_multiplier=100.0,
        ko_barrier=103.0,
        ki_barrier=97.0,
        ko_rate=0.02,
        num_observations=2,
        ko_observation_dates=[2.0 / 365.0, 5.0 / 365.0],
        ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[1.0 / 365.0, 3.0 / 365.0],
        include_principal=True,
    )


def _replay(market_data, engine_config) -> ProductReplay:
    return ProductReplay(
        product=_product(),
        product_quantity=-1.0,
        has_lifecycle=True,
        lifecycle=AutocallableLifecycleState(),
        surface_engine=None,
        event_stats_engine=None,
        engine_config=engine_config,
        market_data=market_data,
        start_date=pd.Timestamp(DATES[0]),
        underlying="CSI1000",
        actions_sink=[],
        event_prob_sink=[],
        daily_event_sink=[],
        surfaces_sink=[],
    )


def _build_env(replay: ProductReplay, dataset, d):
    ts = pd.Timestamp(d)
    market = dataset.get_market_row(ts)
    # the engine's own roll policy: nearest contract that is still live
    selected = FuturesRollPolicy().select_contract(dataset.get_futures_slice(ts), ts)
    return replay.build_env(ts, market, selected)


def _artifact_payload(trade_date: date) -> dict:
    iv_grid = [[atm + 0.05 * (k / SPOT - 1.0) for k in STRIKES] for atm in ATM_VOLS]
    return {
        "trade_date": trade_date.isoformat(),
        "s0": SPOT,
        "strikes": list(STRIKES),
        "maturities": list(MATURITIES),
        "iv_grid": iv_grid,
        "atm_pillars": [
            {
                "T": t,
                "expiry_date": (trade_date + timedelta(days=round(t * 365))).isoformat(),
                "atm_vol": atm,
            }
            for t, atm in zip(MATURITIES, ATM_VOLS)
        ],
        "per_expiry": [
            {
                "T": t,
                "expiry_date": (trade_date + timedelta(days=round(t * 365))).isoformat(),
                "r": RATE,
                "q": q,
                "forward": SPOT * math.exp((RATE - q) * t),
                "df": math.exp(-RATE * t),
            }
            for t, q in zip(MATURITIES, Q_PILLARS)
        ],
        "extrapolation_policy": {
            "beyond_last_listed_expiry": "flat_total_variance",
            "max_listed_T": max(MATURITIES),
        },
        "admission": {"status": "ok"},
    }


@pytest.fixture()
def history_dir(tmp_path) -> Path:
    root = tmp_path / "history"
    (root / "iv_surface").mkdir(parents=True)
    records = []
    for d in DATES:
        payload = _artifact_payload(d.date())
        raw = json.dumps(payload).encode()
        (root / "iv_surface" / f"mo_iv_surface_{d:%Y%m%d}.json").write_bytes(raw)
        records.append(
            {
                "date": f"{d:%Y%m%d}",
                "status": "ok",
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "reason": None,
                "detail": None,
            }
        )
    (root / "surface_manifest.json").write_text(
        json.dumps({"schema_version": 1, "gap_policy": "carry forward", "records": records})
    )
    return root


def _pde_config(**kwargs) -> AutocallableEngineConfig:
    # PDE, like the replay goldens: the QUAD recursion cannot age the 6-day
    # fixture's KO grid day by day.
    return AutocallableEngineConfig(
        pricing_engine_type=EngineType.PDE, pde_params=PDEParams(), **kwargs
    )


def _run(dataset, engine_config, **config_kwargs):
    config = AutocallableBacktestConfig(
        product=_product(),
        market_data=dataset,
        engine_config=engine_config,
        calculate_surfaces=False,
        calculate_event_probabilities=False,
        **config_kwargs,
    )
    return AutocallableBacktestEngine(config).run()


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------
class TestConfig:
    def test_default_dividend_source_is_legacy(self):
        config = AutocallableEngineConfig()
        assert config.dividend_source is None
        assert config.futures_curve_extrapolation == "flat_q"

    def test_unknown_dividend_source_is_rejected(self):
        with pytest.raises(ValidationError):
            AutocallableEngineConfig(dividend_source="front_month")

    def test_unknown_extrapolation_is_rejected(self):
        with pytest.raises(ValidationError):
            AutocallableEngineConfig(
                dividend_source="futures_curve", futures_curve_extrapolation="linear"
            )

    def test_fixed_dividend_yield_conflicts_with_a_term_source(self):
        dataset = _market_data()
        with pytest.raises(ValidationError):
            AutocallableBacktestConfig(
                product=_product(),
                market_data=dataset,
                engine_config=AutocallableEngineConfig(dividend_source="futures_curve"),
                fixed_dividend_yield=0.01,
            )
        with pytest.raises(ValidationError):
            ReplayBacktestConfig(
                products=[ReplayProduct(_product(), -1.0, 0, True)],
                market_data=dataset,
                engine_config=AutocallableEngineConfig(dividend_source="futures_curve"),
                fixed_dividend_yield=0.01,
            )

    def test_surface_grid_conflicts_with_a_term_source(self):
        # record_surfaces re-prices on a FLAT q grid around a scalar centre;
        # under a term-structured dividend that silently swaps the model.
        with pytest.raises(ValidationError):
            AutocallableBacktestConfig(
                product=_product(),
                market_data=_market_data(),
                engine_config=AutocallableEngineConfig(dividend_source="futures_curve"),
                calculate_surfaces=True,
            )


# ---------------------------------------------------------------------------
# build_env
# ---------------------------------------------------------------------------
class TestBuildEnv:
    def test_legacy_env_is_flat_from_the_active_contract(self):
        dataset = _market_data()
        for source in (None, "active_contract"):
            replay = _replay(dataset, AutocallableEngineConfig(dividend_source=source))
            env, basis_yield, implied_q, _ = _build_env(replay, dataset, DATES[0])
            assert isinstance(env.div_yield, SignedDividendYield)
            # simple-compounded, floored: q = max(0, r - (F - S)/S/T)
            contract, expiry, _ = CHAIN[0]
            t = _ttm(expiry, DATES[0])
            f = SPOT * math.exp((RATE - 0.03) * t)
            expected = max(0.0, RATE - (f - SPOT) / SPOT / t)
            assert implied_q == pytest.approx(expected)
            assert env.div_yield.get_yield(0.5) == pytest.approx(expected)

    def test_futures_curve_reprices_every_listed_contract(self):
        dataset = _market_data()
        replay = _replay(dataset, AutocallableEngineConfig(dividend_source="futures_curve"))
        env, basis_yield, implied_q, _ = _build_env(replay, dataset, DATES[0])
        for contract, expiry, q in CHAIN:
            t = _ttm(expiry, DATES[0])
            assert env.div_yield.get_yield(t) == pytest.approx(q, abs=1e-12)
            model_forward = SPOT * math.exp((RATE - env.div_yield.get_yield(t)) * t)
            market_forward = SPOT * math.exp((RATE - q) * t)
            assert model_forward == pytest.approx(market_forward, rel=1e-12)
        # the legacy active-contract record is still produced for the state row
        assert implied_q > 0.0
        assert isinstance(env.vol_surface, FlatVolSurface)

    def test_futures_curve_extrapolates_flat_q_by_default(self):
        dataset = _market_data()
        replay = _replay(dataset, AutocallableEngineConfig(dividend_source="futures_curve"))
        env, *_ = _build_env(replay, dataset, DATES[0])
        assert env.div_yield.get_yield(2.0) == pytest.approx(CHAIN[-1][2], abs=1e-12)
        assert env.div_yield.get_yield(1e-4) == pytest.approx(CHAIN[0][2], abs=1e-12)

    def test_futures_curve_flat_forward_carry_continues_the_last_segment(self):
        dataset = _market_data()
        replay = _replay(
            dataset,
            AutocallableEngineConfig(
                dividend_source="futures_curve",
                futures_curve_extrapolation="flat_forward_carry",
            ),
        )
        env, *_ = _build_env(replay, dataset, DATES[0])
        d = DATES[0]
        t2, t3 = _ttm(CHAIN[1][1], d), _ttm(CHAIN[2][1], d)
        b2, b3 = (RATE - CHAIN[1][2]) * t2, (RATE - CHAIN[2][2]) * t3
        slope = (b3 - b2) / (t3 - t2)
        horizon = 2.0
        b_h = b3 + slope * (horizon - t3)
        expected_q = RATE - b_h / horizon
        assert env.div_yield.get_yield(horizon) == pytest.approx(expected_q, abs=1e-12)
        assert expected_q != pytest.approx(CHAIN[-1][2], abs=1e-6)
        # array evaluation must agree with the scalar path (grid samplers)
        arr = env.div_yield.get_yield(np.array([t2, t3, horizon]))
        assert arr[0] == pytest.approx(CHAIN[1][2], abs=1e-12)
        assert arr[2] == pytest.approx(expected_q, abs=1e-12)

    def test_one_eligible_contract_is_a_flat_continuous_signed_yield(self):
        dataset = _market_data(chain=CHAIN[:1])
        for extrapolation in ("flat_q", "flat_forward_carry"):
            replay = _replay(
                dataset,
                AutocallableEngineConfig(
                    dividend_source="futures_curve",
                    futures_curve_extrapolation=extrapolation,
                ),
            )
            env, *_ = _build_env(replay, dataset, DATES[0])
            for t in (0.01, 0.5, 2.0):
                assert env.div_yield.get_yield(t) == pytest.approx(CHAIN[0][2], abs=1e-12)

    def test_no_eligible_contract_fails_closed(self):
        near = ("IM2312", pd.Timestamp("2024-01-04"), 0.5)
        dataset = _market_data(chain=(near,))
        replay = _replay(dataset, AutocallableEngineConfig(dividend_source="futures_curve"))
        with pytest.raises(ValidationError):
            _build_env(replay, dataset, DATES[0])

    def test_futures_curve_drops_contracts_expiring_today(self):
        expiring_today = ("IM2312", pd.Timestamp(DATES[0]), 0.0)
        dataset = _market_data(chain=(expiring_today,) + CHAIN)
        replay = _replay(dataset, AutocallableEngineConfig(dividend_source="futures_curve"))
        env, *_ = _build_env(replay, dataset, DATES[0])
        t1 = _ttm(CHAIN[0][1], DATES[0])
        assert env.div_yield.get_yield(t1) == pytest.approx(CHAIN[0][2], abs=1e-12)

    def test_surface_forwards_in_scalar_vol_mode(self, history_dir):
        dataset = _market_data(surface_history=VolSurfaceHistory(history_dir))
        replay = _replay(dataset, AutocallableEngineConfig(dividend_source="surface_forwards"))
        env, *_ = _build_env(replay, dataset, DATES[0])
        assert isinstance(env.vol_surface, FlatVolSurface)
        assert env.vol_surface.get_vol(0.0, 0.5, 0.0) == pytest.approx(VOL)
        for t, q in zip(MATURITIES, Q_PILLARS):
            assert env.div_yield.get_yield(t) == pytest.approx(q, abs=1e-9)
        assert replay.last_surface_provenance is not None

    def test_surface_forwards_requires_a_surface_history(self):
        dataset = _market_data()
        replay = _replay(dataset, AutocallableEngineConfig(dividend_source="surface_forwards"))
        with pytest.raises(ValidationError):
            _build_env(replay, dataset, DATES[0])


# ---------------------------------------------------------------------------
# End-to-end replay
# ---------------------------------------------------------------------------
class TestReplayRecords:
    def test_explicit_active_contract_is_identical_to_the_default(self):
        dataset = _market_data(chain=E2E_CHAIN)
        legacy = _run(dataset, _pde_config()).states_df
        explicit = _run(dataset, _pde_config(dividend_source="active_contract")).states_df
        pd.testing.assert_frame_equal(legacy, explicit)

    def test_term_source_changes_the_price_and_records_q_at_remaining_maturity(self):
        dataset = _market_data(chain=E2E_CHAIN)
        legacy = _run(dataset, _pde_config()).states_df
        # min tenor 1 day: keep the 3-day IM2401 as a curve node (the default
        # one-week minimum would drop it, which is the point of that default)
        term = _run(
            dataset,
            _pde_config(dividend_source="futures_curve", futures_curve_min_tenor_days=1),
        ).states_df
        # the active contract is IM2402 (the 3-day IM2401 is inside the roll
        # window); implied_q keeps that legacy meaning in BOTH runs
        assert legacy["active_contract"].iloc[0] == "IM2402"
        assert legacy["pricing_q"].iloc[0] == pytest.approx(0.06, abs=2e-4)
        assert term["implied_q"].iloc[0] == pytest.approx(legacy["implied_q"].iloc[0])
        # pricing_q is the term structure sampled at the product's remaining
        # maturity: 6 days, between the 3-day and 52-day nodes -> linear in q
        d0 = DATES[0]
        nodes_t = [_ttm(expiry, d0) for _, expiry, _ in E2E_CHAIN]
        nodes_q = [q for _, _, q in E2E_CHAIN]
        expected = float(np.interp(6.0 / 365.0, nodes_t, nodes_q))
        assert 0.03 < expected < 0.06
        assert term["pricing_q"].iloc[0] == pytest.approx(expected, abs=1e-12)
        # 3% carry over the product's life vs 6%: a real price difference
        assert abs(term["product_mtm"].iloc[0] - legacy["product_mtm"].iloc[0]) > 0.1
        # once IM2401 has expired the curve starts at IM2402 and q(T) = 6%
        after_expiry = term.index >= pd.Timestamp("2024-01-05")
        assert term.loc[after_expiry, "pricing_q"].iloc[0] == pytest.approx(0.06, abs=1e-12)

    def test_term_source_survives_a_spot_hedge(self, history_dir):
        dataset = _market_data(surface_history=VolSurfaceHistory(history_dir))
        config = ReplayBacktestConfig(
            products=[ReplayProduct(_product(), -1.0, 0, True)],
            market_data=dataset,
            engine_config=_pde_config(dividend_source="surface_forwards"),
            calculate_surfaces=False,
            calculate_event_probabilities=False,
        )
        config.hedge.kind = "spot"
        states = ReplayBacktestEngine(config).run().states_df()
        # remaining maturity (6 days) sits below the first pillar -> flat 1%
        assert states["pricing_q"].iloc[0] == pytest.approx(Q_PILLARS[0], abs=1e-9)


# ---------------------------------------------------------------------------
# Minimum tenor: delivery-week contracts carry no measurable annualised carry
# ---------------------------------------------------------------------------
class TestMinimumTenor:
    def test_default_minimum_tenor_is_one_week(self):
        assert AutocallableEngineConfig().futures_curve_min_tenor_days == 7

    @pytest.mark.parametrize("days", [-1, 0])
    def test_minimum_tenor_below_one_day_is_rejected(self, days):
        # a contract expiring today has T = 0 and therefore no yield
        with pytest.raises(ValidationError):
            AutocallableEngineConfig(
                dividend_source="futures_curve", futures_curve_min_tenor_days=days
            )

    def test_contract_inside_the_minimum_tenor_is_dropped(self):
        # 2 days to expiry with a 1% basis: an annualised yield near 180%
        near = ("IM2312", pd.Timestamp("2024-01-04"), 1.8)
        dataset = _market_data(chain=(near,) + CHAIN)
        replay = _replay(dataset, AutocallableEngineConfig(dividend_source="futures_curve"))
        env, *_ = _build_env(replay, dataset, DATES[0])
        t1 = _ttm(CHAIN[0][1], DATES[0])
        assert env.div_yield.get_yield(t1) == pytest.approx(CHAIN[0][2], abs=1e-12)
        assert env.div_yield.get_yield(1e-3) == pytest.approx(CHAIN[0][2], abs=1e-12)

    def test_minimum_tenor_of_one_day_keeps_a_one_day_contract(self):
        # 1 day out: dropped by the default (7) and kept by min tenor 1
        near = ("IM2312", pd.Timestamp("2024-01-03"), 0.5)
        dataset = _market_data(chain=(near,) + CHAIN)
        default = _replay(dataset, AutocallableEngineConfig(dividend_source="futures_curve"))
        env_default, *_ = _build_env(default, dataset, DATES[0])
        assert env_default.div_yield.get_yield(1.0 / 365.0) == pytest.approx(CHAIN[0][2], abs=1e-12)
        one_day = _replay(
            dataset,
            AutocallableEngineConfig(
                dividend_source="futures_curve", futures_curve_min_tenor_days=1
            ),
        )
        env_one, *_ = _build_env(one_day, dataset, DATES[0])
        assert env_one.div_yield.get_yield(1.0 / 365.0) == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# futures_curve_extrapolation="surface_forward_carry": IM chain in log-forward
# space, option-implied forward carry beyond the last listed contract
# ---------------------------------------------------------------------------
class TestSurfaceForwardCarryTail:
    def test_config_accepts_the_option(self):
        cfg = AutocallableEngineConfig(
            dividend_source="futures_curve", futures_curve_extrapolation="surface_forward_carry"
        )
        assert cfg.uses_term_dividend_source()

    def test_interior_is_the_chain_in_log_forward_space(self, history_dir):
        dataset = _market_data(surface_history=VolSurfaceHistory(history_dir))
        replay = _replay(
            dataset,
            AutocallableEngineConfig(
                dividend_source="futures_curve", futures_curve_extrapolation="surface_forward_carry"
            ),
        )
        env, *_ = _build_env(replay, dataset, DATES[0])
        d = DATES[0]
        # every listed contract repriced exactly, and between nodes the carry
        # is linear in log-forward (identical to the flat_forward_carry interior)
        for contract, expiry, q in CHAIN:
            t = _ttm(expiry, d)
            assert env.div_yield.get_yield(t) == pytest.approx(q, abs=1e-12)
        ref = _replay(
            dataset,
            AutocallableEngineConfig(dividend_source="futures_curve", futures_curve_extrapolation="flat_forward_carry"),
        )
        ref_env, *_ = _build_env(ref, dataset, d)
        t_mid = 0.5 * (_ttm(CHAIN[0][1], d) + _ttm(CHAIN[1][1], d))
        assert env.div_yield.get_yield(t_mid) == pytest.approx(ref_env.div_yield.get_yield(t_mid), abs=1e-12)
        assert replay.last_surface_provenance is not None

    def test_tail_carries_the_option_forward_carry_not_its_level(self, history_dir):
        dataset = _market_data(surface_history=VolSurfaceHistory(history_dir))
        replay = _replay(
            dataset,
            AutocallableEngineConfig(
                dividend_source="futures_curve", futures_curve_extrapolation="surface_forward_carry"
            ),
        )
        env, *_ = _build_env(replay, dataset, DATES[0])
        d = DATES[0]
        t_last = _ttm(CHAIN[-1][1], d)  # ~0.2y, inside the option pillars (0.25, 0.5, 1.0)
        q_last = CHAIN[-1][2]

        def carry(t):
            return (RATE - env.div_yield.get_yield(t)) * t

        def opt_carry(t):
            return float(np.interp(t, [0.0] + MATURITIES, [0.0] + [(RATE - q) * t_ for t_, q in zip(MATURITIES, Q_PILLARS)]))

        b_last = (RATE - q_last) * t_last
        for t in (0.5, 1.0):
            assert carry(t) == pytest.approx(b_last + opt_carry(t) - opt_carry(t_last), abs=1e-12)
        # beyond the last option pillar the last option segment's forward carry continues
        slope = (opt_carry(1.0) - opt_carry(0.5)) / 0.5
        assert carry(2.0) == pytest.approx(carry(1.0) + slope * 1.0, abs=1e-12)
        # the option surface's LEVEL (q = 1%..2%) never enters: the chain's 9% node is kept
        assert env.div_yield.get_yield(t_last) == pytest.approx(q_last, abs=1e-12)

    def test_requires_a_surface_history(self):
        dataset = _market_data()
        replay = _replay(
            dataset,
            AutocallableEngineConfig(
                dividend_source="futures_curve", futures_curve_extrapolation="surface_forward_carry"
            ),
        )
        with pytest.raises(ValidationError):
            _build_env(replay, dataset, DATES[0])

    def test_one_eligible_contract_still_gets_the_option_tail(self, history_dir):
        # the chain's last two contracts are dropped by a 60-day minimum tenor
        # on 2024-01-02 (IM2402 expires in 52 days), leaving IM2401... which is
        # also inside; use 20 days: IM2401 (17d) dropped, IM2402 (52d) and IM2403 kept.
        # A 55-day minimum keeps IM2403 alone: the one-node limit.
        dataset = _market_data(surface_history=VolSurfaceHistory(history_dir))
        replay = _replay(
            dataset,
            AutocallableEngineConfig(
                dividend_source="futures_curve", futures_curve_extrapolation="surface_forward_carry",
                futures_curve_min_tenor_days=55,
            ),
        )
        env, *_ = _build_env(replay, dataset, DATES[0])
        d = DATES[0]
        t3, q3 = _ttm(CHAIN[2][1], d), CHAIN[2][2]
        assert env.div_yield.get_yield(t3) == pytest.approx(q3, abs=1e-12)
        # not a flat yield: the tail follows the option forward carry
        assert env.div_yield.get_yield(1.0) != pytest.approx(q3, abs=1e-6)


class TestSharedDividendRule:
    """The extracted ``term_dividend_yield`` is the engine's own rule."""

    def test_term_dividend_yield_equals_the_engine_for_every_extrapolation(self, history_dir):
        dataset = _market_data(surface_history=VolSurfaceHistory(history_dir))
        d = DATES[0]
        quotes = [
            IndexFuturesQuote(contract=c, maturity=_ttm(e, d), price=SPOT * math.exp((RATE - q) * _ttm(e, d)),
                              multiplier=200.0, expiry_date=e.to_pydatetime())
            for c, e, q in CHAIN
        ]
        artifact = VolSurfaceHistory(history_dir).surface_for(d)
        for extrapolation in ("flat_q", "flat_forward_carry", "surface_forward_carry"):
            replay = _replay(dataset, AutocallableEngineConfig(
                dividend_source="futures_curve", futures_curve_extrapolation=extrapolation,
                futures_curve_min_tenor_days=1))
            env, *_ = _build_env(replay, dataset, d)
            shared = term_dividend_yield(quotes, spot=SPOT, rate_curve=FlatRateCurve(rate=RATE),
                                         extrapolation=extrapolation, underlying="CSI1000", artifact=artifact)
            for t in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
                assert shared.get_yield(t) == pytest.approx(env.div_yield.get_yield(t), abs=1e-15)

    def test_surface_forward_carry_without_an_artifact_fails_closed(self):
        quotes = [IndexFuturesQuote(contract="IM2403", maturity=0.2, price=SPOT * 0.99, multiplier=200.0)]
        with pytest.raises(ValidationError):
            term_dividend_yield(quotes, spot=SPOT, rate_curve=FlatRateCurve(rate=RATE),
                                extrapolation="surface_forward_carry")
        with pytest.raises(ValidationError):
            term_dividend_yield(quotes, spot=SPOT, rate_curve=FlatRateCurve(rate=RATE), extrapolation="cubic")


class TestCarryContextAttachment:
    """``last_carry_context`` is the day's scenario source, or nothing."""

    def _supported_replay(self, dataset, extrapolation):
        return _replay(
            dataset,
            AutocallableEngineConfig(
                dividend_source="futures_curve",
                futures_curve_extrapolation=extrapolation,
                futures_curve_min_tenor_days=1,
            ),
        )

    @pytest.mark.parametrize("extrapolation", ("flat_q", "flat_forward_carry"))
    def test_supported_conventions_keep_the_day_context(self, extrapolation):
        dataset = _market_data()
        replay = self._supported_replay(dataset, extrapolation)
        assert replay.last_carry_context is None
        env, *_ = _build_env(replay, dataset, DATES[0])
        context = replay.last_carry_context
        assert context is not None
        assert context.extrapolation == extrapolation
        assert context.spot == pytest.approx(SPOT)
        assert context.underlying == "CSI1000"
        assert context.contracts == tuple(c for c, _, _ in CHAIN)
        # The pricing dividend IS the context's dividend, not a second build.
        for t in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            assert context.dividend().get_yield(t) == pytest.approx(
                env.div_yield.get_yield(t), abs=1e-15
            )

    def test_the_context_carries_the_eligible_quotes_only(self):
        dataset = _market_data()
        replay = _replay(
            dataset,
            AutocallableEngineConfig(
                dividend_source="futures_curve",
                futures_curve_extrapolation="flat_q",
                futures_curve_min_tenor_days=40,
            ),
        )
        _build_env(replay, dataset, DATES[0])
        # IM2401 expires in 17 calendar days on 2024-01-02 and drops out;
        # the other two (52 and 73 days) stay.
        assert replay.last_carry_context.contracts == ("IM2402", "IM2403")

    def test_the_surface_tail_convention_keeps_no_context(self, history_dir):
        dataset = _market_data(surface_history=VolSurfaceHistory(history_dir))
        replay = self._supported_replay(dataset, "surface_forward_carry")
        env, *_ = _build_env(replay, dataset, DATES[0])
        assert env.div_yield is not None
        assert replay.last_carry_context is None

    def test_a_flat_carry_source_keeps_no_context(self):
        dataset = _market_data()
        replay = _replay(dataset, AutocallableEngineConfig())
        _build_env(replay, dataset, DATES[0])
        assert replay.last_carry_context is None

    def test_yesterdays_context_never_leaks_into_a_new_build(self):
        dataset = _market_data()
        replay = _replay(dataset, AutocallableEngineConfig())
        stale = self._supported_replay(dataset, "flat_q")
        _build_env(stale, dataset, DATES[0])
        replay.last_carry_context = stale.last_carry_context
        _build_env(replay, dataset, DATES[1])
        assert replay.last_carry_context is None

    def test_each_day_rebuilds_its_own_coordinates(self):
        dataset = _market_data()
        replay = self._supported_replay(dataset, "flat_q")
        _build_env(replay, dataset, DATES[0])
        first = replay.last_carry_context
        _build_env(replay, dataset, DATES[3])
        second = replay.last_carry_context
        assert second is not first
        assert second.coordinates() != first.coordinates()
        assert second.valuation_date == pd.Timestamp(DATES[3])
