"""
Tests for the shared helpers of ``example/snowball_q_term_structure``.

The study compares a snowball hedged under a flat dividend yield (one active
futures contract) with the same snowball hedged under the IM chain's implied
q term structure.  These tests pin the pieces the conclusions rest on: the
far-contract roll policy, the static dividend construction (which must be the
very object the replay engine builds), the term sheet, the inception
schedule, the fair-coupon solver, and the paired hedge measures.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.replay import AutocallableMarketDataSet, FuturesRollPolicy
from quantark.backtest.replay.product_replay import ProductReplay
from quantark.util.exceptions import ValidationError

REPO = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO / "example" / "snowball_q_term_structure"


def _load_common():
    path = STUDY_DIR / "_common.py"
    spec = importlib.util.spec_from_file_location("q_term_structure_common", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = _load_common()

RATE = 0.02
SPOT = 7000.0
CHAIN = (
    ("IM2401", pd.Timestamp("2024-01-19"), 0.03),
    ("IM2402", pd.Timestamp("2024-02-23"), 0.06),
    ("IM2403", pd.Timestamp("2024-03-15"), 0.09),
    ("IM2406", pd.Timestamp("2024-06-21"), 0.12),
)


def _ttm(expiry: pd.Timestamp, d: pd.Timestamp) -> float:
    return (expiry - d).days / 365.0


def _chain_frame(dates, chain=CHAIN, spot=SPOT) -> pd.DataFrame:
    rows = []
    for d in dates:
        for contract, expiry, q in chain:
            t = _ttm(expiry, d)
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "contract": contract,
                    "futures_price": spot * math.exp((RATE - q) * t),
                    "expiry_date": expiry,
                    "multiplier": 200.0,
                }
            )
    return pd.DataFrame(rows)


def _dataset(dates, surface_history=None) -> AutocallableMarketDataSet:
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": [SPOT] * len(dates)}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [0.22] * len(dates)}),
        rate_data=pd.DataFrame({"date": dates, "rate": [RATE] * len(dates)}),
        futures_data=_chain_frame(dates),
        surface_history=surface_history,
    )


SURFACE_MATURITIES = [0.25, 0.5, 1.0]
SURFACE_Q = [0.01, 0.015, 0.02]


def _surface_history(dates):
    """A minimal admitted MO-surface history (same shape as the engine tests use)."""
    import hashlib
    import json
    import tempfile
    from datetime import timedelta

    from quantark.param.vol.surface_history import VolSurfaceHistory

    root = Path(tempfile.mkdtemp(prefix="q_term_surface_")) / "history"
    (root / "iv_surface").mkdir(parents=True)
    records = []
    strikes = [90.0, 100.0, 110.0]
    for d in dates:
        td = pd.Timestamp(d).date()
        payload = {
            "trade_date": td.isoformat(),
            "s0": SPOT,
            "strikes": strikes,
            "maturities": SURFACE_MATURITIES,
            "iv_grid": [[0.2 + 0.05 * (k / SPOT - 1.0) for k in strikes] for _ in SURFACE_MATURITIES],
            "atm_pillars": [
                {"T": t, "expiry_date": (td + timedelta(days=round(t * 365))).isoformat(), "atm_vol": 0.2}
                for t in SURFACE_MATURITIES
            ],
            "per_expiry": [
                {"T": t, "expiry_date": (td + timedelta(days=round(t * 365))).isoformat(), "r": RATE, "q": q,
                 "forward": SPOT * math.exp((RATE - q) * t), "df": math.exp(-RATE * t)}
                for t, q in zip(SURFACE_MATURITIES, SURFACE_Q)
            ],
            "extrapolation_policy": {"beyond_last_listed_expiry": "flat_total_variance", "max_listed_T": 1.0},
            "admission": {"status": "ok"},
        }
        raw = json.dumps(payload).encode()
        (root / "iv_surface" / f"mo_iv_surface_{pd.Timestamp(d):%Y%m%d}.json").write_bytes(raw)
        records.append({"date": f"{pd.Timestamp(d):%Y%m%d}", "status": "ok",
                        "artifact_sha256": hashlib.sha256(raw).hexdigest(), "reason": None, "detail": None})
    (root / "surface_manifest.json").write_text(
        json.dumps({"schema_version": 1, "gap_policy": "carry forward", "records": records})
    )
    return VolSurfaceHistory(root)


# ---------------------------------------------------------------------------
# Hedge contract policies
# ---------------------------------------------------------------------------
class TestFarContractRollPolicy:
    def test_fresh_selection_is_the_longest_live_contract(self):
        dates = pd.date_range("2024-01-02", periods=1, freq="D")
        chain = _chain_frame(dates)
        policy = common.FarContractRollPolicy(roll_days_before_expiry=5)
        row = policy.select_contract(chain, dates[0], None)
        assert row["contract"] == "IM2406"

    def test_holding_is_sticky_until_the_roll_window(self):
        policy = common.FarContractRollPolicy(roll_days_before_expiry=5)
        # far from expiry: keep the current contract even though a longer one
        # could be listed later
        d = pd.Timestamp("2024-03-01")
        chain = _chain_frame([d])
        row = policy.select_contract(chain, d, "IM2403")
        assert row["contract"] == "IM2403"
        # inside the window: move to the longest live contract
        d = pd.Timestamp("2024-03-12")
        chain = _chain_frame([d])
        row = policy.select_contract(chain, d, "IM2403")
        assert row["contract"] == "IM2406"

    def test_no_live_contract_fails_closed(self):
        policy = common.FarContractRollPolicy()
        d = pd.Timestamp("2024-07-01")
        with pytest.raises(ValidationError):
            policy.select_contract(_chain_frame([d]), d, None)

    def test_front_policy_is_the_engine_default(self):
        assert isinstance(common.HEDGE_POLICIES["front"](), FuturesRollPolicy)
        assert not isinstance(
            common.HEDGE_POLICIES["front"](), common.FarContractRollPolicy
        )
        assert isinstance(common.HEDGE_POLICIES["far"](), common.FarContractRollPolicy)


# ---------------------------------------------------------------------------
# Static dividend construction == the engine's own build_env
# ---------------------------------------------------------------------------
def _engine_env(dataset, model, d):
    replay = ProductReplay(
        product=common.build_product(
            common.build_terms(date(2024, 1, 2), common.TradingCalendar(
                [date(2024, 1, 2) + pd.Timedelta(days=i) for i in range(400)]
            )),
            s0=SPOT,
            coupon=0.1,
        ),
        product_quantity=-1.0,
        has_lifecycle=True,
        lifecycle=AutocallableLifecycleState(),
        surface_engine=None,
        event_stats_engine=None,
        engine_config=common.engine_config_for(model),
        market_data=dataset,
        start_date=pd.Timestamp(d),
        underlying=common.UNDERLYING_NAME,
        actions_sink=[],
        event_prob_sink=[],
        daily_event_sink=[],
        surfaces_sink=[],
    )
    ts = pd.Timestamp(d)
    market = dataset.get_market_row(ts)
    selected = FuturesRollPolicy().select_contract(dataset.get_futures_slice(ts), ts)
    env, *_ = replay.build_env(ts, market, selected)
    return env, selected


@pytest.mark.parametrize("model_name", ["flat_active", "term_flat_q", "term_flat_fwd", "term_opt_tail"])
def test_static_dividend_matches_the_engine(model_name):
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    dataset = _dataset(dates)
    model = common.Q_MODELS[model_name]
    if model.needs_surface:
        dataset = _dataset(dates, surface_history=_surface_history(dates))
    env, selected = _engine_env(dataset, model, dates[0])
    static = common.dividend_for(
        model,
        valuation=dates[0],
        spot=SPOT,
        rate=RATE,
        chain_slice=dataset.get_futures_slice(dates[0]),
        active_row=selected,
        artifact=dataset.surface_history.surface_for(dates[0]) if model.needs_surface else None,
    )
    for t in (0.01, 0.1, 0.3, 0.5, 1.0, 2.0):
        assert static.get_yield(t) == pytest.approx(env.div_yield.get_yield(t), abs=1e-14)


def test_term_opt_tail_keeps_the_front_contract_and_takes_the_option_tail():
    """The fifth model: min tenor 1 day, log-forward interior, option-forward tail."""
    model = common.Q_MODELS["term_opt_tail"]
    assert model.min_tenor_days == 1
    assert common.engine_config_for(model).futures_curve_min_tenor_days == 1
    assert common.engine_config_for(model).futures_curve_extrapolation == "surface_forward_carry"
    dates = pd.date_range("2024-03-12", periods=1, freq="D")  # IM2403 expires 2024-03-15: 3 days out
    dataset = _dataset(dates, surface_history=_surface_history(dates))
    chain = dataset.get_futures_slice(dates[0])
    div = common.dividend_for(
        model, valuation=dates[0], spot=SPOT, rate=RATE, chain_slice=chain,
        artifact=dataset.surface_history.surface_for(dates[0]),
    )
    # the 3-day contract (IM2403) is kept and repriced exactly (bounded in log-forward space)
    front = common.live_chain(chain, dates[0], 1).sort_values("expiry_date").iloc[0]
    assert front["contract"] == "IM2403"
    t_front = (pd.Timestamp(front["expiry_date"]) - dates[0]).days / 365.0
    assert t_front == pytest.approx(3 / 365)
    f_model = SPOT * math.exp((RATE - div.get_yield(t_front)) * t_front)
    assert f_model == pytest.approx(float(front["futures_price"]), rel=1e-12)
    # the 7-day model drops it
    assert common.curve_quotes(chain, dates[0], 7)[0].contract != front["contract"]
    # beyond the last listed contract the forward carry is the option surface's, not flat
    ref = common.dividend_for(common.Q_MODELS["term_flat_fwd"], valuation=dates[0], spot=SPOT, rate=RATE, chain_slice=chain)
    assert div.get_yield(1.5) != pytest.approx(ref.get_yield(1.5), abs=1e-6)


def test_flat_active_is_floored_and_the_curve_is_signed():
    d = pd.Timestamp("2024-01-02")
    contango = (
        ("IM2401", pd.Timestamp("2024-01-19"), -0.04),
        ("IM2402", pd.Timestamp("2024-02-23"), -0.01),
    )
    chain = _chain_frame([d], chain=contango)
    active = FuturesRollPolicy().select_contract(chain, d)
    flat = common.dividend_for(
        common.Q_MODELS["flat_active"], valuation=d, spot=SPOT, rate=RATE,
        chain_slice=chain, active_row=active,
    )
    term = common.dividend_for(
        common.Q_MODELS["term_flat_q"], valuation=d, spot=SPOT, rate=RATE,
        chain_slice=chain,
    )
    assert flat.get_yield(0.1) == 0.0
    assert term.get_yield(_ttm(contango[0][1], d)) == pytest.approx(-0.04, abs=1e-12)


# ---------------------------------------------------------------------------
# Forward pricing error: the curve reprices every node, the flat model does not
# ---------------------------------------------------------------------------
def test_forward_pricing_error_by_contract():
    d = pd.Timestamp("2024-01-02")
    chain = _chain_frame([d])
    active = FuturesRollPolicy().select_contract(chain, d)
    term = common.dividend_for(
        common.Q_MODELS["term_flat_q"], valuation=d, spot=SPOT, rate=RATE, chain_slice=chain
    )
    flat = common.dividend_for(
        common.Q_MODELS["flat_active"], valuation=d, spot=SPOT, rate=RATE,
        chain_slice=chain, active_row=active,
    )
    err_term = common.forward_pricing_error_bp(term, valuation=d, spot=SPOT, rate=RATE, chain_slice=chain)
    err_flat = common.forward_pricing_error_bp(flat, valuation=d, spot=SPOT, rate=RATE, chain_slice=chain)
    assert set(err_term) == {c for c, _, _ in CHAIN}
    assert all(abs(v) < 1e-8 for v in err_term.values())
    # the active contract is repriced up to compounding; the far ones are not
    assert abs(err_flat["IM2401"]) < 5.0
    assert abs(err_flat["IM2406"]) > 50.0
    assert common.rms(err_flat.values()) > common.rms(err_term.values())


# ---------------------------------------------------------------------------
# Term sheet and inception schedule
# ---------------------------------------------------------------------------
def _weekday_calendar(start: date, days: int) -> "common.TradingCalendar":
    all_days = [start + pd.Timedelta(days=i) for i in range(days)]
    return common.TradingCalendar([d for d in all_days if d.weekday() < 5])


def test_terms_monthly_ko_from_lockout_with_daily_ki():
    calendar = _weekday_calendar(date(2023, 5, 1), 800)
    terms = common.build_terms(date(2023, 6, 1), calendar)
    assert len(terms.ko_times) == common.MATURITY_MONTHS - common.LOCKOUT_MONTHS + 1
    assert terms.ko_times[-1] == pytest.approx(terms.maturity_years)
    assert terms.maturity_date == calendar.next_trading_day(date(2024, 6, 1))
    # first KO on the first trading day on/after the 3-month anniversary
    first = date(2023, 6, 1) + pd.Timedelta(days=round(terms.ko_times[0] * 365))
    assert first == calendar.next_trading_day(date(2023, 9, 1))
    # KI every trading day in (inception, maturity]
    expected_ki = calendar.trading_days_between(date(2023, 6, 1), terms.maturity_date)
    assert len(terms.ki_times) == len(expected_ki)
    assert terms.ki_times[-1] == pytest.approx(terms.maturity_years)


def test_product_sizes_one_unit_to_the_notional():
    calendar = _weekday_calendar(date(2023, 5, 1), 800)
    terms = common.build_terms(date(2023, 6, 1), calendar)
    product = common.build_product(terms, s0=6500.0, coupon=0.11)
    assert product.contract_multiplier == pytest.approx(common.NOTIONAL / 6500.0)
    assert product.initial_price == 6500.0
    assert product.barrier_config.ko_barrier == pytest.approx(common.KO_PCT * 6500.0)
    assert product.barrier_config.ki_barrier == pytest.approx(common.KI_PCT * 6500.0)


def test_enumerate_inceptions_marks_censored_runs():
    calendar = _weekday_calendar(date(2023, 5, 1), 600)  # ~ 1.6 years
    inceptions = common.enumerate_inceptions(calendar, first_month=(2023, 5))
    assert inceptions[0].inception == calendar.first
    assert all(i.inception.day <= 3 for i in inceptions)
    uncensored = [i for i in inceptions if not i.censored]
    censored = [i for i in inceptions if i.censored]
    assert uncensored and censored
    assert all(i.maturity_date <= calendar.last for i in uncensored)
    assert all(i.maturity_date > calendar.last for i in censored)


# ---------------------------------------------------------------------------
# Fair coupon
# ---------------------------------------------------------------------------
def test_fair_coupon_solves_an_affine_pv_in_one_step():
    calls = []

    def pv(coupon):
        calls.append(coupon)
        return -3.0e6 + 2.4e7 * coupon  # root at 0.125

    solution = common.solve_fair_coupon(pv, notional=5.0e7, lower=0.0, upper=0.4)
    assert solution.coupon == pytest.approx(0.125, abs=1e-9)
    assert solution.converged
    # bracket (2) + one false-position step evaluated and checked
    assert len(calls) <= 4


def test_fair_coupon_fails_closed_without_a_bracket():
    with pytest.raises(ValidationError):
        common.solve_fair_coupon(lambda c: 1.0 + c, notional=5.0e7, lower=0.0, upper=0.4)


# ---------------------------------------------------------------------------
# Paired hedge measures
# ---------------------------------------------------------------------------
def _states(total, product, active, pricing_q, mtm):
    n = len(total)
    return pd.DataFrame(
        {
            "total_pnl": total,
            "product_pnl": product,
            "active_contract": active,
            "pricing_q": pricing_q,
            "product_mtm": mtm,
            "transaction_costs": [0.0] * n,
        },
        index=pd.date_range("2024-01-02", periods=n, freq="D"),
    )


def test_hedge_measures_on_a_synthetic_path():
    notional = 1.0e6
    total = [0.0, 100.0, 50.0, 150.0, 120.0]
    product = [0.0, 1000.0, -500.0, 1500.0, 1200.0]
    active = ["A", "A", "B", "B", "B"]
    q = [0.05, 0.05, 0.11, 0.11, 0.11]
    mtm = [-1000.0, -900.0, -1500.0, -1400.0, -1450.0]
    trades = pd.DataFrame(
        {
            "trade_type": ["hedge_rebalance", "roll_close", "roll_open", "hedge_rebalance"],
            "quantity": [10.0, -10.0, 10.0, -2.0],
            "price": [7000.0, 7000.0, 6900.0, 6900.0],
            "multiplier": [200.0] * 4,
            "notional": [1.4e7, 1.4e7, 1.38e7, 2.76e6],
            "transaction_cost": [0.0] * 4,
        }
    )
    m = common.hedge_measures(_states(total, product, active, q, mtm), trades, notional=notional)
    assert m["terminal_pnl_bp"] == pytest.approx(120.0 / notional * 1e4)
    d_total = np.diff(total)
    d_product = np.diff(product)
    assert m["daily_pnl_std_bp"] == pytest.approx(np.std(d_total, ddof=1) / notional * 1e4)
    assert m["variance_reduction_r2"] == pytest.approx(
        1.0 - np.var(d_total, ddof=1) / np.var(d_product, ddof=1)
    )
    # peak 100 at day 2 -> trough 50 is the deepest fall (150 -> 120 is only 30)
    assert m["max_drawdown_bp"] == pytest.approx(50.0 / notional * 1e4)
    assert m["turnover"] == pytest.approx((1.4e7 + 1.4e7 + 1.38e7 + 2.76e6) / notional)
    assert m["rebalance_turnover"] == pytest.approx((1.4e7 + 2.76e6) / notional)
    assert m["roll_turnover"] == pytest.approx((1.4e7 + 1.38e7) / notional)
    # one roll day (index 2): |dMTM| = 600 vs the other days (100, 100, 50)
    assert m["roll_days"] == 1
    assert m["roll_day_mtm_jump_bp"] == pytest.approx(600.0 / notional * 1e4)
    assert m["other_day_mtm_jump_bp"] == pytest.approx(np.mean([100.0, 100.0, 50.0]) / notional * 1e4)
    assert m["roll_day_q_jump"] == pytest.approx(0.06)
    assert m["days"] == 5


def test_hedge_measures_are_nan_safe_on_a_one_day_run():
    m = common.hedge_measures(
        _states([0.0], [0.0], ["A"], [0.05], [-1000.0]),
        pd.DataFrame(columns=["trade_type", "quantity", "price", "multiplier", "notional", "transaction_cost"]),
        notional=1.0e6,
    )
    assert m["days"] == 1
    assert m["terminal_pnl_bp"] == 0.0
    assert math.isnan(m["daily_pnl_std_bp"])
    assert math.isnan(m["variance_reduction_r2"])
    assert m["turnover"] == 0.0


def test_static_curve_applies_the_engine_minimum_tenor():
    near = ("IM2312", pd.Timestamp("2024-01-04"), 1.8)  # 2 days out, 180% annualised
    dates = pd.date_range("2024-01-02", periods=1, freq="D")
    dataset = AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": [SPOT]}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [0.22]}),
        rate_data=pd.DataFrame({"date": dates, "rate": [RATE]}),
        futures_data=_chain_frame(dates, chain=(near,) + CHAIN),
    )
    model = common.Q_MODELS["term_flat_q"]
    env, _ = _engine_env(dataset, model, dates[0])
    static = common.dividend_for(
        model, valuation=dates[0], spot=SPOT, rate=RATE,
        chain_slice=dataset.get_futures_slice(dates[0]),
    )
    for t in (0.001, 0.05, 0.3, 1.0):
        assert static.get_yield(t) == pytest.approx(env.div_yield.get_yield(t), abs=1e-14)
    assert static.get_yield(0.001) == pytest.approx(CHAIN[0][2], abs=1e-12)


def test_run_io_round_trip_keeps_trade_dates(tmp_path):
    dates = pd.date_range("2024-01-02", periods=3, freq="D")

    class _Results:
        states_df = pd.DataFrame({"total_pnl": [0.0, 1.0, 2.0]}, index=pd.Index(dates, name="date"))
        greeks_df = pd.DataFrame({"delta": [1.0, 1.0, 1.0]}, index=pd.Index(dates, name="date"))
        rebalance_df = pd.DataFrame({"target_contracts": [1.0, 1.0, 1.0]}, index=pd.Index(dates, name="date"))
        trades_df = pd.DataFrame(
            {"trade_type": ["hedge_rebalance", "roll_close"], "quantity": [1.0, -1.0], "notional": [1.0, 1.0]},
            index=pd.Index([dates[0], dates[2]], name="date"),
        )
        actions_df = pd.DataFrame({"action_type": ["KO"]}, index=pd.Index([dates[2]], name="date"))

    run_dir = tmp_path / "run"
    common.write_run(run_dir, _Results(), {"notional": 1.0})
    back = common.load_run(run_dir)
    assert list(back["trades"].index) == [dates[0], dates[2]]
    assert list(back["actions"].index) == [dates[2]]
    assert list(back["states"].index) == list(dates)
    assert back["summary"]["notional"] == 1.0


def test_fair_coupon_expands_the_bracket_when_the_root_is_above_upper():
    # root at 60%: above the default upper bound, inside the expansion cap
    solution = common.solve_fair_coupon(
        lambda c: -3.0e7 + 5.0e7 * c, notional=5.0e7, lower=0.0, upper=0.4
    )
    assert solution.coupon == pytest.approx(0.6, abs=1e-9)
    assert solution.converged


def test_fair_coupon_fails_closed_beyond_the_expansion_cap():
    # root at 1000%: no expansion reaches it
    with pytest.raises(ValidationError):
        common.solve_fair_coupon(lambda c: -1.0e7 + 1.0e6 * c, notional=5.0e7, lower=0.0, upper=0.4)


# ---------------------------------------------------------------------------
# Stage 03 pure aggregation logic
# ---------------------------------------------------------------------------
def _load_stage03():
    path = STUDY_DIR / "03_aggregate_and_report.py"
    spec = importlib.util.spec_from_file_location("q_term_structure_stage03", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(inception, model, hedge, **measures):
    base = {k: 0.0 for k, _, _ in _load_stage03().MEASURES}
    base.update(measures)
    return {
        "inception": inception, "model": model, "hedge": hedge,
        "cell": common.cell_name(model, hedge), "termination_reason": "ko",
        "days_replayed": 100, "knocked_in": False, **base,
    }


def test_default_pairs_compare_every_cell_to_the_baseline_and_far_to_front():
    s3 = _load_stage03()
    cells = ["flat_active__front", "term_flat_q__front", "flat_active__far", "term_flat_q__far"]
    pairs = s3.default_pairs(cells)
    assert ("term_flat_q__front", "flat_active__front") in pairs
    assert ("flat_active__far", "flat_active__front") in pairs
    assert ("term_flat_q__far", "term_flat_q__front") in pairs
    assert ("flat_active__front", "flat_active__front") not in pairs
    assert len(pairs) == len(set(pairs))


def test_paired_rows_subtract_on_matched_inceptions_only():
    s3 = _load_stage03()
    rows = [
        _row("2023-05-04", "flat_active", "front", terminal_pnl_bp=10.0, daily_pnl_std_bp=5.0),
        _row("2023-05-04", "term_flat_q", "front", terminal_pnl_bp=4.0, daily_pnl_std_bp=2.0),
        _row("2023-06-01", "term_flat_q", "front", terminal_pnl_bp=1.0),  # no baseline run
    ]
    paired = s3.paired_rows(rows, [("term_flat_q__front", "flat_active__front")])
    assert len(paired) == 1
    assert paired[0]["inception"] == "2023-05-04"
    assert paired[0]["d_terminal_pnl_bp"] == pytest.approx(-6.0)
    assert paired[0]["d_daily_pnl_std_bp"] == pytest.approx(-3.0)
    summary = s3.paired_summary(paired, [("term_flat_q__front", "flat_active__front")])
    assert summary["term_flat_q__front"]["n"] == 1
    assert summary["term_flat_q__front"]["terminal_pnl_bp"]["mean"] == pytest.approx(-6.0)


def test_lifecycle_consistency_flags_a_cell_that_terminated_differently():
    s3 = _load_stage03()
    rows = [
        _row("2023-05-04", "flat_active", "front"),
        _row("2023-05-04", "term_flat_q", "front"),
    ]
    assert s3.lifecycle_consistency(rows)["consistent"]
    rows[1]["days_replayed"] = 101
    check = s3.lifecycle_consistency(rows)
    assert not check["consistent"]
    assert check["mismatched"] == ["2023-05-04"]


# ---------------------------------------------------------------------------
# stage 01: knock-in probability measures


def _load_stage01():
    path = STUDY_DIR / "01_curve_and_static_risk.py"
    spec = importlib.util.spec_from_file_location("q_term_structure_stage01", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_window_centres_on_the_largest_front_q_jump():
    s1 = _load_stage01()
    days = pd.bdate_range("2024-01-01", periods=12)
    q = np.full(len(days), 0.10)
    q[7] = 0.90  # the jump 6 -> 7 is the largest day-to-day move
    anatomy = pd.DataFrame({"q_flat_unfloored_front": q}, index=days)
    window = s1.probe_window(anatomy, before_days=3, after_days=2)
    assert list(window) == list(days[4:10])  # 3 trading days before the jump day, jump day, 2 after
    assert days[7] in window and days[6] in window


def test_probe_window_is_clipped_at_the_history_edges():
    s1 = _load_stage01()
    days = pd.bdate_range("2024-01-01", periods=5)
    anatomy = pd.DataFrame({"q_flat_unfloored_front": [0.1, 0.9, 0.1, 0.1, 0.1]}, index=days)
    window = s1.probe_window(anatomy, before_days=10, after_days=10)
    assert list(window) == list(days)


def test_ki_probe_summary_reports_day_to_day_ki_change_per_model():
    s1 = _load_stage01()
    days = pd.bdate_range("2024-02-01", periods=3)
    probe = pd.DataFrame(
        {
            "date": list(days) * 2,
            "model": ["flat_active"] * 3 + ["term_flat_q"] * 3,
            "q_T": [0.30, 0.90, 0.00, 0.13, 0.19, 0.10],
            "p_ki": [0.55, 1.00, 0.05, 0.20, 0.34, 0.15],
            "p_ko": [0.18, 0.00, 0.68, 0.40, 0.23, 0.49],
            "pv_bp": [-1490.0, -5844.0, 117.0, -269.0, -728.0, -140.0],
        }
    )
    out = s1.ki_probe_summary(probe)
    assert out["n_days"] == 3
    assert out["first_date"] == "2024-02-01" and out["last_date"] == "2024-02-05"
    flat, term = out["models"]["flat_active"], out["models"]["term_flat_q"]
    assert flat["p_ki_daily_abs_change_mean"] == pytest.approx((0.45 + 0.95) / 2)
    assert term["p_ki_daily_abs_change_mean"] == pytest.approx((0.14 + 0.19) / 2)
    assert flat["p_ki"]["max"] == pytest.approx(1.0) and flat["p_ki"]["min"] == pytest.approx(0.05)
    assert flat["q_T"]["max"] == pytest.approx(0.9)
    assert flat["p_ki_range"] == pytest.approx(0.95) and term["p_ki_range"] == pytest.approx(0.19)


def test_static_summary_reports_ki_ko_probabilities_and_their_gap():
    s1 = _load_stage01()
    dates = [date(2024, 1, 2), date(2024, 2, 1)]
    rows = []
    for d, pki_flat, pki_ref, q_flat, q_ref in zip(dates, [0.60, 0.40], [0.30, 0.35], [0.26, 0.05], [0.11, 0.10]):
        for model, pki, q in (("flat_active", pki_flat, q_flat), ("term_flat_q", pki_ref, q_ref)):
            rows.append(
                {
                    "date": d, "model": model, "pv_bp": 0.0, "delta_hands": 20.0, "rhoq_1pct_bp": -50.0,
                    "q_at_maturity": q, "tail_time_share": 0.5, "ko_obs_beyond_last": 6,
                    "p_ki": pki, "p_ko": 1.0 - pki - 0.1,
                }
            )
    risk = pd.DataFrame(rows)
    out = s1.static_summary(risk, pd.DataFrame(), ["flat_active", "term_flat_q"])
    flat = out["models"]["flat_active"]
    assert flat["p_ki"]["mean"] == pytest.approx(0.5)
    assert flat["p_ko"]["mean"] == pytest.approx(0.4)
    assert flat["p_ki_gap_vs_reference"]["mean"] == pytest.approx(((0.60 - 0.30) + (0.40 - 0.35)) / 2)
    assert flat["p_ko_gap_vs_reference"]["mean"] == pytest.approx(-((0.60 - 0.30) + (0.40 - 0.35)) / 2)
    # the KI-probability gap tracks the q-at-maturity gap: both are larger on the first date
    assert flat["p_ki_gap_vs_q_gap_corr"] == pytest.approx(1.0)
    assert "p_ki_gap_vs_reference" not in out["models"]["term_flat_q"]


def test_report_ki_section_renders_only_when_probabilities_exist(tmp_path):
    s3 = _load_stage03()
    d = {"n": 2, "mean": 0.4, "median": 0.4, "std": 0.1, "min": 0.3, "max": 0.5, "share_positive": 1.0, "t_stat": 5.0}
    without = {"grid_dates": 2, "models": {"flat_active": {"pv_bp": d}, "term_flat_q": {"pv_bp": d}}}
    assert s3._ki_section(without, tmp_path) == ""
    with_ki = {
        "grid_dates": 2,
        "models": {
            "term_flat_q": {"p_ki": d, "p_ko": d},
            "flat_active": {"p_ki": d, "p_ko": d, "p_ki_gap_vs_reference": d, "p_ki_gap_vs_q_gap_corr": 0.98},
        },
        "ki_probe": {
            "inception": "2023-05-04", "coupon": 0.045, "n_days": 3, "first_date": "2024-02-01", "last_date": "2024-02-05",
            "models": {"flat_active": {"q_T": d, "p_ki": d, "p_ko": d, "pv_bp": d, "p_ki_range": 0.95,
                                        "p_ki_daily_abs_change_mean": 0.7, "q_T_daily_abs_change_mean": 0.45}},
        },
    }
    html_out = s3._ki_section(with_ki, tmp_path)  # no csv in tmp_path -> no chart, table still renders
    assert "2b." in html_out and "0.98" in html_out and "q-only probe" in html_out and "0.950" in html_out
