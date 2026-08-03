"""Tests for example/mo_volmodels/12_snowball_volmodel_backtest.py (Phase 4).

Covers the pieces that decide what the fleet actually runs: term-sheet parity
with the gate-certified stage 11 product, the inception scheduler, Gate G2
routing, the fair-coupon root finder, engine-config wiring, cost model, and
run classification (KO / matured / censored).

Deliberately excluded: an end-to-end replay. One replay day of the production
3Y snowball costs ~29s of PDE (measured), so an honest end-to-end run belongs
to the fleet, not the test suite. The seams that end-to-end coverage would
protect are covered here directly.
"""

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "example/mo_volmodels/12_snowball_volmodel_backtest.py"
HISTORY_DIR = ROOT / "example/mo_volmodels/data/history"

spec = importlib.util.spec_from_file_location("snowball_volmodel_backtest_12", MODULE_PATH)
s12 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s12  # dataclasses resolve cls.__module__ here
spec.loader.exec_module(s12)

s11 = s12.stage11()

from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weekday_calendar(start: date, end: date):
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return s11.TradingCalendar(days)


@pytest.fixture(scope="module")
def calendar():
    return _weekday_calendar(date(2023, 1, 2), date(2030, 12, 31))


@pytest.fixture(scope="module")
def terms(calendar):
    return s11.build_snowball_terms(date(2023, 5, 4), calendar)


# ---------------------------------------------------------------------------
# Term-sheet parity with the gate-certified product
# ---------------------------------------------------------------------------

def test_backtest_product_matches_gate_term_sheet(terms):
    """The backtest must price the SAME contract stage 11's gate certified.

    Only three things may differ: position size, the solved coupon, and
    initial_date (which the lifecycle tracker needs and a pricing gate does
    not).  Everything contractual must be identical.
    """
    s0 = 6733.97
    gate_product = s11.build_snowball_product(terms, s0)
    bt_product = s12.build_backtest_product(
        terms, initial_spot=s0, coupon=0.15, notional=50_000_000.0
    )

    assert bt_product.initial_price == gate_product.initial_price
    assert bt_product.strike == gate_product.strike
    assert bt_product.maturity == gate_product.maturity
    assert bt_product.is_reverse == gate_product.is_reverse

    gb, bb = gate_product.barrier_config, bt_product.barrier_config
    assert bb.ko_barrier == gb.ko_barrier
    assert bb.ki_barrier == gb.ki_barrier
    assert list(bb.ko_observation_dates) == list(gb.ko_observation_dates)
    assert list(bb.ki_observation_dates) == list(gb.ki_observation_dates)
    assert bb.ko_observation_type == gb.ko_observation_type
    assert bb.ki_observation_type == gb.ki_observation_type
    assert bb.ki_continuous == gb.ki_continuous

    gp, bp = gate_product.payoff_config, bt_product.payoff_config
    assert bp.include_principal == gp.include_principal
    assert bp.protection_type == gp.protection_type
    assert bp.participation_rate == gp.participation_rate


def test_backtest_product_scales_to_notional(terms):
    s0 = 6733.97
    notional = 50_000_000.0
    product = s12.build_backtest_product(
        terms, initial_spot=s0, coupon=0.15, notional=notional
    )
    assert product.contract_multiplier == pytest.approx(notional / s0)
    assert product.initial_price * product.contract_multiplier == pytest.approx(notional)


def test_backtest_product_carries_the_solved_coupon(terms):
    product = s12.build_backtest_product(terms, initial_spot=6733.97, coupon=0.1234)
    assert product.barrier_config.ko_rate == pytest.approx(0.1234)
    assert product.payoff_config.rebate_rate == pytest.approx(0.1234)


def test_backtest_product_sets_initial_date_for_the_lifecycle(terms):
    product = s12.build_backtest_product(terms, initial_spot=6733.97, coupon=0.15)
    assert product.initial_date.date() == terms.inception


# ---------------------------------------------------------------------------
# Inception scheduler
# ---------------------------------------------------------------------------

def test_schedule_is_monthly_and_ordered(calendar):
    out = s12.schedule_inceptions(
        calendar=calendar,
        data_start=date(2023, 5, 4),
        data_end=date(2026, 7, 22),
        first_admitted_surface=date(2023, 5, 4),
        min_observable_months=12,
    )
    assert out == sorted(out)
    assert len(out) == len(set(out)), "no duplicate inceptions"
    months = {(d.year, d.month) for d in out}
    assert len(months) == len(out), "one inception per calendar month"
    assert all(calendar.is_trading_day(d) for d in out)


def test_schedule_honours_min_observable_months(calendar):
    data_end = date(2026, 7, 22)
    for min_months in (0, 12, 24):
        out = s12.schedule_inceptions(
            calendar=calendar,
            data_start=date(2023, 5, 4),
            data_end=data_end,
            first_admitted_surface=date(2023, 5, 4),
            min_observable_months=min_months,
        )
        assert out, f"expected inceptions at min_observable_months={min_months}"
        assert s11.add_months(out[-1], min_months) <= data_end


def test_schedule_shrinks_as_the_requirement_grows(calendar):
    kwargs = dict(
        calendar=calendar,
        data_start=date(2023, 5, 4),
        data_end=date(2026, 7, 22),
        first_admitted_surface=date(2023, 5, 4),
    )
    n0 = len(s12.schedule_inceptions(min_observable_months=0, **kwargs))
    n12 = len(s12.schedule_inceptions(min_observable_months=12, **kwargs))
    n24 = len(s12.schedule_inceptions(min_observable_months=24, **kwargs))
    assert n0 > n12 > n24


def test_schedule_skips_dates_without_an_admitted_surface(calendar):
    out = s12.schedule_inceptions(
        calendar=calendar,
        data_start=date(2023, 5, 4),
        data_end=date(2026, 7, 22),
        first_admitted_surface=date(2024, 1, 2),
        min_observable_months=12,
    )
    assert out and min(out) >= date(2024, 1, 2)


def test_schedule_rejects_negative_requirement(calendar):
    with pytest.raises(ValidationError):
        s12.schedule_inceptions(
            calendar=calendar,
            data_start=date(2023, 5, 4),
            data_end=date(2026, 7, 22),
            first_admitted_surface=date(2023, 5, 4),
            min_observable_months=-1,
        )


@pytest.mark.skipif(
    not (HISTORY_DIR / "surface_manifest.json").exists(),
    reason="IV surface history not built",
)
def test_schedule_on_the_real_window(calendar):
    """The production window must yield the 27 monthly inceptions of record."""
    from quantark.param.vol.surface_history import VolSurfaceHistory

    history = VolSurfaceHistory(HISTORY_DIR)
    spot = s12.load_spot_frame(HISTORY_DIR)
    import pandas as pd

    real_calendar = s11.TradingCalendar.from_spot_csv(
        HISTORY_DIR / "csi1000_spot.csv"
    )
    out = s12.schedule_inceptions(
        calendar=real_calendar,
        data_start=pd.Timestamp(spot["date"].iloc[0]).date(),
        data_end=pd.Timestamp(spot["date"].iloc[-1]).date(),
        first_admitted_surface=history.admitted_dates[0],
        min_observable_months=12,
    )
    assert len(out) == 27
    assert out[0] == date(2023, 5, 4)
    assert out[-1] == date(2025, 7, 1)


# ---------------------------------------------------------------------------
# Gate G2 routing
# ---------------------------------------------------------------------------

def _routing(**overrides):
    """Test double for a gate decision, keyed by study VARIANT name.

    flat_bsm/flat_bsm_quad/ts_bsm/localvol default to "pde" so tests that
    are not specifically exercising 1D/quad routing don't have to spell out
    all four every time; heston/heston_slv are left unset unless a test
    passes them explicitly, since those are what most routing tests vary.

    mc_params carries a representative gate-certified MC config for
    heston/heston_slv regardless of whether a given test actually routes
    them to "mc" -- make_engine_config now fails closed (Task 10) when an
    MC-routed variant has no entry here, so every test that reprices heston
    or heston_slv needs one available even if routing to MC isn't the point
    of that particular test.
    """
    routes = {
        "flat_bsm": "pde",
        "flat_bsm_quad": "pde",
        "ts_bsm": "pde",
        "localvol": "pde",
    }
    routes.update(overrides)
    gated_mc = {
        "paths_per_batch": 8192, "batches": 16, "seed": 20260723,
        "substeps_per_interval": 4, "scheme": "QUADEXP_M",
    }
    return s12.GateRouting(
        decision_path="test",
        evidence_sha256="deadbeef",
        routes=routes,
        pde_params={
            "heston": {
                "n_x": 200,
                "n_v": 60,
                "n_t": 1202,
                "v_grid_power": 2.5,
                "scheme": "cs",
            }
        },
        mc_params={"heston": gated_mc, "heston_slv": gated_mc},
    )


def test_one_dimensional_variants_now_read_their_own_route_from_the_decision():
    """After Task 4, flat_bsm/flat_bsm_quad/ts_bsm/localvol are inside the
    gate's scope too. They share vol_model across the three bsm variants,
    but the decision keys routes by the STUDY VARIANT name, so each can get
    an independent verdict -- proven here by giving them different routes."""
    routing = _routing(flat_bsm="mc", ts_bsm="pde", localvol="mc")
    assert routing.solver_for("flat_bsm") == "mc"
    assert routing.solver_for("ts_bsm") == "pde"
    assert routing.solver_for("localvol") == "mc"


def test_one_dimensional_variant_fails_closed_when_its_route_is_missing():
    """No more short-circuit: an absent 1D/quad entry must raise, never
    silently default to PDE."""
    routing = s12.GateRouting("t", None, {"heston": "mc"}, {})
    with pytest.raises(ValidationError, match="no usable route"):
        routing.solver_for("flat_bsm")
    with pytest.raises(ValidationError, match="no usable route"):
        routing.solver_for("localvol")


def test_one_dimensional_variant_fails_closed_on_an_unrecognised_route_string():
    """Present-but-garbage is just as fatal as absent -- no partial trust."""
    routing = s12.GateRouting("t", None, {"flat_bsm": "quadrature"}, {})
    with pytest.raises(ValidationError, match="no usable route"):
        routing.solver_for("flat_bsm")


def test_routing_reads_the_decision_file_not_a_hardcoded_default():
    assert _routing(heston="mc", heston_slv="mc").solver_for("heston") == "mc"
    assert _routing(heston="pde", heston_slv="mc").solver_for("heston") == "pde"


def test_routing_fails_closed_on_a_missing_or_bad_route():
    with pytest.raises(ValidationError, match="no usable route"):
        _routing(heston_slv="mc").solver_for("heston")
    with pytest.raises(ValidationError, match="no usable route"):
        _routing(heston="quadrature").solver_for("heston")


def test_pde_grid_options_only_flow_on_the_pde_route():
    assert _routing(heston="mc").engine_options_for("heston") == {}
    assert _routing(heston="pde").engine_options_for("heston") == {
        "n_x": 200,
        "n_v": 60,
        "n_t": 1202,
        "v_grid_power": 2.5,
    }, "only supported numerical grid controls are forwarded to the solver"


def test_load_gate_routing_is_actionable_when_absent(tmp_path):
    with pytest.raises(ValidationError, match="run stage 11"):
        s12.load_gate_routing(tmp_path / "nope.json")


def test_load_gate_routing_rejects_malformed_files(tmp_path):
    bad = tmp_path / "gate.json"
    bad.write_text("{not json")
    with pytest.raises(ValidationError, match="not valid JSON"):
        s12.load_gate_routing(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValidationError, match="no 'variants' map"):
        s12.load_gate_routing(empty)


@pytest.mark.skipif(
    not s12.DEFAULT_GATE_DECISION.exists(), reason="gate decision not produced"
)
def test_recorded_gate_decision_routes_both_2d_variants_to_mc():
    routing = s12.load_gate_routing(s12.DEFAULT_GATE_DECISION)
    assert routing.solver_for("heston") == "mc"
    assert routing.solver_for("heston_slv") == "mc"
    assert routing.evidence_sha256, "run manifest must be able to cite the evidence"


# ---------------------------------------------------------------------------
# Fair-coupon root finder (Gate G4)
# ---------------------------------------------------------------------------

def test_affine_root_is_found_in_one_step():
    """PV is affine in the coupon, so false position must land immediately."""
    calls = []

    def pv(coupon):
        calls.append(coupon)
        return -3_606_847.0 + coupon * 23_890_371.0

    result = s12.solve_affine_root(
        pv, lower=0.0, upper=0.8, tolerance=50.0, max_iterations=60
    )
    assert result.solved
    assert result.iterations == 1
    assert len(calls) == 3  # two anchors + the (exact) affine step
    assert abs(pv(result.coupon)) <= 50.0


def test_nonlinear_root_still_converges():
    """A non-affine PV must still converge - the method stays bracketing."""
    result = s12.solve_affine_root(
        lambda x: x ** 3 - 0.02,
        lower=0.0,
        upper=0.8,
        tolerance=1e-9,
        max_iterations=200,
    )
    assert result.solved
    assert result.coupon == pytest.approx(0.02 ** (1 / 3), rel=1e-4)


def test_unbracketed_root_raises_rather_than_returning_a_boundary():
    with pytest.raises(ValidationError, match="not bracketed"):
        s12.solve_affine_root(
            lambda x: 1.0e9, lower=0.0, upper=0.8, tolerance=1.0, max_iterations=10
        )


def test_flat_function_raises_instead_of_dividing_by_zero():
    """Equal endpoint values would make the false-position step divide by 0."""
    with pytest.raises(ValidationError, match="degenerate"):
        s12.solve_affine_root(
            lambda x: 0.0 if x > 0 else -0.0,  # brackets, but with zero slope
            lower=0.0,
            upper=0.8,
            tolerance=-1.0,  # unsatisfiable, so the solver must reach the guard
            max_iterations=5,
        )


def test_exhausted_budget_raises_and_names_gate_g4():
    with pytest.raises(ValidationError, match="Gate G4"):
        s12.solve_affine_root(
            lambda x: x ** 3 - 0.02,
            lower=0.0,
            upper=0.8,
            tolerance=1e-18,
            max_iterations=2,
        )


def test_endpoint_root_is_accepted_without_iterating():
    result = s12.solve_affine_root(
        lambda x: x - 0.0, lower=0.0, upper=0.8, tolerance=1e-6, max_iterations=10
    )
    assert result.solved and result.iterations == 0


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", s12.VARIANTS)
def test_engine_config_matches_the_variant_spec(variant):
    routing = _routing(heston="mc", heston_slv="mc")
    cfg = s12.make_engine_config(variant, routing=routing)
    spec_ = s12.VARIANT_SPECS[variant]
    assert cfg.vol_model == spec_.vol_model
    assert cfg.vol_source == spec_.vol_source
    assert cfg.surface_vol_mode == spec_.surface_vol_mode


def test_every_variant_keeps_the_same_deterministic_reporting_engines():
    """Surface / event-stats engines must be comparable across variants."""
    routing = _routing(heston="mc", heston_slv="mc")
    types = {
        v: s12.make_engine_config(v, routing=routing).resolve_event_stats_engine_type()
        for v in s12.VARIANTS
    }
    assert set(types.values()) == {EngineType.PDE}


def test_mc_method_does_not_leak_into_the_pde_engine_slot():
    """The RQMC method belongs to the vol-model engine, not the PDE engine.

    Putting MonteCarloMethod in ``method`` makes the PDE surface/event-stats
    engines unconstructible ("Invalid method type"), which is why the config
    carries a separate ``vol_model_mc_method`` slot.
    """
    cfg = s12.make_engine_config("heston", routing=_routing(heston="mc"))
    assert cfg.method is None
    assert cfg.resolve_vol_model_mc_method() == MonteCarloMethod.RANDOMIZED_QUASI


def test_bsm_variants_request_no_calibration():
    routing = _routing(heston="mc", heston_slv="mc")
    for variant in ("flat_bsm", "ts_bsm"):
        assert not s12.VARIANT_SPECS[variant].uses_calibration()
    for variant in ("localvol", "heston", "heston_slv"):
        assert s12.VARIANT_SPECS[variant].uses_calibration()
        cfg = s12.make_engine_config(
            variant, routing=routing, calibration_cache_dir=Path("/tmp/cache")
        )
        assert cfg.vol_model_calibration.cache_dir == "/tmp/cache"


def test_quick_mode_only_shrinks_monte_carlo_effort():
    routing = _routing(heston="mc")
    full = s12.make_engine_config("heston", routing=routing, quick=False)
    quick = s12.make_engine_config("heston", routing=routing, quick=True)
    assert quick.mc_params.num_paths < full.mc_params.num_paths
    assert quick.vol_model == full.vol_model
    assert quick.surface_vol_mode == full.surface_vol_mode


def test_mc_params_are_fixed_cost_and_fixed_seed():
    """The gate reference is a fixed batch count at a fixed seed, not adaptive."""
    params = s12.make_mc_params(8192, 16, 20260723)
    assert params.num_paths == 8192
    assert params.rqmc_min_batches == params.rqmc_max_batches == 16
    assert params.seed == 20260723
    assert not params.use_antithetic


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def test_cost_model_uses_the_agreed_cffex_parameters():
    model = s12.make_cost_model(True)
    assert model.proportional_rate == pytest.approx(5e-5)
    assert model.spread_bps == pytest.approx(1.0)
    assert model.fixed_commission == 0.0


def test_costs_can_be_disabled_for_a_no_cost_comparison():
    from quantark.backtest.transaction_costs import ZeroCostModel

    assert isinstance(s12.make_cost_model(False), ZeroCostModel)


def test_costs_are_actually_charged_on_a_futures_trade():
    model = s12.make_cost_model(True)
    cost = model.calculate_cost(
        quantity=10.0,
        price=6000.0,
        notional=10 * 6000.0 * 200.0,
        instrument_type="futures",
        trade_type="buy",
    )
    assert cost > 0.0


# ---------------------------------------------------------------------------
# Run classification
# ---------------------------------------------------------------------------

class _StubResults:
    """Mirrors the engine's real frames: lifecycle flags live in states_df.

    ``actions_df`` carries ``action_type`` with LifecycleEventType VALUES
    ("KO"/"KI"/"COUPON"/"MATURITY") - not a free-text ``action`` column.
    """

    def __init__(self, actions, *, knocked_out=False, knocked_in=False, matured=False):
        import pandas as pd

        self.actions_df = pd.DataFrame(actions)
        self.states_df = pd.DataFrame(
            {
                "total_pnl": [0.0, 1.0],
                "knocked_out": [False, knocked_out],
                "knocked_in": [False, knocked_in],
                "matured": [False, matured],
                "alive": [True, not (knocked_out or matured)],
            }
        )

    def get_summary(self):
        return {"total_pnl": 1.0}


def _summarize(actions, last_date, terms, **flags):
    import pandas as pd

    return s12.summarize_run(
        results=_StubResults(actions, **flags),
        terms=terms,
        inception=terms.inception,
        variant="flat_bsm",
        engine_config=s12.make_engine_config("flat_bsm", routing=_routing()),
        dates=pd.DatetimeIndex([pd.Timestamp(terms.inception), pd.Timestamp(last_date)]),
        coupon=0.15,
        notional=50_000_000.0,
        calibration_records=[],
        elapsed=1.0,
    )


def test_run_reaching_data_end_before_maturity_is_censored(terms):
    summary = _summarize([], date(2024, 6, 3), terms)
    assert summary["lifecycle"]["censored_at_data_end"]
    assert not summary["lifecycle"]["matured"]
    assert not summary["lifecycle"]["knocked_out"]


def test_run_reaching_maturity_is_not_censored(terms):
    summary = _summarize([], terms.maturity_date, terms, matured=True)
    assert summary["lifecycle"]["matured"]
    assert not summary["lifecycle"]["censored_at_data_end"]


def test_knocked_out_run_is_not_censored_and_not_matured(terms):
    """KO ends the trade early; it must not be double-counted as matured."""
    actions = [{"action_type": "KO", "payoff": 1.0}]
    summary = _summarize(actions, date(2023, 9, 4), terms, knocked_out=True)
    assert summary["lifecycle"]["knocked_out"]
    assert not summary["lifecycle"]["censored_at_data_end"]
    assert not summary["lifecycle"]["matured"]


def test_knock_out_date_is_read_from_the_action_log(terms):
    import pandas as pd

    actions = pd.DataFrame(
        [{"action_type": "COUPON"}, {"action_type": "KO"}],
        index=pd.to_datetime(["2023-08-04", "2023-09-04"]),
    )
    summary = s12.summarize_run(
        results=_StubResults(actions, knocked_out=True),
        terms=terms,
        inception=terms.inception,
        variant="flat_bsm",
        engine_config=s12.make_engine_config("flat_bsm", routing=_routing()),
        dates=pd.DatetimeIndex([pd.Timestamp("2023-05-04"), pd.Timestamp("2023-09-04")]),
        coupon=0.15,
        notional=50_000_000.0,
        calibration_records=[],
        elapsed=1.0,
    )
    assert "2023-09-04" in str(summary["lifecycle"]["ko_date"])


def test_lifecycle_flags_come_from_states_not_action_labels(terms):
    """An empty action log must not silently mean 'never knocked out'."""
    summary = _summarize([], date(2023, 9, 4), terms, knocked_out=True)
    assert summary["lifecycle"]["knocked_out"]
    assert summary["lifecycle"]["ko_date"] is None  # no action row to date it


def test_summary_records_the_routing_actually_used(terms):
    summary = _summarize([], date(2024, 6, 3), terms)
    assert summary["vol_model"] == "bsm"
    assert summary["vol_model_solver"] in ("pde", "mc")
    assert summary["surface_vol_mode"] == "flat_atm_remaining"
    assert summary["coupon"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Task fan-out
# ---------------------------------------------------------------------------

def test_tasks_are_the_full_inception_by_variant_cross_product():
    prepared = [
        {"inception": "2023-05-04", "initial_spot": 6733.97, "coupon": 0.15,
         "maturity_date": "2026-05-04"},
        {"inception": "2023-06-01", "initial_spot": 6500.0, "coupon": 0.16,
         "maturity_date": "2026-06-01"},
    ]
    tasks = s12.build_tasks(
        prepared=prepared,
        variants=["flat_bsm", "heston"],
        routing=_routing(heston="mc"),
        history_dir=HISTORY_DIR,
        out_dir=Path("/tmp/out"),
        data_end=date(2026, 7, 22),
        rate=0.02,
        notional=50_000_000.0,
        costs_enabled=True,
        quick=False,
        calculate_surfaces=False,
        calculate_event_probabilities=True,
    )
    assert len(tasks) == 4
    assert {(t["inception"], t["variant"]) for t in tasks} == {
        ("2023-05-04", "flat_bsm"), ("2023-05-04", "heston"),
        ("2023-06-01", "flat_bsm"), ("2023-06-01", "heston"),
    }
    # Every variant of an inception must share ONE coupon (apples-to-apples).
    by_inception = {}
    for task in tasks:
        by_inception.setdefault(task["inception"], set()).add(task["coupon"])
    assert all(len(coupons) == 1 for coupons in by_inception.values())


def test_window_end_is_clipped_to_the_data_end():
    prepared = [{"inception": "2025-07-01", "initial_spot": 6000.0, "coupon": 0.15,
                 "maturity_date": "2028-07-03"}]
    tasks = s12.build_tasks(
        prepared=prepared,
        variants=["flat_bsm"],
        routing=_routing(),
        history_dir=HISTORY_DIR,
        out_dir=Path("/tmp/out"),
        data_end=date(2026, 7, 22),
        rate=0.02,
        notional=50_000_000.0,
        costs_enabled=True,
        quick=False,
        calculate_surfaces=False,
        calculate_event_probabilities=True,
    )
    assert tasks[0]["window_end"] == "2026-07-22"


def test_every_task_carries_the_gate_provenance():
    tasks = s12.build_tasks(
        prepared=[{"inception": "2023-05-04", "initial_spot": 6733.97,
                   "coupon": 0.15, "maturity_date": "2026-05-04"}],
        variants=["heston"],
        routing=_routing(heston="mc"),
        history_dir=HISTORY_DIR,
        out_dir=Path("/tmp/out"),
        data_end=date(2026, 7, 22),
        rate=0.02,
        notional=50_000_000.0,
        costs_enabled=True,
        quick=False,
        calculate_surfaces=False,
        calculate_event_probabilities=True,
    )
    assert tasks[0]["gate"]["evidence_sha256"] == "deadbeef"
    assert tasks[0]["gate"]["routes"]["heston"] == "mc"


def test_failed_runs_are_recorded_not_swallowed():
    def boom(task):
        raise RuntimeError("engine exploded")

    original = s12.run_one
    s12.run_one = boom
    try:
        summaries, failures = s12.execute_tasks(
            [{"inception": "2023-05-04", "variant": "flat_bsm"}], workers=1
        )
    finally:
        s12.run_one = original
    assert summaries == []
    assert len(failures) == 1
    assert failures[0]["error_type"] == "RuntimeError"
    assert "engine exploded" in failures[0]["error"]
    assert failures[0]["traceback"]
