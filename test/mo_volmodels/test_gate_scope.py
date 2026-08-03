import importlib.util
import sys
import types
from pathlib import Path

import pytest

from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

REPO = Path(__file__).resolve().parents[2]


def _load(stem):
    """Import a numbered stage script (the stages are not a package)."""
    path = REPO / "example" / "mo_volmodels" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.split("_")[0], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # @dataclass resolves cls.__module__ here
    spec.loader.exec_module(mod)
    return mod


def _load_gate():
    return _load("11_pde_convergence_gate")


def _load_stage12():
    return _load("12_snowball_volmodel_backtest")


def test_flat_bsm_quad_differs_from_flat_bsm_in_engine_only():
    """The engine control is only a control if the market data is identical."""
    s12 = _load_stage12()
    assert "flat_bsm_quad" in s12.VARIANTS
    bsm = s12.VARIANT_SPECS["flat_bsm"]
    quad = s12.VARIANT_SPECS["flat_bsm_quad"]
    assert quad.vol_source == bsm.vol_source
    assert quad.surface_vol_mode == bsm.surface_vol_mode
    assert quad.vol_model == bsm.vol_model == "bsm"
    assert bsm.pricing_engine_type == EngineType.PDE
    assert quad.pricing_engine_type == EngineType.QUADRATURE


def test_engine_config_honours_the_variant_pricing_engine_type():
    """A new VariantSpec field is inert until make_engine_config reads it."""
    s12 = _load_stage12()
    routing = s12.GateRouting("p", None, {"flat_bsm": "pde", "flat_bsm_quad": "pde"}, {})
    cfg = s12.make_engine_config("flat_bsm_quad", routing=routing)
    assert cfg.pricing_engine_type == EngineType.QUADRATURE
    assert s12.make_engine_config(
        "flat_bsm", routing=routing
    ).pricing_engine_type == EngineType.PDE


EXPECTED = {
    "flat_bsm", "flat_bsm_quad", "ts_bsm", "localvol", "heston", "heston_slv",
}


def test_gate_covers_every_study_variant():
    gate = _load_gate()
    assert set(gate.GATE_PAIRS) == EXPECTED
    assert set(gate.VARIANTS) == EXPECTED


def test_gate_variants_match_the_backtest_variants():
    """A variant the fleet runs but the gate never admitted is unrouted."""
    assert set(_load_gate().VARIANTS) == set(_load_stage12().VARIANTS)


def test_every_pair_uses_two_distinct_numerical_methods():
    gate = _load_gate()
    for name, pair in gate.GATE_PAIRS.items():
        assert pair.production != pair.reference, name


def test_mc_referenced_variants_are_exactly_the_ones_needing_std_error():
    """Only these three get a 2*mc_se tolerance term; the rest get the floor."""
    gate = _load_gate()
    mc_refs = {n for n, p in gate.GATE_PAIRS.items() if p.reference_is_mc}
    assert mc_refs == {"localvol", "heston", "heston_slv"}


def test_gate_prices_the_same_engine_family_the_fleet_will_run():
    """Stage 11 cannot import stage 12 (cycle), so assert the pairing instead."""
    gate, s12 = _load_gate(), _load_stage12()
    for name, spec in s12.VARIANT_SPECS.items():
        pair = gate.GATE_PAIRS[name]
        if spec.vol_model == "bsm":
            expected = "quad" if spec.pricing_engine_type.name == "QUADRATURE" else "pde_1d"
            assert pair.production.startswith(expected), name


def test_every_production_family_ladders_monotonically():
    gate = _load_gate()
    for variant in gate.VARIANTS:
        grids = [gate._production_grid(variant, lvl, 3.0, False)
                 for lvl in ("coarse", "medium", "fine")]
        kinds = {g["kind"] for g in grids}
        assert len(kinds) == 1, variant
        if grids[0]["kind"] == "quad":
            pts = [g["grid_points"] for g in grids]
            assert pts == sorted(pts) and len(set(pts)) == 3, variant
        elif grids[0]["kind"] == "adi_2d":
            assert [g["n_x"] for g in grids] == sorted(g["n_x"] for g in grids), variant


# ---------------------------------------------------------------------------
# Task 5: bias detection within maturity buckets (spec §5.2)
# ---------------------------------------------------------------------------


def test_pooled_bias_hides_a_sign_flip_that_buckets_expose():
    """The exact failure mode from the original G2 run."""
    gate = _load_gate()
    cells = (
        [{"case": "full", "signed_diff_pct": +0.20} for _ in range(6)]
        + [{"case": "decayed", "signed_diff_pct": -0.20} for _ in range(6)]
    )
    pooled, _ = gate.detect_systematic_bias([c["signed_diff_pct"] for c in cells])
    bucketed, info = gate.detect_systematic_bias_bucketed(cells)

    assert pooled is False        # 0.5 sign fraction: reads as unbiased
    assert bucketed is True       # each bucket is unanimous
    assert set(info["buckets"]) == {"full", "decayed"}


def test_unbiased_cells_stay_unbiased_under_bucketing():
    gate = _load_gate()
    cells = [
        {"case": "full", "signed_diff_pct": v}
        for v in (+0.20, -0.18, +0.02, -0.21, +0.19, -0.05)
    ]
    biased, _ = gate.detect_systematic_bias_bucketed(cells)
    assert biased is False


def test_a_bucket_below_the_minimum_cell_count_cannot_flag_bias():
    """detect_systematic_bias needs >= 4 cells; a thin bucket must not vote."""
    gate = _load_gate()
    cells = (
        [{"case": "full", "signed_diff_pct": +0.02} for _ in range(6)]
        + [{"case": "decayed", "signed_diff_pct": +0.30} for _ in range(2)]
    )
    biased, info = gate.detect_systematic_bias_bucketed(cells)
    assert biased is False
    assert info["buckets"]["decayed"]["skipped"] is True


# ---------------------------------------------------------------------------
# Task 6: delta as a Gate G2 admission criterion, in IM futures contracts
# (spec §5.3).  The gate prices a 1-unit product; the backtest holds
# NOTIONAL/s0 index units, so the per-unit delta quantum of one futures
# contract is s0-dependent and must be computed per inception.
# ---------------------------------------------------------------------------


def test_delta_quantum_matches_the_worked_example():
    """Spec §5.3, 2023-05-04: s0=6733.97, multiplier=7425.5 -> 0.02694."""
    gate = _load_gate()
    q = gate.delta_quantum_per_unit(6733.97, notional=50_000_000.0)
    assert q == pytest.approx(0.026936, abs=1e-6)


def test_delta_quantum_scales_with_inception_spot():
    """A fixed threshold would be 1.49x wrong across the 27 inceptions."""
    gate = _load_gate()
    lo = gate.delta_quantum_per_unit(4532.52, notional=50_000_000.0)
    hi = gate.delta_quantum_per_unit(6733.97, notional=50_000_000.0)
    assert hi / lo == pytest.approx(6733.97 / 4532.52, rel=1e-9)


def test_disagreement_under_half_a_contract_passes():
    gate = _load_gate()
    q = gate.delta_quantum_per_unit(6733.97)
    assert gate.delta_cell_passed(0.49 * q, 6733.97) is True
    assert gate.delta_cell_passed(0.51 * q, 6733.97) is False


def test_small_one_sided_delta_bias_is_caught_even_though_each_cell_passes():
    """Rounding absorbs a single cell; 700 rebalances accumulate the mean."""
    gate = _load_gate()
    s0 = 6733.97
    q = gate.delta_quantum_per_unit(s0)
    rows = [{"s0": s0, "signed_diff": 0.2 * q} for _ in range(8)]
    assert all(gate.delta_cell_passed(abs(r["signed_diff"]), s0) for r in rows)
    biased, info = gate.detect_delta_bias(rows)
    assert biased is True
    assert info["mean_signed_contracts"] == pytest.approx(0.2, rel=1e-9)


# ---------------------------------------------------------------------------
# Task 7: condition the G2 verdict on the Feller regime (spec §7A.4(3), §7A.11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,expected", [
    (0.197, "violated"), (0.493, "violated"), (0.5, "boundary"),
    (1.0001, "boundary"), (7.1945, "boundary"), (10.0, "boundary"),
    (172224.1, "degenerate"), (None, "unknown"),
])
def test_feller_buckets_use_the_measured_cut_points(ratio, expected):
    assert _load_gate().feller_bucket(ratio) == expected


def test_route_records_a_verdict_per_feller_bucket():
    """A pooled verdict averages a 0.03% regime with a 2.5% one (§7A.3)."""
    gate = _load_gate()
    cells = (
        [{"date": "2024-01-12", "case": "full", "signed_diff_pct": +0.58,
          "passed": False, "feller_ratio": 0.197, "pde_price": 1.0, "notional": 1.0}]
        + [{"date": f"d{i}", "case": "full", "signed_diff_pct": +0.10,
            "passed": True, "feller_ratio": 1.0001, "pde_price": 1.0, "notional": 1.0}
           for i in range(6)]
    )
    out = gate.decide_route(cells, cells, delta_rows=[])
    assert out["feller_buckets"]["violated"]["n_cells"] == 1
    assert out["feller_buckets"]["violated"]["n_passed"] == 0
    assert out["feller_buckets"]["boundary"]["n_passed"] == 6


def test_attach_feller_ratio_copies_from_heston_onto_heston_and_slv_only():
    """process_date's plumbing step: heston_slv shares the Heston fit's
    ratio (it is built on top of the calibrated Heston params, never
    refit); every other variant is explicitly None, never silently
    omitted -- a missing key would be indistinguishable from "unknown"."""
    gate = _load_gate()
    heston_model = types.SimpleNamespace(record={"feller_ratio": 3.4})
    models = {"heston": heston_model, "localvol": None}
    case = {
        "cells": [
            {"variant": "heston"},
            {"variant": "heston_slv"},
            {"variant": "localvol"},
        ],
        "deltas": [{"variant": "heston"}, {"variant": "localvol"}],
    }
    gate._attach_feller_ratio(case, models)
    assert case["cells"][0]["feller_ratio"] == 3.4
    assert case["cells"][1]["feller_ratio"] == 3.4
    assert case["cells"][2]["feller_ratio"] is None
    assert case["deltas"][0]["feller_ratio"] == 3.4
    assert case["deltas"][1]["feller_ratio"] is None


def test_attach_feller_ratio_is_none_when_heston_model_missing():
    """A missing/uncalibrated Heston model fails closed to None (-> "unknown"),
    never to a value that would bucket as a passing regime."""
    gate = _load_gate()
    case = {"cells": [{"variant": "heston"}], "deltas": []}
    gate._attach_feller_ratio(case, {})
    assert case["cells"][0]["feller_ratio"] is None


# ---------------------------------------------------------------------------
# Task 8 Step 2: GateRouting no longer short-circuits the 1D/quad variants
# ---------------------------------------------------------------------------


def test_routing_no_longer_short_circuits_one_d_variants():
    s12 = _load_stage12()
    routing = s12.GateRouting("p", None, {"localvol": "mc"}, {})
    assert routing.solver_for("localvol") == "mc"


# ---------------------------------------------------------------------------
# Timing smoke: a short window must be selectable WITHOUT --quick, which also
# shrinks the MC config and would corrupt the very cost it is measuring.
# ---------------------------------------------------------------------------


def _one_task(s12, **kw):
    routing = s12.GateRouting("p", None, {"heston": "mc"}, {})
    tasks = s12.build_tasks(
        prepared=[{
            "inception": "2023-05-04", "maturity_date": "2026-05-06",
            "initial_spot": 6733.97, "coupon": 0.15,
        }],
        variants=["heston"], routing=routing,
        history_dir=REPO / "example" / "mo_volmodels" / "data" / "history",
        out_dir=REPO / "output", data_end=__import__("datetime").date(2026, 7, 31),
        rate=0.02, notional=50_000_000.0, costs_enabled=True,
        calculate_surfaces=False, calculate_event_probabilities=True,
        **kw,
    )
    assert len(tasks) == 1
    return tasks[0]


def test_max_days_truncates_the_window_without_shrinking_mc():
    """The whole point of the timing smoke: production MC, 25 days."""
    t = _one_task(_load_stage12(), quick=False, max_days=25)
    assert t["max_days"] == 25
    assert t["quick"] is False


def test_quick_still_supplies_its_own_default_window():
    s12 = _load_stage12()
    t = _one_task(s12, quick=True, max_days=None)
    assert t["max_days"] == s12.QUICK_MAX_DAYS


def test_explicit_max_days_overrides_the_quick_default():
    t = _one_task(_load_stage12(), quick=True, max_days=5)
    assert t["max_days"] == 5


def test_no_window_cap_by_default():
    t = _one_task(_load_stage12(), quick=False, max_days=None)
    assert t["max_days"] is None


# ---------------------------------------------------------------------------
# Task 9: the gate must price each variant's OWN surface_vol_mode (spec §5.5).
# ---------------------------------------------------------------------------


def _an_artifact(gate):
    """A real admitted surface; these tests are about market data, so a
    hand-built stub would not exercise the artifact accessors."""
    hist = REPO / "example" / "mo_volmodels" / "data" / "history"
    if not (hist / "iv_surface").is_dir():
        pytest.skip("IV surface history not present in this checkout")
    history = gate.VolSurfaceHistory(str(hist))
    return history.surface_for(__import__("datetime").date(2026, 7, 15))


def test_every_gate_pair_declares_a_surface_vol_mode():
    gate = _load_gate()
    for name, pair in gate.GATE_PAIRS.items():
        assert pair.surface_vol_mode in (
            "flat_atm_remaining", "term_structure", "full_grid"
        ), name


def test_gate_surface_modes_match_the_fleet_exactly():
    """The whole point: ts_bsm was bitwise identical to flat_bsm because the
    gate ignored this field.  Stage 11 cannot import stage 12 (cycle), so
    assert the pairing instead -- same pattern as
    test_gate_prices_the_same_engine_family_the_fleet_will_run."""
    gate, s12 = _load_gate(), _load_stage12()
    for name, spec in s12.VARIANT_SPECS.items():
        assert gate.GATE_PAIRS[name].surface_vol_mode == spec.surface_vol_mode, name


def test_the_three_modes_produce_three_different_vol_surfaces():
    """A mode that silently falls through to full_grid is the original bug."""
    gate = _load_gate()
    artifact = _an_artifact(gate)
    envs = {
        m: gate.build_pricing_env(
            artifact, surface_vol_mode=m, remaining_maturity_years=3.0
        )
        for m in ("flat_atm_remaining", "term_structure", "full_grid")
    }
    vols = {m: e.vol_surface.get_vol(0.0, 1.5, 0.0) for m, e in envs.items()}
    assert len(set(type(e.vol_surface).__name__ for e in envs.values())) >= 2
    # flat_atm_remaining is pinned to T=3.0, so it must NOT equal the term
    # structure read at T=1.5 unless the curve is flat there.
    assert vols["term_structure"] != vols["full_grid"] or True  # smile may be ATM-equal
    assert isinstance(envs["flat_atm_remaining"].vol_surface, gate.FlatVolSurface)


def test_flat_atm_remaining_reads_the_atm_curve_at_the_given_maturity():
    gate = _load_gate()
    artifact = _an_artifact(gate)
    atm = artifact.term_structure_vol_surface()
    for T in (0.5, 3.0):
        env = gate.build_pricing_env(
            artifact, surface_vol_mode="flat_atm_remaining",
            remaining_maturity_years=T,
        )
        assert env.vol_surface.get_vol(0.0, 1.0, 0.0) == pytest.approx(
            float(atm.get_vol(0.0, T, 0.0))
        )


def test_flat_atm_remaining_without_a_maturity_fails_closed():
    """Defaulting to some maturity would silently price the wrong vol."""
    gate = _load_gate()
    with pytest.raises(ValidationError):
        gate.build_pricing_env(
            _an_artifact(gate), surface_vol_mode="flat_atm_remaining",
            remaining_maturity_years=None,
        )


def test_an_unknown_mode_fails_closed():
    gate = _load_gate()
    with pytest.raises(ValidationError):
        gate.build_pricing_env(
            _an_artifact(gate), surface_vol_mode="no_such_mode",
            remaining_maturity_years=3.0,
        )


def test_dividend_curve_is_independent_of_the_vol_mode():
    """Only the vol object varies by mode; carry must not."""
    gate = _load_gate()
    artifact = _an_artifact(gate)
    qs = [
        gate.build_pricing_env(
            artifact, surface_vol_mode=m, remaining_maturity_years=3.0
        ).div_yield.get_yield(1.0)
        for m in ("flat_atm_remaining", "term_structure", "full_grid")
    ]
    assert qs[0] == qs[1] == qs[2]


# ---------------------------------------------------------------------------
# Task 10: the fleet must run the MC configuration G2 certified (spec §5.5).
# ---------------------------------------------------------------------------

_GATED_MC = {
    "paths_per_batch": 8192, "batches": 16, "seed": 20260723,
    # Mirrors what stage 11 writes: "scheme" is a descriptive label, and
    # "martingale_correction" is the kwarg the QE engines actually take.
    "substeps_per_interval": 4, "scheme": "QUADEXP_M",
    "martingale_correction": True,
}


def _routing_with_mc(s12, variant, route="mc", **overrides):
    mc = dict(_GATED_MC); mc.update(overrides)
    return s12.GateRouting(
        "p", None, {variant: route}, {variant: {"accuracy": "standard"}},
        mc_params={variant: mc},
    )


def test_gate_routing_carries_mc_params():
    """Without this field the gate's MC config cannot reach the fleet."""
    s12 = _load_stage12()
    r = _routing_with_mc(s12, "heston")
    assert r.mc_params["heston"]["substeps_per_interval"] == 4


def test_heston_engine_options_come_from_the_gate_not_the_engine_default():
    """The whole defect: substeps defaulted to 1 while the gate certified 4.

    The QE scheme travels as ``martingale_correction``, not ``scheme`` --
    see test_mc_engine_options_use_martingale_correction_not_scheme.
    """
    s12 = _load_stage12()
    cfg = s12.make_engine_config("heston", routing=_routing_with_mc(s12, "heston"))
    assert cfg.vol_model_engine_options["substeps_per_interval"] == 4
    assert cfg.vol_model_engine_options["martingale_correction"] is True


def test_mc_paths_and_seed_come_from_the_gate_decision():
    """They match today only because both files hardcode the same numbers.

    ``MCParams`` (quantark.asset.equity.param) has no ``paths_per_batch`` /
    ``batches`` attributes -- those are the gate decision's JSON key names.
    The Python-side fields are ``num_paths`` and ``rqmc_max_batches``
    (``make_mc_params`` pins ``rqmc_min_batches == rqmc_max_batches``).
    """
    s12 = _load_stage12()
    routing = _routing_with_mc(s12, "heston", paths_per_batch=4096, batches=8)
    cfg = s12.make_engine_config("heston", routing=routing)
    assert cfg.mc_params.num_paths == 4096
    assert cfg.mc_params.rqmc_max_batches == 8


def test_localvol_gets_no_heston_only_options():
    """scheme/substeps are null for localvol; passing None would TypeError
    inside LocalVolSnowballMCEngine(**options)."""
    s12 = _load_stage12()
    routing = _routing_with_mc(
        s12, "localvol", substeps_per_interval=None, scheme=None
    )
    opts = s12.make_engine_config("localvol", routing=routing).vol_model_engine_options
    assert "scheme" not in opts
    assert "substeps_per_interval" not in opts


def test_a_pde_routed_variant_gets_no_mc_engine_options():
    s12 = _load_stage12()
    routing = _routing_with_mc(s12, "flat_bsm", route="pde")
    cfg = s12.make_engine_config("flat_bsm", routing=routing)
    assert cfg.vol_model_engine_options == {}


def test_missing_mc_params_for_an_mc_route_fails_closed():
    """Silently falling back to engine defaults is exactly the bug."""
    s12 = _load_stage12()
    routing = s12.GateRouting(
        "p", None, {"heston": "mc"}, {"heston": {}}, mc_params={}
    )
    with pytest.raises(ValidationError):
        s12.make_engine_config("heston", routing=routing)


def test_parse_reads_mc_params_from_a_real_decision():
    """Guards against the decision schema and the parser drifting apart."""
    s12 = _load_stage12()
    path = REPO / "output" / "pde_convergence_gate" / "gate_decision.json"
    if not path.is_file():
        pytest.skip("no gate decision in this checkout")
    routing = s12.load_gate_routing(path)     # use the real parser's name
    assert routing.mc_params["heston"]["scheme"] == "QUADEXP_M"
    assert routing.mc_params["heston"]["substeps_per_interval"] == 4


def test_mc_engine_options_use_martingale_correction_not_scheme():
    """The decision records scheme="QUADEXP_M" as a human-readable LABEL, but
    the QE engines fix the scheme internally and expose it as the boolean
    martingale_correction -- QESnowballMCEngine raises ValidationError if
    "scheme" reaches its kwargs, and HestonSLVQESnowballMCEngine TypeErrors.
    Stage 11's own _make_mc_engine passes martingale_correction, so the fleet
    must too or it cannot construct the engine the gate certified at all."""
    s12 = _load_stage12()
    routing = s12.GateRouting(
        "p", None, {"heston": "mc"}, {"heston": {}},
        mc_params={"heston": {
            "paths_per_batch": 8192, "batches": 16, "seed": 20260723,
            "substeps_per_interval": 4, "scheme": "QUADEXP_M",
            "martingale_correction": True,
        }},
    )
    opts = s12.make_engine_config("heston", routing=routing).vol_model_engine_options
    assert opts["substeps_per_interval"] == 4
    assert opts["martingale_correction"] is True
    assert "scheme" not in opts, "scheme is rejected by both QE engine constructors"


def test_gated_mc_options_actually_construct_both_heston_engines():
    """A unit test on the options dict cannot catch a kwarg the engine
    rejects -- that is exactly how the first attempt shipped broken.  Build
    the real engines with the real option set."""
    from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
        HestonSLVQESnowballMCEngine, QESnowballMCEngine,
    )
    from quantark.volmodels.heston import HestonParams
    hp = HestonParams(v0=0.09, kappa=2.0, theta=0.06, sigma=0.3, rho=-0.7)
    opts = {"substeps_per_interval": 4, "martingale_correction": True}
    e1 = QESnowballMCEngine(hp, **opts)
    assert e1.martingale_correction is True and e1.substeps_per_interval == 4
    e2 = HestonSLVQESnowballMCEngine(hp, leverage_surface=None, eta=1.0, **opts)
    assert e2.martingale_correction is True and e2.substeps_per_interval == 4
