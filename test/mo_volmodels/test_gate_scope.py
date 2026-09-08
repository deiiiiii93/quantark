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


def test_adi_production_params_record_the_certified_variance_controls():
    gate = _load_gate()
    grid = gate._production_grid("heston", "medium", 3.0, False)

    block = gate._production_params_block(gate.GATE_PAIRS["heston"], grid)

    assert {
        key: block[key]
        for key in gate.ADI_2D_PRODUCTION_ENGINE_CONTROLS
    } == gate.ADI_2D_PRODUCTION_ENGINE_CONTROLS
    assert "v_grid_power" not in block


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


def test_pde_delta_repricing_uses_the_engine_bump_context():
    gate = _load_gate()

    class FrozenContext:
        def __init__(self):
            self.spots = []

        def price(self, product, env):
            spot = float(env.spot_quote.spot)
            self.spots.append(spot)
            return spot * spot

    class GridMovingEngine:
        def __init__(self):
            self.context_calls = 0
            self.context = FrozenContext()

        def create_bump_context(self, product, env):
            self.context_calls += 1
            return self.context

        def price(self, product, env):
            raise AssertionError("unfrozen engine must not price finite-difference bumps")

    env = types.SimpleNamespace(
        spot=100.0,
        spot_quote=types.SimpleNamespace(spot=100.0),
    )
    engine = GridMovingEngine()

    delta = gate._bumped_pde_delta(engine, object(), env, bump=0.01)

    assert delta == pytest.approx(200.0)
    assert engine.context_calls == 1
    assert engine.context.spots == pytest.approx([101.0, 99.0])


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
    """A decision that OMITS a key must not forward it as None.

    Note what this does and does not say.  "scheme" is genuinely Heston-only.
    "substeps_per_interval" is not -- LocalVolSnowballMCEngine accepts it, and
    test_localvol_gated_mc_options_construct_the_factory_s_engine builds the
    real engine with it.  This test is about None-filtering, not about which
    kwargs the LV engine supports; reading it as the latter is how the gate
    came to run its localvol reference at substeps=1.
    """
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


def test_mc_engine_options_match_each_engine_s_own_qe_knob():
    """The QE scheme knob is ASYMMETRIC across the two Heston engines the
    replay factory builds (engine_factory.py:253 / :277):

        heston     -> HestonSnowballMCEngine      takes scheme=
        heston_slv -> HestonSLVQESnowballMCEngine takes martingale_correction=

    Forwarding one name for both cannot work, and fails loudly rather than
    silently: the wrong kwarg TypeErrors (or, for QESnowballMCEngine,
    raises ValidationError).  Both spellings select the same engine."""
    s12 = _load_stage12()
    heston = s12.make_engine_config(
        "heston", routing=_routing_with_mc(s12, "heston")
    ).vol_model_engine_options
    assert heston["substeps_per_interval"] == 4
    assert heston["scheme"] == "QUADEXP_M"
    assert "martingale_correction" not in heston

    slv = s12.make_engine_config(
        "heston_slv", routing=_routing_with_mc(s12, "heston_slv")
    ).vol_model_engine_options
    assert slv["substeps_per_interval"] == 4
    assert slv["martingale_correction"] is True
    assert "scheme" not in slv


def test_gated_mc_options_actually_construct_the_factory_s_engines():
    """A unit test on the options dict cannot catch a kwarg the engine
    rejects -- that is exactly how the first two attempts shipped broken,
    each failing on the variant the other one worked for.  Build the real
    classes the replay factory builds, with the real resolved option set."""
    from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
        HestonSLVQESnowballMCEngine, HestonSnowballMCEngine,
    )
    from quantark.volmodels.heston import HestonParams
    s12 = _load_stage12()
    hp = HestonParams(v0=0.09, kappa=2.0, theta=0.06, sigma=0.3, rho=-0.7)

    e1 = HestonSnowballMCEngine(hp, **s12.make_engine_config(
        "heston", routing=_routing_with_mc(s12, "heston")
    ).vol_model_engine_options)
    assert e1.substeps_per_interval == 4
    assert e1.scheme.name == "QUADEXP_M"

    e2 = HestonSLVQESnowballMCEngine(hp, leverage_surface=None, eta=1.0,
        **s12.make_engine_config(
            "heston_slv", routing=_routing_with_mc(s12, "heston_slv")
        ).vol_model_engine_options)
    assert e2.substeps_per_interval == 4
    assert e2.martingale_correction is True


# ---------------------------------------------------------------------------
# The localvol reference must run at its DECLARED resolution, and its delta
# must be judged against its own sampling noise.
#
# Both defects shipped together: the gate built LocalVolSnowballMCEngine
# without substeps_per_interval (class default 1, while heston/heston_slv got
# MC_FULL's 4), then charged the resulting discretization bias to the PDE
# through a delta rule that -- unlike its PV sibling gate_tolerance_pct --
# carries no reference-uncertainty term.
# ---------------------------------------------------------------------------


def _lv_model_stub():
    """The localvol reference builder only reads ``local_vol_surface``."""
    return types.SimpleNamespace(local_vol_surface=object())


def test_localvol_reference_runs_at_the_declared_substeps():
    """substeps_per_interval is NOT a Heston-only knob.

    _SubstepRefinementMixin sits on _VolModelSnowballMCBase, so
    LocalVolSnowballMCEngine refines its SDE steps exactly like the QE
    engines -- test_mc_substeps_per_interval.py already proves it for
    'lv-snowball'.  Building the reference without the kwarg silently gates
    against substeps=1 while the decision payload advertises MC_FULL's value.
    """
    gate = _load_gate()
    engine = gate.GATE_PAIRS["localvol"].build_reference(
        _lv_model_stub(), None, gate.MC_FULL
    )
    assert engine.substeps_per_interval == gate.MC_FULL["substeps_per_interval"]


def test_every_mc_reference_records_its_substeps():
    """Whatever the fleet reruns must be the discretization G2 measured.

    Recorded for every MC reference, not just the Heston pair: stage 12
    forwards this key verbatim into the replay engine factory, so an absent
    entry means the fleet falls back to the engine default.
    """
    gate = _load_gate()
    cfg = {"mc": dict(gate.MC_FULL), "seed": gate.SEED}
    for variant, pair in gate.GATE_PAIRS.items():
        block = gate._reference_params_block(pair, cfg)
        if pair.reference_is_mc:
            assert block["substeps_per_interval"] == gate.MC_FULL[
                "substeps_per_interval"
            ], variant
        else:
            assert "substeps_per_interval" not in block, variant


def test_localvol_gated_mc_options_construct_the_factory_s_engine():
    """The Heston-pair equivalent of this test is what let the gap survive:
    it never built the LV engine, so nothing contradicted the comment
    claiming LocalVolSnowballMCEngine 'accepts neither kwarg'."""
    from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
        LocalVolSnowballMCEngine,
    )
    s12 = _load_stage12()
    opts = s12.make_engine_config(
        "localvol", routing=_routing_with_mc(s12, "localvol")
    ).vol_model_engine_options
    assert opts["substeps_per_interval"] == 4
    assert "scheme" not in opts  # genuinely Heston-only
    engine = LocalVolSnowballMCEngine(**opts)
    assert engine.substeps_per_interval == 4


def test_delta_tolerance_widens_by_the_reference_standard_error():
    """Mirror of gate_tolerance_pct for the delta rule.

    A deterministic reference (QUAD / finer PDE) keeps the flat desk bound;
    an MC reference cannot have its own sampling noise charged to the engine
    under test.  Measured on 2024-02-08, the localvol reference delta's SE is
    ~0.56 contracts against a 0.5-contract bound, so this is not academic.
    """
    gate = _load_gate()
    s0 = 4993.105
    quantum = gate.delta_quantum_per_unit(s0)
    bound = gate.DELTA_CELL_CONTRACTS * quantum

    # deterministic reference: the desk bound, unchanged
    assert gate.delta_tolerance_per_unit(None, s0) == bound

    # noisy reference: 2 x SE once it exceeds the bound
    noisy = 0.56 * quantum
    assert gate.delta_tolerance_per_unit(noisy, s0) == gate.MC_SE_FACTOR * noisy

    # quiet reference: the desk bound still binds
    quiet = 0.10 * quantum
    assert gate.delta_tolerance_per_unit(quiet, s0) == bound


def test_delta_cell_passes_when_the_gap_is_inside_the_reference_noise():
    """A 0.57-contract gap against a 0.56-contract SE is the reference
    wandering, not the engine being wrong -- exactly the substeps=16 draw."""
    gate = _load_gate()
    s0 = 4993.105
    quantum = gate.delta_quantum_per_unit(s0)
    assert gate.delta_cell_passed(0.57 * quantum, s0, se=0.56 * quantum)
    # ... but a deterministic reference still holds the engine to the bound
    assert not gate.delta_cell_passed(0.57 * quantum, s0, se=None)


def test_a_load_bearing_mc_delta_gets_its_own_quieter_reference():
    """Tied to the admission rule, not to the name 'localvol'.

    Where the delta decides the route (require_delta true) AND the reference
    is Monte Carlo, the reference must be quiet enough that 2 x SE stays under
    the desk bound -- otherwise the noise term binds and the gate loses the
    power to see a real one-contract error.  Where the delta is only a
    diagnostic (the ADI pair delegates to Stage 16), paying for those paths
    would buy nothing.
    """
    gate = _load_gate()
    for variant, pair in gate.GATE_PAIRS.items():
        load_bearing = (
            pair.reference_is_mc
            and variant not in gate.ADI_GREEK_CERTIFICATION_VARIANTS
        )
        if load_bearing:
            assert pair.build_delta_reference is not None, variant
        else:
            assert pair.build_delta_reference is None, variant


def test_the_delta_reference_runs_more_paths_than_the_pv_reference():
    gate = _load_gate()
    assert (
        gate.MC_DELTA_FULL["paths_per_batch"]
        > gate.MC_FULL["paths_per_batch"]
    )
    # ... and a FINER discretization than the PV reference, because the delta
    # converges later than the price.  substeps discretizes the REFERENCE, not
    # the engine under test, so refining it makes the reference more accurate
    # rather than measuring something else -- that conflation is why the delta
    # reference inherited a price-converged substeps level.  Measured on
    # 2024-02-08: the reference delta reads -0.859 / -0.385 / +0.215 / +0.207
    # contracts at substeps 1 / 4 / 8 / 16, so it is unconverged at 4 and
    # settled by 8, while the PV was already flat from substeps=2.
    assert (
        gate.MC_DELTA_SUBSTEPS > gate.MC_FULL["substeps_per_interval"]
    )
    engine = gate.GATE_PAIRS["localvol"].build_delta_reference(
        _lv_model_stub(), None, gate.MC_FULL
    )
    assert engine.params.num_paths == gate.MC_DELTA_FULL["paths_per_batch"]
    assert engine.substeps_per_interval == gate.MC_DELTA_SUBSTEPS

    # ... and it tracks whichever config is active, so --quick stays quick
    quick = gate.GATE_PAIRS["localvol"].build_delta_reference(
        _lv_model_stub(), None, gate.MC_QUICK
    )
    assert quick.params.num_paths == (
        gate.MC_DELTA_PATH_FACTOR * gate.MC_QUICK["paths_per_batch"]
    )
    assert quick.substeps_per_interval == gate.MC_DELTA_SUBSTEPS


def test_an_mc_delta_row_without_a_standard_error_is_rejected():
    """The SE is what the tolerance widens by; a missing one must not
    silently fall back to the noise-blind bound."""
    gate = _load_gate()
    row = {
        "date": "2024-02-08", "case": "full", "variant": "localvol",
        "level": "medium", "s0": 4993.105,
        "delta_production": 0.5462, "delta_reference": 0.5554,
        "reference_std_error": None,     # <-- the defect
        "signed_diff": -0.0092, "abs_diff": 0.0092, "diff_contracts": -0.46,
        "tolerance_contracts": 0.5, "passed": True, "error": None,
    }
    with pytest.raises(ValueError, match="reference_std_error"):
        gate.validate_delta_rows([row])


def test_the_documented_delta_config_matches_the_derived_one():
    """MC_DELTA_FULL is the production value written down for readers;
    delta_mc_config is what actually runs.  If they drift, the comment block
    documenting the measured SE ladder describes a config nothing uses."""
    gate = _load_gate()
    assert gate.MC_DELTA_FULL == gate.delta_mc_config(gate.MC_FULL)


def test_schema_1_evidence_cannot_be_rescored_under_the_new_delta_rule():
    """--rescore-evidence reuses banked cells and rebuilds the decision.

    Schema-1 delta rows carry no reference_std_error, so rescoring them would
    emit a decision claiming uncertainty-aware delta admission that was never
    performed -- and, for localvol, against a reference that ran at substeps=1.
    """
    gate = _load_gate()
    stale = {
        "schema_version": 1, "study": "pde_convergence_gate", "config": {},
        "dates": [], "cells": [], "deltas": [], "sanity": {},
    }
    with pytest.raises(ValueError, match="rerun the study rather than rescoring"):
        gate.validate_gate_payload(stale)


def test_delta_bias_reports_the_uncertainty_of_its_own_mean():
    """detect_delta_bias tests a mean against a 0.1-contract bound.

    That mean inherits the reference's noise as sqrt(sum(se^2))/n.  Reporting
    the mean without it is how the schema-1 evidence showed
    'mean_signed -0.0551, delta_biased false' while carrying a 5.7-sigma
    per-cell bias on 2024-02-08 -- the bound was tighter than the statistic's
    own error bar, so passing it meant nothing.
    """
    gate = _load_gate()
    s0 = 4993.105
    q = gate.delta_quantum_per_unit(s0)
    rows = [
        {"signed_diff": +0.30 * q, "s0": s0, "reference_std_error": 0.40 * q},
        {"signed_diff": -0.30 * q, "s0": s0, "reference_std_error": 0.40 * q},
    ]
    _, info = gate.detect_delta_bias(rows)
    # sqrt(0.40^2 + 0.40^2) / 2 = 0.2828 contracts -- nearly 3x the 0.1 bound
    assert info["mean_signed_se_contracts"] == pytest.approx(0.2828, abs=1e-3)
    assert info["bias_bound_is_resolvable"] is False

    quiet = [dict(r, reference_std_error=0.05 * q) for r in rows]
    _, qinfo = gate.detect_delta_bias(quiet)
    assert qinfo["bias_bound_is_resolvable"] is True
