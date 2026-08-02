import importlib.util
import sys
from pathlib import Path

import pytest

from quantark.util.enum.engine_enums import EngineType

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
    routing = s12.GateRouting("p", None, {}, {})
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
