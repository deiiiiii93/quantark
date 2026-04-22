"""Tests for the Snowball RFQ KO-rate demo generator."""

from __future__ import annotations

import pytest

from example import generate_snowball_rfq_ko_rate_demo as demo
from util.enum import ProtectionType


def test_stepdown_builder_no_longer_duplicates_ko_barrier() -> None:
    """Stepdown variants should build cleanly with a generated KO schedule."""
    product = demo.build_product_with_barriers(
        ko_rate=1.0,
        variant="stepdown",
        maturity=2.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )

    assert isinstance(product.barrier_config.ko_barrier, list)
    assert len(product.barrier_config.ko_barrier) == 24
    assert product.barrier_config.ko_barrier[0] == 103.0
    assert product.barrier_config.ko_barrier[-1] == pytest.approx(75.0)


def test_partial_variants_tie_protection_to_ki_level() -> None:
    """Partial-protected variants should infer the protection floor from KI."""
    product = demo.build_product_with_barriers(
        ko_rate=1.0,
        variant="standard_partial",
        maturity=2.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )
    financing_leg = demo.build_protected_product_with_barriers(
        variant="standard_partial",
        maturity=2.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )

    assert product.payoff_config.protection_type == ProtectionType.PARTIAL
    assert product.payoff_config.protection_rate == 0.25
    assert financing_leg.payoff_config.protection_type == ProtectionType.PARTIAL
    assert financing_leg.payoff_config.protection_rate == 0.25


def test_daily_ki_schedule_uses_244_business_day_convention() -> None:
    """The demo should build daily KI dates off a 244-business-day year."""
    product = demo.build_product_with_barriers(
        ko_rate=1.0,
        variant="standard",
        maturity=2.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )

    assert product.barrier_config.ki_observation_dates is not None
    assert len(product.barrier_config.ki_observation_dates) == 488
    assert product.barrier_config.ki_observation_dates[-1] == pytest.approx(2.0)


def test_build_cube_includes_tenor_axis_and_tenor_rows(monkeypatch) -> None:
    """The embedded data cube should now be keyed by tenor as the first axis."""

    def fake_solve_fair_ko_rate(
        rate: float,
        div_yield: float,
        vol: float,
        tenor: float,
        variant: str,
        **_: object,
    ) -> dict[str, float]:
        base = tenor + rate + div_yield + vol + len(variant) * 0.001
        return {
            "quoted_ko_rate": base,
            "snowball_target_pv": base + 1.0,
            "interest_component_pv": base + 1.0,
            "protected_snowball_pv": 100.0 - base,
            "combined_pv": 0.0,
        }

    monkeypatch.setattr(demo, "solve_fair_ko_rate", fake_solve_fair_ko_rate)

    cubes, rows = demo.build_cube(pde_params=demo.PDEParams(grid_size=10, time_steps=20))

    assert len(cubes["standard"]["quote"]) == len(demo.TENOR_GRID)
    assert len(cubes["standard"]["quote"][0]) == len(demo.R_GRID)
    assert len(cubes["standard"]["quote"][0][0]) == len(demo.Q_GRID)
    assert len(cubes["standard"]["quote"][0][0][0]) == len(demo.VOL_GRID)
    assert len(rows) == (
        len(demo.VARIANTS)
        * len(demo.TENOR_GRID)
        * len(demo.R_GRID)
        * len(demo.Q_GRID)
        * len(demo.VOL_GRID)
        * len(demo.KO_GRID)
        * len(demo.KI_GRID)
    )
    assert rows[0]["tenor"] in demo.TENOR_GRID
    assert rows[0]["ko_barrier"] in demo.KO_GRID
    assert rows[0]["ki_barrier"] in demo.KI_GRID
    assert rows[0]["interest_protection_type"] in {"FULL", "PARTIAL"}


def test_expand_scenario_rows_with_barriers_changes_exported_barriers() -> None:
    """CSV export rows should enumerate the configured KO/KI grids."""
    anchor_row = {
        "scenario_id": 1,
        "variant": "standard_partial",
        "tenor": 2.0,
        "r": 0.03,
        "q": 0.10,
        "vol": 0.20,
        "ko_barrier": demo.BASE_KO_BARRIER,
        "ki_barrier": demo.BASE_KI_BARRIER,
        "product_protection_type": "PARTIAL",
        "product_protection_rate": 0.25,
        "interest_protection_type": "PARTIAL",
        "interest_protection_rate": 0.25,
        "quoted_ko_rate": 0.5,
        "product_pv": 1.2,
        "interest_pv": 1.2,
        "combined_pv": 0.0,
        "protected_snowball_pv": 98.8,
        "quote_ko_sensitivity": 0.1,
        "quote_ki_sensitivity": -0.2,
        "interest_ko_sensitivity": 0.3,
        "interest_ki_sensitivity": -0.4,
    }

    rows = demo.expand_scenario_rows_with_barriers([anchor_row])

    assert len(rows) == len(demo.KO_GRID) * len(demo.KI_GRID)
    assert {row["ko_barrier"] for row in rows} == set(demo.KO_GRID)
    assert {row["ki_barrier"] for row in rows} == set(demo.KI_GRID)
    assert any(row["ko_barrier"] != demo.BASE_KO_BARRIER for row in rows)
    assert any(row["ki_barrier"] != demo.BASE_KI_BARRIER for row in rows)
