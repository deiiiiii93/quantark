"""Spawn-importable surface-shock base-input factory for Phase 5 tests.

Rebuilds the synthetic cleaned-quote fixture and the DCN product
deterministically in ANY process — the registered-factory replacement for
the solution script's ``_init_worker`` global dict.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # test/

from quantark.asset.equity.riskmeasures.surface_shock_scenarios import (  # noqa: E402
    SurfaceShockInputs,
)
from quantark.execution.scenario import registries  # noqa: E402

SURFACE_SHOCK_TEST_FACTORY_ID = "surface-shock-test-inputs/v1"


def build_surface_shock_test_inputs(payload: dict) -> SurfaceShockInputs:
    from dcn_fixtures import DCN_A, make_dcn, synthetic_cleaned_set

    cleaned, rate_curve, carry_curve, _ = synthetic_cleaned_set()
    settings = [
        ("num_paths", payload["num_paths"]),
        ("seed", payload["seed"]),
        ("engine_workers", payload.get("engine_workers", 1)),
    ]
    if payload.get("max_calibration_rmse_iv") is not None:
        settings.append(
            ("max_calibration_rmse_iv", payload["max_calibration_rmse_iv"])
        )
    if payload.get("heston_calibration_config") is not None:
        settings.append(
            (
                "heston_calibration_config",
                tuple(
                    (key, value)
                    for key, value in sorted(
                        dict(payload["heston_calibration_config"]).items()
                    )
                ),
            )
        )
    return SurfaceShockInputs(
        cleaned=cleaned,
        spot=cleaned.spot,
        rate_curve=rate_curve,
        carry_curve=carry_curve,
        product=make_dcn(DCN_A),
        engine_settings=tuple(sorted(settings)),
    )


registries.register_factory(
    SURFACE_SHOCK_TEST_FACTORY_ID, build_surface_shock_test_inputs
)
