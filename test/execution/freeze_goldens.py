"""Representative cross-family fixtures + golden freeze script (Phase 0).

Run once to (re)generate the checked-in goldens:
    .venv/bin/python test/execution/freeze_goldens.py
Goldens protect later phases against silent serial-path changes; they are
same-machine, version-stamped references, not cross-platform bit claims.
"""
import json
import pathlib
from datetime import datetime

GOLDEN_PATH = pathlib.Path(__file__).parent / "goldens" / "phase0_goldens.json"


def build_representative_cases() -> dict:
    import numpy as np

    from quantark.asset.bond.engine.pde import (
        ConvertibleBondPDEParams,
        ConvertibleBondTFEngine,
    )
    from quantark.asset.bond.product.convertible.convertible_bond import (
        ConvertibleBond,
    )
    from quantark.asset.credit.engine.mc import BasketCDSEngine
    from quantark.asset.credit.product import BasketCDS, BasketType
    from quantark.asset.equity.engine.mc import EuropeanMCEngine
    from quantark.asset.equity.engine.pde import EuropeanPDESolver
    from quantark.asset.equity.param import MCParams, PDEParams
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.fx.engine.mc import FxBarrierMCEngine
    from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
    from quantark.asset.fx.product import CurrencyPair
    from quantark.asset.fx.product.option import FxBarrierOption
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.credit import FlatHazardCurve
    from quantark.priceenv import (
        BasketCreditPricingEnvironment,
        FxPricingEnvironment,
        PricingEnvironment,
    )
    from quantark.util.enum import FxBarrierType, OptionType

    eq_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        valuation_date=datetime(2024, 1, 1),
    )
    eq_option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    fx_env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 15),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )
    fx_option = FxBarrierOption(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0,
    )

    n_names = 5
    corr = np.full((n_names, n_names), 0.3)
    np.fill_diagonal(corr, 1.0)
    cds_product = BasketCDS(
        notional=10_000_000.0, maturity=5.0,
        recovery_rates=[0.4] * n_names, basket_type=BasketType.FTD,
        n_to_default=1, correlation_matrix=corr,
    )
    cds_env = BasketCreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=0.03),
        hazard_curves=[FlatHazardCurve(hazard_rate=0.02)] * n_names,
    )

    cb_env = PricingEnvironment(
        valuation_date=datetime(2024, 6, 1),
        spot_quote=SpotQuote(spot=12.0),
        vol_surface=FlatVolSurface(volatility=0.30),
        rate_curve=FlatRateCurve(rate=0.05),
    )
    cb = ConvertibleBond(
        issue_date=datetime(2024, 1, 1), maturity_date=datetime(2029, 1, 1),
        face_value=100.0, coupon_rate=0.02, conversion_ratio=10.0,
        credit_spread=0.02, hazard_rate=0.01, recovery_rate=0.4,
    )

    return {
        "equity_mc_european": (
            EuropeanMCEngine(params=MCParams(num_paths=2000, seed=42)),
            eq_option, eq_env, "product_env",
        ),
        "equity_pde_european": (
            EuropeanPDESolver(PDEParams()),
            eq_option, eq_env, "product_env",
        ),
        "fx_mc_barrier": (
            FxBarrierMCEngine(
                params=FxMCParams(num_paths=20_000, time_steps=60, seed=3)
            ),
            fx_option, fx_env, "product_env",
        ),
        "credit_mc_basket_cds": (
            BasketCDSEngine(n_simulations=10_000, seed=7),
            cds_product, cds_env, "product_env",
        ),
        "bond_pde_convertible_tf": (
            ConvertibleBondTFEngine(
                cb_env,
                ConvertibleBondPDEParams(num_space_steps=50, num_time_steps=100),
            ),
            cb, None, "env_bound",
        ),
    }


def main() -> None:
    import numpy
    import scipy

    values = {}
    for name, (engine, product, env, call_shape) in build_representative_cases().items():
        if call_shape == "env_bound":
            values[name] = float(engine.price(product))
        else:
            values[name] = float(engine.price(product, env))
    payload = {
        "values": values,
        "versions": {"numpy": numpy.__version__, "scipy": scipy.__version__},
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH} with {len(values)} goldens")


# ---------------------------------------------------------------------------
# Phase 4 pre-refactor PDE goldens (independent oracle for the seam refactor;
# plan-gate finding 2026-07-16). Run ONCE on the pristine tree:
#     .venv/bin/python test/execution/freeze_goldens.py phase4
# ---------------------------------------------------------------------------

PHASE4_GOLDEN_PATH = (
    pathlib.Path(__file__).parent / "goldens" / "pde_phase4_goldens.json"
)

_PHASE4_CASES = (
    "EuropeanPDESolver",
    "SnowballPDESolver",
    "PhoenixPDESolver",
    "KOResetSnowballPDESolver",
    "LocalVolSnowballPDESolver",
    "HestonSnowballPDESolver",
)
# Direct calculate_spot_greeks_curve needs a 1D BasePDESolver._solve that runs
# standalone: 2D ADI solvers have no 1D grid, and the LV snowball's _solve
# requires the _with_surface context the public curve method does not arm.
_PHASE4_CURVE_CASES = (
    "EuropeanPDESolver",
    "SnowballPDESolver",
    "PhoenixPDESolver",
    "KOResetSnowballPDESolver",
)
_PHASE4_EVENT_CASES = (
    "SnowballPDESolver",
    "PhoenixPDESolver",
    "KOResetSnowballPDESolver",
    "LocalVolSnowballPDESolver",
    "HestonSnowballPDESolver",
)
# Refined-resolution oracle (2x the standard profile on both axes, via
# _pdep_refined) for the convergence gate; 1D representatives only.
_PHASE4_REFINED_CASES = ("EuropeanPDESolver", "SnowballPDESolver")


def _phase4_fixtures():
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
    from execution.matrix_fixtures import FIXTURE_BUILDERS, _pdep

    return FIXTURE_BUILDERS, _pdep


def _event_distribution_payload(dist) -> dict:
    return {
        "event_times": [float(t) for t in dist.event_times],
        "survival_probability": [float(p) for p in dist.survival_probability],
        "probabilities": {
            event_type.name: (
                [float(x) for x in probability]
                if hasattr(probability, "__len__")
                else float(probability)
            )
            for event_type, probability in sorted(
                dist.probabilities.items(), key=lambda kv: kv[0].name
            )
        },
    }


def _phase4_case_payload(engine, product, env, with_curve, with_events) -> dict:
    payload = {"price": float(engine.price(product, env))}
    if with_events:
        result = engine.price_with_events(product, env, emit_distribution=True)
        payload["price_with_events"] = {
            "npv": float(result.npv),
            "event_distribution": _event_distribution_payload(
                result.event_distribution
            ),
        }
    if with_curve:
        spot = float(env.spot)
        levels = [0.9 * spot, 1.0 * spot, 1.1 * spot]
        payload["spot_levels"] = levels
        payload["spot_greeks_curve"] = engine.calculate_spot_greeks_curve(
            product, env, levels
        )
    return payload


def freeze_phase4() -> None:
    import numpy
    import scipy

    builders, _pdep = _phase4_fixtures()
    cases = {}
    for name in _PHASE4_CASES:
        engine, product, env, _shape = builders[name]()
        cases[name] = _phase4_case_payload(
            engine, product, env,
            with_curve=name in _PHASE4_CURVE_CASES,
            with_events=name in _PHASE4_EVENT_CASES,
        )
    for name in _PHASE4_REFINED_CASES:
        engine, product, env, _shape = builders[name]()
        from execution.matrix_fixtures import _pdep_refined

        refined = type(engine)(params=_pdep_refined())
        cases[f"{name}::refined"] = _phase4_case_payload(
            refined, product, env,
            with_curve=name in _PHASE4_CURVE_CASES,
            with_events=name in _PHASE4_EVENT_CASES,
        )
    payload = {
        "cases": cases,
        "versions": {"numpy": numpy.__version__, "scipy": scipy.__version__},
    }
    PHASE4_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PHASE4_GOLDEN_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {PHASE4_GOLDEN_PATH} with {len(cases)} goldens")


# ---------------------------------------------------------------------------
# Phase 6 pre-refactor legacy-Dask goldens (bitwise gate for the shared batch
# reducer extraction; spec §17.3). Run ONCE on the pristine tree:
#     .venv/bin/python test/execution/freeze_goldens.py phase6_dask
# ---------------------------------------------------------------------------

PHASE6_DASK_GOLDEN_PATH = (
    pathlib.Path(__file__).parent / "goldens" / "legacy_dask_phase6_goldens.json"
)


def build_phase6_dask_cases() -> dict:
    """Engine/product/env triples exercising every legacy Dask loop variant."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
    from execution.matrix_fixtures import (
        _eq_flat_env,
        _eq_grid_env,
        _mcp,
        _phoenix,
        _snowball,
    )

    from quantark.asset.equity.engine.mc import PhoenixMCEngine, SnowballMCEngine
    from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
        LocalVolSnowballMCEngine,
    )
    from quantark.asset.equity.product.option import create_ko_reset_snowball
    from quantark.util.enum import PostKOScheduleMode
    from quantark.util.enum.engine_enums import MonteCarloMethod

    params = _mcp(num_paths=20_000, seed=42)
    ko_reset = create_ko_reset_snowball(
        initial_price=100.0, strike=100.0, maturity_pre=1.0,
        maturity_post=2.0, post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=False,
    )
    return {
        "snowball-dask": (
            SnowballMCEngine(
                params=params, method=MonteCarloMethod.PSEUDO,
                use_dask=True, num_batches=3,
            ),
            _snowball(), _eq_flat_env(),
        ),
        "ko-reset-dask": (
            SnowballMCEngine(
                params=params, method=MonteCarloMethod.PSEUDO,
                use_dask=True, num_batches=3,
            ),
            ko_reset, _eq_flat_env(),
        ),
        "phoenix-dask": (
            PhoenixMCEngine(
                params=params, method=MonteCarloMethod.PSEUDO,
                use_dask=True, num_batches=3,
            ),
            _phoenix(), _eq_flat_env(),
        ),
        "lv-snowball-dask": (
            LocalVolSnowballMCEngine(
                params=params, use_dask=True, num_batches=3,
            ),
            _snowball(), _eq_grid_env(),
        ),
    }


def _phase6_result_payload(engine, product, env) -> dict:
    engine.price(product, env)
    result = engine.get_last_result()
    return {
        "price": repr(float(result.price)),
        "std_error": repr(float(result.std_error)),
        "num_paths": int(result.num_paths),
        "ko_probability": repr(float(result.ko_probability)),
        "v0_probability": repr(float(result.v0_probability)),
        "v1_probability": repr(float(result.v1_probability)),
        "avg_ko_time": (
            None if result.avg_ko_time is None else repr(float(result.avg_ko_time))
        ),
        "batches_used": int(result.batches_used),
    }


def freeze_phase6_dask() -> None:
    import numpy
    import scipy

    cases = {
        name: _phase6_result_payload(engine, product, env)
        for name, (engine, product, env) in build_phase6_dask_cases().items()
    }
    payload = {
        "cases": cases,
        "versions": {"numpy": numpy.__version__, "scipy": scipy.__version__},
    }
    PHASE6_DASK_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PHASE6_DASK_GOLDEN_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {PHASE6_DASK_GOLDEN_PATH} with {len(cases)} goldens")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "phase4":
        freeze_phase4()
    elif len(sys.argv) > 1 and sys.argv[1] == "phase6_dask":
        freeze_phase6_dask()
    else:
        main()
