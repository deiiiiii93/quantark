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
            EuropeanPDESolver(PDEParams(grid_size=200, time_steps=100)),
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


if __name__ == "__main__":
    main()
