"""Executable fixtures for every concrete inventoried engine (Phase 1 gate).

Recipes are lifted from the authoritative test files (cited per family in
docs/superpowers/plans/2026-07-15-execution-framework-phase1.md). Parameters
are the cheapest observed that still price successfully.
"""
import pathlib
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # test/


# ---------------------------------------------------------------- equity core
def _eq_flat_env():
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.priceenv import PricingEnvironment

    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )


def _eq_grid_env():
    from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.priceenv import PricingEnvironment

    s0 = 100.0
    strikes = list(s0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=s0),
        vol_surface=GridVolSurface(
            strikes, maturities,
            np.full((len(maturities), len(strikes)), 0.20),
        ),
        div_yield=ContinuousDividendYield(0.01),
    )


def _euro():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType

    return EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )


def _hp():
    from quantark.volmodels.heston import HestonParams

    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _unit_leverage(s0=100.0):
    from quantark.volmodels.slv.leverage import LeverageSurface

    ks = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4), strike_grid=ks,
        leverage_grid=np.ones((4, ks.size)),
    )


def _mcp(**kw):
    from quantark.asset.equity.param import MCParams

    return MCParams(**kw)


def _pdep(**kw):
    from quantark.asset.equity.param import PDEParams

    # Pinned to the legacy event discretization the phase0/phase4 frozen
    # goldens were captured with (pre event-projection default flip,
    # 2026-07-23): these fixtures feed exact-equality oracles whose purpose is
    # characterizing the execution-seam refactor, not the event semantics.
    # rannacher_at_events mirrors the pre-decouple gate (event damping only
    # when auto_grid was on).
    kw.setdefault("event_projection", "nodal")
    kw.setdefault("event_rannacher_steps", 1)
    kw.setdefault("rannacher_at_events", bool(kw.get("auto_grid", True)))
    return PDEParams(**kw)


def _pdep_refined(**kw):
    """Refined-resolution variant for the convergence-gate oracle (double the
    standard profile on both axes)."""
    from quantark.asset.equity.engine.pde.grid import GridConfig

    kw.setdefault("grid", GridConfig(points=800, steps_per_day=8.0))
    return _pdep(**kw)


# ------------------------------------------------------------ family products
def _snowball():
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        contract_multiplier=10_000.0, is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
    )


def _phoenix():
    from quantark.asset.equity.product.option.phoenix_config import (
        CouponBarrierConfig,
    )
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_config import (
        BarrierConfig,
        PayoffConfig,
    )
    from quantark.util.calendar.day_counter import DayCountConvention
    from quantark.util.enum import CouponPayType, ObservationType

    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=90.0, coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=False,
        ),
        payoff_config=PayoffConfig(include_principal=True),
    )


def _barrier(maturity=1.0):
    from quantark.asset.equity.product.option.barrier_option import BarrierOption
    from quantark.util.enum import BarrierType, ObservationType, OptionType

    return BarrierOption(
        strike=100.0, option_type=OptionType.CALL, barrier=130.0,
        barrier_type=BarrierType.UP_OUT, maturity=maturity,
        observation_type=ObservationType.CONTINUOUS,
    )


def _dcn():
    from dcn_fixtures import DCN_A, make_dcn

    return make_dcn(DCN_A)


def _dcn_flat_env():
    from dcn_fixtures import FLAT, flat_env

    return flat_env(**FLAT)


def _dcn_grid_env():
    from dcn_fixtures import FLAT, flat_env
    from quantark.param import GridVolSurface

    env = flat_env(**FLAT)
    env.vol_surface = GridVolSurface(
        strikes=[3000.0, 4500.0, 6000.0, 7500.0, 9000.0],
        maturities=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5],
        iv_grid=np.full((6, 5), FLAT["sigma"]),
    )
    return env


def _dcn_hp():
    from quantark.volmodels.heston import HestonParams

    return HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)


# ------------------------------------------------------------------ FX shared
def _fx_env(surface):
    from quantark.param import FlatRateCurve, SpotQuote
    from quantark.priceenv import FxPricingEnvironment

    return FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=surface,
    )


def _fx_flat_env(vol=0.10):
    from quantark.param import FlatVolSurface

    return _fx_env(FlatVolSurface(volatility=vol))


def _fx_grid_env():
    from quantark.param import GridVolSurface

    strikes = list(1.20 * np.exp(np.linspace(-0.3, 0.3, 7)))
    maturities = [0.25, 0.5, 1.0, 1.5]
    return _fx_env(
        GridVolSurface(strikes, maturities, np.full((4, 7), 0.10))
    )


def _fx_vanilla():
    from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
    from quantark.util.enum import OptionType

    return FxVanillaOption(
        strike=1.20, option_type=OptionType.CALL, maturity=1.0,
        notional_foreign=1_000_000.0,
    )


def _fx_unit_leverage():
    from quantark.volmodels.slv.leverage import LeverageSurface

    ks = np.array(list(1.20 * np.exp(np.linspace(-0.4, 0.4, 9))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4), strike_grid=ks,
        leverage_grid=np.ones((4, ks.size)),
    )


def _pair():
    from quantark.asset.fx.product import CurrencyPair

    return CurrencyPair("EUR", "USD")


# ------------------------------------------------------------------- builders
def _build_equity_mc():
    out = {}

    def european():
        from quantark.asset.equity.engine.mc import EuropeanMCEngine

        return (
            EuropeanMCEngine(params=_mcp(num_paths=64, time_steps=4, seed=42)),
            _euro(), _eq_flat_env(), "product_env",
        )

    out["EuropeanMCEngine"] = european

    def local_vol():
        from quantark.asset.equity.engine.mc import LocalVolMCEngine

        return (
            LocalVolMCEngine(params=_mcp(num_paths=2_048, time_steps=24, seed=11)),
            _euro(), _eq_grid_env(), "product_env",
        )

    out["LocalVolMCEngine"] = local_vol

    def heston():
        from quantark.asset.equity.engine.mc import HestonMCEngine
        from quantark.util.enum.engine_enums import HestonMCScheme

        return (
            HestonMCEngine(_hp(), scheme=HestonMCScheme.QUADEXP,
                           params=_mcp(num_paths=2_048, time_steps=24, seed=1)),
            _euro(), _eq_flat_env(), "product_env",
        )

    out["HestonMCEngine"] = heston

    def heston_slv():
        from quantark.asset.equity.engine.mc import HestonSLVMCEngine

        return (
            HestonSLVMCEngine(_hp(), eta=1.0,
                              params=_mcp(num_paths=2_048, time_steps=24, seed=1),
                              leverage_surface=_unit_leverage()),
            _euro(), _eq_grid_env(), "product_env",
        )

    out["HestonSLVMCEngine"] = heston_slv

    def sabr():
        from quantark.asset.equity.engine.mc import SABRMCEngine
        from quantark.param import FlatRateCurve, SpotQuote
        from quantark.param.vol import SABRVolSurface
        from quantark.priceenv import PricingEnvironment

        env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.0),
            valuation_date=datetime(2026, 6, 24),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=SABRVolSurface.from_params(
                alpha=0.2, beta=1.0, rho=-0.4, nu=0.5, maturity=1.0
            ),
        )
        return (
            SABRMCEngine(params=_mcp(num_paths=4_096, time_steps=8, seed=5)),
            _euro(), env, "product_env",
        )

    out["SABRMCEngine"] = sabr

    def american():
        from quantark.asset.equity.engine.mc import AmericanOptionMCEngine
        from quantark.asset.equity.product.option import AmericanOption
        from quantark.util.enum import OptionType
        from quantark.util.enum.engine_enums import MonteCarloMethod

        return (
            AmericanOptionMCEngine(
                params=_mcp(num_paths=2_000, time_steps=50, seed=42),
                method=MonteCarloMethod.QUASI,
            ),
            AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0),
            _eq_flat_env(), "product_env",
        )

    out["AmericanOptionMCEngine"] = american

    def asian():
        from quantark.asset.equity.engine.mc import AsianOptionMCEngine
        from quantark.asset.equity.product.option import AsianOption
        from quantark.asset.equity.product.option.asian_option import (
            AsianObservationRecord,
        )
        from quantark.util.enum import AsianStrikeType, AveragingType, OptionType

        product = AsianOption(
            strike=100.0, option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC, maturity=1.0,
            observation_records=[
                AsianObservationRecord(observation_time=t)
                for t in [0.25, 0.5, 0.75, 1.0]
            ],
        )
        return (
            AsianOptionMCEngine(params=_mcp(num_paths=1_000, seed=3)),
            product, _eq_flat_env(), "product_env",
        )

    out["AsianOptionMCEngine"] = asian

    def digital():
        from quantark.asset.equity.engine.mc import DigitalOptionMCEngine
        from quantark.asset.equity.product.option.digital_option import (
            CashOrNothingDigitalOption,
        )
        from quantark.util.enum import OptionType

        return (
            DigitalOptionMCEngine(params=_mcp(num_paths=1_000, seed=3)),
            CashOrNothingDigitalOption(
                strike=100.0, option_type=OptionType.CALL,
                maturity=1.0, payout=10.0,
            ),
            _eq_flat_env(), "product_env",
        )

    out["DigitalOptionMCEngine"] = digital

    def barrier():
        from quantark.asset.equity.engine.mc import BarrierOptionMCEngine

        return (
            BarrierOptionMCEngine(params=_mcp(num_paths=2_000, time_steps=50, seed=123)),
            _barrier(), _eq_flat_env(), "product_env",
        )

    out["BarrierOptionMCEngine"] = barrier

    def _barrier_vol(kind):
        def build():
            from quantark.asset.equity.engine.mc import (
                HestonBarrierMCEngine,
                HestonSLVBarrierMCEngine,
                LocalVolBarrierMCEngine,
            )

            mcp = _mcp(num_paths=2_048, time_steps=24, seed=2)
            if kind == "lv":
                return (LocalVolBarrierMCEngine(mcp), _barrier(),
                        _eq_grid_env(), "product_env")
            if kind == "heston":
                return (HestonBarrierMCEngine(_hp(), mcp), _barrier(),
                        _eq_grid_env(), "product_env")
            return (HestonSLVBarrierMCEngine(_hp(), _unit_leverage(), mcp),
                    _barrier(), _eq_grid_env(), "product_env")

        return build

    out["LocalVolBarrierMCEngine"] = _barrier_vol("lv")
    out["HestonBarrierMCEngine"] = _barrier_vol("heston")
    out["HestonSLVBarrierMCEngine"] = _barrier_vol("slv")

    def sharkfin_single():
        from quantark.asset.equity.engine.mc import SingleSharkfinOptionMCEngine
        from quantark.asset.equity.product.option import SingleSharkfinOption
        from quantark.util.enum import ObservationType, OptionType

        return (
            SingleSharkfinOptionMCEngine(params=_mcp(num_paths=1_024, time_steps=16, seed=5)),
            SingleSharkfinOption(
                strike=95.0, option_type=OptionType.CALL, barrier=120.0,
                maturity=1.0, participation_rate=0.7, knock_out_rebate=2.0,
                no_hit_rebate=0.5, observation_type=ObservationType.EXPIRY,
            ),
            _eq_flat_env(), "product_env",
        )

    out["SingleSharkfinOptionMCEngine"] = sharkfin_single

    def sharkfin_double():
        from quantark.asset.equity.engine.mc import DoubleSharkfinOptionMCEngine
        from quantark.asset.equity.product.option import DoubleSharkfinOption
        from quantark.util.enum import ObservationType, OptionType

        return (
            DoubleSharkfinOptionMCEngine(params=_mcp(num_paths=1_024, time_steps=16, seed=123)),
            DoubleSharkfinOption(
                strike=100.0, option_type=OptionType.CALL, lower_barrier=70.0,
                upper_barrier=130.0, maturity=1.0, participation_rate=0.8,
                knock_out_rebate=2.0, no_hit_rebate=0.5,
                observation_type=ObservationType.EXPIRY,
            ),
            _eq_flat_env(), "product_env",
        )

    out["DoubleSharkfinOptionMCEngine"] = sharkfin_double

    def range_accrual():
        from quantark.asset.equity.engine.mc import RangeAccrualMCEngine
        from quantark.asset.equity.product.option.range_accrual_config import (
            RangeAccrualConfig,
        )
        from quantark.asset.equity.product.option.range_accrual_option import (
            RangeAccrualOption,
        )

        return (
            RangeAccrualMCEngine(params=_mcp(num_paths=2_000, seed=42)),
            RangeAccrualOption(
                initial_price=100.0,
                range_config=RangeAccrualConfig(
                    upper_barrier=110.0, lower_barrier=90.0,
                    accrual_rate=0.05, is_rate_annualized=True,
                ),
                observation_times=[0.25, 0.5, 0.75, 1.0],
                maturity=1.0, contract_multiplier=10_000.0,
            ),
            _eq_flat_env(), "product_env",
        )

    out["RangeAccrualMCEngine"] = range_accrual

    def accumulator():
        from quantark.asset.equity.engine.mc import AccumulatorMCEngine
        from quantark.asset.equity.product.option import AccumulatorOption
        from quantark.util.enum import AccumulatorKnockOutType, OptionType

        obs = [round(m / 12.0, 6) for m in range(1, 13)]
        return (
            AccumulatorMCEngine(_mcp(num_paths=2_000, seed=7)),
            AccumulatorOption(
                strike=96.0, knock_out_barrier=1.0e6,
                option_type=OptionType.CALL, maturity=1.0,
                daily_share_accumulation=1.0, gearing=2.0,
                knock_out_type=AccumulatorKnockOutType.TERMINATION,
                observation_dates=obs,
            ),
            _eq_flat_env(), "product_env",
        )

    out["AccumulatorMCEngine"] = accumulator

    def snowball_base():
        from quantark.asset.equity.engine.mc import SnowballMCEngine

        return (
            SnowballMCEngine(params=_mcp(num_paths=2_000, time_steps=64, seed=7)),
            _snowball(), _eq_flat_env(), "product_env",
        )

    out["SnowballMCEngine"] = snowball_base

    def phoenix_base():
        from quantark.asset.equity.engine.mc import PhoenixMCEngine

        return (
            PhoenixMCEngine(params=_mcp(num_paths=2_000, seed=7)),
            _phoenix(), _eq_flat_env(), "product_env",
        )

    out["PhoenixMCEngine"] = phoenix_base

    def _autocall_vol(name):
        def build():
            from quantark.asset.equity.engine.mc import (
                HestonPhoenixMCEngine, HestonSLVPhoenixMCEngine,
                HestonSLVQEPhoenixMCEngine, HestonSLVQESnowballMCEngine,
                HestonSLVSnowballMCEngine, HestonSnowballMCEngine,
                LocalVolPhoenixMCEngine, LocalVolSnowballMCEngine,
                QEPhoenixMCEngine, QESnowballMCEngine,
            )

            mcp = _mcp(num_paths=1_024, time_steps=24, seed=19)
            product = _snowball() if "Snowball" in name else _phoenix()
            cls = {
                "LocalVolSnowballMCEngine": lambda: LocalVolSnowballMCEngine(mcp),
                "HestonSnowballMCEngine": lambda: HestonSnowballMCEngine(_hp(), mcp),
                "QESnowballMCEngine": lambda: QESnowballMCEngine(_hp(), mcp),
                "HestonSLVSnowballMCEngine": lambda: HestonSLVSnowballMCEngine(
                    _hp(), params=mcp, leverage_surface=_unit_leverage()),
                "HestonSLVQESnowballMCEngine": lambda: HestonSLVQESnowballMCEngine(
                    _hp(), params=mcp, leverage_surface=_unit_leverage()),
                "LocalVolPhoenixMCEngine": lambda: LocalVolPhoenixMCEngine(mcp),
                "HestonPhoenixMCEngine": lambda: HestonPhoenixMCEngine(_hp(), mcp),
                "QEPhoenixMCEngine": lambda: QEPhoenixMCEngine(_hp(), mcp),
                "HestonSLVPhoenixMCEngine": lambda: HestonSLVPhoenixMCEngine(
                    _hp(), params=mcp, leverage_surface=_unit_leverage()),
                "HestonSLVQEPhoenixMCEngine": lambda: HestonSLVQEPhoenixMCEngine(
                    _hp(), params=mcp, leverage_surface=_unit_leverage()),
            }[name]
            return cls(), product, _eq_grid_env(), "product_env"

        return build

    for name in [
        "LocalVolSnowballMCEngine", "HestonSnowballMCEngine",
        "QESnowballMCEngine", "HestonSLVSnowballMCEngine",
        "HestonSLVQESnowballMCEngine", "LocalVolPhoenixMCEngine",
        "HestonPhoenixMCEngine", "QEPhoenixMCEngine",
        "HestonSLVPhoenixMCEngine", "HestonSLVQEPhoenixMCEngine",
    ]:
        out[name] = _autocall_vol(name)

    def dcn():
        from quantark.asset.equity.engine.mc import DCNMCEngine

        return (DCNMCEngine(num_paths=2**9, seed=42), _dcn(),
                _dcn_flat_env(), "product_env")

    out["DCNMCEngine"] = dcn

    def dcn_lv():
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine

        return (LocalVolDCNMCEngine(num_paths=2**9, seed=42), _dcn(),
                _dcn_grid_env(), "product_env")

    out["LocalVolDCNMCEngine"] = dcn_lv

    def dcn_heston():
        from quantark.asset.equity.engine.mc import HestonDCNMCEngine

        return (HestonDCNMCEngine(model_params=_dcn_hp(), num_paths=2**9, seed=42),
                _dcn(), _dcn_flat_env(), "product_env")

    out["HestonDCNMCEngine"] = dcn_heston

    def dcn_qe():
        from quantark.asset.equity.engine.mc import QEDCNMCEngine

        return (QEDCNMCEngine(_dcn_hp(), num_paths=2**9, seed=7), _dcn(),
                _dcn_flat_env(), "product_env")

    out["QEDCNMCEngine"] = dcn_qe

    def dcn_coupled():
        from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
            coupled_heston_ladder_pair,
        )
        from quantark.util.enum.engine_enums import HestonMCScheme

        coarse, _fine = coupled_heston_ladder_pair(
            _dcn_hp(), 2, HestonMCScheme.QUADEXP_M,
            num_paths=2**9, seed=42, use_sobol=True, num_batches=1,
        )
        return coarse, _dcn(), _dcn_flat_env(), "product_env"

    out["CoupledCoarseHestonDCNMCEngine"] = dcn_coupled
    return out


def _build_equity_pde():
    out = {}

    def _simple(cls_name, product_fn, grid_env):
        def build():
            import quantark.asset.equity.engine.pde as pde_mod

            cls = getattr(pde_mod, cls_name)
            engine = cls(_pdep())
            env = _eq_grid_env() if grid_env else _eq_flat_env()
            return engine, product_fn(), env, "product_env"

        return build

    simple = {
        "EuropeanPDESolver": (_euro, False),
        "LocalVolPDESolver": (_euro, True),
        "SnowballPDESolver": (_snowball, False),
        "LocalVolSnowballPDESolver": (_snowball, True),
        "PhoenixPDESolver": (_phoenix, False),
        "LocalVolPhoenixPDESolver": (_phoenix, True),
        "BarrierPDESolver": (lambda: _barrier(0.5), False),
        "LocalVolBarrierPDESolver": (_barrier, True),
    }
    for name, (product_fn, grid_env) in simple.items():
        out[name] = _simple(name, product_fn, grid_env)

    def american_pde():
        from quantark.asset.equity.engine.pde import AmericanPDESolver
        from quantark.asset.equity.product.option import AmericanOption
        from quantark.util.enum import OptionType

        return (
            AmericanPDESolver(_pdep()),
            AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0),
            _eq_flat_env(), "product_env",
        )

    out["AmericanPDESolver"] = american_pde

    def double_barrier():
        from quantark.asset.equity.engine.pde import DoubleBarrierPDESolver
        from quantark.asset.equity.product.option import DoubleBarrierOption
        from quantark.util.enum import DoubleBarrierType, OptionType

        return (
            DoubleBarrierPDESolver(_pdep()),
            DoubleBarrierOption(
                strike=100.0, option_type=OptionType.CALL, upper_barrier=120.0,
                lower_barrier=80.0, barrier_type=DoubleBarrierType.KNOCK_OUT,
                maturity=0.5,
            ),
            _eq_flat_env(), "product_env",
        )

    out["DoubleBarrierPDESolver"] = double_barrier

    def one_touch():
        from quantark.asset.equity.engine.pde import OneTouchPDESolver
        from quantark.asset.equity.product.option import OneTouchOption
        from quantark.util.enum import BarrierDirection, TouchType

        return (
            OneTouchPDESolver(_pdep()),
            OneTouchOption(
                barrier=110.0, barrier_direction=BarrierDirection.UP,
                maturity=1.0, rebate=100.0, payment_at_hit=True,
                touch_type=TouchType.ONE_TOUCH,
            ),
            _eq_flat_env(), "product_env",
        )

    out["OneTouchPDESolver"] = one_touch

    def double_one_touch():
        from quantark.asset.equity.engine.pde import DoubleOneTouchPDESolver
        from quantark.asset.equity.product.option import DoubleOneTouchOption
        from quantark.util.enum import TouchType

        return (
            DoubleOneTouchPDESolver(_pdep()),
            DoubleOneTouchOption(
                upper_barrier=110.0, lower_barrier=90.0, maturity=1.0,
                rebate=100.0, payment_at_hit=True,
                touch_type=TouchType.DOUBLE_ONE_TOUCH,
            ),
            _eq_flat_env(), "product_env",
        )

    out["DoubleOneTouchPDESolver"] = double_one_touch

    def ko_reset():
        from quantark.asset.equity.engine.pde import KOResetSnowballPDESolver
        from quantark.asset.equity.product.option import create_ko_reset_snowball
        from quantark.util.enum import PostKOScheduleMode

        return (
            KOResetSnowballPDESolver(_pdep()),
            create_ko_reset_snowball(
                initial_price=100.0, strike=100.0, maturity_pre=1.0,
                maturity_post=2.0, post_ko_mode=PostKOScheduleMode.ABSOLUTE,
                ki_continuous=False,
            ),
            _eq_flat_env(), "product_env",
        )

    out["KOResetSnowballPDESolver"] = ko_reset

    def _heston_pde(name):
        def build():
            from quantark.asset.equity.engine.pde import (
                HestonBarrierPDESolver, HestonPDESolver,
                HestonPhoenixPDESolver, HestonSLVBarrierPDESolver,
                HestonSLVPDESolver, HestonSLVPhoenixPDESolver,
                HestonSLVSnowballPDESolver, HestonSnowballPDESolver,
            )

            grid = dict(n_x=48, n_v=18, n_t=16)
            # Autocallable 2D solvers consume event_projection: pin the legacy
            # event discretization these frozen goldens were captured with
            # (see _pdep note). 2D legacy additionally means NO per-event ADI
            # damping (event_rannacher_steps was inert in the ADI loop until
            # 2026-07-23), so pin ers=0 here — _pdep's ers=1 is the 1D legacy.
            # Vanilla/barrier Heston engines have no discrete events (and
            # take engine_params, not params).
            #
            # v0_boundary is pinned for the same reason: the snowball/phoenix
            # 2D solvers moved their default to "degenerate_pde" on 2026-07-31
            # (Feller-boundary evidence, re-baseline spec 7A.6), and these
            # goldens were frozen under the ADI core's "neumann". A deliberate
            # default change is not the seam refactor this oracle exists to
            # police, so pin the frozen value rather than re-baseline it.
            acgrid = dict(
                grid,
                params=_pdep(event_rannacher_steps=0),
                v0_boundary="neumann",
            )
            # Snowball Heston/SLV moved their concentrated variance-grid
            # default to power grading on 2026-08-03. These pre-refactor
            # goldens intentionally preserve the old numerical configuration,
            # just like the v0 boundary pin above.
            snowball_acgrid = dict(acgrid, v_grid_power=0.0)
            table = {
                "HestonPDESolver": lambda: (HestonPDESolver(_hp(), **grid), _euro(), _eq_flat_env()),
                "HestonSLVPDESolver": lambda: (HestonSLVPDESolver(_hp(), _unit_leverage(), eta=1.0, **grid), _euro(), _eq_grid_env()),
                "HestonBarrierPDESolver": lambda: (HestonBarrierPDESolver(_hp(), **grid), _barrier(), _eq_grid_env()),
                "HestonSLVBarrierPDESolver": lambda: (HestonSLVBarrierPDESolver(_hp(), _unit_leverage(), **grid), _barrier(), _eq_grid_env()),
                "HestonSnowballPDESolver": lambda: (HestonSnowballPDESolver(_hp(), **snowball_acgrid), _snowball(), _eq_grid_env()),
                "HestonSLVSnowballPDESolver": lambda: (HestonSLVSnowballPDESolver(_hp(), _unit_leverage(), **snowball_acgrid), _snowball(), _eq_grid_env()),
                "HestonPhoenixPDESolver": lambda: (HestonPhoenixPDESolver(_hp(), grid_style="uniform", **acgrid), _phoenix(), _eq_grid_env()),
                "HestonSLVPhoenixPDESolver": lambda: (HestonSLVPhoenixPDESolver(_hp(), _unit_leverage(), grid_style="uniform", **acgrid), _phoenix(), _eq_grid_env()),
            }
            engine, product, env = table[name]()
            return engine, product, env, "product_env"

        return build

    for name in [
        "HestonPDESolver", "HestonSLVPDESolver", "HestonBarrierPDESolver",
        "HestonSLVBarrierPDESolver", "HestonSnowballPDESolver",
        "HestonSLVSnowballPDESolver", "HestonPhoenixPDESolver",
        "HestonSLVPhoenixPDESolver",
    ]:
        out[name] = _heston_pde(name)

    def dcn_pde():
        from quantark.asset.equity.engine.pde import DCNPDEEngine

        return (DCNPDEEngine(num_space_nodes=301), _dcn(),
                _dcn_flat_env(), "product_env")

    out["DCNPDEEngine"] = dcn_pde

    def dcn_lv_pde():
        from quantark.asset.equity.engine.pde import LocalVolDCNPDEEngine

        return (LocalVolDCNPDEEngine(num_space_nodes=301), _dcn(),
                _dcn_grid_env(), "product_env")

    out["LocalVolDCNPDEEngine"] = dcn_lv_pde

    def dcn_heston_pde():
        from quantark.asset.equity.engine.pde import HestonDCNPDESolver

        return (HestonDCNPDESolver(_dcn_hp(), n_x=151, n_v=41), _dcn(),
                _dcn_flat_env(), "product_env")

    out["HestonDCNPDESolver"] = dcn_heston_pde

    def pde_facade():
        from quantark.asset.equity.engine.pde_engine import PDEEngine

        return (
            PDEEngine(_pdep()),
            _euro(), _eq_flat_env(), "product_env",
        )

    out["PDEEngine"] = pde_facade
    return out


def _build_fx():
    out = {}

    def fx_lv_mc():
        from quantark.asset.fx.engine.mc import FxLocalVolMCEngine

        return (FxLocalVolMCEngine(num_paths=4_000, time_steps=24, seed=5),
                _fx_vanilla(), _fx_grid_env(), "product_env")

    out["FxLocalVolMCEngine"] = fx_lv_mc

    def fx_heston_mc():
        from quantark.asset.fx.engine.mc import FxHestonMCEngine

        return (FxHestonMCEngine(_hp(), num_paths=4_000, time_steps=24, seed=9),
                _fx_vanilla(), _fx_flat_env(), "product_env")

    out["FxHestonMCEngine"] = fx_heston_mc

    def fx_slv_mc():
        from quantark.asset.fx.engine.mc import FxHestonSLVMCEngine

        return (
            FxHestonSLVMCEngine(_hp(), eta=1.0, num_paths=4_000,
                                time_steps=24, seed=9),
            _fx_vanilla(), _fx_grid_env(), "product_env",
        )

    out["FxHestonSLVMCEngine"] = fx_slv_mc

    def fx_lv_pde():
        from quantark.asset.fx.engine.pde import FxLocalVolPDESolver

        return (FxLocalVolPDESolver(grid_size=120, time_steps=48),
                _fx_vanilla(), _fx_grid_env(), "product_env")

    out["FxLocalVolPDESolver"] = fx_lv_pde

    def fx_heston_pde():
        from quantark.asset.fx.engine.pde import FxHestonPDESolver

        return (FxHestonPDESolver(_hp(), n_x=48, n_v=18, n_t=16),
                _fx_vanilla(), _fx_flat_env(), "product_env")

    out["FxHestonPDESolver"] = fx_heston_pde

    def fx_slv_pde():
        from quantark.asset.fx.engine.pde import FxHestonSLVPDESolver

        return (
            FxHestonSLVPDESolver(_hp(), _fx_unit_leverage(), eta=1.0,
                                 n_x=48, n_v=18, n_t=16),
            _fx_vanilla(), _fx_grid_env(), "product_env",
        )

    out["FxHestonSLVPDESolver"] = fx_slv_pde

    def _fx_params(**kw):
        from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams

        return FxMCParams(**kw)

    def fx_range_accrual():
        from quantark.asset.fx.engine.mc import FxRangeAccrualMCEngine
        from quantark.asset.fx.product.option.fx_range_accrual_option import (
            FxRangeAccrualConfig, FxRangeAccrualOption,
        )

        return (
            FxRangeAccrualMCEngine(params=_fx_params(num_paths=2_000, seed=1)),
            FxRangeAccrualOption(
                notional=1_000_000.0,
                range_config=FxRangeAccrualConfig(
                    upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04
                ),
                currency_pair=_pair(), maturity=1.0, num_observations=12,
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxRangeAccrualMCEngine"] = fx_range_accrual

    def fx_barrier():
        from quantark.asset.fx.engine.mc import FxBarrierMCEngine
        from quantark.asset.fx.product.option import FxBarrierOption
        from quantark.util.enum import FxBarrierType, OptionType

        return (
            FxBarrierMCEngine(params=_fx_params(num_paths=4_000, time_steps=24, seed=3)),
            FxBarrierOption(
                strike=1.20, barrier=1.35, is_up=True,
                knock_type=FxBarrierType.KNOCK_OUT,
                option_type=OptionType.CALL,
                currency_pair=_pair(), maturity=1.0,
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxBarrierMCEngine"] = fx_barrier

    def fx_sharkfin():
        from quantark.asset.fx.engine.mc import FxSharkfinMCEngine
        from quantark.asset.fx.product.option import FxSharkfinOption
        from quantark.util.enum import OptionType

        return (
            FxSharkfinMCEngine(params=_fx_params(num_paths=4_000, time_steps=24, seed=3)),
            FxSharkfinOption(
                strike=1.20, barrier=1.35, is_up=True,
                option_type=OptionType.CALL,
                currency_pair=_pair(), maturity=1.0,
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxSharkfinMCEngine"] = fx_sharkfin

    def fx_tarf():
        from quantark.asset.fx.engine.mc import FxTarnForwardMCEngine
        from quantark.asset.fx.product.option import FxTargetRedemptionForward

        return (
            FxTarnForwardMCEngine(params=_fx_params(num_paths=2_000, seed=7)),
            FxTargetRedemptionForward(
                strike=1.20, fixing_times=[0.25, 0.5, 0.75, 1.0],
                currency_pair=_pair(),
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxTarnForwardMCEngine"] = fx_tarf

    def fx_tarn():
        from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine
        from quantark.asset.fx.product.option import FxTargetRedemptionNote

        return (
            FxTargetRedemptionNoteMCEngine(params=_fx_params(num_paths=2_000, seed=11)),
            FxTargetRedemptionNote(
                fixing_times=[0.25, 0.5, 0.75, 1.0], coupon_rate=0.08,
                notional=1.0, strike=1.20, currency_pair=_pair(),
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxTargetRedemptionNoteMCEngine"] = fx_tarn
    return out


def _build_credit_bond():
    out = {}

    def basket_cds():
        from quantark.asset.credit.engine.mc import BasketCDSEngine
        from quantark.asset.credit.product import BasketCDS, BasketType
        from quantark.param import FlatRateCurve
        from quantark.param.credit import FlatHazardCurve
        from quantark.priceenv import BasketCreditPricingEnvironment

        n = 5
        corr = np.full((n, n), 0.3)
        np.fill_diagonal(corr, 1.0)
        return (
            BasketCDSEngine(n_simulations=5_000, seed=11),
            BasketCDS(
                notional=10_000_000.0, maturity=5.0,
                recovery_rates=[0.4] * n, basket_type=BasketType.FTD,
                n_to_default=1, correlation_matrix=corr,
            ),
            BasketCreditPricingEnvironment(
                valuation_date=datetime(2026, 6, 13),
                discount_curve=FlatRateCurve(rate=0.03),
                hazard_curves=[FlatHazardCurve(hazard_rate=0.02)] * n,
            ),
            "product_env",
        )

    out["BasketCDSEngine"] = basket_cds

    def _cb_fixture():
        from quantark.asset.bond.product.convertible.convertible_bond import (
            ConvertibleBond,
        )
        from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
        from quantark.priceenv import PricingEnvironment

        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1), maturity_date=datetime(2029, 1, 1),
            face_value=100.0, coupon_rate=0.02, conversion_ratio=10.0,
            credit_spread=0.02, hazard_rate=0.01, recovery_rate=0.4,
        )
        env = PricingEnvironment(
            valuation_date=datetime(2024, 6, 1),
            spot_quote=SpotQuote(spot=12.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )
        return cb, env

    def cb_jump():
        from quantark.asset.bond.engine.pde import (
            ConvertibleBondJumpDiffusionEngine, ConvertibleBondPDEParams,
        )

        cb, env = _cb_fixture()
        return (
            ConvertibleBondJumpDiffusionEngine(
                env, ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
            ),
            cb, None, "env_bound",
        )

    out["ConvertibleBondJumpDiffusionEngine"] = cb_jump

    def cb_tf():
        from quantark.asset.bond.engine.pde import (
            ConvertibleBondPDEParams, ConvertibleBondTFEngine,
        )

        cb, env = _cb_fixture()
        return (
            ConvertibleBondTFEngine(
                env, ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
            ),
            cb, None, "env_bound",
        )

    out["ConvertibleBondTFEngine"] = cb_tf

    def cb_facade():
        from quantark.asset.bond.engine.convertible import ConvertibleBondEngine
        from quantark.util.enum.engine_enums import (
            ConvertibleBondMethod, EngineType,
        )

        cb, env = _cb_fixture()
        return (
            ConvertibleBondEngine(
                env, method=EngineType.PDE(ConvertibleBondMethod.TF)
            ),
            cb, None, "env_bound",
        )

    out["ConvertibleBondEngine"] = cb_facade
    return out


FIXTURE_BUILDERS = {
    **_build_equity_mc(),
    **_build_equity_pde(),
    **_build_fx(),
    **_build_credit_bond(),
}
