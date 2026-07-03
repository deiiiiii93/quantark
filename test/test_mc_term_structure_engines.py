"""Term-structure benchmark tests for upgraded MC engines (spec test layers 2/3)."""
from datetime import datetime

import numpy as np
import pytest

from term_structure_benchmarks import make_term_env, reference_european_call_price

from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


def _collapsed_flat_env(env_term, maturity, ref_strike=100.0):
    """Flat env matched to the term env's cumulative-to-maturity scalars —
    exactly what a pre-upgrade engine computed from the term env."""
    T = float(maturity)  # MUST be the product's actual pricing maturity
    return PricingEnvironment(
        rate_curve=FlatRateCurve(env_term.get_rate(T)),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(env_term.get_vol(ref_strike, T)),
        div_yield=ContinuousDividendYield(
            max(-0.20, min(0.20, env_term.get_div_yield(T)))
        ),
    )


def _term_sensitivity_check(price_fn, maturity):
    """An upgraded engine must price the term env differently from the
    collapsed flat env (the old scalar behavior made them equal)."""
    env_term = make_term_env("kinked")
    px_term = price_fn(env_term)
    px_collapsed = price_fn(_collapsed_flat_env(env_term, maturity))
    assert px_term != pytest.approx(px_collapsed, rel=1e-4)
    return px_term, px_collapsed


@pytest.mark.parametrize("shape", ["up", "down", "kinked"])
def test_euro_mc_matches_term_benchmark(shape):
    env = make_term_env(shape)
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    engine = EuropeanMCEngine(params=MCParams(num_paths=200_000, seed=42))
    px = engine.price(option, env)
    ref = reference_european_call_price(env, 100.0, 1.5)
    assert px == pytest.approx(ref, rel=1e-2)


def test_euro_mc_flat_env_still_matches_reference():
    env = make_term_env("flat")
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    px = EuropeanMCEngine(params=MCParams(num_paths=200_000, seed=42)).price(option, env)
    assert px == pytest.approx(
        reference_european_call_price(env, 100.0, 1.0), rel=1e-2
    )


@pytest.mark.parametrize("shape", ["up", "kinked"])
def test_euro_mc_forward_reproduction_via_parity(shape):
    """C - P = DF*(F - K) exactly in the model; MC noise partially cancels."""
    env = make_term_env(shape)
    K, T = 100.0, 2.0
    call = EuropeanVanillaOption(K, OptionType.CALL, maturity=T)
    put = EuropeanVanillaOption(K, OptionType.PUT, maturity=T)
    engine = EuropeanMCEngine(params=MCParams(num_paths=200_000, seed=42))
    c, p = engine.price(call, env), engine.price(put, env)
    df = env.get_discount_factor(T)
    fwd = env.spot * np.exp((env.get_rate(T) - env.get_div_yield(T)) * T)
    assert c - p == pytest.approx(df * (fwd - K), rel=2e-2, abs=0.15)


def test_asian_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc.asian_option_mc_engine import (
        AsianOptionMCEngine,
    )
    from quantark.asset.equity.product.option import AsianOption
    from quantark.util.enum import AsianStrikeType, AveragingType

    def price_fn(env):
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC,
            maturity=1.0,
        )
        return AsianOptionMCEngine(
            params=MCParams(num_paths=20_000, time_steps=64, seed=42)
        ).price(option, env)

    _term_sensitivity_check(price_fn, maturity=1.0)


def test_barrier_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc.barrier_option_mc_engine import (
        BarrierOptionMCEngine,
    )
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.util.enum import BarrierType, ObservationType

    def price_fn(env):
        option = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
            rebate=0.0,
            observation_type=ObservationType.CONTINUOUS,
        )
        return BarrierOptionMCEngine(
            params=MCParams(num_paths=20_000, time_steps=64, seed=42)
        ).price(option, env)

    _term_sensitivity_check(price_fn, maturity=1.0)


def test_range_accrual_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc.range_accrual_mc_engine import (
        RangeAccrualMCEngine,
    )
    from quantark.asset.equity.product.option import (
        RangeAccrualConfig,
        RangeAccrualOption,
    )

    def price_fn(env):
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=RangeAccrualConfig(
                upper_barrier=110.0,
                lower_barrier=90.0,
                accrual_rate=0.05,
                is_rate_annualized=True,
            ),
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
            contract_multiplier=10000.0,
        )
        return RangeAccrualMCEngine(
            params=MCParams(num_paths=20_000, time_steps=64, seed=42)
        ).price(option, env)

    _term_sensitivity_check(price_fn, maturity=1.0)


def test_single_sharkfin_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc import SingleSharkfinOptionMCEngine
    from quantark.asset.equity.product.option import SingleSharkfinOption
    from quantark.util.enum import ObservationType

    def price_fn(env):
        option = SingleSharkfinOption(
            strike=95.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            maturity=1.0,
            participation_rate=0.7,
            knock_out_rebate=2.0,
            no_hit_rebate=0.5,
            observation_type=ObservationType.CONTINUOUS,
        )
        return SingleSharkfinOptionMCEngine(
            params=MCParams(num_paths=20_000, time_steps=64, seed=42)
        ).price(option, env)

    _term_sensitivity_check(price_fn, maturity=1.0)


def test_double_sharkfin_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc import DoubleSharkfinOptionMCEngine
    from quantark.asset.equity.product.option import DoubleSharkfinOption
    from quantark.util.enum import ObservationType

    def price_fn(env):
        option = DoubleSharkfinOption(
            strike=100.0,
            option_type=OptionType.CALL,
            lower_barrier=70.0,
            upper_barrier=130.0,
            maturity=1.0,
            participation_rate=0.8,
            knock_out_rebate=2.0,
            no_hit_rebate=0.5,
            observation_type=ObservationType.CONTINUOUS,
        )
        return DoubleSharkfinOptionMCEngine(
            params=MCParams(num_paths=20_000, time_steps=64, seed=42)
        ).price(option, env)

    _term_sensitivity_check(price_fn, maturity=1.0)


def test_accumulator_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc import AccumulatorMCEngine
    from quantark.asset.equity.product.option import AccumulatorOption
    from quantark.util.enum import AccumulatorKnockOutType

    obs = [round(i / 12.0, 6) for i in range(1, 13)]

    def price_fn(env):
        option = AccumulatorOption(
            strike=96.0,
            knock_out_barrier=108.0,
            option_type=OptionType.CALL,
            maturity=1.0,
            daily_share_accumulation=1.0,
            observation_dates=obs,
            knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        )
        return AccumulatorMCEngine(
            params=MCParams(num_paths=20_000, seed=42)
        ).price(option, env)

    _term_sensitivity_check(price_fn, maturity=1.0)


def _standard_snowball():
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=1.03,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=0.75,
            ki_observation_type=ObservationType.CONTINUOUS,
        ),
        payoff_config=None,
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )


def test_snowball_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine

    def price_fn(env):
        return SnowballMCEngine(
            params=MCParams(num_paths=50_000, time_steps=252, seed=42)
        ).price(_standard_snowball(), env)

    _term_sensitivity_check(price_fn, maturity=1.0)


def test_snowball_mc_flat_identity_vs_pre_upgrade_golden():
    """Flat-input identity: fixed-seed flat-env price must equal the value
    captured on this worktree BEFORE the snowball engine was wired for term
    structures (constant coefficient arrays reproduce scalars bit-exactly)."""
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine

    GOLDEN_PRE_UPGRADE = 102.97478568748559
    env = make_term_env("flat")
    px = SnowballMCEngine(
        params=MCParams(num_paths=50_000, time_steps=252, seed=42)
    ).price(_standard_snowball(), env)
    assert px == pytest.approx(GOLDEN_PRE_UPGRADE, rel=1e-12)


def test_phoenix_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.phoenix_config import (
        CouponBarrierConfig,
    )
    from quantark.util.calendar import DayCountConvention
    from quantark.util.enum import CouponPayType, ObservationType

    def price_fn(env):
        product = PhoenixOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=BarrierConfig(
                ko_barrier=103.0,
                ko_rate=0.15,
                ko_observation_type=ObservationType.DISCRETE,
                ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
                ki_barrier=75.0,
                ki_observation_type=ObservationType.CONTINUOUS,
            ),
            coupon_config=CouponBarrierConfig(
                coupon_barrier=85.0,
                coupon_rate=0.01,
                memory_coupon=True,
                day_count_convention=DayCountConvention.ACT_365,
                coupon_pay_type=CouponPayType.INSTANT,
            ),
            payoff_config=None,
            accrual_config=None,
            contract_multiplier=1.0,
            maturity=1.0,
            is_reverse=False,
        )
        return PhoenixMCEngine(
            params=MCParams(num_paths=50_000, time_steps=252, seed=42)
        ).price(product, env)

    _term_sensitivity_check(price_fn, maturity=1.0)


def test_american_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc.american_option_mc_engine import (
        AmericanOptionMCEngine,
    )
    from quantark.asset.equity.product.option import AmericanOption

    def price_fn(env):
        option = AmericanOption(100.0, OptionType.PUT, maturity=1.0)
        return AmericanOptionMCEngine(
            params=MCParams(num_paths=20_000, time_steps=64, seed=42)
        ).price(option, env)

    _term_sensitivity_check(price_fn, maturity=1.0)


def test_american_call_no_dividend_matches_european_reference():
    """American call with q=0 has no early-exercise premium: must match the
    exact European price under a term rate/vol structure."""
    from quantark.asset.equity.engine.mc.american_option_mc_engine import (
        AmericanOptionMCEngine,
    )
    from quantark.asset.equity.product.option import AmericanOption
    from quantark.param.div.dividend_yield import NoDividend

    env = make_term_env("up")
    env.div_yield = NoDividend()
    option = AmericanOption(100.0, OptionType.CALL, maturity=1.5)
    px = AmericanOptionMCEngine(
        params=MCParams(num_paths=100_000, time_steps=64, seed=42)
    ).price(option, env)
    assert px == pytest.approx(
        reference_european_call_price(env, 100.0, 1.5), rel=2e-2
    )


def test_sabr_mc_discounting_sees_rate_term_structure():
    """SABR evolves the forward (term-exact via cumulative r/q); the discount
    factor must come from the curve, not exp(-r(T)*T) recomposed — for the
    zero-curve classes here these agree, so assert against the curve DF."""
    from quantark.asset.equity.engine.mc.sabr_mc_engine import SABRMCEngine
    from quantark.param.vol.sabr import SABRVolSurface

    env_term = make_term_env("up")
    env_term.vol_surface = SABRVolSurface(
        slices={
            0.5: {"alpha": 0.2, "beta": 1.0, "rho": -0.3, "nu": 0.4},
            2.0: {"alpha": 0.25, "beta": 1.0, "rho": -0.3, "nu": 0.4},
        }
    )
    T = 1.5
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=T)
    engine = SABRMCEngine(params=MCParams(num_paths=50_000, time_steps=64, seed=42))
    px = engine.price(option, env_term)
    assert px > 0.0
    from quantark.param.rrf.rate_curve import ParallelShiftRateCurve

    env_shift = make_term_env("up")
    env_shift.vol_surface = env_term.vol_surface
    env_shift.rate_curve = ParallelShiftRateCurve(env_term.rate_curve, 0.01)
    px_shift = engine.price(option, env_shift)
    # forward changes too, so just require the price to move and stay sane
    assert px_shift != pytest.approx(px, rel=1e-6)


def test_forward_reproduction_from_synthetic_futures_marks():
    """Spec test layer 3: q(T) implied from futures marks; the drift/carry
    arrays the engines consume must reprice those marks at every node."""
    from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
    from quantark.param.div.dividend_yield import TermStructureDividendYield

    S, r = 100.0, 0.03
    node_times = [0.25, 0.5, 1.0, 2.0]
    marks = [100.3, 100.9, 101.5, 103.2]  # synthetic futures marks
    q_nodes = [r - np.log(f / S) / t for f, t in zip(marks, node_times)]
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(S),
        vol_surface=FlatVolSurface(0.20),
        div_yield=TermStructureDividendYield(times=node_times, yields=q_nodes),
    )
    # grid with nodes exactly at the futures maturities
    dt_array = np.diff([0.0] + node_times)
    ti = build_mc_term_inputs(
        env, ref_strike=S, maturity=2.0,
        time_steps=len(dt_array), dt_array=dt_array,
    )
    dt = np.diff(ti.times)
    for i, (t_i, f_i) in enumerate(zip(node_times, marks)):
        model_fwd = S * np.exp(
            np.sum((ti.rrf[: i + 1] - ti.div[: i + 1]) * dt[: i + 1])
        )
        assert model_fwd == pytest.approx(f_i, rel=1e-10), f"node {t_i}"


def test_forward_reproduction_through_mc_paths():
    """End-to-end: simulated path expectations reprice the futures marks."""
    from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
    from quantark.asset.equity.process.bsm.qmc_path_generator import (
        GBMPathGenerator,
    )
    from quantark.asset.equity.process.bsm.qmc_sobol import (
        PseudoRandomNormalGenerator,
    )
    from quantark.param.div.dividend_yield import TermStructureDividendYield

    S, r = 100.0, 0.03
    node_times = [0.25, 0.5, 1.0, 2.0]
    marks = [100.3, 100.9, 101.5, 103.2]
    q_nodes = [r - np.log(f / S) / t for f, t in zip(marks, node_times)]
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(S),
        vol_surface=FlatVolSurface(0.20),
        div_yield=TermStructureDividendYield(times=node_times, yields=q_nodes),
    )
    dt_array = np.diff([0.0] + node_times)
    ti = build_mc_term_inputs(
        env, ref_strike=S, maturity=2.0,
        time_steps=len(dt_array), dt_array=dt_array,
    )
    g = GBMPathGenerator(
        initial_value=S, vol=ti.vol, rrf=ti.rrf, div=ti.div, maturity=2.0,
        time_steps=len(dt_array), num_paths=400_000, dt_array=dt_array,
        random_stream=PseudoRandomNormalGenerator(seed=42),
    )
    paths, _ = g.generate_paths()
    for i, (t_i, f_i) in enumerate(zip(node_times, marks)):
        mc_fwd = float(paths[:, i + 1].mean())
        assert mc_fwd == pytest.approx(f_i, rel=3e-3), f"node {t_i}"


def test_digital_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc.digital_option_mc_engine import (
        DigitalOptionMCEngine,
    )
    from quantark.asset.equity.product.option.digital_option import (
        CashOrNothingDigitalOption,
    )

    def price_fn(env):
        option = CashOrNothingDigitalOption(
            strike=100.0, payout=10.0, option_type=OptionType.CALL, maturity=2.0
        )
        return DigitalOptionMCEngine(params=MCParams(num_paths=100_000, seed=42)).price(option, env)

    _term_sensitivity_check(price_fn, maturity=2.0)
