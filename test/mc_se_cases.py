"""Fixed-seed pricing cases for the antithetic standard-error tests (T7 / spec I3, A9).

Each case builds an engine + product on a fixed seed and returns ``(price, std_error)``.
``run_case(name, antithetic, n_paths)`` is also used to capture the pre-change values.
"""
from __future__ import annotations

from datetime import datetime

from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

_VAL = datetime(2024, 1, 1)


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=_VAL,
    )


def _fx_env():
    from quantark.priceenv import FxPricingEnvironment

    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 15),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )


def _mc(n, antithetic, steps=24):
    from quantark.asset.equity.param import MCParams

    return MCParams(num_paths=n, time_steps=steps, seed=11, use_antithetic=antithetic)


def _se_attr(engine):
    if hasattr(engine, "get_last_std_error") and engine.get_last_std_error() is not None:
        return float(engine.get_last_std_error())
    return float(engine.get_last_result().std_error)


def _euro(n, a):
    from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType
    from quantark.util.enum.engine_enums import MonteCarloMethod

    e = EuropeanMCEngine(params=_mc(n, a), method=MonteCarloMethod.PSEUDO)
    p = e.price(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0), _env())
    return p, _se_attr(e)


def _binary(n, a):
    from quantark.asset.equity.engine.mc.digital_option_mc_engine import DigitalOptionMCEngine
    from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
    from quantark.util.enum import OptionType
    from quantark.util.enum.engine_enums import MonteCarloMethod

    e = DigitalOptionMCEngine(params=_mc(n, a), method=MonteCarloMethod.PSEUDO)
    p = e.price(CashOrNothingDigitalOption(strike=100.0, payout=10.0, option_type=OptionType.CALL, maturity=1.0), _env())
    return p, _se_attr(e)


def _asian(n, a):
    from quantark.asset.equity.engine.mc import AsianOptionMCEngine
    from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
    from quantark.util.enum import OptionType
    from quantark.util.enum.engine_enums import MonteCarloMethod

    times = [0.25, 0.5, 0.75, 1.0]
    opt = AsianOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0,
        observation_records=[AsianObservationRecord(observation_time=t) for t in times],
    )
    e = AsianOptionMCEngine(params=_mc(n, a), method=MonteCarloMethod.PSEUDO)
    p = e.price(opt, _env())
    return p, _se_attr(e)


def _range_accrual(n, a):
    from quantark.asset.equity.engine.mc import RangeAccrualMCEngine
    from quantark.asset.equity.product.option import RangeAccrualConfig, RangeAccrualOption
    from quantark.util.enum.engine_enums import MonteCarloMethod

    cfg = RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.05, is_rate_annualized=True)
    opt = RangeAccrualOption(initial_price=100.0, range_config=cfg, observation_times=[0.25, 0.5, 0.75, 1.0],
                             maturity=1.0, contract_multiplier=10000.0)
    e = RangeAccrualMCEngine(params=_mc(n, a), method=MonteCarloMethod.PSEUDO)
    p = e.price(opt, _env())
    return p, _se_attr(e)


def _american(n, a):
    from quantark.asset.equity.engine.mc import AmericanOptionMCEngine
    from quantark.asset.equity.product.option import AmericanOption
    from quantark.util.enum import OptionType
    from quantark.util.enum.engine_enums import MonteCarloMethod

    e = AmericanOptionMCEngine(params=_mc(n, a), method=MonteCarloMethod.PSEUDO)
    p = e.price(AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0), _env())
    return p, _se_attr(e)


def _barrier(obs, n, a):
    from quantark.asset.equity.engine.mc.barrier_option_mc_engine import BarrierOptionMCEngine
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.util.enum import BarrierType, ObservationType, OptionType
    from quantark.util.enum.engine_enums import MonteCarloMethod

    opt = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=125.0,
                        barrier_type=BarrierType.UP_OUT, maturity=1.0, rebate=0.0,
                        observation_type=getattr(ObservationType, obs),
                        observation_dates=[i / 12.0 for i in range(1, 13)] if obs == "DISCRETE" else None)
    e = BarrierOptionMCEngine(params=_mc(n, a), method=MonteCarloMethod.PSEUDO)
    p = e.price(opt, _env())
    return p, _se_attr(e)


def _accumulator(n, a):
    from quantark.asset.equity.engine.mc import AccumulatorMCEngine
    from quantark.asset.equity.product.option import AccumulatorOption
    from quantark.util.enum import AccumulatorKnockOutType, OptionType
    from quantark.util.enum.engine_enums import MonteCarloMethod

    acc = AccumulatorOption(
        strike=96.0, knock_out_barrier=107.0, option_type=OptionType.CALL, maturity=1.0,
        daily_share_accumulation=1.0, gearing=2.0,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        observation_dates=[round((i + 1) / 12.0, 10) for i in range(12)],
    )
    e = AccumulatorMCEngine(_mc(n, a), method=MonteCarloMethod.PSEUDO)
    p = e.price(acc, _env())
    return p, _se_attr(e)


def _single_shark(obs, n, a):
    from quantark.asset.equity.engine.mc import SingleSharkfinOptionMCEngine
    from quantark.asset.equity.product.option import SingleSharkfinOption
    from quantark.util.enum import ObservationFrequency, ObservationType, OptionType
    from quantark.util.enum.engine_enums import MonteCarloMethod

    kw = dict(strike=100.0, option_type=OptionType.CALL, barrier=120.0, maturity=1.0,
              participation_rate=1.0, knock_out_rebate=1.0, no_hit_rebate=0.25,
              observation_type=getattr(ObservationType, obs))
    if obs == "DISCRETE":
        kw["observation_frequency"] = ObservationFrequency.MONTHLY
    e = SingleSharkfinOptionMCEngine(params=_mc(n, a), method=MonteCarloMethod.PSEUDO)
    p = e.price(SingleSharkfinOption(**kw), _env())
    return p, _se_attr(e)


def _double_shark(obs, n, a):
    from quantark.asset.equity.engine.mc import DoubleSharkfinOptionMCEngine
    from quantark.asset.equity.product.option import DoubleSharkfinOption
    from quantark.util.enum import ObservationFrequency, ObservationType, OptionType
    from quantark.util.enum.engine_enums import MonteCarloMethod

    kw = dict(strike=100.0, option_type=OptionType.CALL, lower_barrier=70.0, upper_barrier=130.0,
              maturity=1.0, participation_rate=0.8, knock_out_rebate=2.0, no_hit_rebate=0.5,
              observation_type=getattr(ObservationType, obs))
    if obs == "DISCRETE":
        kw["observation_frequency"] = ObservationFrequency.MONTHLY
    e = DoubleSharkfinOptionMCEngine(params=_mc(n, a), method=MonteCarloMethod.PSEUDO)
    p = e.price(DoubleSharkfinOption(**kw), _env())
    return p, _se_attr(e)


def _fx_mc(n, a):
    from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams

    return FxMCParams(num_paths=n, time_steps=24, seed=3, use_antithetic=a)


def _fx_barrier(mon, n, a):
    from quantark.asset.fx.engine.mc import FxBarrierMCEngine
    from quantark.asset.fx.product import CurrencyPair
    from quantark.asset.fx.product.option import FxBarrierOption
    from quantark.util.enum import FxBarrierType, ObservationType, OptionType

    kw = dict(strike=1.20, barrier=1.35, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
              option_type=OptionType.CALL, currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0)
    if mon == "DISCRETE":
        kw.update(monitoring=ObservationType.DISCRETE, observation_times=[0.25, 0.5, 0.75, 1.0])
    e = FxBarrierMCEngine(params=_fx_mc(n, a))
    p = e.price(FxBarrierOption(**kw), _fx_env())
    return p, float(e.get_last_result().std_error)


def _fx_shark(mon, n, a):
    from quantark.asset.fx.engine.mc import FxSharkfinMCEngine
    from quantark.asset.fx.product import CurrencyPair
    from quantark.asset.fx.product.option import FxSharkfinOption
    from quantark.util.enum import ObservationType, OptionType

    kw = dict(strike=1.20, barrier=1.35, is_up=True, option_type=OptionType.CALL,
              currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0)
    if mon == "DISCRETE":
        kw.update(monitoring=ObservationType.DISCRETE, observation_times=[0.25, 0.5, 0.75, 1.0])
    e = FxSharkfinMCEngine(params=_fx_mc(n, a))
    p = e.price(FxSharkfinOption(**kw), _fx_env())
    return p, float(e.get_last_result().std_error)


def _fx_range(n, a):
    from quantark.asset.fx.engine.mc import FxRangeAccrualMCEngine
    from quantark.asset.fx.product import CurrencyPair
    from quantark.asset.fx.product.option import FxRangeAccrualConfig, FxRangeAccrualOption

    opt = FxRangeAccrualOption(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04),
        currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0, num_observations=12,
    )
    e = FxRangeAccrualMCEngine(num_paths=n, seed=3, use_antithetic=a)
    p = e.price(opt, _fx_env())
    return p, float(e.get_last_result().std_error)


def _fx_tarf(n, a):
    from quantark.asset.fx.engine.mc import FxTarnForwardMCEngine
    from quantark.asset.fx.product import CurrencyPair
    from quantark.asset.fx.product.option import FxTargetRedemptionForward
    from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionState

    opt = FxTargetRedemptionForward(strike=1.20, fixing_times=[0.25, 0.5, 0.75, 1.0],
                                    currency_pair=CurrencyPair("EUR", "USD"), state=TargetRedemptionState(0.0))
    e = FxTarnForwardMCEngine(params=_fx_mc(n, a))
    p = e.price(opt, _fx_env())
    return p, float(e.get_last_result().std_error)


CASES = {
    "euro": _euro,
    "binary": _binary,
    "asian": _asian,
    "range_accrual": _range_accrual,
    "american": _american,
    "barrier_continuous": lambda n, a: _barrier("CONTINUOUS", n, a),
    "barrier_discrete": lambda n, a: _barrier("DISCRETE", n, a),
    "barrier_expiry": lambda n, a: _barrier("EXPIRY", n, a),
    "accumulator": _accumulator,
    "single_sharkfin_expiry": lambda n, a: _single_shark("EXPIRY", n, a),
    "single_sharkfin_discrete": lambda n, a: _single_shark("DISCRETE", n, a),
    "double_sharkfin_expiry": lambda n, a: _double_shark("EXPIRY", n, a),
    "double_sharkfin_discrete": lambda n, a: _double_shark("DISCRETE", n, a),
    "fx_barrier_continuous": lambda n, a: _fx_barrier("CONTINUOUS", n, a),
    "fx_barrier_discrete": lambda n, a: _fx_barrier("DISCRETE", n, a),
    "fx_sharkfin_continuous": lambda n, a: _fx_shark("CONTINUOUS", n, a),
    "fx_sharkfin_discrete": lambda n, a: _fx_shark("DISCRETE", n, a),
    "fx_range_accrual": _fx_range,
    "fx_tarf": _fx_tarf,
}


# -- autocallables (vol-model engines pair; the plain GBM engine draws iid and must not) -------------------------

_S0 = 100.0


def _grid_env():
    import numpy as np

    from quantark.param import GridVolSurface

    strikes = list(_S0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=_S0),
        vol_surface=GridVolSurface(strikes, maturities, np.full((len(maturities), len(strikes)), 0.22)),
        div_yield=ContinuousDividendYield(0.01),
    )


def _snowball_product():
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    return SnowballOption(
        initial_price=_S0, strike=_S0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=[i / 12.0 for i in range(1, 13)], ki_continuous=False,
        ),
    )


def _phoenix_product():
    from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
    from quantark.util.calendar.day_counter import DayCountConvention
    from quantark.util.enum import CouponPayType, ObservationType

    return PhoenixOption(
        initial_price=_S0, strike=_S0,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.5, 1.0], ki_barrier=None,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=[80.0, 80.0], coupon_rate=0.02, coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365, memory_coupon=False,
        ),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        contract_multiplier=1.0, maturity=1.0,
    )


def _heston():
    from quantark.volmodels.heston import HestonParams

    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _autocall(kind, model, n, a):
    from quantark.asset.equity.engine import mc as m

    engine_cls = {
        ("snowball", "base"): m.SnowballMCEngine,
        ("snowball", "lv"): m.LocalVolSnowballMCEngine,
        ("snowball", "heston"): m.HestonSnowballMCEngine,
        ("phoenix", "lv"): m.LocalVolPhoenixMCEngine,
        ("phoenix", "heston"): m.HestonPhoenixMCEngine,
    }[(kind, model)]
    kw = {"model_params": _heston()} if model == "heston" else {}
    e = engine_cls(params=_mc(n, a, steps=12), **kw)
    product = _snowball_product() if kind == "snowball" else _phoenix_product()
    p = e.price(product, _grid_env())
    return p, float(e.get_last_result().std_error)


def _dcn(model, n, a):
    from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
    from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine, LocalVolDCNMCEngine

    kw = dict(num_paths=2 * n, seed=42, use_sobol=False, use_antithetic=a, num_batches=2)  # n paths per batch
    if model == "gbm":
        e = DCNMCEngine(**kw)
    elif model == "lv":
        e = LocalVolDCNMCEngine(**kw)
    else:
        e = HestonDCNMCEngine(model_params=_heston(), **kw)
    env = flat_env(**FLAT)
    if model == "lv":
        import numpy as np

        from quantark.param import GridVolSurface

        strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
        maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
        env.vol_surface = GridVolSurface(strikes=strikes, maturities=maturities,
                                         iv_grid=np.full((len(maturities), len(strikes)), FLAT["sigma"]))
    r = e.price_detailed(make_dcn(DCN_A), env)
    return float(r.pv), float(r.std_error)


CASES.update({
    "snowball_base_iid": lambda n, a: _autocall("snowball", "base", n, a),
    "snowball_lv": lambda n, a: _autocall("snowball", "lv", n, a),
    "snowball_heston": lambda n, a: _autocall("snowball", "heston", n, a),
    "phoenix_lv": lambda n, a: _autocall("phoenix", "lv", n, a),
    "phoenix_heston": lambda n, a: _autocall("phoenix", "heston", n, a),
    "dcn_gbm": lambda n, a: _dcn("gbm", n, a),
    "dcn_lv": lambda n, a: _dcn("lv", n, a),
    "dcn_heston": lambda n, a: _dcn("heston", n, a),
})

# Cases whose engine draws iid paths whatever ``use_antithetic`` says: the s.e. must stay the iid one.
UNPAIRED = {"snowball_base_iid"}
# Cases reduced per batch (DCN: 2 batches of n paths each): checked through ``pathwise_stderr`` pooling.
BATCHED = {"dcn_gbm": 2, "dcn_lv": 2, "dcn_heston": 2}


def run_case(name, antithetic, n_paths):
    return CASES[name](n_paths, antithetic)

# The autocallable vol engines simulate 2 * ceil(n / 2) rows (the draws are [z; -z] over an even row count).
EVEN_ROWS = {"snowball_lv", "snowball_heston", "phoenix_lv", "phoenix_heston"}


# -- CP3 additions: FX TARN note, KO-reset snowball (plain + vol), QE / Heston-SLV snowball and Phoenix ---------------

def _fx_tarn(n, a):
    from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine
    from quantark.asset.fx.product import CurrencyPair
    from quantark.asset.fx.product.option import FxTargetRedemptionNote
    from quantark.asset.fx.product.option.target_redemption_state import TargetRedemptionState

    note = FxTargetRedemptionNote(fixing_times=[0.25, 0.5, 0.75, 1.0], coupon_rate=0.08, notional=1.0, strike=1.20,
                                  currency_pair=CurrencyPair("EUR", "USD"), state=TargetRedemptionState(0.0),
                                  accrual_start=0.0)
    e = FxTargetRedemptionNoteMCEngine(params=_fx_mc(n, a))
    p = e.price(note, _fx_env())
    return p, float(e.get_last_result().std_error)


def _ko_reset(model, n, a):
    from test_dated_expiry_settles import HESTON, SLV_LEVERAGE
    from test_mc_time_node_merge import env, ko_reset
    from quantark.asset.equity.engine import mc as m

    params = _mc(n, a, steps=12)
    e = {
        "base": lambda: m.SnowballMCEngine(params=params),
        "lv": lambda: m.LocalVolSnowballMCEngine(params=params),
        "heston": lambda: m.HestonSnowballMCEngine(HESTON, params=params),
        "slv": lambda: m.HestonSLVSnowballMCEngine(HESTON, params=params, leverage_surface=SLV_LEVERAGE),
    }[model]()
    p = e.price(ko_reset("dated", 0), env(100.0, 0, grid=True))
    return p, float(e.get_last_result().std_error)


def _autocall_more(kind, model, n, a):
    from test_dated_expiry_settles import HESTON, SLV_LEVERAGE
    from quantark.asset.equity.engine import mc as m

    params = _mc(n, a, steps=12)
    names = {
        ("snowball", "qe"): lambda: m.QESnowballMCEngine(HESTON, params=params),
        ("snowball", "slv"): lambda: m.HestonSLVSnowballMCEngine(HESTON, params=params, leverage_surface=SLV_LEVERAGE),
        ("snowball", "slv_qe"): lambda: m.HestonSLVQESnowballMCEngine(HESTON, params=params, leverage_surface=SLV_LEVERAGE),
        ("phoenix", "qe"): lambda: m.QEPhoenixMCEngine(HESTON, params=params),
        ("phoenix", "slv"): lambda: m.HestonSLVPhoenixMCEngine(HESTON, params=params, leverage_surface=SLV_LEVERAGE),
        ("phoenix", "slv_qe"): lambda: m.HestonSLVQEPhoenixMCEngine(HESTON, params=params, leverage_surface=SLV_LEVERAGE),
    }
    e = names[(kind, model)]()
    product = _snowball_product() if kind == "snowball" else _phoenix_product()
    p = e.price(product, _grid_env())
    return p, float(e.get_last_result().std_error)


CASES.update({
    "fx_tarn": _fx_tarn,
    "ko_reset_base_iid": lambda n, a: _ko_reset("base", n, a),
    "ko_reset_lv": lambda n, a: _ko_reset("lv", n, a),
    "ko_reset_heston": lambda n, a: _ko_reset("heston", n, a),
    "ko_reset_slv": lambda n, a: _ko_reset("slv", n, a),
    "snowball_qe": lambda n, a: _autocall_more("snowball", "qe", n, a),
    "snowball_slv": lambda n, a: _autocall_more("snowball", "slv", n, a),
    "snowball_slv_qe": lambda n, a: _autocall_more("snowball", "slv_qe", n, a),
    "phoenix_qe": lambda n, a: _autocall_more("phoenix", "qe", n, a),
    "phoenix_slv": lambda n, a: _autocall_more("phoenix", "slv", n, a),
    "phoenix_slv_qe": lambda n, a: _autocall_more("phoenix", "slv_qe", n, a),
})
UNPAIRED.add("ko_reset_base_iid")
EVEN_ROWS.update({"ko_reset_lv", "ko_reset_heston", "ko_reset_slv", "snowball_qe", "snowball_slv", "snowball_slv_qe",
                  "phoenix_qe", "phoenix_slv", "phoenix_slv_qe"})
