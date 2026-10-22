import pytest
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError


def test_fx_mc_params_defaults():
    p = FxMCParams()
    assert p.num_paths == 200_000
    assert p.seed == 42
    assert p.use_antithetic is True
    assert p.method == MonteCarloMethod.PSEUDO


def test_fx_mc_params_rejects_nonpositive_paths():
    with pytest.raises(ValidationError):
        FxMCParams(num_paths=0)


# ---------------------------------------------------------------------------
# Task 4: FxBarrierOption monitoring fields
# ---------------------------------------------------------------------------

from quantark.asset.fx.product.option import FxBarrierOption
from quantark.util.enum import OptionType, FxBarrierType, ObservationType


def _barrier(**kw):
    base = dict(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        maturity=1.0,
    )
    base.update(kw)
    return FxBarrierOption(**base)


def test_barrier_defaults_to_continuous():
    opt = _barrier()
    assert opt.monitoring == ObservationType.CONTINUOUS
    assert opt.observation_times is None


def test_discrete_requires_sorted_unique_in_range_times():
    ok = _barrier(monitoring=ObservationType.DISCRETE,
                  observation_times=[0.25, 0.5, 0.75, 1.0])
    assert ok.observation_times == [0.25, 0.5, 0.75, 1.0]
    with pytest.raises(ValidationError):
        _barrier(monitoring=ObservationType.DISCRETE, observation_times=None)
    with pytest.raises(ValidationError):
        _barrier(monitoring=ObservationType.DISCRETE, observation_times=[0.5, 0.25])
    with pytest.raises(ValidationError):
        _barrier(monitoring=ObservationType.DISCRETE, observation_times=[0.5, 0.5])
    with pytest.raises(ValidationError):
        _barrier(monitoring=ObservationType.DISCRETE, observation_times=[0.5, 1.5])


# ---------------------------------------------------------------------------
# Tasks 5 + 6: continuous barrier MC + analytic discrete rejection
# ---------------------------------------------------------------------------

from datetime import datetime
import math

from quantark.asset.fx.engine.mc import FxBarrierMCEngine, FxBarrierMCResult
from quantark.asset.fx.engine.analytical import VannaVolgaBarrierEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.param import FlatRateCurve, SpotQuote
from quantark.param.vol.vannavolga import (
    DeltaConvention, FXEnv, SmileQuotes, VannaVolgaVolSurface,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.exceptions import PricingError

_VAL = datetime(2026, 6, 15)


def _smile_env(atm=0.10, rr=0.0, bf=0.0):
    # Flat smile (rr=bf=0) => VV reduces to flat Black-Scholes, so the GK MC
    # (constant vol) and the analytic Reiner-Rubinstein agree.
    surface = VannaVolgaVolSurface(
        FXEnv(spot=1.20, rd=0.05, rf=0.03, tau=1.0),
        SmileQuotes(sigma_atm=atm, rr25=rr, bf25_2vol=bf),
        DeltaConvention.SPOT,
    )
    return FxPricingEnvironment(
        valuation_date=_VAL, spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03), vol_surface=surface,
    )


def _ccy_barrier(**kw):
    base = dict(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0,
    )
    base.update(kw)
    return FxBarrierOption(**base)


def test_analytic_engine_rejects_discrete():
    opt = _ccy_barrier(monitoring=ObservationType.DISCRETE,
                       observation_times=[0.5, 1.0])
    with pytest.raises(PricingError, match="continuous"):
        VannaVolgaBarrierEngine().price(opt, _smile_env())


def test_continuous_up_out_call_matches_analytic_flat():
    env = _smile_env()
    opt = _ccy_barrier()  # up-and-out call, zero rebate
    analytic = VannaVolgaBarrierEngine().price(opt, env)
    eng = FxBarrierMCEngine(params=FxMCParams(num_paths=120_000, time_steps=120, seed=3))
    mc = eng.price(opt, env)
    se = eng.get_last_result().std_error
    assert se is not None and se > 0.0
    assert abs(mc - analytic) <= 5.0 * se


def test_continuous_down_in_put_matches_analytic_flat():
    env = _smile_env()
    opt = _ccy_barrier(
        is_up=False, barrier=1.05, knock_type=FxBarrierType.KNOCK_IN,
        option_type=OptionType.PUT,
    )
    analytic = VannaVolgaBarrierEngine().price(opt, env)
    eng = FxBarrierMCEngine(params=FxMCParams(num_paths=120_000, time_steps=120, seed=3))
    mc = eng.price(opt, env)
    se = eng.get_last_result().std_error
    assert abs(mc - analytic) <= 5.0 * se


def test_ki_plus_ko_equals_vanilla_zero_rebate():
    from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
    from quantark.asset.fx.product.option import FxVanillaOption
    env = _smile_env()
    ko = _ccy_barrier(knock_type=FxBarrierType.KNOCK_OUT)
    ki = _ccy_barrier(knock_type=FxBarrierType.KNOCK_IN)
    eng = FxBarrierMCEngine(params=FxMCParams(num_paths=120_000, time_steps=120, seed=5))
    s = eng.price(ko, env) + eng.price(ki, env)
    vanilla = FxVanillaOption(strike=1.20, option_type=OptionType.CALL,
                              maturity=1.0, notional_foreign=1.0)
    v = GarmanKohlhagenEngine().price(vanilla, env)
    assert s == pytest.approx(v, rel=2e-2)


def test_grid_invariance_continuous():
    env = _smile_env()
    opt = _ccy_barrier()
    p_coarse = FxBarrierMCEngine(params=FxMCParams(num_paths=80_000, time_steps=60, seed=9)).price(opt, env)
    p_fine = FxBarrierMCEngine(params=FxMCParams(num_paths=80_000, time_steps=240, seed=9)).price(opt, env)
    assert abs(p_coarse - p_fine) / max(p_fine, 1e-6) < 0.05


def test_quasi_reports_no_statistical_se():
    env = _smile_env()
    opt = _ccy_barrier()
    eng = FxBarrierMCEngine(
        params=FxMCParams(num_paths=65_536, time_steps=120, seed=1,
                          method=MonteCarloMethod.QUASI))
    eng.price(opt, env)
    assert eng.get_last_result().std_error is None


def test_rqmc_not_implemented_yet():
    env = _smile_env()
    opt = _ccy_barrier()
    eng = FxBarrierMCEngine(
        params=FxMCParams(method=MonteCarloMethod.RANDOMIZED_QUASI))
    with pytest.raises(NotImplementedError):
        eng.price(opt, env)


# ---------------------------------------------------------------------------
# Task 7: discrete monitoring
# ---------------------------------------------------------------------------


def test_discrete_converges_to_continuous_no_rebate():
    # Discrete->continuous barrier convergence is O(sqrt(dt)) (Broadie-
    # Glasserman-Kou), so the robust gate is that denser monitoring moves the
    # price toward continuous, and that discrete KO >= continuous (fewer KO
    # chances make the KO worth more), not an absolute tolerance.
    env = _smile_env()
    p_cont = FxBarrierMCEngine(
        params=FxMCParams(num_paths=120_000, time_steps=240, seed=4)
    ).price(_ccy_barrier(), env)

    def disc_price(n):
        times = [round(k / n, 6) for k in range(1, n + 1)]
        opt = _ccy_barrier(monitoring=ObservationType.DISCRETE, observation_times=times)
        return FxBarrierMCEngine(
            params=FxMCParams(num_paths=120_000, time_steps=240, seed=4)
        ).price(opt, env)

    p_sparse = disc_price(20)
    p_dense = disc_price(120)

    assert p_sparse >= p_cont - 5e-4
    assert p_dense >= p_cont - 5e-4
    assert abs(p_dense - p_cont) < abs(p_sparse - p_cont)


def test_discrete_fewer_fixings_worth_more_for_ko():
    env = _smile_env()
    sparse = _ccy_barrier(monitoring=ObservationType.DISCRETE,
                          observation_times=[0.5, 1.0])
    dense = _ccy_barrier(monitoring=ObservationType.DISCRETE,
                         observation_times=[round(0.1 * k, 4) for k in range(1, 11)])
    eng = lambda: FxBarrierMCEngine(params=FxMCParams(num_paths=120_000, time_steps=200, seed=8))
    assert eng().price(sparse, env) >= eng().price(dense, env) - 5e-4


# ---------------------------------------------------------------------------
# Task 8: Greeks smoke
# ---------------------------------------------------------------------------


def test_greeks_finite():
    env = _smile_env()
    opt = _ccy_barrier()
    greeks = FxBarrierMCEngine(
        params=FxMCParams(num_paths=60_000, time_steps=100, seed=2)
    ).calculate_greeks(opt, env)
    for key in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
        assert key in greeks and math.isfinite(greeks[key])
