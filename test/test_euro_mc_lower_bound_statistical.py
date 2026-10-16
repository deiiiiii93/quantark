"""EuropeanMCEngine's no-arbitrage lower-bound check is statistical (W13, user decision 2026-10-10).

A deep in-the-money European one day from expiry is worth its discounted intrinsic bound (Black-Scholes minus the
bound is 0.0 in double precision), so an unbiased Monte Carlo estimate lands below the bound about half the time
(docs/superpowers/specs/2026-10-10-mc-lower-bound-diagnosis/). The engine refuses only an estimate more than
``max(1e-6, 4 * std_error)`` below the bound -- the std error it reports, in contract units -- and keeps the hard
1e-6 check when the std error is zero or undefined. The estimator is unchanged: every price that passed before is
bit-identical (pinned below against 0d1e4133).
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime

import pytest

from golden_compare import assert_close
from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import PricingError

from test_expiry_carried_knock_in import BARRIER_PRODUCTS, BEFORE, _mc, env as carried_env

VALUATION = datetime(2026, 1, 5)
_TERM_PILLARS = [(0.0, 0.01), (0.01, 0.02), (0.25, 0.03), (1.0, 0.045), (2.0, 0.05)]


def _env(spot: float, term: bool = False) -> PricingEnvironment:
    curve = LinearRateCurve(_TERM_PILLARS) if term else FlatRateCurve(rate=0.03)
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.20),
                              rate_curve=curve, div_yield=ContinuousDividendYield(0.01), valuation_date=VALUATION)


def _option(option_type, days, lag_days=0, multiplier=1.0) -> EuropeanVanillaOption:
    kw = dict(strike=100.0, option_type=option_type, maturity=days / 365, contract_multiplier=multiplier)
    if lag_days:
        kw["settlement_convention"] = SettlementConvention(lag=lag_days / 365.0,
                                                           lag_unit=SettlementLagUnit.YEAR_FRACTION)
    return EuropeanVanillaOption(**kw)


_CONFIGS = {
    "plain256": (MonteCarloMethod.PSEUDO, dict(num_paths=256)),
    "anti256": (MonteCarloMethod.PSEUDO, dict(num_paths=256, use_antithetic=True)),
    "qmc1024": (MonteCarloMethod.QUASI, dict(num_paths=1024)),
    "rqmc256": (MonteCarloMethod.RANDOMIZED_QUASI, dict(num_paths=256)),
}


def _engine(config: str) -> EuropeanMCEngine:
    method, extra = _CONFIGS[config]
    return EuropeanMCEngine(params=MCParams(seed=7, **extra), method=method)


def _price(engine, product, env) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(engine.price(product, env))


# -- the refused one-day deep in-the-money cells now price ------------------------------------------------------------

@pytest.mark.parametrize("spot,expected", [(90.0, 9.993453009252772), (75.0, 24.993174367657804)])
def test_one_day_deep_itm_put_prices_its_estimate(spot, expected):
    """256 antithetic paths, seed 7: the estimate sits 0.012 std errors below the bound and is now returned as is."""
    assert_close(_price(_engine("anti256"), _option(OptionType.PUT, 1), _env(spot)), expected)


@pytest.mark.parametrize("form,spot,expected", [
    ("dated", 90.0, 9.993453009252772), ("dated", 75.0, 24.993174367657804),
    # the numeric twin aged by time_shift: maturity 1 - 364/365, a few ULPs off 1/365
    ("numeric", 90.0, 9.993453009252768), ("numeric", 75.0, 24.993174367657808),
])
def test_carried_knock_in_put_the_day_before_prices_through_the_barrier_mc(form, spot, expected):
    """The test_expiry_carried_knock_in day-before cell (mc, down_in_put, carried) that used to raise."""
    build = dict((k, b) for k, b, _, _ in BARRIER_PRODUCTS)["down_in_put"]
    product = build(form, True, BEFORE)
    assert_close(_price(BarrierOptionMCEngine(params=_mc()), product, carried_env(spot, BEFORE)), expected)


# -- the check still refuses a clearly wrong estimate ---------------------------------------------------------------

def _patched(engine, estimate_fn):
    """Replace the estimator: ``estimate_fn(df)`` -> (price, std_error) before the contract multiplier."""
    engine._price_mc_or_qmc = lambda *a, df=None, **k: estimate_fn(df)
    return engine


def _bound(product, env) -> float:
    eng = _engine("plain256")
    cap = {}
    orig = eng._european_lower_bound
    eng._european_lower_bound = lambda *a, **k: (cap.setdefault("b", orig(*a, **k)), -math.inf)[1]
    _price(eng, product, env)
    return cap["b"]


@pytest.mark.parametrize("multiplier", [1.0, 100.0])
def test_estimate_ten_std_errors_below_the_bound_raises(multiplier):
    product, env = _option(OptionType.PUT, 1, multiplier=multiplier), _env(90.0)
    bound_per_unit = _bound(product, env) / multiplier
    se = 0.01
    engine = _patched(_engine("plain256"), lambda df: (bound_per_unit - 10 * se, se))
    with pytest.raises(PricingError, match="discounted European lower bound"):
        _price(engine, product, env)


@pytest.mark.parametrize("multiplier", [1.0, 100.0])
def test_tolerance_is_four_reported_std_errors_in_contract_units(multiplier):
    """3.9 std errors below passes (returned unchanged), 4.1 raises -- the std error scaled by the multiplier."""
    product, env = _option(OptionType.PUT, 1, multiplier=multiplier), _env(90.0)
    bound_per_unit = _bound(product, env) / multiplier
    se = 0.01
    inside = _patched(_engine("plain256"), lambda df: (bound_per_unit - 3.9 * se, se))
    assert _price(inside, product, env) == (bound_per_unit - 3.9 * se) * multiplier
    outside = _patched(_engine("plain256"), lambda df: (bound_per_unit - 4.1 * se, se))
    with pytest.raises(PricingError, match="discounted European lower bound"):
        _price(outside, product, env)


@pytest.mark.parametrize("se", [0.0, float("nan")])
def test_zero_or_undefined_std_error_keeps_the_hard_check(se):
    product, env = _option(OptionType.PUT, 1), _env(90.0)
    bound = _bound(product, env)
    engine = _patched(_engine("plain256"), lambda df: (bound - 1e-5, se))
    with pytest.raises(PricingError, match="discounted European lower bound"):
        _price(engine, product, env)


def test_tiny_std_error_keeps_the_hard_floor():
    """4 * se below the 1e-6 floor: the floor decides (1e-5 below still raises)."""
    product, env = _option(OptionType.PUT, 1), _env(90.0)
    bound = _bound(product, env)
    engine = _patched(_engine("plain256"), lambda df: (bound - 1e-5, 1e-8))
    with pytest.raises(PricingError, match="discounted European lower bound"):
        _price(engine, product, env)


# -- ATM and longer-dated prices are bit-identical to 0d1e4133 --------------------------------------------------------

#: (term curve, spot, option, days, lag days, multiplier, config) -> price at 0d1e4133
#: (docs/superpowers/specs/2026-10-10-mc-lower-bound-diagnosis/05_bit_identity_before.out)
_PINNED = [
    (False, 100.0, OptionType.PUT, 365, 0, 1.0, "plain256", 7.4581551171068785),
    (False, 100.0, OptionType.PUT, 365, 0, 1.0, "anti256", 7.342541096448935),
    (False, 100.0, OptionType.PUT, 365, 0, 1.0, "qmc1024", 7.033283729493982),
    (False, 100.0, OptionType.PUT, 365, 0, 1.0, "rqmc256", 6.7800858139573394),
    (False, 100.0, OptionType.CALL, 30, 2, 100.0, "plain256", 219.20424204487804),
    (False, 100.0, OptionType.CALL, 30, 2, 100.0, "anti256", 255.94136328780078),
    (False, 100.0, OptionType.CALL, 30, 2, 100.0, "qmc1024", 242.61308514948695),
    (False, 100.0, OptionType.CALL, 30, 2, 100.0, "rqmc256", 233.40878123693898),
    (False, 110.0, OptionType.CALL, 365, 0, 1.0, "plain256", 14.491172451255881),
    (False, 110.0, OptionType.CALL, 365, 0, 1.0, "anti256", 16.246139355209454),
    (False, 110.0, OptionType.CALL, 365, 0, 1.0, "qmc1024", 15.645925622189845),
    (False, 110.0, OptionType.CALL, 365, 0, 1.0, "rqmc256", 15.335652480936353),
    (True, 90.0, OptionType.PUT, 365, 2, 1.0, "plain256", 11.936574093072919),
    (True, 90.0, OptionType.PUT, 365, 2, 1.0, "anti256", 11.412120308213643),
    (True, 90.0, OptionType.PUT, 365, 2, 1.0, "qmc1024", 11.166077462810943),
    (True, 90.0, OptionType.PUT, 365, 2, 1.0, "rqmc256", 10.977827460756894),
    (True, 100.0, OptionType.CALL, 1, 0, 1.0, "plain256", 0.3870775776704728),
    (True, 100.0, OptionType.CALL, 1, 0, 1.0, "anti256", 0.45132579439007325),
    (True, 100.0, OptionType.CALL, 1, 0, 1.0, "qmc1024", 0.428603632501648),
    (True, 100.0, OptionType.CALL, 1, 0, 1.0, "rqmc256", 0.41238646332938883),
]


@pytest.mark.parametrize("term,spot,option_type,days,lag,multiplier,config,expected", _PINNED,
                         ids=[f"{'term' if c[0] else 'flat'}-S{c[1]:g}-{c[2].name.lower()}-{c[3]}d-lag{c[4]}-"
                              f"x{c[5]:g}-{c[6]}" for c in _PINNED])
def test_passing_prices_are_bit_identical(term, spot, option_type, days, lag, multiplier, config, expected):
    product = _option(option_type, days, lag_days=lag, multiplier=multiplier)
    assert_close(_price(_engine(config), product, _env(spot, term)), expected)
