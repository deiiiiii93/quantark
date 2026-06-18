"""End-to-end MC exposure -> regulatory CVA with real quant-ark objects (spec §3.2).

Builds a real PricingEnvironment / EuropeanVanillaOption / BlackScholesEngine, wraps
them in the SA-CVA portfolio model, runs MonteCarloExposureEngine to an EE profile,
and feeds RegulatoryCVAEngine. No fakes for the priced trade.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv.pricing_environment import PricingEnvironment
from quantark.asset.equity.product.option.european_vanilla_option import (
    EuropeanVanillaOption,
)
from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.util.enum import OptionType
from quantark.util.calendar import DayCountConvention
from quantark.util.exceptions import ValidationError

from quantark.sacva.cva.engine import RegulatoryCVAEngine
from quantark.sacva.exposure.engine import Measure
from quantark.sacva.exposure.simulator import (
    MonteCarloExposureConfig,
    MonteCarloExposureEngine,
)
from quantark.sacva.models.enums import CreditQuality
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.trade import CVATrade

VAL = datetime(2026, 6, 17)
EXP_1Y = datetime(2027, 6, 17)


class FlatCreditCurve:
    def __init__(self, hazard, recovery=0.4):
        self.h, self.R = hazard, recovery

    def get_survival_probability(self, t):
        return float(np.exp(-self.h * t))

    @property
    def recovery_rate(self):
        return self.R


def _env(spot=100.0, vol=0.2, rate=0.03, asset="ACME"):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate),
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=spot, asset_name=asset),
        vol_surface=FlatVolSurface(vol),
        div_yield=ContinuousDividendYield(0.0),
        day_count_convention=DayCountConvention.CALENDAR_DAYS,
    )


def _call_trade(trade_id="t1", strike=100.0, qty=1.0, env=None):
    env = env or _env()
    opt = EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL,
                                exercise_date=EXP_1Y)
    return CVATrade(trade_id=trade_id, product=opt, engine=BlackScholesEngine(),
                    env=env, quantity=qty, trade_currency="USD")


def _counterparty(netting_sets, hazard=0.02):
    return Counterparty(name="CP", netting_sets=netting_sets,
                        credit_curve=FlatCreditCurve(hazard), bucket=2,
                        credit_quality=CreditQuality.IG)


def test_mc_profile_t0_matches_black_scholes_and_feeds_cva():
    trade = _call_trade()
    cp = _counterparty([NettingSet(set_id="n1", trades=[trade])])
    eng = MonteCarloExposureEngine(MonteCarloExposureConfig(num_paths=20000, n_steps=12,
                                                            seed=7))
    prof = eng.compute(cp)

    assert prof.measure is Measure.RISK_NEUTRAL and prof.regulatory_eligible is True
    assert prof.times[0] == 0.0 and prof.times[-1] == pytest.approx(1.0, abs=0.01)
    assert np.all(prof.epe_discounted >= 0.0)
    # t0 exposure is deterministic (all paths start at spot) -> exact BS value
    bs0 = BlackScholesEngine().price(trade.product, trade.env)
    assert prof.epe_discounted[0] == pytest.approx(bs0, rel=1e-6)
    # a long call value is always >= 0, so discounted EE = discounted E[C_t] = C0 for
    # all t (martingale): the profile is ~flat at C0 (MC + day-rounding tolerance)
    assert np.allclose(prof.epe_discounted, bs0, rtol=0.05)

    cva = RegulatoryCVAEngine().compute(cp.credit_curve, prof)
    assert 0.0 < cva < prof.epe_discounted.max()


def test_mc_netting_reduces_exposure_for_offsetting_trades():
    env = _env()
    long_call = _call_trade(trade_id="L", qty=1.0, env=env)
    short_call = _call_trade(trade_id="S", qty=-1.0, env=env)
    cp_net = _counterparty([NettingSet("n1", [long_call, short_call],
                                       netting_enforceable=True)])
    cp_gross = _counterparty([NettingSet("n1", [_call_trade("L2", qty=1.0, env=env),
                                                _call_trade("S2", qty=-1.0, env=env)],
                                         netting_enforceable=False)])
    eng = MonteCarloExposureEngine(MonteCarloExposureConfig(num_paths=8000, n_steps=8,
                                                            seed=3))
    net = eng.compute(cp_net).epe_discounted
    gross = eng.compute(cp_gross).epe_discounted
    # a perfectly offsetting pair nets to ~0 EE; gross (sum of positive parts) does not
    assert net.max() < 1e-6
    assert gross.max() > 1.0


def test_mc_rejects_stateful_grid_engine():
    class GridEngine:
        supports_spot_greeks_grid = True

        def price(self, product, env):
            return 1.0

    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                exercise_date=EXP_1Y)
    trade = CVATrade(trade_id="sb", product=opt, engine=GridEngine(), env=_env(),
                     quantity=1.0)
    cp = _counterparty([NettingSet("n1", [trade])])
    with pytest.raises(ValidationError):
        MonteCarloExposureEngine().compute(cp)


def test_mc_multi_underlying_requires_correlation():
    a = _call_trade(trade_id="A", env=_env(spot=100.0, asset="AAA"))
    b = _call_trade(trade_id="B", env=_env(spot=50.0, asset="BBB"))
    cp = _counterparty([NettingSet("n1", [a, b])])
    with pytest.raises(ValidationError):
        MonteCarloExposureEngine().compute(cp)
