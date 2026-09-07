"""Greek-bump component summaries are VALUE identities of the market objects.

The planner attributes a cell's mutation to a tag by fingerprinting each
declared component before and after the bump (spec 10.2). A summary must
therefore (a) differ whenever the object's values differ (no misattribution)
and (b) be unfingerprintable for a family it does not understand, so the
planner falls back to conservative invalidation instead of reading two
unknown objects as "unchanged" (fingerprint(None) == fingerprint(None)).
"""
from datetime import datetime

import pytest

from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.greeks import (
    GREEK_BUMP_TRANSFORMER_ID, TradeState, _div_summary, _rate_summary, _vol_summary,
    apply_greek_bump,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div.dividend_yield import NoDividend, TermStructureDividendYield
from quantark.param.rrf.rate_curve import (
    CubicSplineRateCurve, LinearRateCurve, ParallelShiftRateCurve, RateCurve,
)
from quantark.param.vol.vol_surface import TermStructureVolSurface
from quantark.priceenv import PricingEnvironment

VD = datetime(2024, 1, 15)
PILLARS = [(0.5, 0.03), (1.0, 0.035), (2.0, 0.04)]


def _env(rate_curve=None, vol_surface=None, div_yield=None):
    return PricingEnvironment(
        spot_quote=SpotQuote(100.0),
        vol_surface=vol_surface or FlatVolSurface(0.2),
        rate_curve=rate_curve or FlatRateCurve(0.03),
        div_yield=div_yield if div_yield is not None else ContinuousDividendYield(0.01),
        valuation_date=VD,
    )


# --- rate --------------------------------------------------------------------

def test_rate_summary_flat_curve_is_the_rate_float():
    assert _rate_summary(_env(FlatRateCurve(0.0356))) == 0.0356          # bitwise


def test_rate_summary_interpolated_curve_encodes_tenors():
    a = _rate_summary(_env(LinearRateCurve([(0.5, 0.03), (1.0, 0.035), (2.0, 0.04)])))
    b = _rate_summary(_env(LinearRateCurve([(1.0, 0.03), (2.0, 0.035), (5.0, 0.04)])))
    assert a != b                                   # same rates, different tenors
    assert try_fingerprint(a) is not None and try_fingerprint(b) is not None


def test_rate_summary_interpolated_curve_encodes_interpolation_scheme():
    a = _rate_summary(_env(LinearRateCurve(PILLARS)))
    b = _rate_summary(_env(CubicSplineRateCurve(PILLARS)))
    assert a != b


def test_rate_summary_parallel_shift_is_fingerprintable_and_encodes_the_shift():
    base = LinearRateCurve(PILLARS)
    s0 = _rate_summary(_env(base))
    s1 = _rate_summary(_env(ParallelShiftRateCurve(base, 1e-4)))
    s2 = _rate_summary(_env(ParallelShiftRateCurve(base, 2e-4)))
    assert s1 is not None and try_fingerprint(s1) is not None
    assert s1 != s0 and s1 != s2
    # value identity, not object identity
    assert s1 == _rate_summary(_env(ParallelShiftRateCurve(LinearRateCurve(PILLARS), 1e-4)))


class _MysteryCurve(RateCurve):
    def get_rate(self, t):
        return 0.02

    def get_discount_factor(self, t):
        import math
        return math.exp(-0.02 * t)


def test_rate_summary_unknown_family_is_not_fingerprintable():
    """Unknown -> conservative invalidation, never 'unchanged'."""
    assert try_fingerprint(_rate_summary(_env(_MysteryCurve()))) is None


def test_rate_up_cell_on_term_curve_yields_a_distinct_fingerprintable_rate_component():
    from quantark.asset.equity.engine.analytical import BlackScholesEngine
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.equity.riskmeasures import GreeksCalculator
    from quantark.util.enum import OptionType

    gc = GreeksCalculator()
    env = _env(LinearRateCurve(PILLARS))
    state = TradeState(
        product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
        pricing_env=env, cash_legs=(), engine=BlackScholesEngine(), streams=None,
        quantity=1.0, greeks_params=gc.params,
    )
    bumped = apply_greek_bump("rate_up", state, gc).pricing_env
    s_base, s_bumped = _rate_summary(env), _rate_summary(bumped)
    assert s_bumped is not None and try_fingerprint(s_bumped) is not None
    assert s_bumped != s_base


def test_planner_attributes_rate_up_on_term_curve_to_the_rate_tag():
    from quantark.asset.equity.engine.analytical import BlackScholesEngine
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.equity.riskmeasures import GreeksCalculator
    from quantark.execution.contracts import ScenarioSpec
    from quantark.execution.scenario.planner import plan_scenarios
    from quantark.util.enum import OptionType

    gc = GreeksCalculator()
    state = TradeState(
        product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
        pricing_env=_env(LinearRateCurve(PILLARS)), cash_legs=(), engine=BlackScholesEngine(),
        streams=None, quantity=1.0, greeks_params=gc.params,
    )
    spec = ScenarioSpec(scenario_id="rate_up", transformer_id=GREEK_BUMP_TRANSFORMER_ID,
                        parameters=(("bump_id", "rate_up"),), mutation_tags=frozenset({"rate"}))
    plan = plan_scenarios(state, [spec], None)
    assert plan.cells[0].changed_tags == frozenset({"rate"})


# --- vol / div: same contract ----------------------------------------------------

def test_vol_summary_term_structure_encodes_times():
    a = _vol_summary(_env(vol_surface=TermStructureVolSurface(times=[0.5, 1.0], vols=[0.2, 0.25])))
    b = _vol_summary(_env(vol_surface=TermStructureVolSurface(times=[1.0, 2.0], vols=[0.2, 0.25])))
    assert a != b and try_fingerprint(a) is not None


def test_div_summary_term_structure_encodes_times():
    a = _div_summary(_env(div_yield=TermStructureDividendYield(times=[0.5, 1.0], yields=[0.01, 0.02])))
    b = _div_summary(_env(div_yield=TermStructureDividendYield(times=[1.0, 2.0], yields=[0.01, 0.02])))
    assert a != b and try_fingerprint(a) is not None


def test_div_summary_no_dividend_is_zero():
    assert _div_summary(_env(div_yield=NoDividend())) == 0.0


class _MysterySurface:
    def get_vol(self, strike, t):
        return 0.2


def test_vol_summary_unknown_family_is_not_fingerprintable():
    env = _env()
    env.vol_surface = _MysterySurface()
    assert try_fingerprint(_vol_summary(env)) is None
