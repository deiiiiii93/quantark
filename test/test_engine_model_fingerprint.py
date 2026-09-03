"""BaseEngine.model_fingerprint: declared-attribute identity for params-only engines (patch spec §7)."""
from quantark.asset.equity.engine.analytical import (
    AmericanOptionAnalyticalEngine,
    BarrierAnalyticalEngine,
    BlackScholesEngine,
    DeltaOneEngine,
    DoubleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.param import EngineParams
from quantark.pnlexplain import engines_equivalent


def test_params_only_engines_fingerprint_by_class_and_params():
    a, b = BlackScholesEngine(), BlackScholesEngine()
    assert a.model_fingerprint() == b.model_fingerprint() is not None
    assert BlackScholesEngine(EngineParams(bus_days_in_year=244)).model_fingerprint() != a.model_fingerprint()
    assert DeltaOneEngine(use_market_price=True).model_fingerprint() != DeltaOneEngine().model_fingerprint()
    fp = BarrierAnalyticalEngine().model_fingerprint()
    assert fp is not None and any(name == "_one_touch_engine" for name, _ in fp[2])   # sub-engine folded in
    assert fp[:2] == (BarrierAnalyticalEngine.__module__, "BarrierAnalyticalEngine")


def test_engines_without_a_declaration_have_no_fingerprint():
    assert EuropeanMCEngine().model_fingerprint() is None


def test_engines_equivalent_rules():
    a, b = BlackScholesEngine(), BlackScholesEngine()
    assert engines_equivalent(a, a) and engines_equivalent(a, b)
    assert not engines_equivalent(a, BlackScholesEngine(EngineParams(bus_days_in_year=244)))
    assert not engines_equivalent(a, DeltaOneEngine())                                  # different class
    assert not engines_equivalent(a, None) and not engines_equivalent(None, a)
    mc = EuropeanMCEngine()
    assert engines_equivalent(mc, mc) and not engines_equivalent(mc, EuropeanMCEngine())  # identity only
    assert engines_equivalent(AmericanOptionAnalyticalEngine(), AmericanOptionAnalyticalEngine())
    assert engines_equivalent(DoubleSharkfinOptionAnalyticalEngine(), DoubleSharkfinOptionAnalyticalEngine())
    assert not engines_equivalent(DoubleSharkfinOptionAnalyticalEngine(quad_points=16),
                                  DoubleSharkfinOptionAnalyticalEngine())
    assert not engines_equivalent(object(), object())                                    # no fingerprint hook
