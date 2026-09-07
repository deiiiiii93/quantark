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


def test_every_params_only_analytical_engine_opted_in():
    """The rule in BaseEngine is 'instance state == construction arguments'; apply it uniformly.

    Leaving a qualifying engine at identity makes MODEL-change semantics differ by engine: a
    fresh-but-identical engine would raise a phantom MODEL row and break transition substitution
    for that product only (Kimi review 2026-09-03).
    """
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import (
        AccumulatorAnalyticalEngine,
    )
    from quantark.asset.equity.engine.analytical.asian_option_analytical_engine import (
        AsianOptionAnalyticalEngine,
    )
    from quantark.asset.equity.engine.analytical.range_accrual_analytical_engine import (
        RangeAccrualAnalyticalEngine,
    )
    for cls in (AsianOptionAnalyticalEngine, AccumulatorAnalyticalEngine, RangeAccrualAnalyticalEngine):
        assert engines_equivalent(cls(), cls()), cls.__name__
        assert not engines_equivalent(cls(EngineParams(bus_days_in_year=244)), cls()), cls.__name__
    # the Asian engine's method is part of its model, the range-accrual result cache is not
    assert not engines_equivalent(AsianOptionAnalyticalEngine(method="kemna_vorst"),
                                  AsianOptionAnalyticalEngine(method="levy"))


def test_engines_equivalent_never_raises_on_a_broken_fingerprint():
    """A subclass may declare attributes it never sets; comparison must fail closed, not explode."""
    class Broken(BlackScholesEngine):
        MODEL_FINGERPRINT_ATTRS = ('params', 'never_set')

    a, b = Broken(), Broken()
    assert engines_equivalent(a, a)                          # identity still short-circuits
    assert engines_equivalent(a, b) is False                 # and the AttributeError is absorbed
    assert engines_equivalent(a, BlackScholesEngine()) is False
