"""Execution-framework migration demo 2: typed scenarios on processes.

The old worker-globals pattern (mutable module state + os.environ + parsing
scenario names) becomes: register an importable factory/transformer/runner
by string id, describe scenarios as typed ``ScenarioSpec``s, and run the
SAME plan serial or on spawn processes. ``compare_scenario_outcomes``
proves the complete normalized payload matches field-for-field.

This module doubles as the spawn-import module: workers rebuild the
factory/transformer/runner by importing it via ``CallableRef``, so all
registered callables are module-level and the executable part is guarded by
``__main__``.

Run:  python example/execution_scenarios_demo.py    (finishes in seconds)
Docs: docs/execution/README.md
"""
import dataclasses

from quantark.execution.scenario import registries

# ----------------------------------------------------------- registered parts


@dataclasses.dataclass(frozen=True)
class MarketInputs:
    """JSON-buildable base snapshot (travels to workers as payload pairs)."""

    spot: float
    vol: float
    rate: float
    strike: float
    maturity: float


def build_market_inputs(payload):
    return MarketInputs(
        spot=payload["spot"], vol=payload["vol"], rate=payload["rate"],
        strike=payload["strike"], maturity=payload["maturity"],
    )


def bump_spot(base, parameters):
    return dataclasses.replace(base, spot=base.spot * (1.0 + parameters["rel"]))


def mc_engine_factory(parameters):
    """Receives the CELL's parameters; this demo builds one fixed engine."""
    from quantark.asset.equity.engine.mc import EuropeanMCEngine
    from quantark.asset.equity.param import MCParams

    return EuropeanMCEngine(params=MCParams(num_paths=20_000, seed=42))


def price_european(cell, resolved, child_context):
    """value_kind='float' runner: eligible for processes/dask backends."""
    from datetime import datetime

    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.util.enum import OptionType

    m = resolved.transformed
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=m.spot),
        vol_surface=FlatVolSurface(volatility=m.vol),
        rate_curve=FlatRateCurve(rate=m.rate),
        valuation_date=datetime(2024, 1, 1),
    )
    option = EuropeanVanillaOption(
        strike=m.strike, option_type=OptionType.CALL, maturity=m.maturity
    )
    pv = float(resolved.engine.price(option, env))
    economics = (("pv", pv), ("spot", m.spot), ("numerical.vol", m.vol))
    return pv, economics, None


# Registration must happen ONLY under this module's importable identity:
# spawn workers rebuild every callable by importing `execution_scenarios_demo`
# and the framework fingerprints the rebuilt base against the parent's. Both
# a `python example/execution_scenarios_demo.py` run (`__main__`) and the
# spawn bootstrap's re-execution of the launching script (`__mp_main__`)
# execute this file under NON-importable names — neither may register, or the
# worker's canonical import would see the same ids bound to different
# objects. `__main__` therefore delegates to the imported module below.
if __name__ == "execution_scenarios_demo":
    registries.register_factory("demo-market/v1", build_market_inputs)
    registries.register_factory("demo-euro-mc/v1", mc_engine_factory)
    registries.register_transformer(
        "demo-bump-spot/v1", bump_spot,
        allowed_tags=frozenset({"spot"}),
        components=(("spot", lambda b: b.spot), ("vol_surface", lambda b: b.vol)),
        covered_fields=("spot",),
    )
    registries.register_runner(
        "demo-price/v1", price_european, value_kind="float"
    )


# ----------------------------------------------------------------- demo run


def main():
    from quantark.execution import PricingSession, ScenarioSpec
    from quantark.execution.scenario.contracts import BaseInputsRef
    from quantark.execution.scenario.validate import compare_scenario_outcomes

    base = BaseInputsRef(
        factory_id="demo-market/v1",
        payload=(
            ("spot", 100.0), ("vol", 0.20), ("rate", 0.03),
            ("strike", 100.0), ("maturity", 1.0),
        ),
    )
    specs = [
        ScenarioSpec(
            scenario_id=f"spot{int(rel * 100):+d}%",
            transformer_id="demo-bump-spot/v1",
            parameters=(("rel", rel),),
            mutation_tags=frozenset({"spot"}),
            required_capabilities=frozenset({"runner:demo-price/v1"}),
        )
        for rel in (-0.10, -0.05, 0.0, 0.05, 0.10)
    ]

    def run(backend, workers):
        from quantark.execution import ExecutionPolicy, ExecutorSelection
        from quantark.execution import default_context

        context = dataclasses.replace(
            default_context(),
            execution_policy=ExecutionPolicy(
                scenario=ExecutorSelection(backend=backend, workers=workers)
            ),
        )
        with PricingSession(context) as session:
            return session.run_scenarios(base, specs, "demo-euro-mc/v1")

    serial = run("serial", 1)
    procs = run("processes", 2)

    for outcome in serial:
        print(f"  {outcome.scenario_id:>9}: pv = {outcome.value:.6f}")

    report = compare_scenario_outcomes(serial, procs)
    print(
        f"serial vs processes: {report.scenarios_matching}/"
        f"{report.scenarios_compared} scenarios, "
        f"{report.fields_matching}/{report.fields_compared} fields match"
    )
    assert report.all_scenarios_match


if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from execution_scenarios_demo import main as _canonical_main

    _canonical_main()
