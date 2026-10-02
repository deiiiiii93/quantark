"""Write far_field_before_g.json on the commit BEFORE flip (g) (spec 3A §5.10; Task 15 step 2).

The European call and the American put on the four term-curve shapes, and the monthly down-and-out
call 85 on the flat curve: values flip (g) must not move (the call rows) or moves by rounding only
(the flat barrier call, F2). Run from the repository root with the tree to capture on PYTHONPATH:

    PYTHONPATH=. python test/pde_3a/fixtures/capture_far_field_before_g.py
"""
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))           # test/: term_structure_benchmarks

from quantark.asset.equity.engine.pde import AmericanPDESolver, BarrierPDESolver, EuropeanPDESolver  # noqa: E402
from quantark.asset.equity.param import PDEParams  # noqa: E402
from quantark.asset.equity.product.option import AmericanOption, BarrierOption, EuropeanVanillaOption  # noqa: E402
from quantark.util.enum import BarrierType, ObservationType, OptionType  # noqa: E402
from term_structure_benchmarks import make_term_env  # noqa: E402

OBS = [k / 12.0 for k in range(1, 13)]


def main():
    values = {}
    for shape in ("flat", "up", "down", "kinked"):
        env = make_term_env(shape)
        values[shape] = {
            "european_call": float(EuropeanPDESolver(PDEParams()).price(
                EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0), env)).hex(),
            "american_put": float(AmericanPDESolver(PDEParams()).price(
                AmericanOption(100.0, OptionType.PUT, maturity=1.0), env)).hex(),
        }
    barrier_call = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=85.0,
                                 barrier_type=BarrierType.DOWN_OUT, observation_type=ObservationType.DISCRETE,
                                 observation_dates=OBS, maturity=1.0)
    values["flat"]["barrier_call"] = float(BarrierPDESolver(PDEParams()).price(
        barrier_call, make_term_env("flat"))).hex()
    out = {"machine": platform.machine(), "values": values}
    (HERE / "far_field_before_g.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
