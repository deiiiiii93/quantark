"""PDE term-structure performance benchmark (spec Component 4 gate).

Times the standard 1y snowball PDE on a flat environment vs a term
environment whose curves have 4 pillars (far fewer than the solver's time
steps — the worst case for operator-cache reuse, since forward parameters
then differ on every step). Budget: term/flat median ratio <= 1.20.

Run: PYTHONPATH=. .venv/bin/python example/pde_term_structure_benchmark.py
(from the repo root; the test/ directory must be importable for the shared
benchmark environments).
"""
import statistics
import sys
import time

sys.path.insert(0, "test")

from term_structure_benchmarks import make_term_env  # noqa: E402

from quantark.asset.equity.engine.pde import SnowballPDESolver  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import (  # noqa: E402
    BarrierConfig,
)
from quantark.asset.equity.product.option.snowball_option import (  # noqa: E402
    SnowballOption,
)
from quantark.util.enum import ObservationType  # noqa: E402


def _product() -> SnowballOption:
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


def _median_time(env, runs: int = 5) -> float:
    times = []
    for _ in range(runs):
        product = _product()
        start = time.perf_counter()
        SnowballPDESolver().price(product, env)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> None:
    flat = _median_time(make_term_env("flat"))
    term = _median_time(make_term_env("kinked"))
    ratio = term / flat
    print(f"{'case':<12}{'median (s)':>12}")
    print(f"{'flat':<12}{flat:>12.4f}")
    print(f"{'term (4p)':<12}{term:>12.4f}")
    print(f"{'ratio':<12}{ratio:>12.3f}   (budget <= 1.20)")
    if ratio > 1.20:
        print("BUDGET EXCEEDED — profile with solver.enable_profiling(True)")
        sys.exit(1)


if __name__ == "__main__":
    main()
