"""The replay's maturity probe env carries the run's clock (0.16.0, DCN/dated run D8).

``ProductReplay._remaining_maturity_years`` measures the product's remaining maturity on a throwaway probe env; the
day's pricing env comes from ``ProductReplay.build_env``. Both are built on the default calendar-day clock today (no
backtest code sets one), so the probe already measures on the run's clock. The guard keeps it that way: a clock field
passed to one construction and not the other fails here.
"""
import ast
import inspect
import textwrap

from quantark.backtest.replay.product_replay import ProductReplay

CLOCK_KEYWORDS = {"day_count_convention", "calendar", "bus_days_in_year", "holiday_time_adjustment",
                  "trading_clock_horizon", "rate_curve_quoting", "div_yield_quoting", "basis_yield_quoting"}


def _env_clock_keywords(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "PricingEnvironment"]
    assert calls, f"{method.__name__} builds no PricingEnvironment"
    return [frozenset(k.arg for k in call.keywords if k.arg in CLOCK_KEYWORDS) for call in calls]


def test_probe_env_and_built_env_share_the_clock():
    built = set(_env_clock_keywords(ProductReplay.build_env))
    probe = set(_env_clock_keywords(ProductReplay._remaining_maturity_years))
    assert built == probe, f"build_env clock keywords {built} != probe {probe}"
