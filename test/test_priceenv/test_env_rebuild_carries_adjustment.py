"""AC10: a PricingEnvironment rebuilt field by field keeps ``holiday_time_adjustment``.

A rebuild that copies ``day_count_convention`` from another environment but not the adjustment would silently put a
FOLLOWING environment back on PRECEDING under a bump, a scenario or a snapshot (spec D12'). The guard scans every
``PricingEnvironment(...)`` call in ``quantark/``.
"""
import ast
from pathlib import Path

import quantark

ROOT = Path(quantark.__file__).resolve().parent


def _rebuild_calls():
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name != "PricingEnvironment":
                continue
            kwargs = {k.arg: k.value for k in node.keywords if k.arg}
            copied = kwargs.get("day_count_convention")
            if isinstance(copied, ast.Attribute) and copied.attr == "day_count_convention":
                yield path.relative_to(ROOT.parent), node.lineno, kwargs


def test_there_are_rebuild_sites_to_guard():
    assert len(list(_rebuild_calls())) >= 5


def test_every_rebuild_copies_the_adjustment():
    missing = [f"{path}:{line}" for path, line, kwargs in _rebuild_calls()
               if "holiday_time_adjustment" not in kwargs]
    assert not missing, f"PricingEnvironment rebuilds dropping holiday_time_adjustment: {missing}"


#: 0.15.0 (cash-curve quoting spec D12): a rebuild that drops the calendar or a quoting declaration turns a
#: calendar-quoted curve into an undeclared native one (refused) or leaves the environment unable to re-anchor.
CLOCK_FIELDS = ("calendar", "rate_curve_quoting", "div_yield_quoting", "basis_yield_quoting", "trading_clock_horizon")


def test_every_rebuild_copies_the_clock_fields():
    missing = [f"{path}:{line} {name}" for path, line, kwargs in _rebuild_calls()
               for name in CLOCK_FIELDS if name not in kwargs]
    assert not missing, f"PricingEnvironment rebuilds dropping clock fields: {missing}"
