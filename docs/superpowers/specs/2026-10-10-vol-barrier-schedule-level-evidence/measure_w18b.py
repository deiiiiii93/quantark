"""Measure every W18b cell on one tree: the Black-Scholes barrier MC (and the three vol-model barrier MC routes, which
return its expiry value at T = 0) deciding at the valuation instant, built as
``test/test_barrier_mc_record_level_at_valuation.py`` builds them.

Groups (cell prefix):
  expiry     exercise date, records at L != product.barrier       -- moves only where the spot lies between the levels
  expiry0    exercise date, records at product.barrier (plain)     -- must not move
  payoff     exercise date, records at L paying 4.5               -- moves where the decision or the cash changes
  mid        day 91 (an observation today), records at L          -- moves where the spot lies between the levels
  mid0       day 91, plain                                        -- must not move
  next       day 92 (no observation today), records at L          -- moves where the spot lies between the levels
  next0      day 92, plain                                        -- must not move

    env PYTHONPATH=<tree>:<branch>/test <venv>/bin/python measure_w18b.py > <out>.txt
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import quantark  # noqa: E402
import test_barrier_mc_record_level_at_valuation as t  # noqa: E402
from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver  # noqa: E402
from quantark.asset.equity.param import PDEParams  # noqa: E402


def attempt(fn):
    try:
        return repr(float(fn()))
    except Exception as exc:  # noqa: BLE001 -- the evidence records the refusal
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:110]}"


def row(cell, form, quantity, value):
    print(f"{cell} | {form} | {quantity} | {value}")


SPOT_NAMES = ("between", "beyond", "inside")
print(f"# quantark from {quantark.__file__}")

for route, (factory, grid) in t.MC_ROUTES.items():
    for kind, (_, _, barrier, level, spots) in t.KINDS.items():
        for form in ("dated", "numeric"):
            for schedule in t.SCHEDULES:
                for name, spot in zip(SPOT_NAMES, spots):
                    e = t.env(spot, t.EXERCISE, grid=grid)
                    for group, lvl in (("expiry", level), ("expiry0", barrier)):
                        row(f"{group}:{route}-{kind}-{schedule}-{name}", form, "price", attempt(lambda: t._quiet(
                            lambda: factory().price(t._settled(kind, form, barrier, lvl, schedule), e))))
                    if schedule == "single":
                        row(f"payoff:{route}-{kind}-{schedule}-{name}", form, "price", attempt(lambda: t._quiet(
                            lambda: factory().price(t._settled(kind, form, barrier, level, schedule, payoff=4.5),
                                                    e))))

for kind, (_, _, barrier, level, spots) in t.KINDS.items():
    for form in ("dated", "numeric"):
        for name, spot in zip(SPOT_NAMES, spots):
            for day, keep, group in ((91, True, "mid"), (92, False, "next")):
                e = t.env(spot, t._on(day))
                for g, lvl in ((group, level), (group + "0", barrier)):
                    row(f"{g}:bs_mc-{kind}-d{day}-{name}", form, "price", attempt(lambda: t._quiet(
                        lambda: t._engine().price(t._mid(kind, form, barrier, lvl, day, keep), e))))
                if group == "mid":
                    row(f"midpde:bs_pde-{kind}-d{day}-{name}", form, "price", attempt(lambda: t._quiet(
                        lambda: BarrierPDESolver(PDEParams()).price(t._mid(kind, form, barrier, level, day, keep),
                                                                    e))))
