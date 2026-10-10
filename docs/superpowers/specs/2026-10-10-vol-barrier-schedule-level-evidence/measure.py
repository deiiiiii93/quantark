"""Measure every W18 cell on one tree: the vol-model barrier routes (local-vol / Heston / SLV MC, Heston / SLV PDE)
on discrete schedules, dated and numeric, built as ``test/test_vol_barrier_schedule_level.py`` builds them.

Groups (cell prefix):
  plain     records at product.barrier, payoff unset, STOP_FIRST_HIT       -- must not move
  nolevel   records without a level (product.barrier by normalisation)       -- must not move
  level     records at a common level L != product.barrier                    -- moves (value -> value)
  terms     records at L paying 4.5 != product.rebate 3                       -- moves (value -> value)
  payoff    records at product.barrier paying 4.5 != product.rebate 3         -- knock-outs move, knock-ins do not
  greeks    up-and-out call, level L, day 90, price / delta / gamma           -- moves
  agg       ACCUMULATE / BEST / WORST, records at product.barrier             -- moves (value -> refused)
  agglimit  the same on the Heston / SLV PDE deterministic-variance limit     -- moves (value -> refused)

Run against any tree with the BRANCH's test directory second on the path (the cells are the branch's):

    env PYTHONPATH=<tree>:<branch>/test <venv>/bin/python measure.py > <out>.txt
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import quantark  # noqa: E402
import test_vol_barrier_schedule_level as t  # noqa: E402


def attempt(fn):
    try:
        return repr(float(fn()))
    except Exception as exc:  # noqa: BLE001 -- the evidence records the refusal
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:110]}"


def row(cell, form, quantity, value):
    print(f"{cell} | {form} | {quantity} | {value}")


print(f"# quantark from {quantark.__file__}")

for route in t.BARRIER_ROUTES:
    for kind, (_, _, barrier, level) in t.KINDS.items():
        for form in ("dated", "numeric"):
            for day in t.DAYS:
                def px(product, day=day):
                    return attempt(lambda: t._price(route, product, day))
                row(f"plain:{route}-{kind}-d{day}", form, "price", px(t._product(kind, form, barrier, barrier)))
                row(f"level:{route}-{kind}-d{day}", form, "price", px(t._product(kind, form, barrier, level)))
                if day == 10:
                    row(f"nolevel:{route}-{kind}-d{day}", form, "price", px(t._product(kind, form, barrier, None)))
                    row(f"terms:{route}-{kind}-d{day}", form, "price",
                        px(t._product(kind, form, barrier, level, payoff=4.5)))
                    row(f"payoff:{route}-{kind}-d{day}", form, "price",
                        px(t._product(kind, form, barrier, barrier, payoff=4.5)))

for route in t.BARRIER_ROUTES:
    factory, grid = t.BARRIER_ROUTES[route]
    for form in ("dated", "numeric"):
        e = t.env(100.0, t._on(90), grid=grid)
        for key in ("price", "delta", "gamma"):
            row(f"greeks:{route}-up_out_call-d90", form, f"greeks_{key}", attempt(lambda: t._quiet(
                lambda: factory().calculate_greeks(t._aged(t._product("up_out_call", form, 120.0, 115.0), 90),
                                                   e)[key])))

for route, kind, form, mode in t.AGG_CASES:
    _, _, barrier, _ = t.KINDS[kind]
    row(f"agg:{route}-{kind}-{mode.value}-d10", form, "price",
        attempt(lambda: t._price(route, t._product(kind, form, barrier, barrier, aggregation=mode), 10)))

for name, make in (("heston_pde_limit", lambda: t.HestonBarrierPDESolver(t._flat())),
                   ("slv_pde_limit", lambda: t.HestonSLVBarrierPDESolver(t._flat(), t._flat_unit_leverage()))):
    for mode in t.NON_FIRST_HIT:
        row(f"agglimit:{name}-up_out_call-{mode.value}-d10", "numeric", "price", attempt(lambda: t._quiet(
            lambda: make().price(t._aged(t._product("up_out_call", "numeric", 120.0, 120.0, aggregation=mode), 10),
                                 t.env(100.0, t._on(10))))))
