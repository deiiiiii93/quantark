"""The knock-in-side far-field row (spec 3C §5.5; decisions 5, 15): under discretely observed knock-in (European
included) the not-knocked-in edge on the knock-in side takes the knocked-in surface's row, while a knock-in observation
remains at or after the row's time.

Product builders and reference calls copied from the evidence
(``docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-evidence``), with ``lab3b.``/``lab.`` dropped (their
``env``, ``make_snowball``, ``make_phoenix``, ``SNOWBALL``, ``PHOENIX`` and ``KO_RESET`` are banked in
``pde_3a.products`` and ``pde_statistics.cases``), ``exact_stats.exact_with_error(...)["pv"]`` replaced by the banked
``pde_statistics.reference.exact_event_stats(...)["pv"]`` (its fine run) and ``lab3b.params(...)`` by ``PDEParams``:

  ``m3_far_field_accuracy.py`` lines 39-41   the European-knock-in Snowball cells
  ``r1_ki_row_followups.py`` lines 37-46     ``phoenix_european``; lines 68-71 the reverse Snowball against QUAD
  ``x4_ki_row_after_last_observation.py`` lines 23-34, 71-72   ``snowball(ki_dates)`` and the knock-in date sets
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pde_3a.products import KO_RESET, PHOENIX, SNOWBALL, env
from pde_statistics.reference import exact_event_stats

from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.param.engine_params import QuadParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.modelvalidation.builders import equity_phoenix
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from quantark.util.enum import ObservationType

SB, PH = SNOWBALL, PHOENIX
CAPTURE_PATH = Path(__file__).with_name("ki_side_capture.json")


def _exact(make, make_env):
    return lambda: float(exact_event_stats(make(), make_env())["pv"])


# --- m3_far_field_accuracy.py lines 39-41: the European-knock-in Snowball -------------------------------------------
def _sb_european():
    return make_snowball(dict(SB, ki_monitoring="european"))


EUROPEAN_KI_CASES = [
    dict(name="spot_100", product=_sb_european, env=env, exact=_exact(_sb_european, env)),
    dict(name="spot_86", product=_sb_european, env=lambda: env(spot=86.0), exact=_exact(_sb_european, lambda: env(spot=86.0))),
    dict(name="vol_35", product=_sb_european, env=lambda: env(vol=0.35), exact=_exact(_sb_european, lambda: env(vol=0.35))),
    dict(name="bounds_60", product=_sb_european, env=env, exact=_exact(_sb_european, env)),
]


# --- r1_ki_row_followups.py lines 37-46: the certified Phoenix with its knock-in observed at maturity only -----------
def phoenix_european(over=None):
    """The certified Phoenix with its knock-in observed at maturity only."""
    spec = dict(PH, ki_monitoring="discrete", **(over or {}))
    orig = equity_phoenix._ki_kwargs
    equity_phoenix._ki_kwargs = lambda s: {"ki_continuous": False, "ki_observation_type": ObservationType.DISCRETE,
                                           "ki_observation_dates": [float(s["maturity"])]}
    try:
        return make_phoenix(spec)
    finally:
        equity_phoenix._ki_kwargs = orig


PHOENIX_CASES = [
    dict(name=f"phoenix_european_spot_{s:g}", solver=lambda: PhoenixPDESolver(PDEParams()), product=phoenix_european,
         env=lambda s=s: env(spot=s), exact=_exact(phoenix_european, lambda s=s: env(spot=s)))
    for s in (100.0, 80.0)
]


# --- r1_ki_row_followups.py lines 68-71: the reverse Snowball (knock-out below at 97, knock-in above at 115) --------
def _reverse_european():
    # reverse: knock-out below (97), knock-in above (115)
    return make_snowball(dict(SB, ko_barrier=97.0, ki_barrier=115.0, is_reverse=True, ki_monitoring="european"))


def _quad_32001(make, make_env):
    return lambda: float(SnowballQuadEngine(QuadParams(grid_points=32001)).price(make(), make_env()))


REVERSE_CASES = [
    dict(name=f"reverse_european_spot_{s:g}", solver=lambda: SnowballPDESolver(PDEParams()), product=_reverse_european,
         env=lambda s=s: env(spot=s), exact=_quad_32001(_reverse_european, lambda s=s: env(spot=s)))
    for s in (100.0, 114.0, 98.0)
]


# --- x4_ki_row_after_last_observation.py lines 23-34, 71-72: knock-in dates that stop before maturity ---------------
REF = make_snowball(dict(SNOWBALL, ki_monitoring="discrete"))


def snowball(ki_dates):
    bc = REF.barrier_config
    return SnowballOption(
        initial_price=REF.initial_price, strike=REF.strike,
        barrier_config=BarrierConfig(ko_barrier=bc.ko_barrier, ko_rate=bc.ko_rate, ko_observation_type=bc.ko_observation_type,
                                     ko_observation_dates=list(bc.ko_observation_dates), ki_barrier=bc.ki_barrier,
                                     ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=list(ki_dates)),
        payoff_config=REF.payoff_config, contract_multiplier=1.0, maturity=1.0, is_reverse=False)


KI_DATES = {"early3": [k / 12.0 for k in (1, 2, 3)], "month6": [0.5], "monthly": [k / 12.0 for k in range(1, 13)],
            "none": []}

GATED_CASES = [
    dict(name=f"{tag}_spot_{s:g}", product=lambda d=dates: snowball(d), env=lambda s=s: env(spot=s),
         exact=_exact(lambda d=dates: snowball(d), lambda s=s: env(spot=s)))
    for tag, dates in KI_DATES.items() if tag != "monthly" for s in (100.0, 88.0)
]

#: the bit-identity cells: the exhausted schedule (the gate withholds the row everywhere) and the control
BUILD = {f"{tag}|{s:g}": (lambda d=KI_DATES[tag], s=s: (snowball(d), env(spot=s)))
         for tag in ("none", "monthly") for s in (100.0, 88.0)}


def KO_RESET_AFTER_FIRST():
    """The certified KO-reset study's ``discrete_ki`` cell (``example/modelvalidation/ko_reset_flat_bsm.yaml``): monthly
    knock-in to the end of its first schedule (one year), a second schedule to two years. Its march runs a year past
    its last knock-in observation, where the gate withholds the row (6,114 times in r4_anchors_gated.log)."""
    return make_ko_reset(dict(KO_RESET, ki_continuous=False)), env()


def _capture():
    return json.loads(CAPTURE_PATH.read_text())


@pytest.mark.parametrize("accuracy", ["standard", "high"])
@pytest.mark.parametrize("case", EUROPEAN_KI_CASES[:3], ids=[c["name"] for c in EUROPEAN_KI_CASES[:3]])
def test_european_knock_in_against_the_exact_recursion(case, accuracy):
    got = SnowballPDESolver(PDEParams(accuracy=accuracy)).price(case["product"](), case["env"]())
    assert abs(got - case["exact"]()) <= 5e-4, case["name"]           # measured 2.0e-4; today +4.4e-3


@pytest.mark.parametrize("case", PHOENIX_CASES + REVERSE_CASES, ids=lambda c: c["name"])
def test_phoenix_and_reverse(case):
    got = case["solver"]().price(case["product"](), case["env"]())
    assert abs(got - case["exact"]()) <= 5e-4, case["name"]


def test_reconciliation_identity_holds():
    case = EUROPEAN_KI_CASES[0]
    stats = SnowballPDESolver().calculate_event_stats(case["product"](), case["env"]())
    assert abs(stats.reconciliation_error) <= 1e-10                  # measured 6.0e-14


def test_bounds_close_to_the_barrier():
    case = next(c for c in EUROPEAN_KI_CASES if c["name"] == "bounds_60")
    got = SnowballPDESolver(PDEParams(grid=GridConfig(bounds=(60.0, None)))).price(case["product"](), case["env"]())
    assert abs(got - case["exact"]()) <= 5e-3                         # measured 1.3e-3; today 1.25


@pytest.mark.parametrize("case", GATED_CASES, ids=lambda c: c["name"])
def test_gated_row(case):
    got = SnowballPDESolver().price(case["product"](), case["env"]())
    assert abs(got - case["exact"]()) <= 1e-3                         # measured 3.3e-4 (ungated: 4.29e-3)


def test_exhausted_schedule_is_bit_identical():
    for tag, value in _capture()["snowball"].items():
        product, e = BUILD[tag]()
        assert SnowballPDESolver().price(product, e).hex() == value, tag


def test_ko_reset_after_its_first_schedule_is_unmoved():
    """The gate withholds the row once the first schedule has ended (no knock-in observable): the KO-reset of
    §4.9's certified cell, whose march runs past its first schedule, prices as before the row."""
    product, e = KO_RESET_AFTER_FIRST()
    assert KOResetSnowballPDESolver().price(product, e).hex() == _capture()["ko_reset"]
