"""MC event statistics come from ``price()``'s own paths (unified event statistics spec AC2, AC4, D14, D23).

Across the route grid of ``mc_price_path_routes`` (PSEUDO, QMC, Dask batches, adaptive and fixed RQMC, the
conditional-moment estimators, the vol-model subclasses, KO-reset ABSOLUTE / REBASED, DCN, a carried
knock-in, a knock-out decided at valuation, zero time to maturity):

- ``calculate_event_stats(p, env).pv == price(p, env)`` bitwise, and
  ``price_with_events(p, env).npv == price(p, env)`` bitwise -- each call on a fresh engine with the same
  parameters, so no state cached by one call can help another;
- the recorder holds one batch record per batch the route priced;
- the outcomes partition the paths: knock-out counts plus the two maturity branches equal the path count;
- the recorded outcome cash adds up to the cash the price discounted, batch by batch (native kernels);
- the statistics are the price's own path statistics (the engine's last result reports the same total
  knock-out probability on the serial and Dask routes).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from test_asset.equity.engine_mc.mc_price_path_routes import DASK_ROUTES, ROUTES

warnings.filterwarnings("ignore", category=DeprecationWarning)

STATS_ROUTES = [name for name, route in ROUTES.items() if route.family != "dcn"]
DCN_ROUTES = [name for name, route in ROUTES.items() if route.family == "dcn"]
#: summed-cash identity bound (relative to the batch's absolute discounted cash): round-off
ROUND_OFF = 1e-12


def _fresh(name):
    route = ROUTES[name]
    return route, route.engine(), route.product(), route.env()


def _price(name):
    _, engine, product, env = _fresh(name)
    return engine.price(product, env)


def _expected_batches(route, engine):
    if route.batches == "adaptive":
        return int(engine.get_last_result().batches_used)
    return int(route.batches)


# --- AC2: the statistics' pv is price() -------------------------------------------------------------

@pytest.mark.parametrize("name", STATS_ROUTES)
def test_event_stats_pv_is_the_price_bitwise(name):
    price = _price(name)
    route, engine, product, env = _fresh(name)
    stats = engine.calculate_event_stats(product, env)
    if route.expired:
        assert stats is None          # zero time to maturity: the documented expired rule
        return
    assert stats.pv == price, f"{name}: stats.pv {stats.pv!r} != price {price!r}"


@pytest.mark.parametrize("name", DCN_ROUTES)
def test_dcn_event_stats_pv_is_the_price_bitwise(name):
    price = _price(name)
    _, engine, product, env = _fresh(name)
    result = engine.price_detailed(product, env)
    assert result.event_stats.pv == price


# --- AC4 / D14: price_with_events returns price() ------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES))
def test_price_with_events_npv_is_the_price_bitwise(name):
    price = _price(name)
    route, engine, product, env = _fresh(name)
    result = engine.price_with_events(product, env)          # REBASED included (spec D13, T9)
    assert result.npv == price, f"{name}: npv {result.npv!r} != price {price!r}"


# --- one record per priced batch ----------------------------------------------------------------------

@pytest.mark.parametrize("name", STATS_ROUTES)
def test_recorder_holds_one_record_per_priced_batch(name):
    route, engine, product, env = _fresh(name)
    engine.calculate_event_stats(product, env)
    records = engine._last_stats_records
    assert len(records) == _expected_batches(route, engine)
    if route.batches == "adaptive":
        assert 2 <= len(records) <= 12


@pytest.mark.parametrize("name", DCN_ROUTES)
def test_dcn_recorder_holds_one_record_per_batch(name):
    route, engine, product, env = _fresh(name)
    engine.price_detailed(product, env)
    assert len(engine._last_stats_records) == int(route.batches)
    assert sum(int(r["n"]) for r in engine._last_stats_records) == engine.num_paths


def test_adaptive_routes_stop_before_their_maximum():
    """The adaptive grid entries really adapt (else they would only repeat the fixed-count routes)."""
    stopped = []
    for name, route in ROUTES.items():
        if route.batches == "adaptive":
            _, engine, product, env = _fresh(name)
            engine.price(product, env)
            stopped.append(engine.get_last_result().batches_used)
    assert stopped and all(2 <= b < 12 for b in stopped)


# --- partition and cash identities per record ---------------------------------------------------------

def _ko_total(stats) -> float:
    """Total knock-out probability: KO rows; REBASED: the termination distribution's knock-outs."""
    if stats.termination is not None:
        events = np.asarray(stats.termination.event)
        return float(stats.termination.probability[(events == "ko_pre") | (events == "ko_post")].sum())
    return float(np.sum(stats.observations.probability))


def _ko_rows(record):
    if "pre_count" in record:
        return record["pre_count"].sum() + record["post_count"].sum()
    return record["ko_count"].sum()


@pytest.mark.parametrize("name", [n for n in STATS_ROUTES if not ROUTES[n].expired])
def test_outcomes_partition_every_batch(name):
    _, engine, product, env = _fresh(name)
    stats = engine.calculate_event_stats(product, env)
    total_n = 0
    terminal = 0.0
    for record in engine._last_stats_records:
        n = int(record["n"])
        total_n += n
        closed = _ko_rows(record) + record["v0_count"] + record["v1_count"]
        if record["kind"] == "native":
            assert closed == n
        else:
            assert closed == pytest.approx(n, abs=1e-9 * n)
        terminal += record["v0_count"] + record["v1_count"]
    assert _ko_total(stats) + terminal / total_n == pytest.approx(1.0, abs=1e-12)


def _outcome_cash(record):
    cash = record.get("ko_disc_cash", record.get("ko_redemption_disc_cash"))
    if "pre_disc_cash" in record:
        cash = record["pre_disc_cash"].sum() + record["post_disc_cash"].sum()
    else:
        cash = cash.sum()
    cash += record.get("coupon_disc_cash", np.zeros(1)).sum()
    return cash + record["v0_disc_cash"] + record["v1_disc_cash"]


@pytest.mark.parametrize("name", [n for n in STATS_ROUTES if not ROUTES[n].expired])
def test_recorded_outcome_cash_adds_up_to_the_priced_cash(name):
    _, engine, product, env = _fresh(name)
    engine.calculate_event_stats(product, env)
    for record in engine._last_stats_records:
        scale = max(abs(record["pv_sum"]), 1.0)
        assert abs(_outcome_cash(record) - record["pv_sum"]) <= ROUND_OFF * scale * 10


@pytest.mark.parametrize("name", [n for n in STATS_ROUTES
                                  if not ROUTES[n].expired and "rqmc" not in n and "conditional" not in n
                                  and "control" not in n and "affine" not in n and "adaptive" not in n])
def test_statistics_are_the_prices_own_path_statistics(name):
    """The serial and Dask routes report their KO weight in the result: the statistics agree on it."""
    _, engine, product, env = _fresh(name)
    stats = engine.calculate_event_stats(product, env)
    result = engine.get_last_result()
    assert _ko_total(stats) == pytest.approx(result.ko_probability, abs=1e-14)


# --- the Dask route records on every scheduler it can select --------------------------------------------

@pytest.mark.parametrize("scheduler", ["threads", "processes", "synchronous"])
@pytest.mark.parametrize("name", [n for n in DASK_ROUTES if n in ("sb_cki_dask", "kr_rebased_dask",
                                                                  "ph_mem_instant_dask")])
def test_dask_batches_record_on_every_scheduler(name, scheduler):
    dask = pytest.importorskip("dask")
    with dask.config.set(scheduler=scheduler):
        price = _price(name)
        route, engine, product, env = _fresh(name)
        stats = engine.calculate_event_stats(product, env)
    assert stats.pv == price
    assert len(engine._last_stats_records) == int(route.batches)
    assert sum(int(r["n"]) for r in engine._last_stats_records) == 2048


def test_dask_records_do_not_depend_on_the_scheduler():
    dask = pytest.importorskip("dask")
    seen = []
    for scheduler in ("threads", "processes", "synchronous"):
        with dask.config.set(scheduler=scheduler):
            _, engine, product, env = _fresh("ph_mem_instant_dask")
            stats = engine.calculate_event_stats(product, env)
        seen.append((stats.pv, tuple(stats.observations.probability), tuple(stats.coupons.probability),
                     tuple(stats.coupons.discounted_cash), tuple(stats.ki_table.probability)))
    assert seen[0] == seen[1] == seen[2]
