"""The Greek harness sweeps the daily-KI fixture on its own ladder and profile and labels its certificates."""
from collections import Counter

from intraday.gate_c import cells as C
from intraday.gate_c.greek_harness import demonstrated, fixture_horizons, greek_groups


def test_daily_ki_groups_are_desk_only_on_their_own_ladder():
    groups = [g for g in greek_groups() if g.product == "snowball_daily_ki"]
    assert len(groups) == 154
    assert {g.profile for g in groups} == {"desk"}
    assert {g.horizon for g in groups} == set(C.DAILY_KI_HORIZONS)
    assert {g.barrier for g in groups} == {"ko", "ki"} and {g.offset for g in groups} == set(C.SPOT_OFFSETS)


def test_existing_fixture_groups_are_unchanged():
    counts = Counter(g.product for g in greek_groups())
    assert (counts["snowball_discrete_ki"], counts["snowball_long_gap"], counts["digital"]) == (528, 220, 352)


def _row(fixture, horizon_s, offset, barrier, status, identity):
    cell = {"product": fixture, "engine": "quad_v2", "profile": "desk", "horizon": horizon_s, "offset": offset,
            "barrier": barrier, "id": f"{fixture}-quad_v2-desk-{horizon_s}s-{offset}-{barrier}"}
    return {"cell": cell, "route": "QuadV2Route", "product": "SnowballOption", "settings": {"engine": "E"},
            "market": {}, "economic_identity": identity,
            "measures": [{"measure": "point_delta", "status": status, "measure_settings": {}}]}


def _sweep(fixture, identity, fail=None):
    return [_row(fixture, int(h.total_seconds()), o, b,
                 "failed" if (int(h.total_seconds()), o, b) == fail else "passed", identity)
            for h in fixture_horizons(fixture) for o in C.SPOT_OFFSETS for b in C.BARRIERS[fixture]]


def test_demonstrated_labels_rows_with_their_fixtures_and_keeps_families_apart():
    rows = demonstrated(_sweep("snowball_daily_ki", "daily") + _sweep("snowball_discrete_ki", "monthly"))
    by_identity = {r["economic_identity"]: r for r in rows}
    assert by_identity["daily"]["fixtures"] == ["snowball_daily_ki"]
    assert (by_identity["daily"]["horizon_s"], by_identity["daily"]["horizon_max_s"]) == (1, 21600)
    assert by_identity["monthly"]["fixtures"] == ["snowball_discrete_ki"]
    assert by_identity["monthly"]["horizon_max_s"] == 29 * 86400


def test_a_daily_ki_miss_splits_only_the_daily_window():
    rows = demonstrated(_sweep("snowball_daily_ki", "daily", fail=(300, "eq", "ki"))
                        + _sweep("snowball_discrete_ki", "monthly"))
    daily = sorted((r["horizon_s"], r["horizon_max_s"]) for r in rows if r["economic_identity"] == "daily")
    assert daily == [(1, 60), (900, 21600)]
    assert [(r["horizon_s"], r["horizon_max_s"]) for r in rows if r["economic_identity"] == "monthly"] == [(1, 29 * 86400)]


def test_only_the_daily_ki_reference_is_one_grid_level_finer():
    from intraday.gate_c.greek_harness import PROXY_REFERENCE_POINTS, proxy_points_for, reference_points_for
    from intraday.reference.budgets import GATE_C_POINTS
    assert reference_points_for("snowball_daily_ki") == (16001, 32001, 64001)
    assert proxy_points_for("snowball_daily_ki") == (8001, 16001)
    for fixture in ("snowball_discrete_ki", "snowball_long_gap", "digital"):
        assert reference_points_for(fixture) == GATE_C_POINTS and proxy_points_for(fixture) == PROXY_REFERENCE_POINTS
