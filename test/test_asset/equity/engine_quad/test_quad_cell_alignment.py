"""Integer-cell barrier alignment: ``QuadParams.align_cell_stretch``.

A uniform lattice has one degree of freedom, so it can pin exactly ONE
barrier onto a node. ``align_priority`` decides which, and because it picks
the barrier nearest to spot, the choice flips where spot crosses the
geometric mean of two barriers. Either side of that crossover the price is
computed on a differently placed lattice, so a spot bump that straddles it
measures a grid change on top of a market change and delta is
discontinuous. Evidence: docs/bucket-futures-hedge/gates.md, "The alignment
crossover".

Widening the cell so the separations between barriers are whole numbers of
cells puts every barrier on a node at once. The pin then selects among
lattices that are the same set of points, so the choice -- and the
discontinuity -- stops existing.

Opt-in, because it moves barrier prices and so rebases goldens.
"""

from datetime import datetime
import math

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.quad_math import (
    QuadratureMath,
    snap_cell_width,
)
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.param import FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

SPOT = 100.0
KI = 75.0
KO = 103.0
#: Where the barrier nearest to spot changes hands, so where the alignment
#: target switches: log(S/KI) == log(KO/S) at S = sqrt(KI * KO).
CROSSOVER = math.sqrt(KI * KO)
#: 1001 points leaves about 49 cells between the barriers, so reaching a
#: whole number of them costs well under 1%. Coarser grids need more; see
#: test_a_budget_too_small_to_reach_a_whole_cell_leaves_the_grid_alone.
GRID = 1001
STRETCH = 0.02


def _env(spot: float = SPOT) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot)),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.03),
        valuation_date=datetime(2026, 1, 2),
    )


def _product():
    return create_standard_snowball(
        initial_price=SPOT,
        strike=SPOT,
        maturity=1.0,
        num_observations=6,
        ko_barrier=KO,
        ki_barrier=KI,
        include_principal=False,
    )


def _price(
    spot: float,
    priority: str,
    stretch: float | None,
    grid_points: int = GRID,
) -> float:
    kwargs = {"grid_points": grid_points, "align_priority": priority}
    if stretch is not None:
        kwargs["align_cell_stretch"] = stretch
    return SnowballQuadEngine(params=QuadParams(**kwargs)).price(
        _product(), _env(spot)
    )


def _math(stretch: float | None, barrier_logs, grid_x: int = GRID) -> QuadratureMath:
    return QuadratureMath(
        grid_x=grid_x,
        spot=SPOT,
        maturity=1.0,
        vol_max=0.22,
        align_log=barrier_logs[0] if barrier_logs else None,
        barrier_logs=barrier_logs,
        cell_stretch=stretch,
    )


def _barrier_logs(spot: float = SPOT):
    return (math.log(KI / spot), math.log(KO / spot))


# ---------------------------------------------------------------------------
# the option itself
# ---------------------------------------------------------------------------
def test_the_option_is_off_unless_asked_for():
    """Prices move when it is on, so nothing already banked may opt in by default."""
    assert QuadParams().align_cell_stretch is None


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_a_stretch_outside_its_range_is_refused(bad):
    with pytest.raises(ValidationError, match="align_cell_stretch"):
        QuadParams(align_cell_stretch=bad)


def test_off_builds_exactly_the_grid_it_always_did():
    """Byte-for-byte guard on the default path, against the pre-change arithmetic."""
    logs = _barrier_logs()
    built = _math(None, logs)

    log_c = math.log(built.constant_c)
    h = 2.0 * log_c / (GRID - 1)
    idx = int(round((logs[0] + log_c) / h))
    idx = max(0, min(idx, GRID - 1))
    shift = logs[0] - (-log_c + idx * h)
    expected = np.linspace(-log_c, log_c, GRID) + shift

    assert built.h == h
    assert built.cell_snap_ratio == 1.0
    assert np.array_equal(built.grid, expected)


# ---------------------------------------------------------------------------
# the cell-width search
# ---------------------------------------------------------------------------
def test_fewer_than_two_levels_have_no_separation_to_fix():
    h0 = 0.0064484
    assert snap_cell_width(h0, [], 0.05) == h0
    assert snap_cell_width(h0, [math.log(0.75)], 0.05) == h0


def test_two_barriers_land_on_nodes_together():
    logs = _barrier_logs()
    snapped = _math(STRETCH, logs)
    cells = abs(logs[1] - logs[0]) / snapped.h

    assert cells == pytest.approx(round(cells), abs=1e-9)
    assert snapped.barrier_cell_offset == pytest.approx(0.0, abs=1e-12)
    # and it was genuinely off-node before, or this proves nothing
    assert _math(None, logs).barrier_cell_offset > 1e-4


def test_the_cell_never_shrinks_and_the_node_count_never_changes():
    """The guard against a grid that grows to fit the barriers.

    Holding the domain and shrinking the cell would need proportionally more
    nodes. Widening instead keeps grid_x fixed and lets the domain grow with
    the cell, which makes the truncation bound safer, not weaker.
    """
    logs = _barrier_logs()
    plain, snapped = _math(None, logs), _math(STRETCH, logs)

    assert snapped.grid_x == plain.grid_x == GRID
    assert snapped.grid.size == plain.grid.size
    assert snapped.h >= plain.h
    assert snapped.h <= plain.h * (1.0 + STRETCH)
    # domain widened with the cell rather than the grid growing
    assert np.ptp(snapped.grid) >= np.ptp(plain.grid)


def test_a_budget_too_small_to_reach_a_whole_cell_leaves_the_grid_alone():
    """It declines rather than overspending, and says so through the ratio.

    A coarse grid puts few cells between the barriers, and the widening
    needed is roughly one part in that count -- so the same budget that is
    ample at 1001 points is not at 301.
    """
    logs = _barrier_logs()
    coarse = _math(0.005, logs, grid_x=301)

    assert coarse.cell_snap_ratio == 1.0
    assert coarse.barrier_cell_offset > 1e-4
    assert _math(0.08, logs, grid_x=301).cell_snap_ratio > 1.0


def test_three_levels_cannot_all_land_but_the_worst_offset_still_improves():
    """One spacing cannot satisfy two independent separations exactly.

    Two levels give one separation and one unknown, so it is solvable. A
    third adds a second constraint with no second unknown, so the search
    minimises the WORST off-node distance instead of reaching zero -- and
    must not claim otherwise.
    """
    logs = (*_barrier_logs(), math.log(92.0 / SPOT))
    plain, snapped = _math(None, logs), _math(0.05, logs)

    assert snapped.barrier_cell_offset < plain.barrier_cell_offset
    assert snapped.barrier_cell_offset > 0.0


# ---------------------------------------------------------------------------
# what it is for
# ---------------------------------------------------------------------------
def test_the_alignment_target_stops_changing_the_price():
    """With every barrier on a node, the pin selects among identical lattices."""
    off_ki, off_ko = _price(SPOT, "ki", None), _price(SPOT, "ko", None)
    on_ki, on_ko = _price(SPOT, "ki", STRETCH), _price(SPOT, "ko", STRETCH)

    assert off_ki != off_ko
    assert on_ko == pytest.approx(on_ki, rel=1e-9)


#: The delta test runs COARSE on purpose. How far off-node a barrier sits
#: barely changes with resolution, but how much the price cares shrinks
#: fast: the gap between the two pinned branches measures 0.0255 at 201
#: points and 0.0010 at 1001. So the defect is a coarse-grid problem, and a
#: fine grid would leave nothing for the assertion to see. Reaching a whole
#: cell costs about one part in the number of cells between the barriers --
#: roughly 9 here -- so the budget has to be far wider than 2%.
KINK_GRID = 201
KINK_STRETCH = 0.12


def test_delta_stops_kinking_where_the_alignment_target_switches():
    """The defect this exists for, isolated with a second difference.

    Scanned across sqrt(KI * KO), where `auto` hands the pin from one
    barrier to the other. The spread of the secants alone will not do: it is
    dominated by real gamma, which is present either way. The change in the
    secant from one step to the next cancels smooth curvature and leaves the
    kink, which is what a discontinuity actually is.
    """
    spots = CROSSOVER + np.array([-1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5])

    def worst_kink(stretch):
        prices = np.array(
            [_price(s, "auto", stretch, KINK_GRID) for s in spots]
        )
        delta = np.diff(prices) / np.diff(spots)
        return float(np.abs(np.diff(delta)).max())

    off, on = worst_kink(None), worst_kink(KINK_STRETCH)
    assert off > 10.0 * on
