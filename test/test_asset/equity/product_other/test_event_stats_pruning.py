"""Event-stats stream pruning + npv passthrough (Phase 2a, plan Task 2.4 slice).

Golden invariant: pruning the KI indicator columns and/or Phoenix coupon columns
(when no leg requests them) leaves ko_probability / survival / pv bit-identical —
the KI regime jump still runs; only auxiliary columns are dropped [§11.1]. And a
caller-supplied npv reproduces the internal self.price() result exactly.
"""

from datetime import datetime

import numpy as np

from golden_compare import assert_close

from quantark.asset.equity.engine.pde import PhoenixPDESolver, SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg.event_distribution import EventType
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _snowball():
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _phoenix():
    ko_dates = [i / 4 for i in range(1, 5)]
    bc = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=ko_dates,
        ki_barrier=None,
    )
    cc = CouponBarrierConfig(
        coupon_barrier=85.0,
        coupon_rate=0.02,
        coupon_pay_type=CouponPayType.INSTANT,
        day_count_convention=DayCountConvention.ACT_365,
        memory_coupon=True,
    )
    pf = PayoffConfig(rebate_rate=0.0, include_principal=True)
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=bc,
        coupon_config=cc,
        payoff_config=pf,
        contract_multiplier=100.0,
        maturity=1.0,
    )


# Golden values captured from the pre-2a code (commit before stream pruning),
# regenerated 2026-07 after the Rannacher terminal-step off-by-one fix
# (rannacher_steps=1 now applies one implicit-Euler step, shifting the
# discretization by ~3e-6 relative), and regenerated 2026-07-23 for the
# corrected default event semantics (event_projection="cell_average" +
# event_rannacher_steps=2): PV moved -3.4e-4 relative, KI +1.5e-3 absolute —
# the half-cell trigger-phase bias the projection removes (see
# pde_auto_grid_investigation.md and test_pde_event_projection.py), and
# nudged -1e-5 relative on 2026-07-23 when the straddling cell switched to
# the full two-branch cell average (review finding 1: envelope-preserving
# projection; moves further toward the QUAD reference).
# Re-frozen 2026-08-18 for the FIRST_PASSAGE continuous-KI correction: the
# per-step nodal jump missed intra-step barrier crossings, understating KI.
# Verification against SnowballMCEngine at 262,144 paths (exact first-passage
# ki_ever 0.150738): ki_ever moved 0.147043 -> 0.150637, closing the gap to
# ~1e-4; pv is steps-per-day invariant to 0.5c/unit post-fix (was O(sqrt(dt))).
# Re-frozen 2026-10-04 for spec 3B stage 2 (a): the statistics take the value policy (allocated rows,
# VALUE theta, the adapter's stages); moves ki -6.0e-9, ki_ever -6.3e-9; pv unchanged.
# Unified event statistics T12: the statistics are ``EventStatistics``; SNOWBALL_KI is the KI settlement row
# (knocked in and not knocked out, the marched KIS column — unmoved, bitwise). The KI-ever golden is dropped: a
# knock-in after a knock-out is no knock-in, and P(KI before KO) comes from the forward pass (T13).
SNOWBALL_PV = 987699.4565124855  # re-frozen with the FIRST_PASSAGE default
SNOWBALL_KI = 0.1137292604459367


def test_snowball_full_stream_matches_golden():
    stats = SnowballPDESolver(PDEParams()).calculate_event_stats(
        _snowball(), _env()
    )
    # These are frozen on ONE machine, so they compare at the suite's
    # cross-architecture tolerance rather than an absolute epsilon. `< 1e-6` on
    # a PV of ~9.9e5 is 1e-12 RELATIVE -- tighter than the 2.2e-11 drift
    # measured between x86_64 and ARM64 on these solvers, so it was passing on
    # margin rather than by design. Probabilities are O(0.1), where the same
    # relative bound is far tighter in absolute terms than the old 1e-9.
    assert_close(float(stats.pv), SNOWBALL_PV, msg="snowball pv")
    assert_close(float(stats.ki_settlement.probability), SNOWBALL_KI, msg="snowball ki")


def test_snowball_ki_pruning_preserves_ko_and_pv():
    solver = SnowballPDESolver(PDEParams())
    env = _env()
    product = _snowball()
    full = solver.calculate_event_stats(product, env, streams=None)
    lean = solver.calculate_event_stats(
        product, env, streams=frozenset({EventType.KO})
    )
    # KO/survival/pv bit-identical whether or not the KI stream is requested.
    assert np.array_equal(
        np.asarray(full.observations.probability), np.asarray(lean.observations.probability)
    )
    assert np.array_equal(
        np.asarray(full.observations.survival),
        np.asarray(lean.observations.survival),
    )
    assert float(full.pv) == float(lean.pv)
    # T12: the KI settlement unit is never pruned (it is a maturity outcome, AC7), so it is the same number;
    # the KI-stream fields are "not requested", not reported as 0.
    assert lean.ki_settlement.probability == full.ki_settlement.probability > 0.0
    assert lean.knock_in.probability is None and lean.ki_table is None
    assert lean.unavailable["knock_in.probability"] == "not requested"


def test_npv_passthrough_skips_internal_price():
    solver = SnowballPDESolver(PDEParams())
    env = _env()
    product = _snowball()
    internal = solver.calculate_event_stats(product, env)
    sentinel = 123456.0
    passed = solver.calculate_event_stats(product, env, npv=sentinel)
    assert float(passed.pv) == sentinel
    # the leg is marched (spec 3B decision 3): the supplied npv moves the residual, not the leg
    assert passed.maturity.discounted_cash == internal.maturity.discounted_cash
    assert passed.ki_settlement.discounted_cash == internal.ki_settlement.discounted_cash
    assert np.array_equal(passed.ledger.discounted_cash, internal.ledger.discounted_cash)
    assert abs(passed.reconciliation_error - (sentinel - float(np.sum(internal.ledger.discounted_cash)))) < 1e-8


def test_phoenix_coupon_pruning_preserves_ko_and_pv():
    solver = PhoenixPDESolver(PDEParams())
    env = _env()
    product = _phoenix()
    full = solver.calculate_event_stats(product, env, streams=None)
    lean = solver.calculate_event_stats(
        product, env, streams=frozenset({EventType.KO})
    )
    assert np.array_equal(
        np.asarray(full.observations.probability), np.asarray(lean.observations.probability)
    )
    assert float(full.pv) == float(lean.pv)
    # Coupon trigger columns pruned => coupon probability "not requested"; the coupon cash (a ledger flow) stays.
    assert np.asarray(full.coupons.probability).size > 0
    assert lean.coupons.probability is None
    assert lean.unavailable["coupons.probability"] == "not requested"
    assert np.array_equal(full.coupons.discounted_cash, lean.coupons.discounted_cash)
