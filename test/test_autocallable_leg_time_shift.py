"""AutocallableCashLeg.time_shift mirrors the product drop rules [§11.9]."""

import numpy as np

from quantark.cashleg.autocallable_leg import (
    AccrualBasis,
    AutocallableCashLeg,
    AutocallableLegType,
)
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventType


def _leg(**over):
    kw = dict(
        direction=LegDirection.BUYER_RECEIVES,
        leg_type=AutocallableLegType.REBATE,
        notional=100.0,
        rate=0.05,
        observation_schedule=(0.25, 0.5, 0.75, 1.0),
        accrual_factors=(0.25, 0.5, 0.75, 1.0),
        settlement_schedule=(0.27, 0.52, 0.77, 1.02),
        terminal_accrual_factor=1.0,
        terminal_settlement_time=1.0,
        accrual_basis=AccrualBasis.KO_MATURITY,
        terminal_events=frozenset({EventType.MATURITY_NO_KO}),
    )
    kw.update(over)
    return AutocallableCashLeg(**kw)


def test_shift_drops_past_observations_and_shifts_survivors():
    shifted = _leg().time_shift(0.3)
    assert shifted is not None
    # 0.25 drops (0.25-0.3 <= 0); 0.5/0.75/1.0 survive shifted by -0.3
    assert np.allclose(shifted.observation_schedule, [0.2, 0.45, 0.7])
    # accrual factors for the dropped observation are removed
    assert np.allclose(shifted.accrual_factors, [0.5, 0.75, 1.0])
    # settlement schedule shifts with the survivors
    assert np.allclose(shifted.settlement_schedule, [0.22, 0.47, 0.72])
    # terminal settlement shifts too
    assert abs(shifted.terminal_settlement_time - 0.7) < 1e-12


def test_shift_returns_none_when_all_observations_drop():
    assert _leg().time_shift(1.5) is None


def test_shift_boundary_observation_drops_at_tolerance():
    # obs exactly at dt drops (obs - dt == 0, not > 0)
    shifted = _leg(observation_schedule=(0.3, 0.6),
                   accrual_factors=(1.0, 1.0),
                   settlement_schedule=(0.3, 0.6)).time_shift(0.3)
    assert shifted is not None
    assert np.allclose(shifted.observation_schedule, [0.3])  # only 0.6 -> 0.3 survives


def test_zero_or_negative_shift_is_identity():
    leg = _leg()
    assert leg.time_shift(0.0) is leg
    assert leg.time_shift(-0.1) is leg


def test_terminal_only_leg_survives_shift():
    # A leg with no periodic observations (terminal branch only) is not dropped.
    leg = _leg(
        observation_schedule=(),
        accrual_factors=(),
        settlement_schedule=(),
    )
    shifted = leg.time_shift(0.4)
    assert shifted is not None
    assert shifted.observation_schedule == ()
    assert abs(shifted.terminal_settlement_time - 0.6) < 1e-12


def test_shifted_leg_still_valid_and_required_streams_preserved():
    shifted = _leg().time_shift(0.3)
    assert shifted.required_event_types() == frozenset(
        {EventType.KO, EventType.MATURITY_NO_KO}
    )
