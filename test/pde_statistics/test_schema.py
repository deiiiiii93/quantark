"""Statistics schema pieces shared by every producer: provenance, refusal, and the event distribution's explicit
maturity determination. (The legacy additive-schema checks went with the legacy types: unified event statistics
T18; the hand-built ``EventStatistics`` -> ``EventDistribution`` contract is pinned in
``test_cashleg/test_event_distribution_statistics.py``.)"""
import dataclasses

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import StatisticsProvenance, StatisticsRefused
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.util.exceptions import PricingError


def test_provenance_is_implementation_only():
    p = StatisticsProvenance("flat_vol_1d", "heston_adi_2d")
    assert p.surrogate and not StatisticsProvenance("flat_vol_1d", "flat_vol_1d").surrogate
    assert {f.name for f in dataclasses.fields(p)} == {"statistics_model", "price_model"}


def test_refusal_is_a_pricing_error():
    assert issubclass(StatisticsRefused, PricingError)


def test_explicit_maturity_determination_relaxes_the_terminal_check():
    # a maturity outcome determined at 0.5 and paid at 0.6, before the last event time 1.0
    d = EventDistribution(
        stream_times={EventType.KO: np.array([0.5, 1.0])},
        probabilities={EventType.KO: np.array([0.2, 0.1]), EventType.MATURITY_NO_KO: 0.3,
                       EventType.MATURITY_WITH_KI: 0.4},
        survival_times=np.array([0.0, 0.5, 1.0]), survival_probability=np.array([1.0, 0.8, 0.7]),
        payment_times={EventType.KO: np.array([0.5, 1.0]), EventType.MATURITY_NO_KO: 0.6,
                       EventType.MATURITY_WITH_KI: 1.0},
        determination_times={EventType.MATURITY_NO_KO: 0.5})
    assert d.determination_times[EventType.MATURITY_NO_KO] == 0.5
    with pytest.raises(Exception, match="precedes"):
        dataclasses.replace(d, determination_times={})   # today's rule: 0.6 < last event 1.0
