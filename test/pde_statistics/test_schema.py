"""The 3B schema is additive (spec 3B §6, §10 step 5): every new field defaults to today's meaning."""
import dataclasses

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import (
    AutocallableEventStats, KOResetEventStats, StatisticsProvenance, StatisticsRefused,
)
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.util.exceptions import PricingError


def _stats(**kw):
    base = dict(pv=1.0, ko_times=np.array([0.5, 1.0]), ko_probability=np.array([0.2, 0.1]),
                survival_probability=np.array([0.8, 0.7]), expected_discounted_ko_cashflow=np.array([0.2, 0.1]),
                ki_probability=0.1, expected_discounted_maturity_cashflow=0.7)
    base.update(kw)
    return AutocallableEventStats(**base)


def test_provenance_is_implementation_only():
    p = StatisticsProvenance("flat_vol_1d", "heston_adi_2d")
    assert p.surrogate and not StatisticsProvenance("flat_vol_1d", "flat_vol_1d").surrogate
    assert {f.name for f in dataclasses.fields(p)} == {"statistics_model", "price_model"}


def test_refusal_is_a_pricing_error():
    assert issubclass(StatisticsRefused, PricingError)


def test_new_fields_default_to_none():
    assert _stats().provenance is None
    s = KOResetEventStats(**dataclasses.asdict(_stats()))
    for name in ("pre_ko_payment_times", "post_ko_payment_times", "expected_discounted_pre_maturity_cashflow",
                 "expected_discounted_post_maturity_cashflow", "pre_maturity_probability", "post_maturity_probability",
                 "pre_maturity_time", "post_maturity_time", "pre_maturity_payment_time", "post_maturity_payment_time"):
        assert getattr(s, name) is None
    assert s.post_ko_times_are_offsets is False


def test_distribution_carries_provenance_and_defaults_determination():
    p = StatisticsProvenance("flat_vol_1d", "flat_vol_1d")
    d = EventDistribution.from_autocallable_stats(_stats(provenance=p))
    assert d.provenance == p
    assert d.determination_times == {}


def test_explicit_maturity_determination_relaxes_the_terminal_check():
    # a maturity outcome determined at 0.5 and paid at 0.6, before the last event time 1.0
    d = EventDistribution(
        event_times=np.array([0.5, 1.0]), event_dates=None,
        probabilities={EventType.KO: np.array([0.2, 0.1]), EventType.MATURITY_NO_KO: 0.3,
                       EventType.MATURITY_WITH_KI: 0.4},
        survival_probability=np.array([1.0, 0.8, 0.7]),
        payment_times={EventType.KO: np.array([0.5, 1.0]), EventType.MATURITY_NO_KO: 0.6,
                       EventType.MATURITY_WITH_KI: 1.0},
        determination_times={EventType.MATURITY_NO_KO: 0.5})
    assert d.determination_times[EventType.MATURITY_NO_KO] == 0.5
    with pytest.raises(Exception, match="precedes"):
        dataclasses.replace(d, determination_times={})   # today's rule: 0.6 < last event 1.0
