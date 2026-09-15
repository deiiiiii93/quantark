from datetime import datetime

import pytest

from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, flat_env


def test_timed_record_needs_matching_date_and_aware_tz():
    d = datetime(2026, 10, 15)
    ok = ObservationRecord(observation_date=d, observation_timestamp=datetime(2026, 10, 15, 14, 0, tzinfo=SHANGHAI),
                           barrier=103.0)
    ok.validate()
    with pytest.raises(ValidationError, match="timezone-aware"):
        ObservationRecord(observation_date=d, observation_timestamp=datetime(2026, 10, 15, 14, 0), barrier=103.0).validate()
    with pytest.raises(ValidationError, match="observation_date"):
        ObservationRecord(observation_timestamp=datetime(2026, 10, 15, 14, 0, tzinfo=SHANGHAI), barrier=103.0).validate()
    with pytest.raises(ValidationError, match="same local date"):
        ObservationRecord(observation_date=d, observation_timestamp=datetime(2026, 10, 16, 14, 0, tzinfo=SHANGHAI),
                          barrier=103.0).validate()
    with pytest.raises(ValidationError, match="settlement_timestamp"):
        ObservationRecord(observation_date=d, settlement_timestamp=datetime(2026, 10, 17, 15, 0), barrier=103.0).validate()


def test_legacy_resolution_ignores_the_timestamp():
    d = datetime(2026, 10, 15)
    rec = ObservationRecord(observation_date=d, observation_timestamp=datetime(2026, 10, 15, 14, 0, tzinfo=SHANGHAI),
                            barrier=103.0)
    env = flat_env(datetime(2026, 9, 15))          # naive legacy env
    env.spot_quote.timestamp = None
    assert rec.resolve_time(env) == (d - datetime(2026, 9, 15)).days / 365.0
    resolved = ObservationSchedule(records=[rec]).resolve(pricing_env=env, default_barrier=103.0, require_single=True)
    assert resolved[0].observation_time == rec.resolve_time(env)
