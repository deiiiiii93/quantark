from datetime import date

import pytest

from quantark.intraday.profile import SegmentKind, VarianceProfile
from quantark.util.exceptions import ValidationError


def test_weights_must_normalize_to_one():
    with pytest.raises(ValidationError, match="sum to 1"):
        VarianceProfile("bad", "1", 244, overnight_weight=0.5, session_weights=(0.3, 0.3), break_weights=(0.0,))
    with pytest.raises(ValidationError, match="non-negative"):
        VarianceProfile("bad", "1", 244, overnight_weight=-0.1, session_weights=(0.6, 0.5), break_weights=(0.0,))
    p = VarianceProfile("desk", "1", 244, overnight_weight=0.25, session_weights=(0.35, 0.35), break_weights=(0.05,))
    assert p.weights_for(2) == ((SegmentKind.OVERNIGHT, 0.25), (SegmentKind.SESSION, 0.35),
                                (SegmentKind.BREAK, 0.05), (SegmentKind.SESSION, 0.35))
    with pytest.raises(ValidationError, match="sessions"):
        p.weights_for(3)


def test_uniform_profile_is_constant_calendar_rate(sse_sessions):
    p = VarianceProfile.uniform(sse_sessions, 244, reference_date=date(2026, 9, 15))
    # Mon close 15:00 -> Tue open 09:30 = 18.5h; sessions 2h + 2h; lunch 1.5h; total 24h
    assert p.overnight_weight == pytest.approx(18.5 / 24)
    assert p.session_weights == pytest.approx((2.0 / 24, 2.0 / 24))
    assert p.break_weights == pytest.approx((1.5 / 24,))
    assert sum((p.overnight_weight, *p.session_weights, *p.break_weights)) == pytest.approx(1.0, abs=1e-12)


def test_sessions_only_profile_has_exact_zero_overnight_and_break(sse_sessions):
    p = VarianceProfile.sessions_only(sse_sessions, 244)
    assert p.overnight_weight == 0.0 and p.break_weights == (0.0,)
    assert p.session_weights == (0.5, 0.5)


def test_identity_includes_version_and_weights():
    a = VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))
    b = VarianceProfile("desk", "2", 244, 0.25, (0.35, 0.35), (0.05,))
    assert a.identity() != b.identity() and hash(a.identity())
