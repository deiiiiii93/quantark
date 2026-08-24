"""Forward-density event-stats mode (spec 2026-08-24). Battery grows task by task."""
import pytest

from quantark.asset.equity.param import QuadParams
from quantark.util.exceptions import ValidationError


def test_event_stats_mode_default_is_stacked():
    assert QuadParams().event_stats_mode == "stacked"


def test_event_stats_mode_accepts_forward_density():
    assert QuadParams(event_stats_mode="forward_density").event_stats_mode == "forward_density"


def test_event_stats_mode_rejects_unknown():
    with pytest.raises(ValidationError):
        QuadParams(event_stats_mode="fwd")
