from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import DigitalOptionAnalyticalEngine
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.execution.errors import CapabilityError
from quantark.intraday.capability import find_capability, render_capability_matrix, require_capability
from intraday.conftest import dated_snowball, digital


def test_plan1_routes_are_declared(sse_calendar):
    snow = dated_snowball(sse_calendar, datetime(2026, 3, 16))
    cap = require_capability(snow, SnowballQuadEngineV2(), monitoring="discrete")
    assert cap.status == "supported" and "price" in cap.outputs
    assert find_capability(digital(datetime(2026, 12, 15)), DigitalOptionAnalyticalEngine(), monitoring="terminal") is not None


def test_unsupported_engine_names_limitation_and_alternatives(sse_calendar):
    snow = dated_snowball(sse_calendar, datetime(2026, 3, 16))
    with pytest.raises(CapabilityError) as ei:
        require_capability(snow, SnowballQuadEngine(), monitoring="discrete")      # legacy QUAD V1: no intraday route
    msg = str(ei.value)
    assert "SnowballQuadEngine has no intraday route" in msg and "SnowballQuadEngineV2" in msg and "intraday inventory" in msg


def test_requesting_an_output_the_route_lacks_is_a_capability_error(sse_calendar):
    snow = dated_snowball(sse_calendar, datetime(2026, 3, 16))
    with pytest.raises(CapabilityError, match="vanna"):
        require_capability(snow, SnowballQuadEngineV2(), monitoring="discrete", outputs=("price", "vanna"))


def test_subclasses_do_not_inherit_a_route(sse_calendar):
    class TunedQuad(SnowballQuadEngineV2):
        pass
    with pytest.raises(CapabilityError):
        require_capability(dated_snowball(sse_calendar, datetime(2026, 3, 16)), TunedQuad(), monitoring="discrete")


def test_matrix_renders_every_row():
    md = render_capability_matrix()
    assert "| Product | Engine | Monitoring | Profiles | Outputs | Status | Note |" in md and "SnowballQuadEngineV2" in md
