"""A quadrature candidate's declared readout must reach the engine.

``params()`` spreads the study's declared settings into the recorded
configuration, so any key a YAML names moves the identity hash. If the
candidate then builds its engine from defaults, the certificate claims
to cover a mode it never measured. That is worse than not supporting
the key at all, so this pins the thread from YAML to engine.
"""
from __future__ import annotations

import pytest

from quantark.modelvalidation.builders.equity_ko_reset import KOResetQuadCandidate
from quantark.modelvalidation.builders.equity_phoenix import PhoenixQuadCandidate
from quantark.modelvalidation.builders.equity_snowball import SnowballQuadCandidate
from quantark.util.exceptions import ValidationError

CANDIDATES = (SnowballQuadCandidate, PhoenixQuadCandidate, KOResetQuadCandidate)
#: Only the snowball engine implements the transition readout. The other two
#: refuse it, so declaring it for them must fail when the study loads rather
#: than as an ERROR cell after the references have been paid for.
SUPPORTS_TRANSITION = (SnowballQuadCandidate,)


#: The arms take (environment, product, quantities, params); only the last
#: matters here, and nothing in these tests prices anything.
def _make(cls, **params):
    return cls({}, {}, ("pv",), dict(params))


@pytest.mark.parametrize("cls", SUPPORTS_TRANSITION)
def test_the_declared_readout_is_recorded_in_the_identity(cls):
    default = _make(cls).params()["grid"]
    transition = _make(cls, readout="transition").params()["grid"]
    assert default["readout"] == "legacy_linear"
    assert transition["readout"] == "transition"
    assert default != transition, "two readouts must not share one identity"


@pytest.mark.parametrize("cls", SUPPORTS_TRANSITION)
def test_the_declared_readout_reaches_the_engine(cls):
    """The engine the candidate prices with must carry the declared mode."""
    assert _make(cls, readout="transition")._engine_params(1001).readout == "transition"


@pytest.mark.parametrize("cls", CANDIDATES)
def test_the_default_readout_reaches_the_engine(cls):
    assert _make(cls)._engine_params(1001).readout == "legacy_linear"


@pytest.mark.parametrize("cls", [c for c in CANDIDATES if c not in SUPPORTS_TRANSITION])
def test_an_engine_that_cannot_do_a_readout_refuses_it_at_load_time(cls):
    with pytest.raises(ValidationError, match="does not support readout"):
        _make(cls, readout="transition").params()


@pytest.mark.parametrize("cls", CANDIDATES)
def test_an_unknown_readout_is_refused_when_the_candidate_is_built(cls):
    with pytest.raises(ValidationError):
        _make(cls, readout="nope").params()


NAMES = {
    SnowballQuadCandidate: "equity.snowball.quad",
    PhoenixQuadCandidate: "equity.phoenix.quad",
    KOResetQuadCandidate: "equity.ko_reset_snowball.quad",
}


@pytest.mark.parametrize("cls", SUPPORTS_TRANSITION)
def test_a_non_default_readout_is_a_differently_named_candidate(cls):
    """One study may certify both readouts, so they cannot share a name.

    A decision is recorded per candidate name, so two candidates named
    alike would overwrite one another. The default keeps the name it has
    always had, so banked certificates still match.
    """
    base = NAMES[cls]
    assert _make(cls).name() == base
    assert _make(cls, readout="legacy_linear").name() == base
    assert _make(cls, readout="transition").name() == f"{base}.transition"
