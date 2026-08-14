"""Engine-release certification: stochastic benchmarks certify deterministic engines.

This package standardizes how complex pricing engines (PDE, quadrature) are
released: a statistically-controlled stochastic benchmark certifies the
deterministic engine over a set of scenarios, the run banks a schema-versioned
evidence package, and cheap deterministic anchors extracted from that package
guard the certified behavior in CI afterwards.

See ``docs/modelvalidation/RELEASE_PROCEDURE.md`` for the release procedure.
"""

from quantark.modelvalidation.pipeline import Certificate, certify, validate_payload
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import (
    QUANTITIES,
    CaseSpec,
    CertificationStudy,
    EconomicScale,
    GateBounds,
    HedgeContractScale,
    SamplingPolicy,
)

__all__ = [
    "QUANTITIES",
    "CaseSpec",
    "Certificate",
    "CertificationStudy",
    "EconomicScale",
    "GateBounds",
    "HedgeContractScale",
    "SamplingPolicy",
    "certify",
    "register_builder",
    "validate_payload",
]
