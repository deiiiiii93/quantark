"""Reproducibility manifest skeleton (spec section 14.3).

Phase 0 stamps identity and dependency versions. Request and plan
fingerprints are populated from Phase 1 onward; ``None`` means
"fingerprint unavailable" (an uncacheable, legacy-adapted request).
"""
import platform as _platform
from dataclasses import dataclass

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ReproducibilityManifest",
    "build_versions",
    "platform_tag",
]

MANIFEST_SCHEMA_VERSION = "execution-manifest/0"


@dataclass(frozen=True)
class ReproducibilityManifest:
    schema_version: str
    request_fingerprint: str | None
    plan_fingerprint: str | None
    adapter_id: str
    adapter_version: str
    engine_class_path: str
    versions: tuple           # (("python", "3.12.1"), ...)
    platform: str
    resolved_policy: tuple    # (("batch.backend", "serial"), ...)
    preparation_fingerprint: str | None = None


def build_versions() -> tuple:
    """Stamp interpreter and numerical dependency versions."""
    import numpy
    import scipy

    import quantark

    return (
        ("python", _platform.python_version()),
        ("quantark", getattr(quantark, "__version__", "unknown")),
        ("numpy", numpy.__version__),
        ("scipy", scipy.__version__),
    )


def platform_tag() -> str:
    return f"{_platform.system()}-{_platform.machine()}"
