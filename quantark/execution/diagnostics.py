"""Operational diagnostics records and sinks (spec section 16).

Diagnostics are immutable after outcome construction and never participate in
economic equality or plan fingerprints.
"""
from dataclasses import dataclass

__all__ = ["RunDiagnostics", "InMemoryDiagnosticsSink"]


@dataclass(frozen=True)
class RunDiagnostics:
    """Immutable per-run operational record."""

    adapter_id: str
    timings: tuple = ()          # (("execute_seconds", 0.12), ...)
    policy_sources: tuple = ()   # (("batch.backend", "default"), ...)
    records: tuple = ()          # free-form operational note strings


class InMemoryDiagnosticsSink:
    """Default library sink: appends records to a list (spec section 16)."""

    def __init__(self):
        self.entries: list = []

    def emit(self, diagnostics: RunDiagnostics) -> None:
        self.entries.append(diagnostics)
