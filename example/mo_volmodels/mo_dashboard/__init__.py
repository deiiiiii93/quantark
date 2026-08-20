"""Read-only progress dashboard for the snowball vol-model study.

Every module here reads; none writes anywhere under ``output/`` except the
single HTML file the CLI is told to produce, and none imports pricing code.
See docs/superpowers/specs/2026-08-03-snowball-progress-dashboard-design.md.
"""
from .payload import SCHEMA_VERSION, collect

__all__ = ["collect", "SCHEMA_VERSION"]
