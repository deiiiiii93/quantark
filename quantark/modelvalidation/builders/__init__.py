"""Builtin builders.

Importing this package registers every builder the shipped studies need. The
registry grows one engine family at a time: a new certification adds its own
module here, and nothing is registered speculatively.
"""

from __future__ import annotations

from quantark.modelvalidation.builders import european_selftest  # noqa: F401

__all__: list[str] = ["european_selftest"]
