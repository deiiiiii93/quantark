"""Builtin builders.

Importing this package registers every builder the shipped studies need. The
registry grows one engine family at a time: a new certification adds its own
module here, and nothing is registered speculatively.
"""

from __future__ import annotations

from quantark.modelvalidation.builders import equity_ko_reset  # noqa: F401
from quantark.modelvalidation.builders import equity_phoenix  # noqa: F401
from quantark.modelvalidation.builders import equity_snowball  # noqa: F401
from quantark.modelvalidation.builders import european_selftest  # noqa: F401

__all__: list[str] = [
    "equity_ko_reset",
    "equity_phoenix",
    "equity_snowball",
    "european_selftest",
]
