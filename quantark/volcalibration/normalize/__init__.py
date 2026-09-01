"""Quote normalizers: the one place a quoting convention is interpreted.

Each normalizer consumes a :class:`~quantark.volcalibration.snapshot.QuoteSnapshot`
and produces a :class:`~quantark.volcalibration.quotes.QuoteSet`.
"""

from typing import Protocol

from quantark.volcalibration.quotes import QuoteSet
from quantark.volcalibration.snapshot import QuoteSnapshot


class QuoteNormalizer(Protocol):
    """Turn one convention's quotes into the convention-neutral QuoteSet."""

    convention: str

    def normalize(self, snapshot: QuoteSnapshot) -> QuoteSet:
        ...


__all__ = ["QuoteNormalizer"]
