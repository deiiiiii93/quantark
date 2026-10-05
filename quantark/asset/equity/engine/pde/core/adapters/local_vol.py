"""The local-vol vanilla, barrier and DCN families on the core (spec 3C §5.1, §5.3): the parent family's events,
rows and readout; one local-vol row per step (LocalVolAdapterMixin); the banded backend (decision 12)."""
from __future__ import annotations

from ..groups import VALUE, Backend, ColumnGroup
from .barrier import BarrierAdapter
from .dcn import DCNAdapter
from .european import EuropeanAdapter
from .snowball import LocalVolAdapterMixin

__all__ = ["LocalVolEuropeanAdapter", "LocalVolBarrierAdapter", "LocalVolDCNAdapter"]


class _LocalVolCoreAdapter(LocalVolAdapterMixin):
    def market_checks(self, product, env, tau) -> None:
        """The local vol is the model: there is no constant-vol collapse to guard (spec 3C §5.3)."""

    def groups(self, product, layout, request) -> tuple:
        # one coefficient set per step defeats the factorization cache: banded (decision 12)
        return (ColumnGroup("value", ("value",), Backend.BANDED, VALUE),)


class LocalVolEuropeanAdapter(_LocalVolCoreAdapter, EuropeanAdapter):
    """``LocalVolPDESolver``."""


class LocalVolBarrierAdapter(_LocalVolCoreAdapter, BarrierAdapter):
    """``LocalVolBarrierPDESolver``."""


class LocalVolDCNAdapter(LocalVolAdapterMixin, DCNAdapter):
    """``LocalVolDCNPDEEngine``: two surfaces, banded (``DCNAdapter.groups``)."""
