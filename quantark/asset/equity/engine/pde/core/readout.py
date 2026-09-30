"""Reading a march (spec 2 §5.3): the adapter maps the final columns to the operation's outputs."""
from __future__ import annotations

__all__ = ["read"]


def read(prepared, result):
    return prepared.adapter.read(prepared, result)
