"""Batch execution backends (spec section 12). All backends consume the same
immutable BatchPlan and yield outcomes in canonical batch-index order; they
own scheduling only, never numerical meaning."""
from quantark.execution.backends import serial, threads

__all__ = ["serial", "threads"]
