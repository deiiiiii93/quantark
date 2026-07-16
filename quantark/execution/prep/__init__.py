"""Shared preparation artifact states for engine-family adapters.

These helpers sit BETWEEN the execution kernel and the engine families: the
kernel never imports them (adapters do), and they import only shared model
infrastructure (``quantark.volmodels``), following the precedent of
``execution.cache.draws`` importing ``quantark.montecarlo``.
"""
from quantark.execution.prep.dupire import dupire_surface_state

__all__ = ["dupire_surface_state"]
