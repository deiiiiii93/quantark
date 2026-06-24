"""Guard against silently collapsing a smile surface to a single strike vol.

Constant-vol numerical engines (flat-vol MC / PDE) read a single
``sigma = env.get_vol(K, T)`` and diffuse with it. That is correct for a flat
or ATM term-structure surface, but a SABR / Vanna-Volga smile carries skew and
forward-smile information that such an engine throws away. This helper makes the
loss explicit instead of silent.
"""

from __future__ import annotations

import warnings

from quantark.util.exceptions import PricingError


class SmileCollapseWarning(UserWarning):
    """A smile surface was collapsed to one strike vol by a constant-vol engine."""


def guard_constant_vol(surface, engine_name: str, *, strict: bool = False) -> None:
    """Warn (or raise in strict mode) when a smile surface meets a flat-vol engine.

    Args:
        surface: The volatility surface the engine is about to collapse.
        engine_name: Engine class name, for the message.
        strict: When True, raise PricingError instead of warning.

    Raises:
        PricingError: If ``strict`` and the surface is a smile.
    """
    if not getattr(surface, "is_smile", False):
        return
    msg = (
        f"{engine_name} is a numerical Black/GK engine using a single "
        f"strike-selected constant vol; it collapses the smile surface "
        f"{type(surface).__name__} and ignores skew / forward-smile dynamics. "
        f"For smile-consistent path pricing use a Local-Vol / SLV / Heston / "
        f"SABR Monte-Carlo engine."
    )
    if strict:
        raise PricingError(msg)
    warnings.warn(msg, SmileCollapseWarning, stacklevel=3)
