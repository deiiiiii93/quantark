"""Who prices an un-knocked REBASED KO-reset (spec 2026-10-08 KO-reset lifecycle, section 6a).

A REBASED second schedule holds offsets from the knock-in time. Only ``SnowballMCEngine`` places it after each
simulated knock-in; the PDE and QUAD routes raise and QUAD V2 refuses relative resets. A knocked-in contract is
priced by every engine through the tracker's ABSOLUTE copy, so only the un-knocked state is restricted. The
refusal is raised where a consumer is configured, never mid-run on the first un-knocked day.
"""
from __future__ import annotations

from typing import Any, Mapping

from quantark.util.enum.engine_enums import EngineType
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.exceptions import ValidationError

__all__ = ["is_unknocked_rebased", "require_rebased_provider"]


def is_unknocked_rebased(product: Any) -> bool:
    """A KO-reset with a REBASED schedule 2 that is not (carried) knocked in."""
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption

    return (
        isinstance(product, KnockOutResetSnowballOption)
        and product.post_ko_mode == PostKOScheduleMode.REBASED
        and not getattr(product, "_otc_lifecycle_knocked_in", False)
    )


def _is_mc(engine: Any) -> bool:
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine

    return engine == EngineType.MONTE_CARLO or isinstance(engine, SnowballMCEngine)


def require_rebased_provider(product: Any, *, engines: Mapping[str, Any], consumer: str, position: Any) -> None:
    """Refuse an un-knocked REBASED KO-reset unless every engine named in ``engines`` is Monte Carlo."""
    if not is_unknocked_rebased(product):
        return
    off = {role: engine for role, engine in engines.items() if not _is_mc(engine)}
    if off:
        names = ", ".join(
            f"{role}={getattr(engine, 'name', None) or type(engine).__name__ if not isinstance(engine, EngineType) else engine.name}"
            for role, engine in off.items()
        )
        raise ValidationError(
            f"{consumer}: position {position} is an un-knocked KO-reset snowball with a REBASED schedule 2, which "
            "only Monte Carlo prices (SnowballMCEngine places schedule 2 after each simulated knock-in); "
            f"got {names}"
        )
