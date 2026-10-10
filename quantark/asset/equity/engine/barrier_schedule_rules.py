"""Contract rules on a barrier's discrete observation schedule shared by every pricing route (MC, PDE, QUAD).

Neutral module: no engine of any kind is imported here, so the PDE core, the MC engines and the QUAD adapters read
one rule and raise one message.
"""
from __future__ import annotations

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close


def refuse_knock_in_record_payoff(product) -> None:
    """A knock-in has one rebate (spec 3C §5.2 (a)): it is paid when NO observation hits, so no record's payoff can
    apply. A schedule record paying anything other than ``product.rebate`` is refused; a record without a payoff, or
    one paying the rebate, is the same contract. Knock-outs (a record's payoff is the hit cash) are not concerned.

    First the 1D PDE core's rule (``_KnockInAdapter.validate``); every route reads it since W18b (user ruling
    2026-10-10, option (a)).
    """
    if not getattr(product, "is_knock_in", False):
        return
    schedule = getattr(product, "observation_schedule", None)
    if schedule is None:
        return
    for rec in schedule.records:
        if rec.payoff is not None and not is_close(rec.payoff, product.rebate):
            raise ValidationError(
                f"a knock-in has one rebate ({product.rebate}); a schedule record pays {rec.payoff} "
                "(spec 3C §5.2 (a))")
