"""European vanilla (``european_pde_solver.py``): the facade's checks, lifecycle and expiry outcomes."""
from __future__ import annotations

from quantark.util.exceptions import PricingError

from ..prepare import Operation, TerminalOutcome
from .base import FamilyAdapter

__all__ = ["EuropeanAdapter"]

#: The operations the solver's facades define; the session seam follows ``price`` (spec 3C decision 29).
_FACADE = (Operation.PRICE, Operation.GREEKS, Operation.SESSION_OUTPUTS)


class EuropeanAdapter(FamilyAdapter):
    """``price`` / ``calculate_greeks`` moved verbatim."""

    def validate(self, product, env, request, lifecycle_state) -> None:
        if request.operation not in _FACADE:
            return
        from quantark.asset.equity.engine.settlement_support import validate_settlement_capability
        from quantark.asset.equity.product.option import EuropeanVanillaOption

        if request.operation is Operation.PRICE and not isinstance(product, EuropeanVanillaOption):
            raise PricingError(
                f"EuropeanPDESolver only supports EuropeanVanillaOption, "
                f"got {type(product).__name__}"
            )
        validate_settlement_capability(self.solver, product, lifecycle_state)

    def terminal_outcome(self, product, env, request, lifecycle_state):
        if request.operation not in _FACADE:
            return super().terminal_outcome(product, env, request, lifecycle_state)
        from quantark.asset.equity.engine.settlement_support import (pending_receivable_pv,
                                                                     resolve_terminal_timing,
                                                                     terminal_lifecycle_pv)
        solver = self.solver
        greeks = request.operation is Operation.GREEKS
        session = request.operation is Operation.SESSION_OUTPUTS       # as PRICE, with no grid (spec 3C §5.6)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, env)
        if fixed_pv is not None:
            if session:
                return TerminalOutcome("lifecycle", self._session_outputs(fixed_pv, None))
            return TerminalOutcome("lifecycle",
                                   {"price": fixed_pv, "delta": 0.0, "gamma": 0.0} if greeks else fixed_pv)
        timing = resolve_terminal_timing(product, env)
        if product.get_maturity(env) > 0.0:
            return None
        if greeks:
            out = {
                "price": solver._calculate_intrinsic(product, env.spot) * timing.delay_df,
                "delta": solver._intrinsic_delta(product, env.spot) * timing.delay_df,
                "gamma": 0.0,
            }
            out["price"] += pending_receivable_pv(lifecycle_state, env)
            return TerminalOutcome("expired", out)
        contingent = solver._calculate_intrinsic(product, env.spot) * timing.delay_df
        if session:
            return TerminalOutcome("expired", self._session_outputs(
                contingent + pending_receivable_pv(lifecycle_state, env), None))
        return TerminalOutcome("expired", contingent + pending_receivable_pv(lifecycle_state, env))

    def read(self, prepared, result):
        out = super().read(prepared, result)
        if prepared.request.operation not in _FACADE:
            return out
        from quantark.asset.equity.engine.settlement_support import pending_receivable_pv

        # where price / calculate_greeks added it after the base result
        pending = pending_receivable_pv(prepared.lifecycle_state, prepared.env)
        if prepared.request.operation is Operation.SESSION_OUTPUTS:
            if not pending:                                       # stateless: the npv keeps its bits
                return out
            solution = out.solution if out.solution is None else out.solution._replace(cash=pending)
            return self._session_outputs(out.npv + pending, solution)
        if prepared.request.operation is Operation.GREEKS:
            out["price"] += pending
            return out
        return out + pending
