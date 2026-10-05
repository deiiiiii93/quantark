"""A prepared session's outcome, through the real dispatch (spec 3C §5.6)."""
from quantark.execution import PricingRequest, PricingSession
from quantark.execution.contracts import OutputKind


def session(engine, product, env, outputs, *, lifecycle_state=None, levels=None):
    options = () if levels is None else (("grid_spot_levels", tuple(float(x) for x in levels)),)
    with PricingSession() as s:
        res = s.execute(engine, PricingRequest(product=product, pricing_env=env, outputs=frozenset(outputs),
                                               lifecycle_state=lifecycle_state, operation_options=options))
    return res.value


PV, GRID = OutputKind.PV, OutputKind.GRID
