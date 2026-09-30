"""Component stand-ins for the slot tests (spec 2 §4.1, §5.8)."""
from dataclasses import dataclass

import numpy as np

from quantark.asset.equity.engine.pde.core.components import (
    CentralDifference,
    CrankNicolson,
    StandardReadout,
)


@dataclass(frozen=True)
class BackwardEuler:
    """Fully implicit: θ=1 everywhere, through the built-in arithmetic."""

    def schedule(self, layout):
        return np.ones(layout.time.actual_steps)

    def step(self, state, s, dt, theta, boundaries):
        return CrankNicolson().step(state, s, dt, 1.0, boundaries)

    def evidence(self):
        return {"kind": "backward_euler"}


@dataclass(frozen=True)
class ShiftedOperator(CentralDifference):
    """The built-in stencil with ``shift`` added to every diagonal: a custom
    operator whose ``evidence()`` is empty but whose value differs."""

    shift: float = 0.0

    def assemble(self, samples, layout):
        sc = super().assemble(samples, layout)
        return sc._replace(lcu_sets=[(l, c + self.shift, u) for (l, c, u) in sc.lcu_sets])


@dataclass(frozen=True)
class NearestNodeReadout:
    def price(self, column, spot, layout):
        return float(column[int(np.argmin(np.abs(layout.spatial.s - spot)))])

    def delta_gamma(self, column, spot, layout):
        return StandardReadout().delta_gamma(column, spot, layout)

    def evidence(self):
        return {}


@dataclass(frozen=True)
class NaNStepper(BackwardEuler):
    def step(self, state, s, dt, theta, boundaries):
        out = super().step(state, s, dt, theta, boundaries)
        out[0] = np.nan
        return out


@dataclass(frozen=True)
class WrongShapeStepper(BackwardEuler):
    def step(self, state, s, dt, theta, boundaries):
        return super().step(state, s, dt, theta, boundaries)[:-1]


@dataclass(frozen=True)
class InPlaceStepper(BackwardEuler):
    """Writes into the state it was handed: the march owns it (§5.8)."""

    def step(self, state, s, dt, theta, boundaries):
        state.v_next[0] = 0.0
        return super().step(state, s, dt, theta, boundaries)


class PlainStepper:          # not a dataclass: no value fingerprint
    schedule = BackwardEuler.schedule
    step = BackwardEuler.step
    evidence = BackwardEuler.evidence


ShiftedOperatorStub = ShiftedOperator
