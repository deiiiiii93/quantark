"""Local-vol transition QUAD: backward recursion over event dates on a barrier-aligned grid.

Values are carried per lifecycle channel (snowball: 0 = never knocked in,
1 = knocked in) as point values at the cell centres, per unit notional S0.
Between consecutive event dates a kernel applies one interval transition; at an
event date the KI switch and the KO payment act on whole cells (the barriers are
cell edges). Readout is the s0 cell; native delta/gamma use the uniform stencil
around it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from quantark.util.numerical import Tolerance, is_close

from .contract import Contract
from .grid import GridConfig, LogGrid, UnresolvedGrid, build_grid
from .kernels import interior_rows
from .market import Market


def event_times(c: Contract) -> list[float]:
    return sorted(set(c.ko_times) | set(c.ki_times) | {c.maturity})


def terminal_values(c: Contract, grid: LogGrid) -> np.ndarray:
    """Maturity payoff per channel at the cell centres, BEFORE maturity events."""
    s = np.exp(grid.centres)
    if c.kind == "vanilla":
        payoff = np.maximum(s - c.strike, 0.0) if c.option == "call" else np.maximum(c.strike - s, 0.0)
        return (payoff / c.s0)[None, :]
    if c.kind == "up_out":
        return (np.maximum(s - c.strike, 0.0) / c.s0)[None, :]
    never = np.full(grid.size, 1.0 + c.rebate_rate)
    knocked = 1.0 + np.minimum(s - c.strike, 0.0) / c.s0
    return np.stack((never, knocked))


def apply_events(values: np.ndarray, c: Contract, grid: LogGrid, t: float) -> np.ndarray:
    """KI switch then KO payment at observation time ``t`` (KO wins; regions are disjoint)."""
    out = values.copy()
    if c.kind == "snowball" and t in c.ki_times:
        below = grid.below(c.ki)
        out[0, below] = out[1, below]
    if t in c.ko_times:
        above = grid.above(c.ko)
        out[:, above] = c.ko_payment(t)
    return out


def readout(values: np.ndarray, grid: LogGrid) -> dict:
    """PV, delta = dPV/dS*S0 and gamma = d2PV/dS2*S0**2 from the uniform centred stencil."""
    i = grid.spot_index
    h = grid.widths[i]
    v = values[0]
    vx = (v[i + 1] - v[i - 1]) / (2 * h)
    vxx = (v[i + 1] - 2 * v[i] + v[i - 1]) / h ** 2
    return {"pv": float(v[i]), "delta": float(vx), "gamma": float(vxx - vx)}


def _require_anchor(c: Contract, market: Market) -> None:
    if not is_close(market.spot, c.s0):
        raise ValueError("the grid is anchored on the contract's s0; price at market.spot == s0")


def reference_vol(market: Market, c: Contract) -> float:
    """Root time-averaged local variance at s0 over the contract life (grid width scale)."""
    return math.sqrt(float(market.surface.time_avg_var(c.s0, 0.0, c.maturity)))


PV_BUDGET = 1e-4          # per unit notional (spec Contracts, Accuracy)
IDENTITY_FRACTION = 1e-2  # per-interval identities sum to <= 1% of the PV budget over the life (D16, ruled)
THETA_DT = 1.0 / 365.0
OUTPUTS = ("pv", "delta", "gamma", "vega", "theta", "rho", "carry")


def call_from_put(put_pv: float, c: Contract, market: Market) -> float:
    """Put-call parity per unit notional: C = P + e^{-qT} - (K/S0) e^{-rT}."""
    return put_pv + math.exp(-market.q * c.maturity) - c.strike / c.s0 * math.exp(-market.r * c.maturity)


@dataclass
class LocalVolQuad:
    """Transition QUAD. In BOTH modes every interval carries the probes [1, S/S0] through the
    same kernel application and checks mass and forward on the interior rows against
    ``IDENTITY_FRACTION * PV_BUDGET / intervals`` (``UnresolvedGrid`` otherwise). ``validate``
    additionally runs the kernel's own per-operator checks (CTMC semigroup) — untimed use only."""

    kernel: object
    grid_config: GridConfig = field(default_factory=GridConfig)
    validate: bool = False

    def grid(self, c: Contract, market: Market) -> LogGrid:
        return build_grid(c, reference_vol(market, c), self.grid_config)

    def _check_probes(self, moved: np.ndarray, grid: LogGrid, market: Market, h: float, tolerance: float,
                      record: dict) -> None:
        sd = math.sqrt(float(np.max(market.surface.lv_grid)) ** 2 * h)
        rows = interior_rows(grid, max(sd, Tolerance.ZERO))
        k = self.grid_config.readout_cells
        block = slice(grid.spot_index - k, grid.spot_index + k + 1)
        if not rows[block].all():
            # a long interval: 8 sd exceed the domain; check at least the readout block and
            # every row as far from both edges as it (horizon recorded)
            depth = np.minimum(grid.centres - grid.edges[0], grid.edges[-1] - grid.centres)
            rows = depth >= float(np.min(depth[block]))
            record["horizon_sd"] = min(record.get("horizon_sd", math.inf), float(np.min(depth[block])) / sd)
        s = np.exp(grid.centres[rows] - grid.centres[grid.spot_index])
        mass = float(np.max(np.abs(moved[0, rows] * math.exp(market.r * h) - 1.0)))
        forward = float(np.max(np.abs(moved[1, rows] * math.exp(market.q * h) / s - 1.0)))
        record["mass"] = max(record["mass"], mass)
        record["forward"] = max(record["forward"], forward)
        if max(mass, forward) > tolerance:
            raise UnresolvedGrid(f"interval identity residual (mass {mass:.3g}, forward {forward:.3g}) exceeds "
                                 f"{tolerance:.3g} over {h:.6f}y; refine the grid")

    def rollback(self, c: Contract, market: Market, grid: LogGrid | None = None,
                 diagnostics: dict | None = None) -> tuple[np.ndarray, LogGrid]:
        grid = self.grid(c, market) if grid is None else grid
        times = event_times(c)
        tolerance = IDENTITY_FRACTION * PV_BUDGET / len(times)
        record = {"mass": 0.0, "forward": 0.0, "intervals": len(times), "tolerance": tolerance}
        probes = np.vstack((np.ones(grid.size), np.exp(grid.centres - grid.centres[grid.spot_index])))
        channels = c.channels
        saved = getattr(self.kernel, "validate", None), getattr(self.kernel, "tolerance", None)
        if self.validate:
            self.kernel.validate, self.kernel.tolerance = True, tolerance
        try:
            values = apply_events(terminal_values(c, grid), c, grid, times[-1])
            for right, left in zip(times[::-1], (times[-2::-1] + [0.0])):
                moved = self.kernel.propagate(np.vstack((values, probes)), grid, market, left, right)
                values = moved[:channels]
                self._check_probes(moved[channels:], grid, market, right - left, tolerance, record)
                if left > 0.0:
                    values = apply_events(values, c, grid, left)
        finally:
            if saved[0] is not None:
                self.kernel.validate, self.kernel.tolerance = saved
        if diagnostics is not None:
            diagnostics.update(record)
        return values, grid

    def price(self, c: Contract, market: Market) -> dict:
        _require_anchor(c, market)
        start = perf_counter()
        diagnostics: dict = {}
        values, grid = self.rollback(c, market, diagnostics=diagnostics)
        out = readout(values, grid)
        return dict(values=out, seconds=perf_counter() - start, cells=grid.size, diagnostics=diagnostics)

    def full_risk(self, c: Contract, market: Market) -> dict:
        """Eight valuations on ONE grid (spec D9): base (PV, native delta/gamma), vega+-, theta, rho+-, carry+-.

        Units: PV per unit notional; delta = dPV/dS*S0; gamma = d2PV/dS2*S0**2; vega per +0.01
        LV; theta per calendar day (production semantics, D8); rho and carry per +1bp.
        """
        _require_anchor(c, market)
        timings: dict = {}

        def timed(name, contract, m):
            start = perf_counter()
            values, _ = self.rollback(contract, m, grid)
            timings[name] = perf_counter() - start
            return values

        start = perf_counter()
        grid = self.grid(c, market)
        timings["grid"] = perf_counter() - start
        out = readout(timed("base", c, market), grid)
        i = grid.spot_index
        for kind in ("vega", "rho", "carry"):
            up = timed(f"{kind}+", c, market.bumped(kind, 1))[0, i]
            down = timed(f"{kind}-", c, market.bumped(kind, -1))[0, i]
            out[kind] = float((up - down) / 2)
        out["theta"] = float(timed("theta", c.shifted(THETA_DT), market)[0, i] - out["pv"])
        return dict(values={k: out[k] for k in OUTPUTS}, timings=timings, cells=grid.size)
