"""Per-path barrier state machine (spec §3.2).

Propagates discrete contractual state (alive / knocked_in) along each simulated
path. Between-node barrier transitions are pathwise-sampled with a Brownian
bridge (common random numbers), not endpoint-only — removing the systematic
one-directional KI under-count. v1 supports continuous KI via the bridge;
continuous KO (needs a crossing time for settlement) is deferred and raises.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from quantark.util.exceptions import ValidationError


@dataclass
class BarrierStateMachine:
    ki_barrier: Optional[float] = None
    ki_direction: str = "down"
    ko_barrier: Optional[float] = None
    ko_direction: str = "up"
    monitoring_idx: List[int] = field(default_factory=list)
    ki_monitoring_idx: Optional[List[int]] = None
    ko_monitoring_idx: Optional[List[int]] = None
    # KO-reset: a knocked-in path follows a DIFFERENT (post-KI) KO barrier/schedule.
    # When post_ko_barrier is None the trade is a plain snowball and the single KO
    # (ko_barrier/ko_monitoring_idx) applies to every alive path regardless of KI.
    # When set, the pre-KI barrier applies only to not-yet-KI paths and the post-KI
    # barrier only to knocked-in paths (mirrors the engine's v_out/v_in KO split).
    post_ko_barrier: Optional[float] = None
    post_ko_monitoring_idx: Optional[List[int]] = None
    times: object = None
    seed: int = 999
    continuous: bool = False
    continuous_ko: bool = False
    vol: float = 0.0
    initial_knocked_in: bool = False   # seasoned: knocked in before valuation

    def __post_init__(self) -> None:
        if self.ki_direction not in ("up", "down"):
            raise ValidationError("ki_direction must be 'up'/'down'")
        if self.ko_direction not in ("up", "down"):
            raise ValidationError("ko_direction must be 'up'/'down'")
        if self.continuous_ko:
            raise ValidationError("continuous KO is deferred in v1 (needs crossing time)")
        for name, b in (("ki_barrier", self.ki_barrier), ("ko_barrier", self.ko_barrier),
                        ("post_ko_barrier", self.post_ko_barrier)):
            if b is not None and not (np.isfinite(b) and b > 0):
                raise ValidationError(f"{name} must be a positive finite level")
        if self.post_ko_barrier is not None and self.ki_barrier is None:
            raise ValidationError(
                "post_ko_barrier (KO reset) requires a ki_barrier: the reset is "
                "triggered by knock-in")
        if not (np.isfinite(self.vol) and self.vol >= 0):
            raise ValidationError("vol must be non-negative and finite")
        if self.continuous and self.vol <= 0:
            raise ValidationError("continuous monitoring requires vol > 0 (bridge variance)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValidationError("seed must be an int (deterministic CRN)")
        if not isinstance(self.initial_knocked_in, bool):
            raise ValidationError("initial_knocked_in must be a bool")
        if self.times is None:
            raise ValidationError("times is required")
        t = np.asarray(self.times, dtype=float)
        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValidationError("times must be a finite, strictly-increasing 1-D array")
        self.times = t
        # KI and KO may follow different schedules (e.g. snowball: continuous daily
        # KI vs discrete monthly KO); each falls back to the shared monitoring_idx.
        self._ki_idx = (self.ki_monitoring_idx if self.ki_monitoring_idx is not None
                        else self.monitoring_idx)
        self._ko_idx = (self.ko_monitoring_idx if self.ko_monitoring_idx is not None
                        else self.monitoring_idx)
        self._post_ko_idx = (self.post_ko_monitoring_idx
                             if self.post_ko_monitoring_idx is not None else [])
        # a set barrier with an empty schedule would be silently never monitored
        if self.ki_barrier is not None and len(self._ki_idx) == 0:
            raise ValidationError("ki_barrier set but its monitoring schedule is empty")
        if self.ko_barrier is not None and len(self._ko_idx) == 0:
            raise ValidationError("ko_barrier set but its monitoring schedule is empty")
        if self.post_ko_barrier is not None and len(self._post_ko_idx) == 0:
            raise ValidationError(
                "post_ko_barrier set but its monitoring schedule is empty")

    def run(self, spots: np.ndarray) -> dict:
        spots = np.asarray(spots, dtype=float)
        if spots.ndim != 2:
            raise ValidationError("spots must be 2-D (num_paths, n_times)")
        if not np.all(np.isfinite(spots)):  # NaN would defeat the <= 0 / barrier checks
            raise ValidationError("spots must be finite")
        if np.any(spots <= 0):
            raise ValidationError("spots must be positive")
        n_paths, n_t = spots.shape
        if self.times.shape != (n_t,):
            raise ValidationError("times length must match spots' time axis")
        for nm, idx in (("ki", self._ki_idx), ("ko", self._ko_idx),
                        ("post_ko", self._post_ko_idx)):
            for j in idx:
                if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
                    raise ValidationError(f"{nm}_monitoring_idx entries must be integers")
                if j < 0 or j >= n_t:
                    raise ValidationError(f"{nm}_monitoring_idx out of range")
        ki_set = set(self._ki_idx)
        # Continuous KI samples the bridge on every interval of its window; a gapped
        # schedule would silently skip intervals and undercount first passages.
        if self.continuous and self.ki_barrier is not None:
            mi = sorted(ki_set)
            if mi and mi != list(range(mi[0], mi[-1] + 1)):
                raise ValidationError(
                    "continuous KI requires a contiguous monitoring schedule (no gaps)")
        knocked_in = np.zeros((n_paths, n_t), dtype=bool)
        alive = np.ones((n_paths, n_t), dtype=bool)
        ko_idx = np.full(n_paths, -1, dtype=int)
        rng = np.random.default_rng(self.seed)
        # seasoned trades may already be knocked in at valuation (the QUAD engine
        # prices such a trade from v_in); seed the KI history so t0 and every later
        # node select v_in even if the spot has since recovered above the barrier.
        ki = np.full(n_paths, self.initial_knocked_in, dtype=bool)
        dead = np.zeros(n_paths, dtype=bool)
        # ko_post[p] records whether path p knocked out under the POST-KI (reset)
        # barrier (vs the pre-KI barrier), so the receivable can pick the right payoff.
        ko_post = np.zeros(n_paths, dtype=bool)
        post_active = self.post_ko_barrier is not None
        post_ko_set = set(self._post_ko_idx)
        for j in range(n_t):
            # KI is resolved FIRST so a path that knocks in at this node already uses
            # the post-KI (reset) KO barrier here, matching the engine's v_out<-v_in
            # copy that happens after the per-surface KO at the same observation.
            if self.ki_barrier is not None and j in ki_set:
                hit = ((spots[:, j] <= self.ki_barrier) if self.ki_direction == "down"
                       else (spots[:, j] >= self.ki_barrier))
                # bridge only across an interval interior to the KI window (both
                # endpoints monitored) so the pre-activation interval is not sampled
                if self.continuous and j > 0 and (j - 1) in ki_set:
                    hit = hit | self._bridge_cross(
                        spots[:, j - 1], spots[:, j], self.ki_barrier,
                        float(self.times[j] - self.times[j - 1]), self.ki_direction, rng)
                ki = ki | hit
            if self.ko_barrier is not None and j in self._ko_idx:
                hit = ((spots[:, j] >= self.ko_barrier) if self.ko_direction == "up"
                       else (spots[:, j] <= self.ko_barrier))
                # pre-KI barrier applies to NOT-yet-KI paths when a reset is configured;
                # to ALL alive paths for a plain snowball (post_ko_barrier is None).
                if post_active:
                    hit = hit & ~ki
                newly = hit & ~dead
                ko_idx[newly] = j
                dead = dead | hit
            if post_active and j in post_ko_set:
                hit = ((spots[:, j] >= self.post_ko_barrier)
                       if self.ko_direction == "up"
                       else (spots[:, j] <= self.post_ko_barrier))
                hit = hit & ki                 # post-KI barrier applies to KI'd paths
                newly = hit & ~dead
                ko_idx[newly] = j
                ko_post[newly] = True
                dead = dead | hit
            knocked_in[:, j] = ki
            alive[:, j] = ~dead
        return {"knocked_in": knocked_in, "alive": alive, "ko_idx": ko_idx,
                "ko_post": ko_post}

    def _bridge_cross(self, s0, s1, barrier, dt, direction, rng):
        return _brownian_bridge_cross(s0, s1, barrier, (self.vol ** 2) * dt,
                                      direction, rng)


def _brownian_bridge_cross(s0, s1, barrier, var, direction, rng):
    """First-passage indicator over a step whose BOTH endpoints are strictly on the
    safe side (wrong-side endpoints are caught by the node check). One uniform per path
    is drawn (not per safe-subset) so the CRN stream stays aligned across base/bumped
    re-runs even when the safe set shifts — preserving variance reduction."""
    out = np.zeros(np.asarray(s0).shape[0], dtype=bool)
    if var <= 0:
        return out
    x0, x1, b = np.log(s0), np.log(s1), np.log(barrier)
    if direction == "down":          # barrier below; safe side = above b
        safe = (x0 > b) & (x1 > b)
        arg = -2.0 * (x0 - b) * (x1 - b) / var
    else:                            # up barrier; safe side = below b
        safe = (x0 < b) & (x1 < b)
        arg = -2.0 * (b - x0) * (b - x1) / var
    p = np.zeros_like(s0)
    p[safe] = np.exp(np.minimum(arg[safe], 0.0))
    if np.any(p[safe] < -1e-9) or np.any(p[safe] > 1.0 + 1e-9):
        raise ValidationError("bridge crossing probability out of [0,1]")
    u = rng.random(np.asarray(s0).shape[0])
    out[safe] = u[safe] < np.clip(p[safe], 0.0, 1.0)
    return out


@dataclass
class PhoenixStateMachine:
    """Per-path Phoenix state: KI (alive/knocked-in), KO termination, and accumulated
    missed-coupon memory.

    Coupon and KO share the observation schedule (``obs_idx``). At each observation: KI
    is resolved first (continuous via Brownian bridge / scalar node check), then the KO
    barrier terminates a path if crossed, else the coupon condition resolves the memory
    (pay -> reset to 0, miss -> +1). The memory RECORDED per node is the POST-resolution
    memory, matching the engine's EX-coupon per-memory surfaces (indexed by the memory a
    survivor carries INTO the next period). ``direction`` is the KO/coupon trigger side
    ("up" standard: pay/KO when spot >= barrier; "down" reverse). Memory is bounded by
    ``num_obs`` (cannot miss more coupons than observations)."""

    coupon_barrier: float
    ko_barrier: float
    direction: str                  # "up" (standard) / "down" (reverse)
    obs_idx: List[int]
    num_obs: int
    ki_barrier: Optional[float] = None
    ki_direction: str = "down"
    ki_monitoring_idx: List[int] = field(default_factory=list)
    times: object = None
    seed: int = 999
    continuous: bool = False
    vol: float = 0.0
    use_memory: bool = True
    initial_knocked_in: bool = False

    def __post_init__(self) -> None:
        if self.direction not in ("up", "down"):
            raise ValidationError("direction must be 'up'/'down'")
        if self.ki_direction not in ("up", "down"):
            raise ValidationError("ki_direction must be 'up'/'down'")
        for nm, b in (("coupon_barrier", self.coupon_barrier),
                      ("ko_barrier", self.ko_barrier)):
            if not (np.isfinite(b) and b > 0):
                raise ValidationError(f"{nm} must be a positive finite level")
        if self.ki_barrier is not None and not (np.isfinite(self.ki_barrier)
                                                and self.ki_barrier > 0):
            raise ValidationError("ki_barrier must be a positive finite level")
        if not (np.isfinite(self.vol) and self.vol >= 0):
            raise ValidationError("vol must be non-negative and finite")
        if self.continuous and self.ki_barrier is not None and self.vol <= 0:
            raise ValidationError("continuous KI requires vol > 0 (bridge variance)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValidationError("seed must be an int (deterministic CRN)")
        if not isinstance(self.use_memory, bool):
            raise ValidationError("use_memory must be a bool")
        if self.times is None:
            raise ValidationError("times is required")
        t = np.asarray(self.times, dtype=float)
        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)) \
                or np.any(np.diff(t) <= 0):
            raise ValidationError("times must be a finite, strictly-increasing 1-D array")
        self.times = t
        if not self.obs_idx:
            raise ValidationError("Phoenix requires a non-empty observation schedule")

    def run(self, spots: np.ndarray) -> dict:
        spots = np.asarray(spots, dtype=float)
        if spots.ndim != 2:
            raise ValidationError("spots must be 2-D (num_paths, n_times)")
        if not np.all(np.isfinite(spots)) or np.any(spots <= 0):
            raise ValidationError("spots must be finite and positive")
        n_paths, n_t = spots.shape
        if self.times.shape != (n_t,):
            raise ValidationError("times length must match spots' time axis")
        for nm, idx in (("obs", self.obs_idx), ("ki", self.ki_monitoring_idx)):
            for j in idx:
                if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
                    raise ValidationError(f"{nm}_idx entries must be integers")
                if j < 0 or j >= n_t:
                    raise ValidationError(f"{nm}_idx out of range")
        ki_set = set(self.ki_monitoring_idx)
        if self.continuous and self.ki_barrier is not None:
            mi = sorted(ki_set)
            if mi and mi != list(range(mi[0], mi[-1] + 1)):
                raise ValidationError(
                    "continuous KI requires a contiguous monitoring schedule (no gaps)")
        obs_set = set(self.obs_idx)
        up = self.direction == "up"
        rng = np.random.default_rng(self.seed)

        knocked_in = np.zeros((n_paths, n_t), dtype=bool)
        alive = np.ones((n_paths, n_t), dtype=bool)
        memory_hist = np.zeros((n_paths, n_t), dtype=int)
        ko_idx = np.full(n_paths, -1, dtype=int)
        ki = np.full(n_paths, self.initial_knocked_in, dtype=bool)
        dead = np.zeros(n_paths, dtype=bool)
        memory = np.zeros(n_paths, dtype=int)
        for j in range(n_t):
            if self.ki_barrier is not None and j in ki_set:
                hit = ((spots[:, j] <= self.ki_barrier) if self.ki_direction == "down"
                       else (spots[:, j] >= self.ki_barrier))
                if self.continuous and j > 0 and (j - 1) in ki_set:
                    hit = hit | _brownian_bridge_cross(
                        spots[:, j - 1], spots[:, j], self.ki_barrier,
                        (self.vol ** 2) * float(self.times[j] - self.times[j - 1]),
                        self.ki_direction, rng)
                ki = ki | hit
            if j in obs_set:
                ko_hit = ((spots[:, j] >= self.ko_barrier) if up
                          else (spots[:, j] <= self.ko_barrier))
                newly = ko_hit & ~dead
                ko_idx[newly] = j
                dead = dead | ko_hit
                survivors = ~dead
                pay = ((spots[:, j] >= self.coupon_barrier) if up
                       else (spots[:, j] <= self.coupon_barrier))
                if self.use_memory:
                    nxt = np.minimum(memory + 1, self.num_obs)
                    memory = np.where(survivors & pay, 0,
                                      np.where(survivors & ~pay, nxt, memory))
                else:
                    memory = np.where(survivors, 0, memory)
            memory_hist[:, j] = memory
            knocked_in[:, j] = ki
            alive[:, j] = ~dead
        return {"alive": alive, "knocked_in": knocked_in, "ko_idx": ko_idx,
                "memory": memory_hist}
