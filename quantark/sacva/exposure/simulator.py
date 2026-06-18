"""Monte-Carlo regulatory exposure engine (spec §3.2).

Risk-neutral MC: vectorized GBM spot paths (one constant-vol factor per underlying)
+ deterministic value-surface repricing at each exposure node, aggregated to a
discounted EPE profile per MAR50.35 (netting within enforceable sets, summed across
sets). The profile is RISK_NEUTRAL / regulatory-eligible and feeds RegulatoryCVAEngine.

v1 scope: equity (and reporting-vs-foreign FX) spot underlyings, deterministic
rates, single reporting currency, uncollateralized. Vanilla (single-state) trades
are priced via the analytic value surface (full vol surface re-evaluated, terminal
payoff exact). Stateful trades whose engine advertises ``supports_spot_greeks_grid``
(snowball) take the backward-grid value surface + ``BarrierStateMachine`` path (see
``_compute_stateful`` and ``snowball_surface``); one such trade defines the exposure
grid and vanillas may net on it. Unsupported products / out-of-scope features raise,
never approximate.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.sacva.exposure.asof import equity_asof_env
from quantark.sacva.exposure.correlation import CorrelationModel
from quantark.sacva.exposure.engine import (
    ExposureEngine,
    ExposureProfile,
    Measure,
    aggregate_epe,
)
from quantark.sacva.exposure.grid import ExposureGrid
from quantark.sacva.exposure.paths import StatePathGenerator
from quantark.sacva.exposure.phoenix_surface import (
    PhoenixTerminatedAtValuation,
    build_phoenix_surface,
)
from quantark.sacva.exposure.repricer import reprice_phoenix, reprice_trade
from quantark.sacva.exposure.snowball_surface import (
    SnowballTerminatedAtValuation,
    build_snowball_surface,
)
from quantark.sacva.exposure.statemachine import BarrierStateMachine, PhoenixStateMachine
from quantark.sacva.exposure.value_surface import AnalyticValueSurface
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.util.exceptions import ValidationError

# tau below this (in years, ~1 calendar day) is treated as the terminal node and
# valued by the contractual payoff, sidestepping the engine's exercise-date guard
_TAU_FLOOR = 1.0 / 365.0


@dataclass
class MonteCarloExposureConfig:
    num_paths: int = 20000
    n_steps: int = 24
    seed: int = 12345
    correlation: Optional[CorrelationModel] = None  # required iff >1 underlying


class MonteCarloExposureEngine(ExposureEngine):
    """Risk-neutral MC exposure for one counterparty -> discounted EPE profile."""

    measure = Measure.RISK_NEUTRAL
    regulatory_eligible = True

    def __init__(self, config: Optional[MonteCarloExposureConfig] = None):
        self.config = config or MonteCarloExposureConfig()

    # -- risk-factor extraction -------------------------------------------------
    def _underlying_key(self, trade):
        sq = getattr(trade.env, "spot_quote", None)
        key = getattr(sq, "asset_name", None) if sq is not None else None
        if not key:
            raise ValidationError(
                f"{trade.trade_id}: env.spot_quote.asset_name is required to identify "
                "the underlying risk factor")
        return key

    def _trade_maturity(self, trade):
        T = float(trade.product.get_maturity(trade.env))
        if not (T > 0):
            raise ValidationError(f"{trade.trade_id}: non-positive maturity {T}")
        return T

    def compute(self, counterparty) -> ExposureProfile:
        trades = [t for ns in counterparty.netting_sets for t in ns.trades]
        if not trades:
            raise ValidationError(f"{counterparty.name}: no trades")

        # Phoenix (memory-coupon) is a distinct 3-D state (spot, KI, memory) and takes
        # its own path; it must be checked before the snowball two-regime dispatch
        # (PhoenixQuadEngine subclasses SnowballQuadEngine).
        phoenix = [t for t in trades if isinstance(t.engine, PhoenixQuadEngine)]
        if phoenix:
            if len(phoenix) != 1:
                raise ValidationError(
                    f"{counterparty.name}: v1 supports at most one Phoenix trade per "
                    "counterparty")
            return self._compute_phoenix(counterparty, trades, phoenix[0])

        # stateful (grid) trades take the snowball backward-grid + state-machine path
        stateful = [t for t in trades
                    if getattr(t.engine, "supports_spot_greeks_grid", False)]
        if stateful:
            return self._compute_stateful(counterparty, trades, stateful)

        return self._compute_grid(counterparty, trades)

    def _compute_phoenix(self, counterparty, trades, phoenix) -> ExposureProfile:
        """Phoenix exposure: per-(t, KI, memory) surface + PhoenixStateMachine.

        v1 prices a single Phoenix trade per counterparty (co-netting with other trades
        is deferred — the snowball shared-grid path handles vanilla co-netting). The
        memory dimension makes the surface 3-D, so the value is selected per path by
        (KI state, accumulated missed-coupon count).
        """
        if len(trades) != 1:
            raise ValidationError(
                f"{counterparty.name}: Phoenix exposure v1 supports a single Phoenix "
                "trade per counterparty (co-netting is deferred)")
        try:
            spec = build_phoenix_surface(phoenix)
        except PhoenixTerminatedAtValuation:
            return self._compute_grid(counterparty, trades, zero_trades=[phoenix])
        times = spec.times
        self._check_market_consistency(trades, times)
        key = self._underlying_key(phoenix)
        env = phoenix.env
        horizon = float(times[-1])
        paths = StatePathGenerator(
            keys=[key], spots=[float(env.spot)], vols=[spec.vol],
            rates=[float(env.get_rate(horizon))], divs=[float(env.get_div_yield(horizon))],
            corr=[[1.0]], grid_times=times, num_paths=self.config.num_paths,
            seed=self.config.seed).generate()
        state = PhoenixStateMachine(
            coupon_barrier=spec.coupon_barrier, ko_barrier=spec.ko_barrier,
            direction=spec.direction, obs_idx=spec.obs_idx, num_obs=spec.num_obs,
            ki_barrier=spec.ki_barrier, ki_direction=spec.ki_direction,
            ki_monitoring_idx=spec.ki_monitoring_idx, times=times,
            seed=self.config.seed + 1, continuous=spec.ki_continuous, vol=spec.vol,
            use_memory=spec.use_memory,
            initial_knocked_in=spec.initial_knocked_in).run(paths[key])
        snow_value = reprice_phoenix(
            spec.surface, paths[key], state, times, float(phoenix.quantity),
            exposure_idx=list(range(len(times))))
        df = np.array([float(env.get_discount_factor(float(t))) for t in times])
        values = {id(phoenix): snow_value}
        epe = np.zeros(len(times))
        for ns in counterparty.netting_sets:
            arrays = [values[id(t)] for t in ns.trades]
            epe = epe + aggregate_epe(arrays, enforceable=ns.netting_enforceable, df=df)
        return ExposureProfile(times=times, epe_discounted=epe,
                               measure=Measure.RISK_NEUTRAL, regulatory_eligible=True)

    def _compute_grid(self, counterparty, trades, zero_trades=()) -> ExposureProfile:
        """Vanilla (analytic-surface) exposure on a uniform grid over ``trades``.

        ``zero_trades`` are carried at zero value (used for a snowball that terminated
        at valuation): they still appear in their netting set so the set aggregation is
        correct, but contribute no exposure. If nothing is left to price, the profile
        is a single t0 node of zero exposure (CVA integrates to zero).
        """
        zero_ids = {id(t) for t in zero_trades}
        priced = [t for t in trades if id(t) not in zero_ids]
        if not priced:
            return ExposureProfile(times=np.array([0.0]), epe_discounted=np.array([0.0]),
                                   measure=Measure.RISK_NEUTRAL, regulatory_eligible=True)

        # per-underlying market (constant-vol GBM factor), taken from the first trade
        # on that key; horizon = max maturity across the priced trades
        horizon = max(self._trade_maturity(t) for t in priced)
        keys, spots, vols, rates, divs = [], [], [], [], []
        market_env = {}
        for t in priced:
            k = self._underlying_key(t)
            if k in market_env:
                if abs(market_env[k].spot - t.env.spot) > 1e-9:
                    raise ValidationError(
                        f"inconsistent spot for underlying {k} across trades")
                continue
            market_env[k] = t.env
            spot = float(t.env.spot)
            keys.append(k)
            spots.append(spot)
            vols.append(float(t.env.get_vol(spot, horizon)))   # ATM term vol
            rates.append(float(t.env.get_rate(horizon)))
            divs.append(float(t.env.get_div_yield(horizon)))

        corr = self._corr_matrix(keys)

        grid = ExposureGrid.build(horizon=horizon, n_steps=self.config.n_steps,
                                  event_times=[])
        times = grid.times
        self._check_market_consistency(priced, times)
        gen = StatePathGenerator(keys=keys, spots=spots, vols=vols, rates=rates,
                                 divs=divs, corr=corr, grid_times=times,
                                 num_paths=self.config.num_paths, seed=self.config.seed)
        paths = gen.generate()

        # discount factors on the reporting curve (single currency v1)
        ref_curve = priced[0].env
        df = np.array([float(ref_curve.get_discount_factor(float(ti))) for ti in times])

        # pathwise undiscounted values per trade; zero_trades carried at zero
        values = {id(t): self._trade_value_array(t, paths, times) for t in priced}
        for t in zero_trades:
            values[id(t)] = np.zeros((self.config.num_paths, len(times)))

        # counterparty EPE = sum over sets of set-level discounted EPE (MAR50.35)
        epe = np.zeros(len(times))
        for ns in counterparty.netting_sets:
            arrays = [values[id(t)] for t in ns.trades]
            epe = epe + aggregate_epe(arrays, enforceable=ns.netting_enforceable, df=df)

        return ExposureProfile(times=times, epe_discounted=epe,
                               measure=Measure.RISK_NEUTRAL, regulatory_eligible=True)

    def _check_market_consistency(self, trades, times):
        """Reject ambiguous portfolios so exposure is independent of trade order.

        A counterparty's trades must share ONE reporting discount curve (v1 single
        reporting currency), and all trades on the same underlying must agree on the
        GBM factor inputs (spot/rate/div/vol over the horizon) — there is exactly one
        simulated process per underlying, so disagreeing market data is ill-defined
        rather than something to silently pick one of.
        """
        horizon = float(times[-1])
        ref_df = np.array([float(trades[0].env.get_discount_factor(float(t)))
                           for t in times])
        by_key = {}
        for tr in trades:
            dfk = np.array([float(tr.env.get_discount_factor(float(t))) for t in times])
            if not np.allclose(dfk, ref_df, rtol=1e-9, atol=1e-12):
                raise ValidationError(
                    f"{tr.trade_id}: all trades in a counterparty must share one "
                    "reporting discount curve (v1 single reporting currency)")
            k = self._underlying_key(tr)
            sp = float(tr.env.spot)
            market = (sp, float(tr.env.get_rate(horizon)),
                      float(tr.env.get_div_yield(horizon)),
                      float(tr.env.get_vol(sp, horizon)))
            if k in by_key:
                if not np.allclose(by_key[k], market, rtol=1e-9, atol=1e-12):
                    raise ValidationError(
                        f"trades on underlying {k} must agree on spot/rate/div/vol "
                        "(one GBM factor per underlying); differing market data is "
                        "ambiguous")
            else:
                by_key[k] = market

    def _corr_matrix(self, keys):
        """Principal correlation submatrix for ``keys`` (identity for one factor).

        The configured matrix may be a SUPERSET of the underlyings needed for a given
        run: the requested keys are looked up by name and the principal submatrix is
        extracted in requested order. This lets one config serve both the full stateful
        run and a terminated-snowball fallback that prices only the remaining vanillas
        (a strict whole-key match would make the bumped terminated state unpriceable).
        """
        if len(keys) == 1:
            return [[1.0]]
        if self.config.correlation is None:
            raise ValidationError(
                "multi-underlying counterparty requires a correlation matrix in "
                "MonteCarloExposureConfig (independence is not assumed)")
        cm = self.config.correlation
        cm_keys = list(cm.keys)
        missing = [k for k in keys if k not in cm_keys]
        if missing:
            raise ValidationError(
                f"correlation matrix missing underlyings {missing}; "
                f"available keys {cm_keys}")
        idx = [cm_keys.index(k) for k in keys]
        return np.asarray(cm.matrix, dtype=float)[np.ix_(idx, idx)].tolist()

    def _compute_stateful(self, counterparty, trades, stateful) -> ExposureProfile:
        """Snowball exposure: backward-grid surface + per-path KI/KO state machine.

        One stateful (snowball) trade defines the exposure grid (its observation
        times). Vanilla trades in the counterparty ride on the SAME grid — their
        analytic value surface evaluates at any node — so a snowball nets with
        vanillas. Multiple stateful trades (different observation schedules) would
        need a merged grid and raise rather than guessing one.
        """
        if len(stateful) != 1:
            raise ValidationError(
                f"{counterparty.name}: v1 supports at most one stateful (snowball) trade "
                "per counterparty; multiple autocallables need a merged observation grid")
        snow = stateful[0]
        try:
            spec = build_snowball_surface(snow)
        except SnowballTerminatedAtValuation:
            # the snowball is already terminated at valuation (immediate KO / zero
            # maturity): zero future exposure, but co-netted vanillas still price.
            return self._compute_grid(counterparty, trades, zero_trades=[snow])
        times = spec.times
        self._check_market_consistency(trades, times)
        snow_key = self._underlying_key(snow)
        horizon = float(times[-1])

        for t in trades:
            if t is snow:
                continue
            # a co-netted trade maturing past the snowball horizon would have exposure
            # beyond the shared grid that we cannot see -> raise (needs a merged grid)
            if self._trade_maturity(t) > horizon + 1e-9:
                raise ValidationError(
                    f"{t.trade_id}: maturity exceeds the stateful trade's horizon "
                    f"{horizon}; netting a longer-dated trade with a snowball needs a "
                    "merged exposure grid (deferred)")
            # a co-netted trade on the SNOWBALL underlying is repriced via its own vol
            # surface but rides the snowball's constant-vol (spec.vol) path; require the
            # surface to be flat at spec.vol there (true for a flat surface), else the
            # path and the repricing vols silently disagree under skew.
            if self._underlying_key(t) == snow_key:
                tv = float(t.env.get_vol(float(t.env.spot), horizon))
                if not np.isclose(tv, spec.vol, rtol=1e-9, atol=1e-12):
                    raise ValidationError(
                        f"{t.trade_id}: co-netted on the snowball underlying but its vol "
                        f"{tv:g} != the snowball path vol {spec.vol:g}; co-netting on a "
                        "snowball underlying requires a flat vol at the snowball pricing vol")

        # per-underlying market on the SHARED grid. The snowball underlying is seeded
        # FIRST and sourced entirely from snow.env (spot/rate/div + the surface's strike
        # vol) so the simulated drift matches the env the value surface was priced under,
        # regardless of trade order. Co-netted trades must agree on the full market data
        # for a shared underlying (enforced by _check_market_consistency + the vol guard
        # above). Other underlyings use their own env at term vol.
        keys, spots, vols, rates, divs = [], [], [], [], []
        snow_spot = float(snow.env.spot)
        seen = {snow_key: snow_spot}
        keys.append(snow_key)
        spots.append(snow_spot)
        vols.append(spec.vol)
        rates.append(float(snow.env.get_rate(horizon)))
        divs.append(float(snow.env.get_div_yield(horizon)))
        for t in trades:
            k = self._underlying_key(t)
            if k in seen:
                if abs(seen[k] - float(t.env.spot)) > 1e-9:
                    raise ValidationError(
                        f"inconsistent spot for underlying {k} across trades")
                continue
            seen[k] = float(t.env.spot)
            keys.append(k)
            spots.append(float(t.env.spot))
            vols.append(float(t.env.get_vol(float(t.env.spot), horizon)))
            rates.append(float(t.env.get_rate(horizon)))
            divs.append(float(t.env.get_div_yield(horizon)))
        paths = StatePathGenerator(
            keys=keys, spots=spots, vols=vols, rates=rates, divs=divs,
            corr=self._corr_matrix(keys), grid_times=times,
            num_paths=self.config.num_paths, seed=self.config.seed).generate()

        # bridge RNG on a separate deterministic stream (seed+1) so the knock-in
        # bridge uniforms are not coupled to the path-generator draws (MC hygiene),
        # while staying common across base/bumped re-runs.
        snow_state = BarrierStateMachine(
            ki_barrier=spec.ki_barrier, ki_direction=spec.ki_direction,
            ko_barrier=spec.ko_barrier, ko_direction=spec.ko_direction,
            ki_monitoring_idx=spec.ki_monitoring_idx,
            ko_monitoring_idx=spec.ko_monitoring_idx,
            post_ko_barrier=spec.post_ko_barrier,
            post_ko_monitoring_idx=(spec.post_ko_monitoring_idx
                                    if spec.post_ko_barrier is not None else None),
            times=times, seed=self.config.seed + 1,
            continuous=spec.ki_continuous, vol=spec.vol,
            initial_knocked_in=spec.initial_knocked_in).run(paths[snow_key])

        ref_curve = trades[0].env
        df = np.array([float(ref_curve.get_discount_factor(float(ti))) for ti in times])

        snow_value = reprice_trade(
            spec.surface, paths[snow_key], snow_state, times, float(snow.quantity),
            exposure_idx=list(range(len(times))),
            state_labels=("alive", "knocked_in"))
        # delayed-settlement KO redemptions: a knocked-out path is owed its redemption
        # until settlement, so carry payoff*DF(t_j, settle) on [obs, settle) as a
        # deterministic receivable (df[j] from the aggregator then yields the constant
        # t0 value payoff*DF(0, settle)). Immediate settlement (settle==obs) adds nothing.
        # KO-reset: a path that knocked out under the post-KI barrier (ko_post) uses the
        # post-leg payoff/settlement; otherwise the pre-leg.
        ko_idx = snow_state["ko_idx"]
        ko_post = snow_state.get("ko_post")
        pre_mask = ~ko_post if ko_post is not None else None
        recv = self._ko_receivable(
            ko_idx, df, spec.ko_monitoring_idx, spec.ko_payoffs, spec.ko_settle_idx,
            regime_mask=pre_mask)
        if spec.post_ko_barrier is not None:
            recv = recv + self._ko_receivable(
                ko_idx, df, spec.post_ko_monitoring_idx, spec.post_ko_payoffs,
                spec.post_ko_settle_idx, regime_mask=ko_post)
        snow_value = snow_value + float(snow.quantity) * recv
        values = {id(snow): snow_value}
        for t in trades:
            if t is snow:
                continue
            values[id(t)] = self._trade_value_array(t, paths, times)

        epe = np.zeros(len(times))
        for ns in counterparty.netting_sets:
            arrays = [values[id(t)] for t in ns.trades]
            epe = epe + aggregate_epe(arrays, enforceable=ns.netting_enforceable, df=df)
        return ExposureProfile(times=times, epe_discounted=epe,
                               measure=Measure.RISK_NEUTRAL, regulatory_eligible=True)

    def _ko_receivable(self, ko_idx, df, monitoring_idx, payoffs, settle_idx,
                       regime_mask=None):
        """Pending-receivable exposure for one delayed-settlement KO leg, (n_paths, n_t).

        A path that knocks out at observation grid index ``obs`` with settlement at grid
        index ``settle > obs`` is owed ``payoff`` until settlement; its time-t_j value is
        ``payoff*DF(t_j, settle) = payoff*df[settle]/df[j]``, carried on ``[obs, settle)``.
        The value is left UNDISCOUNTED-to-t0 (aggregator convention): the aggregator's
        ``df[j]`` collapses it to the constant t0 value ``payoff*df[settle]``. Immediate
        settlement (``settle == obs``) contributes nothing — the redemption is paid at KO.

        ``regime_mask`` (KO-reset) restricts the leg to paths that knocked out under its
        regime (pre-KI vs post-KI), so a node KO'd under the other regime is not paid here.
        """
        ko_idx = np.asarray(ko_idx)
        n_t = df.shape[0]
        if np.any(df <= 0.0):  # df divides below; a non-positive curve point is invalid
            raise ValidationError("discount factors must be positive for KO receivable")
        out = np.zeros((ko_idx.shape[0], n_t), dtype=float)
        for obs, payoff, settle in zip(monitoring_idx, payoffs, settle_idx):
            if not (0 <= obs < n_t) or not (0 <= settle < n_t):
                raise ValidationError(
                    f"KO receivable obs/settle index out of range (obs={obs}, "
                    f"settle={settle}, n_t={n_t})")
            if settle <= obs:
                continue
            sel = ko_idx == obs
            if regime_mask is not None:
                sel = sel & regime_mask
            if not sel.any():
                continue
            for j in range(obs, settle):
                out[sel, j] = payoff * df[settle] / df[j]
        return out

    def _trade_value_array(self, trade, paths, times):
        """Pathwise UNDISCOUNTED reporting-currency value, shape (num_paths, n_t)."""
        key = self._underlying_key(trade)
        spots = paths[key]                       # (num_paths, n_t)
        T = self._trade_maturity(trade)
        surface = AnalyticValueSurface(
            engine=trade.engine, product=trade.product, base_env=trade.env,
            as_of_env=equity_asof_env, currency=trade.trade_currency)
        out = np.zeros_like(spots)
        for j, tj in enumerate(times):
            tau = T - float(tj)
            col = spots[:, j]
            if tau >= _TAU_FLOOR:
                out[:, j] = surface.value_at(col, float(tj), None)
            elif tau >= 0.0:                     # terminal node: contractual payoff
                if not hasattr(trade.product, "get_payoff"):
                    raise ValidationError(
                        f"{trade.trade_id}: product lacks get_payoff for terminal node")
                out[:, j] = np.array([trade.product.get_payoff(float(s)) for s in col])
            # tau < 0: matured/settled -> 0
        return out * float(trade.quantity)
