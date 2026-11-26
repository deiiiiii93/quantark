"""
Quadrature pricing engine for Phoenix (autocallable) options.

Implements a two-state (knocked-in / not-knocked-in) quadrature recursion
with coupon jumps at observation times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from quantark.asset.equity.engine.event_stats import CouponTable
from quantark.asset.equity.engine.pde.grid.events import project_piecewise_event
from quantark.asset.equity.engine.quad.first_step import pointwise_piecewise_event
from quantark.asset.equity.engine.quad.quad_adapters import (
    resolve_structured_payment_timings,
)
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v1_coupons_forward import CouponGroupStep, ForwardCouponRows, NodalGroupStep
from quantark.priceenv.term_sampling import make_df_fn
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.observation_schedule import at_valuation_instant
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import Tolerance, is_close, safe_log, same_instant


@dataclass(frozen=True)
class _CouponCashRows:
    """Coupon-cash columns of the stacked event-statistics recursion.

    The coupon a path is paid at observation ``k`` depends on its memory state there (the arrears it
    carries) and, under EXPIRY pay, on when the note ends. Both stay linear, so two columns per row
    carry them without a memory dimension. With ``A`` the arrears carried into a time ``t``, the
    discounted coupon cash of row ``k`` seen from ``t`` is ``U_k + A * N_k``:

    * ``U_k`` -- the row's cash on paths that carry no arrears into ``t``. At observation ``j < k``
      a missed coupon (no knock-out) adds ``c_j`` to the arrears, so ``U_k += c_j * N_k`` there;
    * ``N_k`` (memory only) -- the discounted unit value of the arrears released at ``k``: by a
      coupon paid at ``k``, or by a knock-out at ``k`` whatever the coupon barrier says (the release
      rule the price applies). A coupon paid, or a knock-out, at ``j < k`` releases the arrears
      first, so ``N_k = 0`` there.

    The discount is the price's own: INSTANT pays at the observation's settlement; EXPIRY pays when the
    note ends, so a coupon paid without a knock-out is worth its termination-value column ``W`` (the
    price's ``w_out`` / ``w_in``, one unit paid at the knock-out settlement or at the terminal payment)
    and one paid with a knock-out its settlement discount.

    ``carried_arrears`` are the arrears carried into the first future observation (the contract's
    initial arrears, or the state the valuation instant leaves); ``decided_coupon`` is a coupon paid
    by an observation decided at the valuation instant without a knock-out.

    This is the stacked statistics route's layout (the ``n_ko`` knock-out rows come first); the forward
    route reads these columns off the adjoint forward pass (``v1_coupons_forward.ForwardCouponRows``).
    """

    n_ko: int
    memory: bool
    expiry: bool
    amounts: np.ndarray
    delay_dfs: np.ndarray
    terminal_delay_df: float
    carried_arrears: float
    decided_index: Optional[int]
    decided_coupon: float

    @property
    def indicator_row(self) -> int:
        """First coupon-trigger indicator row (after the ``n_ko`` KO rows)."""
        return self.n_ko

    @property
    def cash_row(self) -> int:
        """First ``U`` row (after the KO and coupon-indicator rows)."""
        return 2 * self.n_ko

    @property
    def release_row(self) -> int:
        """First ``N`` row (memory only)."""
        return 3 * self.n_ko

    @property
    def termination_row(self) -> int:
        """The ``W`` row (EXPIRY only)."""
        return (4 if self.memory else 3) * self.n_ko

    @property
    def n_buckets(self) -> int:
        """EXPIRY payment buckets: one per knock-out record (paid at its settlement), then the maturity."""
        return self.n_ko + 1

    @property
    def bucket_unit_row(self) -> int:
        """First ``T_b`` row (EXPIRY): one unit paid when the note ends at bucket ``b``."""
        return self.termination_row + 1

    @property
    def bucket_cash_row(self) -> int:
        """First ``D_b`` row (EXPIRY): the coupon cash paid at bucket ``b`` on paths carrying no arrears."""
        return self.bucket_unit_row + self.n_buckets

    @property
    def bucket_release_row(self) -> int:
        """First ``R_b`` row (EXPIRY memory): the unit value of carried arrears released and paid at bucket ``b``."""
        return self.bucket_cash_row + self.n_buckets

    @property
    def earned_row(self) -> int:
        """First ``E_k`` row (EXPIRY memory): the coupon earned at row ``k``, valued at ``k`` (undiscounted there)."""
        return self.bucket_release_row + self.n_buckets

    @property
    def earned_release_row(self) -> int:
        """First ``Q_k`` row (EXPIRY memory): the unit value at ``k`` of carried arrears released at ``k``."""
        return self.earned_row + self.n_ko

    @property
    def n_extra_rows(self) -> int:
        """Rows beyond the ``n_ko`` KO rows: indicators, ``U``, ``N`` (memory), ``W`` (EXPIRY), then the EXPIRY payment
        buckets ``T``, ``D`` (and ``R`` with memory) and, with memory, the earned-coupon columns ``E``, ``Q``."""
        rows = (3 if self.memory else 2) * self.n_ko
        if self.expiry:
            rows += 1 + (3 if self.memory else 2) * self.n_buckets + (2 * self.n_ko if self.memory else 0)
        return rows

    def initialize(self, v_out, v_in) -> None:
        """Terminal values: no coupon cash after maturity; ``W`` is one unit paid at the terminal
        payment, valued at maturity (the price's ``terminal_w``); so is the maturity bucket's ``T``."""
        if self.expiry:
            v_out[self.termination_row] = self.terminal_delay_df
            v_in[self.termination_row] = self.terminal_delay_df
            last = self.bucket_unit_row + self.n_ko
            v_out[last] = self.terminal_delay_df
            v_in[last] = self.terminal_delay_df

    def write_observation(
        self, j, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss
    ) -> None:
        """Fill the branch values of observation ``j`` in place (rows x grid arrays).

        ``pay_*`` continue after a paid coupon, ``miss_*`` after a missed one, ``breach_*`` are the
        knock-out values with and without the coupon. Every branch array starts as the columns before
        the observation. The indicator row is one on the paid branch, continuing or knocked out.
        """
        n = self.n_ko
        row = self.indicator_row + j
        pay_out[row] = pay_in[row] = 1.0
        breach_pay[row] = 1.0
        c = float(self.amounts[j])
        ko_df = float(self.delay_dfs[j])
        u = self.cash_row + j
        if self.expiry:
            w = self.termination_row
            paid_out, paid_in = pay_out[w].copy(), pay_in[w].copy()
            breach_pay[w] = breach_miss[w] = ko_df
        else:
            paid_out = paid_in = ko_df
        pay_out[u] = c * paid_out
        pay_in[u] = c * paid_in
        breach_pay[u] = c * ko_df
        if self.memory:
            r = self.release_row + j
            later_u = slice(u + 1, self.cash_row + n)
            later_r = slice(r + 1, self.release_row + n)
            pay_out[r] = paid_out
            pay_in[r] = paid_in
            breach_pay[r] = breach_miss[r] = ko_df
            pay_out[later_r] = 0.0
            pay_in[later_r] = 0.0
            miss_out[later_u] += c * miss_out[later_r]
            miss_in[later_u] += c * miss_in[later_r]
        if self.expiry:
            self._write_buckets(j, c, ko_df, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss)
            if self.memory:
                self._write_earned(j, c, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss)

    def _write_buckets(self, j, c, ko_df, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss) -> None:
        """The EXPIRY payment buckets at observation ``j``: a knock-out here ends the note at bucket ``j``; a coupon
        paid here is paid at the bucket the note ends at (``T_b``), and so are the arrears it releases (``R_b``); a
        missed memory coupon joins the arrears released later (``D_b += c R_b``)."""
        units = slice(self.bucket_unit_row, self.bucket_unit_row + self.n_buckets)
        cash = slice(self.bucket_cash_row, self.bucket_cash_row + self.n_buckets)
        t_out, t_in = pay_out[units].copy(), pay_in[units].copy()
        pay_out[cash] += c * t_out
        pay_in[cash] += c * t_in
        breach_pay[units] = breach_miss[units] = 0.0
        breach_pay[self.bucket_unit_row + j] = breach_miss[self.bucket_unit_row + j] = ko_df
        breach_pay[cash] = breach_miss[cash] = 0.0
        breach_pay[self.bucket_cash_row + j] = c * ko_df
        if self.memory:
            release = slice(self.bucket_release_row, self.bucket_release_row + self.n_buckets)
            miss_out[cash] += c * miss_out[release]
            miss_in[cash] += c * miss_in[release]
            pay_out[release] = t_out
            pay_in[release] = t_in
            breach_pay[release] = breach_miss[release] = 0.0
            breach_pay[self.bucket_release_row + j] = breach_miss[self.bucket_release_row + j] = ko_df

    def _write_earned(self, j, c, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss) -> None:
        """The coupon earned at row ``j`` valued at ``j`` (EXPIRY memory): the ``U``/``N`` recursion with one unit paid
        at the observation itself, whatever the note's end."""
        n = self.n_ko
        e = self.earned_row + j
        q = self.earned_release_row + j
        pay_out[e] = pay_in[e] = c
        breach_pay[e] = c
        pay_out[q] = pay_in[q] = 1.0
        breach_pay[q] = breach_miss[q] = 1.0
        later_e = slice(e + 1, self.earned_row + n)
        later_q = slice(q + 1, self.earned_release_row + n)
        pay_out[later_q] = 0.0
        pay_in[later_q] = 0.0
        miss_out[later_e] += c * miss_out[later_q]
        miss_in[later_e] += c * miss_in[later_q]

    def readout(self, values_at_spot: np.ndarray, termination_at_spot: Optional[float]) -> np.ndarray:
        """Per-row discounted coupon cash from the columns read at the valuation spot."""
        n = self.n_ko
        cash = np.array(values_at_spot[self.cash_row:self.cash_row + n], dtype=float)
        if self.memory:
            cash += self.carried_arrears * np.asarray(
                values_at_spot[self.release_row:self.release_row + n], dtype=float
            )
        if self.decided_index is not None:
            discount = (
                termination_at_spot if self.expiry else float(self.delay_dfs[self.decided_index])
            )
            cash[self.decided_index] += self.decided_coupon * float(discount)
        return cash

    def readout_buckets(self, values_at_spot: np.ndarray) -> np.ndarray:
        """EXPIRY: the coupon cash paid at each ending event (knock-out records, then the maturity), discounted."""
        b = self.n_buckets
        units = np.asarray(values_at_spot[self.bucket_unit_row:self.bucket_unit_row + b], dtype=float)
        cash = np.array(values_at_spot[self.bucket_cash_row:self.bucket_cash_row + b], dtype=float)
        if self.memory:
            cash += self.carried_arrears * np.asarray(
                values_at_spot[self.bucket_release_row:self.bucket_release_row + b], dtype=float
            )
        if self.decided_index is not None:
            cash += self.decided_coupon * units
        return cash

    def readout_earned(self, values_at_spot: np.ndarray) -> np.ndarray:
        """EXPIRY memory: the coupon earned at each row, valued at the row (``E[DF(t_k) earned_k]``)."""
        n = self.n_ko
        earned = np.array(values_at_spot[self.earned_row:self.earned_row + n], dtype=float)
        earned += self.carried_arrears * np.asarray(
            values_at_spot[self.earned_release_row:self.earned_release_row + n], dtype=float
        )
        if self.decided_index is not None:
            earned[self.decided_index] += self.decided_coupon
        return earned


class PhoenixQuadEngine(SnowballQuadEngine):
    """
    Quadrature pricing engine for Phoenix options with coupon jumps.
    """

    # This engine reads its price off its OWN surfaces, so the
    # transition readout is not implemented here yet; refuse it rather
    # than report a legacy-readout price under another name.
    supported_readouts = ("legacy_linear",)

    def __init__(self, params: Optional[QuadParams] = None):
        super().__init__(params=params)

    def _validate_product(self, product: PhoenixOption) -> None:
        if product.barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise PricingError("PhoenixQuadEngine requires discrete KO monitoring.")

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        if self.params.auto_converge:
            return self._price_with_convergence(product, pricing_env)
        return self._price_once(product, pricing_env)

    def _price_once(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        fold=None,
    ) -> float:
        if not isinstance(product, PhoenixOption):
            raise PricingError(
                f"PhoenixQuadEngine only supports PhoenixOption, got {type(product).__name__}"
            )
        if pricing_env is None:
            raise PricingError("PricingEnvironment is required for PhoenixQuadEngine.")

        self._validate_product(product)

        # Opt-in per-memory backward-grid capture for the CVA exposure layer. Phoenix
        # pays coupons WHILE ALIVE, so the exposure value at an observation must be the
        # EX-coupon continuation: we record the surfaces at the TOP of each step (before
        # that observation's coupon/KO jump), indexed by the post-resolution memory state
        # (= the diffused surfaces from the next observation). Stored as
        # observation_time -> (spot_grid, [v_in_k...], [v_out_k...]). Reset BEFORE any
        # early return so a terminated bumped re-price cannot leave stale surfaces.
        record_grids = getattr(self, "record_backward_grids", False)
        if record_grids:
            self._backward_grids = {}

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        self._require_resolvable_time(maturity)
        if maturity <= 0.0:
            self._refuse_expired_fold(fold)
            return self._decided_at_valuation(product, pricing_env, float(spot))

        # An observation AT the valuation instant is decided on the known spot: the recursion below
        # starts after it (``_merge_times`` drops time zero). The engine used to skip it -- no
        # knock-out, no coupon, no missed-coupon memory and no knock-in from an observation now.
        carried_knock_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        decision = product.decide_observations_at_valuation(
            float(spot), pricing_env, knocked_in=carried_knock_in
        )
        if decision.knocked_out:
            if fold is not None:
                self._refuse_fold_on_shared_instants(product, product.resolve_ko_observations(pricing_env))
                in_life = self._fold_in_life(
                    product.resolve_ko_observations(pricing_env), maturity
                )
                fold_ko = self._bind_fold_ko(fold, [("main", rec) for rec in in_life])
                decided = self._match_record(decision.ko_record.observation_time, in_life)
                return self._decided_at_valuation(
                    product, pricing_env, float(spot),
                    extra_cash=self._folded_amount(fold_ko, decided),
                )
            return self._decided_at_valuation(product, pricing_env, float(spot))

        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        if vol <= 0:
            raise ValidationError(f"Volatility must be positive, got {vol}")
        if not np.isfinite(div):
            raise ValidationError(f"Dividend yield must be finite, got {div}")
        if vol > 5.0:
            raise ValidationError(f"Volatility too high for quadrature stability: {vol}")

        # An empty side (spec 2026-10-07 theta boundary: aged past its last date) folds away:
        # no knock-out records = no autocall before maturity; no knock-in records = no knock-in.
        ko_records = product.resolve_ko_observations(pricing_env)
        payment_timings = resolve_structured_payment_timings(
            product, pricing_env, ko_records
        )
        event_delay_by_record = {
            id(record): float(delay_df)
            for record, delay_df in zip(
                ko_records, payment_timings.event_delay_dfs
            )
        }
        fold_ko: Optional[dict] = None
        fold_terminal = None
        if fold is not None:
            self._refuse_fold_on_shared_instants(product, ko_records)
            fold_ko = self._bind_fold_ko(
                fold, [("main", rec) for rec in self._fold_in_life(ko_records, maturity)]
            )
            terminal = payment_timings.terminal
            fold_terminal = {
                branch: fold.bind_terminal(
                    branch, terminal.determination_time, terminal.payment_time
                )
                for branch in ("no_ki", "ki")
            }

        ki_continuous = product.has_ki_barrier and (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        ki_records: Sequence = []
        if product.has_ki_barrier and not ki_continuous:
            ki_records = product.resolve_ki_observations(pricing_env)

        times = self._merge_times(
            [rec.observation_time for rec in ko_records],
            [rec.observation_time for rec in ki_records],
            maturity,
        )
        if ki_continuous:
            from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid
            contract_times = times
            times, self._continuous_ki_model = continuous_time_grid(pricing_env, times, self.params)
            # the knots are engine-added nodes: a contract instant is never displaced by one (contract clock D2)
            times = self._merge_times(times, (), maturity, anchors=contract_times)
        if not times:
            raise PricingError("Observation time grid is empty for PhoenixQuadEngine.")

        coupon_barrier = product.coupon_config.coupon_barrier
        if isinstance(coupon_barrier, list):
            if len(coupon_barrier) != len(ko_records):
                raise ValidationError(
                    "Coupon barrier schedule length does not match KO observations."
                )
            coupon_barriers = np.array(coupon_barrier, dtype=float)
        else:
            coupon_barriers = np.full(len(ko_records), float(coupon_barrier))

        rate_vec, div_vec, vol_vec = self._term_step_params(
            pricing_env, product.strike, times, rate, div, vol
        )
        vol_max_val = float(np.max(vol_vec))
        df_local = make_df_fn(pricing_env)
        dt = self._build_dt(times)
        reachable_ko = self._reachable_ko_observations(
            ko_records,
            times,
            spot,
            rate_vec,
            div_vec,
            vol_vec,
            dt,
            is_reverse=product.is_reverse,
        )
        self._last_ignored_ko_observation_indices = tuple(
            index for index, active in enumerate(reachable_ko) if not active
        )

        align_log = self._select_alignment_log(
            spot,
            [rec for rec, active in zip(ko_records, reachable_ko) if active],
            coupon_barriers=coupon_barriers,
            ki_records=ki_records,
            product=product,
        )

        barrier_logs = self._barrier_logs(
            spot,
            [record.barrier for record in ko_records],
            coupon_barriers,
            [record.barrier for record in ki_records],
            product.barrier_config.ki_barrier,
        )
        fft_padding_factor = self._resolve_fft_padding_factor()
        fft_filter_alpha, fft_filter_power = self._resolve_fft_filter()
        grid_points = self._resolve_grid_points(
            maturity, vol_max_val, [rec.observation_time for rec in ki_records]
        )
        math_utils = QuadratureMath(
            grid_x=grid_points,
            spot=spot,
            maturity=maturity,
            vol_max=vol_max_val,
            num_std_devs=self.params.num_std_devs,
            align_log=align_log,
            barrier_logs=barrier_logs,
            cell_stretch=self._resolve_align_cell_stretch(),
            integration_rule=self.params.integration_rule,
            fft_padding_factor=fft_padding_factor,
            fft_filter_alpha=fft_filter_alpha,
            fft_filter_power=fft_filter_power,
        )
        grid = math_utils.grid
        spot_grid = spot * np.exp(grid)
        tau = 0.5 * vol_vec * vol_vec * dt
        self._require_resolved_kernels(
            times, tau, math_utils.h,
            levels=self._observed_levels(
                times, ko_records, ki_records,
                [(r.observation_time, coupon_barriers[i]) for i, r in enumerate(ko_records)],
            ),
        )
        if np.any(tau[1:] <= 0.0):
            raise ValidationError("time step too small for quadrature solver.")

        alpha_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) / (
            vol_vec * vol_vec
        )
        beta_vec = (rate_vec - div_vec - 0.5 * vol_vec * vol_vec) ** 2 / (
            vol_vec**4
        ) + 2.0 * rate_vec / (vol_vec * vol_vec)

        v_in = np.array(
            [
                product.get_maturity_payoff_v1(spot_value, pricing_env=pricing_env)
                for spot_value in spot_grid
            ],
            dtype=float,
        )
        v_out = np.array(
            [
                product.get_maturity_payoff_v0(spot_value, pricing_env=pricing_env)
                for spot_value in spot_grid
            ],
            dtype=float,
        )

        log_ki_barrier = None
        if product.has_ki_barrier and ki_continuous:
            if product.barrier_config.ki_barrier is None:
                raise PricingError("KI barrier configuration is missing.")
            if isinstance(product.barrier_config.ki_barrier, list):
                raise PricingError("Continuous KI requires scalar ki_barrier.")
            log_ki_barrier = safe_log(product.barrier_config.ki_barrier / spot)

        ko_times = np.array([rec.observation_time for rec in ko_records], dtype=float)
        period_year_fractions = np.array(
            product.get_coupon_period_year_fractions(pricing_env)[:len(ko_records)],
            dtype=float,
        )
        coupon_amounts = np.array(
            [
                product.get_coupon_payoff(i, year_fraction=period_year_fractions[i])
                for i in range(len(ko_records))
            ],
            dtype=float,
        )
        coupon_cumulative = np.concatenate(([0.0], np.cumsum(coupon_amounts)))

        use_memory = product.has_memory_coupon
        if use_memory and len(ko_records) > 50:
            raise ValidationError(
                f"Too many observations ({len(ko_records)}) for Memory Phoenix Quad engine. "
                "Limit is 50 to prevent performance degradation. Use MC engine instead."
            )

        # Initialize value function vectors.
        # v_in_list[k] = Value(S, t, memory=k)
        # At maturity, we initialize for the maximum possible accumulated coupons.
        # Max accumulation at maturity is N (if we missed all N-1 previous and miss the last one?
        # Actually, at maturity, we just pay what is due.
        # Let's track surfaces based on "number of missed coupons entering the next period".
        # At T (step N), we construct N+1 surfaces for having missed 0, 1, ..., N coupons.
        num_obs = len(ko_records)
        v_in_list = []
        v_out_list = []

        # Termination-value surfaces for coupon_pay_type=EXPIRY. W is the value
        # of one unit paid when the note ENDS -- at the knock-out settlement if
        # it knocks out, at maturity otherwise -- so a coupon of c earned here
        # is worth c*W rather than c*DF(now -> maturity). Every rolled-up coupon
        # shares one termination date, so W does not depend on how many have
        # been earned: two surfaces, not one per earned-count. At maturity the
        # note ends however it got there, so W starts at one.
        expiry_coupons = (
            product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
        )
        # A note surviving to maturity pays at the RESOLVED terminal payment,
        # so W's terminal condition is the value at maturity of one unit paid
        # there (exactly 1.0 when no settlement terms exist).
        terminal_w = float(payment_timings.terminal.delay_df)
        w_out = (
            np.full_like(spot_grid, terminal_w, dtype=float)
            if expiry_coupons
            else None
        )
        w_in = (
            np.full_like(spot_grid, terminal_w, dtype=float)
            if expiry_coupons
            else None
        )

        # Memory coupons already owed at the pricing date. ``missed == obs_idx`` is the state in which
        # EVERY period since the pricing date has missed, so nothing has paid yet and those arrears are
        # still outstanding; in any other state a coupon has paid and released them. No extra memory
        # dimension is needed. (The PDE solver keeps two memory states instead -- state 0 and one unit of
        # arrears more -- and carries these arrears at the first observation: ``_state_arrears``.)
        # The engine used to price from zero arrears whatever the contract carried.
        initial_arrears = (
            float(product.coupon_config.initial_coupon_arrears) if use_memory else 0.0
        )

        def accumulated_before(obs_idx: int, missed: int) -> float:
            carried = initial_arrears if missed == obs_idx else 0.0
            if missed <= 0 or obs_idx <= 0:
                return carried
            start = max(obs_idx - missed, 0)
            return float(coupon_cumulative[obs_idx] - coupon_cumulative[start] + carried)

        # Terminal condition for each memory state (base payoff; coupons added via jumps). The base payoff does
        # not depend on the memory state (no coupon accumulated: get_maturity_payoff_v1 takes none, v0 gets 0.0),
        # so it is evaluated once and every state starts from the same arrays (T25: the same floats as one
        # evaluation per state; neither array is written below).
        def terminal_lists(spot_values):
            """The per-memory-state surfaces at maturity on any points."""
            base_in_cash = np.array(
                [
                    product.get_maturity_payoff_v1(spot_value, pricing_env=pricing_env)
                    for spot_value in spot_values
                ],
                dtype=float,
            )
            base_out_cash = np.array(
                [
                    product.get_maturity_payoff_v0(
                        spot_value,
                        accumulated_coupons=0.0,
                        pricing_env=pricing_env,
                    )
                    for spot_value in spot_values
                ],
                dtype=float,
            )
            ins, outs = [], []
            for k in range(num_obs + 1):
                v_in_cash, v_out_cash = base_in_cash, base_out_cash
                if fold_terminal is not None:
                    v_in_cash = v_in_cash + fold_terminal["ki"]
                    v_out_cash = v_out_cash + fold_terminal["no_ki"]
                v_in_k = v_in_cash * payment_timings.terminal.delay_df
                v_out_k = v_out_cash * payment_timings.terminal.delay_df
                ins.append(v_in_k)
                outs.append(v_out_k)
            return ins, outs

        v_in_list, v_out_list = terminal_lists(spot_grid)

        full_p_lr, full_p_ur, full_p0 = 0, len(grid) - 1, (len(grid) - 1) % 2
        omega_grid = math_utils.z_grid
        disable_ko_after_ki = product.barrier_config.disable_ko_after_ki
        smoothing_width = self._resolve_event_smoothing_width(math_utils, product)

        # records on one env instant (0.16.0 D14): each record's index -> the instant's record indices
        group_of_index = self._shared_ko_indices(product, ko_records)

        def observe_group(indices, grid, spot_grid, smoothing_width, v_in_list, v_out_list, w_in, w_out,
                          discrete_ki_record):
            """The records of one env instant (D14 zero-gap rule): consecutive observations with zero elapsed time --
            the same spot, contract order; record k, still alive, pays its coupon (memory: with the arrears since the
            last paid coupon) and then tests its knock-out (``_zero_gap_group``). Returns ``(v_in_list, v_out_list,
            w_in, w_out, union knock-out weight, discrete knock-in applied)``."""
            records = [
                (j, float(ko_records[j].payoff or 0.0) + (fold_ko[id(ko_records[j])] if fold_ko is not None else 0.0),
                 float(event_delay_by_record[id(ko_records[j])]), float(coupon_amounts[j]),
                 float(coupon_barriers[j]),
                 float(ko_records[j].barrier) if reachable_ko[j] and ko_records[j].barrier is not None else None)
                for j in indices
            ]
            return self._zero_gap_group(
                product, grid, spot, smoothing_width, records, v_in_list, v_out_list, w_in, w_out,
                ki_barrier=None if discrete_ki_record is None else float(discrete_ki_record.barrier),
                use_memory=use_memory, expiry=expiry_coupons, accumulated_before=accumulated_before,
            )

        def observe(obs_time, grid, spot_grid, smoothing_width, v_in_list, v_out_list, w_in, w_out):
            """The observation at ``obs_time`` on every memory state (and the termination surfaces), on any points
            (the lattice, or the exact first step's quadrature nodes)."""
            discrete_ki_record = (
                self._match_record(obs_time, ki_records) if ki_records else None
            )
            ki_applied_jointly = False

            # Identify if this time step is a KO/Coupon observation
            ko_weight = None
            ko_index = None
            ko_record = None
            for idx, rec in enumerate(ko_records):
                if same_instant(obs_time, rec.observation_time):
                    ko_index = idx
                    ko_record = rec
                    break
            if group_of_index and ko_index is not None and ko_index in group_of_index:
                # the instant holds several records: one zero-gap stage replaces the single record's (D14)
                v_in_list, v_out_list, w_in, w_out, ko_weight, ki_applied_jointly = observe_group(
                    group_of_index[ko_index], grid, spot_grid, smoothing_width, v_in_list, v_out_list, w_in, w_out,
                    discrete_ki_record,
                )
                ko_record = ko_index = None

            # If this is an observation $i$ (0-based index `ko_index`),
            # we are moving from $t_{i+1}$ (where we have valid surfaces) to $t_i$.
            # The surfaces coming from diffusion are $U[k]$ for $k \in \{0, \dots, i+1\}$ (missed 0 to i+1 coupons).
            # We need to construct surfaces $W[k]$ for $k \in \{0, \dots, i\}$ (missed 0 to i coupons).

            if ko_record is not None and ko_index is not None:
                ko_is_reachable = bool(reachable_ko[ko_index])
                barrier_val = coupon_barriers[ko_index]
                coupon_amt = float(coupon_amounts[ko_index])

                pay_weight = self._event_weight(
                    grid,
                    barrier_val,
                    spot,
                    smoothing_width,
                    trigger_is_down=product.is_reverse,
                )
                ko_weight = (
                    self._event_weight(
                        grid,
                        ko_record.barrier,
                        spot,
                        smoothing_width,
                        trigger_is_down=product.is_reverse,
                    )
                    if ko_is_reachable
                    else np.zeros_like(grid)
                )

                ko_discount = event_delay_by_record[id(ko_record)]
                base_ko_payoff = float(ko_record.payoff or 0.0)
                if fold_ko is not None:
                    base_ko_payoff += fold_ko[id(ko_record)]
                
                # INSTANT pays the coupon at its record's settlement (the
                # observation itself when no settlement terms exist); EXPIRY
                # pays it at termination, whose value is the regime's own W
                # surface — never a fixed terminal discount, which is wrong
                # whenever a knock-out is possible.
                if expiry_coupons:
                    coupon_discount_out, coupon_discount_in = w_out, w_in
                else:
                    coupon_discount_out = coupon_discount_in = (
                        event_delay_by_record[id(ko_record)]
                    )

                new_v_in_list = []
                new_v_out_list = []

                # Max possible missed coupons entering this step is `ko_index`.
                # If use_memory is False, we only have state 0.
                max_k = ko_index if use_memory else 0

                for k in range(max_k + 1):
                    # We are in state "missed k coupons" just before observation.
                    
                    # Branch 1: Coupon Condition Met (Pay)
                    # Payoff = Current + (k * accumulated)
                    # Transition -> State 0 (reset memory)
                    accumulated_pay = (
                        accumulated_before(ko_index, k) if use_memory else 0.0
                    )
                    earned = coupon_amt + accumulated_pay

                    # Continuation value if paid: comes from state 0 of the next step
                    # Wait, diffused surfaces `v_out_list` are indexed by *future* memory state.
                    # If we pay now, the future starts with 0 memory.
                    # So we use `v_out_list[0]`.
                    val_pay_out = v_out_list[0] + earned * coupon_discount_out
                    val_pay_in = v_in_list[0] + earned * coupon_discount_in

                    # Branch 2: Coupon Condition Missed
                    # Payoff = 0
                    # Transition -> State k+1 (accumulate)
                    # Use `v_out_list[k+1]`
                    # If memory is off, we stay in state 0 (or technically state 0 again for next step).
                    next_k_miss = k + 1 if use_memory else 0
                    val_miss_out = v_out_list[next_k_miss]
                    val_miss_in = v_in_list[next_k_miss]

                    if self._use_cell_average_events():
                        combined_out, combined_in = self._project_joint_phoenix_event(
                            grid=grid,
                            spot=spot,
                            is_reverse=product.is_reverse,
                            coupon_barrier=float(barrier_val),
                            ko_barrier=(
                                float(ko_record.barrier)
                                if ko_is_reachable
                                else None
                            ),
                            ki_barrier=(
                                float(discrete_ki_record.barrier)
                                if discrete_ki_record is not None
                                else None
                            ),
                            val_miss_out=val_miss_out,
                            val_pay_out=val_pay_out,
                            val_miss_in=val_miss_in,
                            val_pay_in=val_pay_in,
                            ko_value_without_coupon=(
                                base_ko_payoff + accumulated_pay
                            )
                            * ko_discount,
                            ko_value_with_coupon=(
                                base_ko_payoff + accumulated_pay + coupon_amt
                            )
                            * ko_discount,
                            disable_ko_after_ki=disable_ko_after_ki,
                            pointwise=self._pointwise_events,
                        )
                        ki_applied_jointly = discrete_ki_record is not None
                    else:
                        # Legacy smoothed/nodal composition.
                        combined_out = (
                            pay_weight * val_pay_out
                            + (1.0 - pay_weight) * val_miss_out
                        )
                        combined_in = (
                            pay_weight * val_pay_in
                            + (1.0 - pay_weight) * val_miss_in
                        )
                        ko_val_grid = np.full_like(
                            combined_out, base_ko_payoff * ko_discount
                        )
                        ko_val_grid += accumulated_pay * ko_discount
                        ko_val_grid += pay_weight * coupon_amt * ko_discount
                        combined_out = (
                            ko_weight * ko_val_grid
                            + (1.0 - ko_weight) * combined_out
                        )
                        if not disable_ko_after_ki:
                            combined_in = (
                                ko_weight * ko_val_grid
                                + (1.0 - ko_weight) * combined_in
                            )

                    new_v_in_list.append(combined_in)
                    new_v_out_list.append(combined_out)

                v_in_list = new_v_in_list
                v_out_list = new_v_out_list

                # A knock-out ends the note, so W becomes the discount to its
                # settlement. Blended by the same weight the value surfaces use,
                # and left alone on the knocked-in regime when the knock-out no
                # longer applies there.
                if expiry_coupons:
                    w_out = ko_weight * ko_discount + (1.0 - ko_weight) * w_out
                    if not disable_ko_after_ki:
                        w_in = ko_weight * ko_discount + (1.0 - ko_weight) * w_in

            # KI Logic (Discrete/Continuous) applied to ALL surfaces
            if ki_continuous and log_ki_barrier is not None:
                ki_mask = (
                    spot_grid >= product.barrier_config.ki_barrier
                    if product.is_reverse
                    else spot_grid <= product.barrier_config.ki_barrier
                )
                for i in range(len(v_out_list)):
                    v_out_list[i][ki_mask] = v_in_list[i][ki_mask]
                if expiry_coupons:
                    # A knock-in switches regime for the termination law too.
                    w_out[ki_mask] = w_in[ki_mask]
            elif discrete_ki_record is not None and not ki_applied_jointly:
                ki_record = discrete_ki_record
                for i in range(len(v_out_list)):
                    v_out_list[i] = self._blend_ki_transition(
                        v_out_list[i],
                        v_in_list[i],
                        grid,
                        spot_grid,
                        ki_record.barrier,
                        spot,
                        smoothing_width,
                        product.is_reverse,
                        # the knock-out wins a tie on the not-knocked-in surface whatever the
                        # flag: v_out already carries it, so the knock-in must not overwrite it
                        ko_weight=ko_weight,
                    )
                if expiry_coupons:
                    w_out = self._blend_ki_transition(
                        w_out, w_in, grid, spot_grid, ki_record.barrier, spot,
                        smoothing_width, product.is_reverse,
                        ko_weight=ko_weight,
                    )

            return v_in_list, v_out_list, w_in, w_out

        first_exact = self._first_step_exact(float(tau[1]), math_utils.h)
        first_pre = None
        first_levels = []

        for step_index in range(len(times), 0, -1):
            obs_time = times[step_index - 1]

            # EX-coupon recording: capture the per-memory continuation surfaces BEFORE
            # this observation's coupon/KO jump, indexed by post-resolution memory k
            # (= the value a path carries INTO the next period). A KO'd path is handled
            # by the state machine (dead -> 0), so these surfaces value only survivors.
            if record_grids:
                self._backward_grids[float(obs_time)] = (
                    spot_grid.copy(),
                    [arr.copy() for arr in v_in_list],
                    [arr.copy() for arr in v_out_list],
                )

            if step_index == 1 and first_exact:
                # the exact first step (S3b): the pre-event states (the maturity seed when the first event is maturity)
                first_pre = None if len(times) == 1 else (
                    [arr.copy() for arr in v_in_list], [arr.copy() for arr in v_out_list],
                    None if w_in is None else w_in.copy(), None if w_out is None else w_out.copy(),
                )
                first_ko_index = next(
                    (idx for idx, rec in enumerate(ko_records) if same_instant(obs_time, rec.observation_time)), None
                )
                first_ko = None if first_ko_index is None else ko_records[first_ko_index]
                first_ki = self._match_record(obs_time, ki_records) if ki_records else None
                first_levels = [
                    level for level in (
                        first_ko.barrier if first_ko is not None else None,
                        coupon_barriers[first_ko_index] if first_ko is not None else None,
                        first_ki.barrier if first_ki is not None else None,
                        product.barrier_config.ki_barrier if ki_continuous and log_ki_barrier is not None else None,
                    ) if level is not None
                ]
                for j in group_of_index.get(first_ko_index, ())[1:]:
                    # every record of the instant: its knock-out and coupon barriers are breakpoints (D6/D14)
                    first_levels += [level for level in (ko_records[j].barrier, coupon_barriers[j])
                                     if level is not None]
            v_in_list, v_out_list, w_in, w_out = observe(
                obs_time, grid, spot_grid, smoothing_width, v_in_list, v_out_list, w_in, w_out
            )

            # Diffusion Step
            tau_step = float(tau[step_index])
            alpha = float(alpha_vec[step_index])
            beta = float(beta_vec[step_index])
            vol_step = float(vol_vec[step_index])
            prefactor = self._step_prefactor(
                math.exp(-beta * tau_step) / math.sqrt(math.pi * tau_step) / 2.0, tau_step, alpha, math_utils.h)
            omega_array = np.exp(-(omega_grid**2) / (4.0 * tau_step) - alpha * omega_grid)

            # Diffuse all surfaces
            for i in range(len(v_in_list)):
                # Retained pre-diffusion for the bridge correction below.
                v_in_at_obs = v_in_list[i]
                v_in_list[i] = self._diffuse_fft(
                    v_in_list[i],
                    math_utils,
                    omega_array,
                    prefactor,
                    full_p_lr,
                    full_p_ur,
                    full_p0,
                    alpha,
                    beta,
                    tau_step,
                )

                if ki_continuous:
                    v_out_list[i] = self._diffuse_with_bridge(
                        v_out_list[i],
                        v_in_at_obs,
                        math_utils,
                        omega_array,
                        prefactor,
                        full_p_lr,
                        full_p_ur,
                        full_p0,
                        log_ki_barrier,
                        alpha,
                        beta,
                        vol_step,
                        dt[step_index],
                        tau_step,
                        product.is_reverse,
                    )
                else:
                    v_out_list[i] = self._diffuse_fft(
                        v_out_list[i],
                        math_utils,
                        omega_array,
                        prefactor,
                        full_p_lr,
                        full_p_ur,
                        full_p0,
                        alpha,
                        beta,
                        tau_step,
                    )

            # The termination-value surfaces ride the same operator, including
            # the knock-in bridge, so the discount a rolled-up coupon earns is
            # transported exactly as the values it is applied to.
            if expiry_coupons:
                w_in_at_obs = w_in
                w_in = self._diffuse_fft(
                    w_in, math_utils, omega_array, prefactor,
                    full_p_lr, full_p_ur, full_p0, alpha, beta, tau_step,
                )
                if ki_continuous:
                    w_out = self._diffuse_with_bridge(
                        w_out, w_in_at_obs, math_utils, omega_array, prefactor,
                        full_p_lr, full_p_ur, full_p0, log_ki_barrier, alpha,
                        beta, vol_step, dt[step_index], tau_step,
                        product.is_reverse,
                    )
                else:
                    w_out = self._diffuse_fft(
                        w_out, math_utils, omega_array, prefactor,
                        full_p_lr, full_p_ur, full_p0, alpha, beta, tau_step,
                    )

        first_values = None
        if first_exact:
            first_values, v_in_list, v_out_list, w_in, w_out = self._phoenix_first_step(
                product, math_utils, spot, float(tau[1]), float(alpha_vec[1]), float(beta_vec[1]),
                first_levels, first_pre,
                terminal_lists, terminal_w, expiry_coupons, observe, times[0], v_in_list, v_out_list, w_in, w_out,
                touch=(log_ki_barrier, product.is_reverse) if ki_continuous and log_ki_barrier is not None else None,
                pricing_env=pricing_env,
            )

        # t0 surfaces (memory 0, no observation at valuation): the exposure grid's
        # valuation-date slice. v_*_list has only the memory-0 state here.
        if record_grids:
            self._backward_grids[0.0] = (
                spot_grid.copy(),
                [arr.copy() for arr in v_in_list],
                [arr.copy() for arr in v_out_list],
            )

        # The state the valuation instant leaves behind. With no observation now that is memory
        # state 0 on the carried knock-in regime. An observation now has been decided above: its
        # coupon is paid here, a missed memory coupon starts the claim in memory state 1 (the
        # surfaces are indexed by the periods missed since the pricing date, this one included),
        # and a discrete knock-in observed now reads the knocked-in regime. A continuously
        # monitored level stays with the surfaces, which already carry it.
        observed_now = any(
            at_valuation_instant(record.observation_time) for record in ko_records
        )
        missed_now = observed_now and decision.coupon_index is None
        memory_state = 1 if (use_memory and missed_now) else 0
        now = [j for j, record in enumerate(ko_records) if at_valuation_instant(record.observation_time)]
        if use_memory and len(now) >= 2:
            # several records at the valuation instant (D14 zero-gap): the periods missed there after the last
            # one paid (all of them when none paid) -- the surfaces' memory index
            memory_state = (len(now) if decision.coupon_index is None
                            else sum(1 for j in now if j > decision.coupon_index))
        knocked_in_now = carried_knock_in or (decision.knocked_in and not ki_continuous)
        value_surface = (
            v_in_list[memory_state] if knocked_in_now else v_out_list[memory_state]
        )
        self._last_spot_greeks_grid = (spot_grid.copy(), value_surface.copy())
        if first_values is not None:
            value = first_values["in" if knocked_in_now else "out"][memory_state]
        else:
            value = math_utils.interpolate(value_surface, x=0.0)
        if decision.coupon:
            if expiry_coupons:
                # paid at termination: the regime's own termination-value surface discounts it
                if first_values is not None:
                    coupon_discount = first_values["w_in" if knocked_in_now else "w_out"]
                else:
                    coupon_discount = math_utils.interpolate(
                        w_in if knocked_in_now else w_out, x=0.0
                    )
            else:
                coupon_discount = event_delay_by_record[
                    id(ko_records[decision.coupon_index])
                ]
            value += float(decision.coupon) * float(coupon_discount)
        return value

    def _shared_ko_instants(self, product, ko_records) -> dict:
        """``{id(record): the instant's records}`` for every record on an env instant holding two or more records
        (0.16.0 D14): every Phoenix record is its own coupon date, so none is dropped -- only its knock-out test can be
        inert. Empty when no instant holds two records (the recursion runs exactly as before)."""
        if not isinstance(product, PhoenixOption):
            return {}
        return self._instants_of(ko_records, lambda records: records)

    def _shared_ko_indices(self, product, ko_records) -> dict:
        """``{index: indices of the instant's records}`` (contract order) of ``_shared_ko_instants``."""
        shared = self._shared_ko_instants(product, ko_records)
        if not shared:
            return {}
        index_of = {id(rec): j for j, rec in enumerate(ko_records)}
        out = {}
        for j, rec in enumerate(ko_records):
            if id(rec) in shared:
                out[j] = tuple(sorted(index_of[id(r)] for r in shared[id(rec)]))
        return out

    def _zero_gap_group(self, product, grid, spot, width, records, v_in_list, v_out_list, w_in, w_out, *,
                        ki_barrier, use_memory, expiry, accumulated_before):
        """The records of one env instant on every memory state (0.16.0 D14 zero-gap rule).

        ``records``: per record in contract order ``(index, knock-out redemption, settlement delay, coupon, coupon
        barrier, knock-out barrier or None when its test is inert)``; ``v_*_list`` the states entering the next
        observation (memory: the periods missed), ``w_*`` the termination rows (EXPIRY), ``ki_barrier`` a discrete
        knock-in observed at the instant. Record k, still alive, pays its coupon plus the arrears since the last paid
        coupon where the spot meets its coupon barrier (memory moves to 0, else it grows by one), then knocks out
        paying its redemption, the arrears carried into it and its coupon where paid; the knock-out wins a tie with
        the knock-in, which switches the regime of a note no record knocked out. Cell-average events: one pass of the
        piecewise projection whose branches are that chain on each region's indicators (``_project_joint_phoenix_event``
        for several records); nodal events: the records' blends in reverse contract order, then the knock-in.
        Returns ``(v_in_list, v_out_list, w_in, w_out, union knock-out weight, discrete knock-in applied)``."""
        reverse = bool(product.is_reverse)
        disable = product.barrier_config.disable_ko_after_ki

        if self._use_cell_average_events():
            def log_of(level):
                return safe_log(float(level) / float(spot))

            levels = [r[4] for r in records] + [r[5] for r in records if r[5] is not None]
            if ki_barrier is not None:
                levels.append(ki_barrier)
            breaks = sorted({log_of(level) for level in levels if level > 0.0})
            samples = ([breaks[0] - 1.0] + [0.5 * (a + b) for a, b in zip(breaks[:-1], breaks[1:])]
                       + [breaks[-1] + 1.0])

            def hit(sample, level):
                return sample <= log_of(level) if reverse else sample >= log_of(level)

            def chain(coupon_hits, ko_hits, knocked_in, regime_out):
                lists, w = ((v_out_list, w_out) if regime_out and not knocked_in else (v_in_list, w_in))
                ko_active = regime_out or not disable
                for pos in range(len(records) - 1, -1, -1):
                    j, base, delay, coupon, _, _ = records[pos]
                    paid, ko = coupon_hits[pos], ko_active and ko_hits[pos]
                    paid_discount = w if expiry else delay
                    new = []
                    for k in range((j if use_memory else 0) + 1):
                        arrears = accumulated_before(j, k) if use_memory else 0.0
                        if ko:
                            new.append((base + arrears + (coupon if paid else 0.0)) * delay)
                        elif paid:
                            new.append(lists[0] + (coupon + arrears) * paid_discount)
                        else:
                            new.append(lists[k + 1 if use_memory else 0])
                    lists = new
                    if expiry and ko:
                        w = delay
                return lists, w

            out_branches, in_branches = [], []
            for sample in samples:
                coupon_hits = [hit(sample, r[4]) for r in records]
                ko_hits = [r[5] is not None and hit(sample, r[5]) for r in records]
                knocked_in = ki_barrier is not None and (
                    sample >= log_of(ki_barrier) if reverse else sample <= log_of(ki_barrier))
                out_branches.append(chain(coupon_hits, ko_hits, knocked_in, True))
                in_branches.append(chain(coupon_hits, ko_hits, False, False))
            project = pointwise_piecewise_event if self._pointwise_events else project_piecewise_event

            def projected(branches, k):
                return np.asarray(project(grid, breaks, [b[0][k] for b in branches]), dtype=float)

            n_states = len(out_branches[0][0])
            new_out = [projected(out_branches, k) for k in range(n_states)]
            new_in = [projected(in_branches, k) for k in range(n_states)]
            if expiry:
                w_out = np.asarray(project(grid, breaks, [b[1] for b in out_branches]), dtype=float)
                w_in = np.asarray(project(grid, breaks, [b[1] for b in in_branches]), dtype=float)
            union = None
            ko_levels = [r[5] for r in records if r[5] is not None]
            if ko_levels:
                extreme = max(ko_levels) if reverse else min(ko_levels)
                union = self._event_weight(grid, extreme, spot, width, trigger_is_down=reverse)
            return new_in, new_out, w_in, w_out, union, ki_barrier is not None

        survive = np.ones_like(np.asarray(grid, dtype=float))
        for pos in range(len(records) - 1, -1, -1):
            j, base, delay, coupon, coupon_barrier, ko_barrier = records[pos]
            pay_weight = self._event_weight(grid, coupon_barrier, spot, width, trigger_is_down=reverse)
            ko_weight = (self._event_weight(grid, ko_barrier, spot, width, trigger_is_down=reverse)
                         if ko_barrier is not None else np.zeros_like(survive))
            paid_out, paid_in = (w_out, w_in) if expiry else (delay, delay)
            new_out, new_in = [], []
            for k in range((j if use_memory else 0) + 1):
                arrears = accumulated_before(j, k) if use_memory else 0.0
                miss = k + 1 if use_memory else 0
                out = pay_weight * (v_out_list[0] + (coupon + arrears) * paid_out) + (1.0 - pay_weight) * v_out_list[miss]
                inn = pay_weight * (v_in_list[0] + (coupon + arrears) * paid_in) + (1.0 - pay_weight) * v_in_list[miss]
                ko_value = (base + arrears) * delay + pay_weight * coupon * delay
                out = ko_weight * ko_value + (1.0 - ko_weight) * out
                if not disable:
                    inn = ko_weight * ko_value + (1.0 - ko_weight) * inn
                new_out.append(out)
                new_in.append(inn)
            v_out_list, v_in_list = new_out, new_in
            if expiry:
                w_out = ko_weight * delay + (1.0 - ko_weight) * w_out
                if not disable:
                    w_in = ko_weight * delay + (1.0 - ko_weight) * w_in
            survive = survive * (1.0 - ko_weight)
        return v_in_list, v_out_list, w_in, w_out, 1.0 - survive, False

    def _phoenix_first_step(self, product, math_utils, spot, tau1, alpha1, beta1, levels, pre, terminal_lists,
                            terminal_w, expiry, observe, first_time, v_in_list, v_out_list, w_in, w_out, *, touch=None,
                            pricing_env=None):
        """The exact first step (S3b) on every memory state and the termination surfaces: the price's own
        observation (``observe``) on the quadrature nodes, pointwise. Returns ``(values, v_in_list, v_out_list, w_in,
        w_out)``: F at the spot per state (``values['out'][k]``, ``['in'][k]``, ``['w_out']``, ``['w_in']``) and the
        t=0 surfaces with the band nodes replaced by F."""
        fs = self._first_step_quadrature(math_utils, spot, tau1, alpha1, beta1, levels, touch=touch,
                                         terminal=product if pre is None else None, pricing_env=pricing_env)
        y = fs.y
        if pre is None:
            ins_y, outs_y = terminal_lists(spot * np.exp(y))
            w_in_y = np.full(y.size, terminal_w, dtype=float) if expiry else None
            w_out_y = np.full(y.size, terminal_w, dtype=float) if expiry else None
        else:
            ins_y = [fs.lift(arr) for arr in pre[0]]
            outs_y = [fs.lift(arr) for arr in pre[1]]
            w_in_y = None if pre[2] is None else fs.lift(pre[2])
            w_out_y = None if pre[3] is None else fs.lift(pre[3])
        with self._pointwise():
            ins_y, outs_y, w_in_y, w_out_y = observe(
                first_time, y, spot * np.exp(y), 0.0, ins_y, outs_y, w_in_y, w_out_y
            )
        if len(ins_y) != len(v_in_list):
            raise PricingError("the exact first step's memory states do not match the lattice's")
        values = {"out": [], "in": []}
        v_in_list = [np.array(arr, dtype=float, copy=True) for arr in v_in_list]
        v_out_list = [np.array(arr, dtype=float, copy=True) for arr in v_out_list]
        for k, (in_y, out_y) in enumerate(zip(ins_y, outs_y)):
            f_out, f_in = fs.expect_bridged(out_y, in_y), fs.expect(in_y)
            v_out_list[k][fs.band] = f_out[1:]
            v_in_list[k][fs.band] = f_in[1:]
            values["out"].append(float(f_out[0]))
            values["in"].append(float(f_in[0]))
        if expiry:
            f_w_out, f_w_in = fs.expect_bridged(w_out_y, w_in_y), fs.expect(w_in_y)
            w_out = np.array(w_out, dtype=float, copy=True)
            w_in = np.array(w_in, dtype=float, copy=True)
            w_out[fs.band] = f_w_out[1:]
            w_in[fs.band] = f_w_in[1:]
            values["w_out"], values["w_in"] = float(f_w_out[0]), float(f_w_in[0])
        self._last_first_step = fs
        return values, v_in_list, v_out_list, w_in, w_out

    def _decided_at_valuation(
        self, product, pricing_env, spot: float, extra_cash: Optional[float] = None
    ) -> float:
        """Zero time to maturity: the observations still pending at this instant are decided on the spot.

        The backward induction decides an observation at the valuation instant on the known spot when
        more of the claim lies ahead. The shortcut used to return the redemption of a never-knocked-in
        claim instead: no last coupon, no memory arrears, no knock-out, no knock-in, not even a knock-in
        carried from an earlier day.

        ``extra_cash`` is folded cash paid with a knock-out decided now (ship spec D4).
        """
        from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing

        def discounted(cash: float, payment_time) -> float:
            if payment_time is not None and payment_time > 0.0:
                return float(cash) * float(pricing_env.get_discount_factor(payment_time))
            return float(cash)

        decision = product.decide_observations_at_valuation(
            spot,
            pricing_env,
            knocked_in=bool(getattr(product, "_otc_lifecycle_knocked_in", False)),
        )
        if decision.knocked_out:
            record = decision.ko_record
            cash = float(record.payoff or 0.0) + decision.coupon
            if extra_cash is not None:
                cash += float(extra_cash)
            return discounted(cash, record.settlement_time)
        redemption = product.get_payoff(spot, knocked_in=decision.knocked_in, pricing_env=pricing_env)
        value = discounted(redemption, resolve_terminal_timing(product, pricing_env).payment_time)
        if decision.coupon:
            if product.coupon_config.coupon_pay_type == CouponPayType.INSTANT:
                records = product.resolve_ko_observations(pricing_env)
                payment_time = records[decision.coupon_index].settlement_time
            else:
                payment_time = resolve_terminal_timing(product, pricing_env).payment_time
            value += discounted(decision.coupon, payment_time)
        return value

    def _decided_ko_statistics(self, product, pricing_env, ko_records, ko_record_0, *, streams=None):
        """A knock-out decided at valuation, with the coupon decided at the same instant (DESIGN §4.5).

        ``price`` pays the redemption and the coupon decided now (``_decided_at_valuation``): the knock-out row
        carries the redemption alone, the coupon is its own row of the coupon table, paid with the knock-out
        (AC11), so the ledger sums to the pv."""
        from quantark.asset.equity.engine.pde.core.statistics import event_statistics as es

        decision = product.decide_observations_at_valuation(
            float(pricing_env.spot),
            pricing_env,
            knocked_in=bool(getattr(product, "_otc_lifecycle_knocked_in", False)),
        )
        decided = ko_record_0
        if decision.knocked_out:
            if self._shared_ko_instants(product, ko_records) and decision.ko_index is not None:
                # several records at the valuation instant: the decision's own record, by contract index (D4)
                decided = ko_records[decision.ko_index]
            else:
                decided = self._match_record(decision.ko_record.observation_time, ko_records) or ko_record_0
        pv = float(self._decided_at_valuation(product, pricing_env, float(pricing_env.spot)))
        unit = self._decided_ko_unit(pricing_env, ko_records, decided)
        observations = es.observation_table(pricing_env, ko_records, unit,
                                            is_coupon_obs=np.ones(len(ko_records), dtype=bool))
        index = int(np.flatnonzero(unit)[0])
        coupon_cash = np.zeros(len(ko_records), dtype=float)
        coupon_cash[index] = pv - float(observations.discounted_cash[index])
        coupon_probability = np.zeros(len(ko_records), dtype=float)
        coupon_probability[index] = 1.0 if decision.coupon_index is not None else 0.0
        payment = observations.payment_times
        unavailable = {} if es.want_coupon_probability(streams) else {"coupons.probability": es.NOT_REQUESTED}
        coupons = CouponTable(
            probability=coupon_probability if es.want_coupon_probability(streams) else None,
            survival=np.concatenate([[1.0], observations.survival[:-1]]), payment_times=payment,
            payment_dates=observations.payment_dates, discounted_cash=coupon_cash,
            cash_to_pay=es.to_pay(pricing_env, coupon_cash, payment),
            payment_is_path_dependent=np.zeros(len(ko_records), dtype=bool),
        )
        stats = self._decided_statistics(
            product, pricing_env, observations, pv=pv, streams=streams, coupons=coupons,
            terminal=resolve_structured_payment_timings(product, pricing_env, ko_records).terminal,
            unavailable=unavailable,
        )
        return stats

    def _compute_event_stats(self, product, pricing_env, *, streams=None, pv=None, _route=None):
        """The Snowball recursion, with this product's coupon-cash columns (``_CouponCashRows``).

        The columns depend on the contract (memory, pay type, amounts, the arrears carried into the
        first future observation) and on the statistics route (whether knock-out rows come first),
        which the row hooks below are not given; they are resolved here, for this call only.
        """
        self._validate_product(product)
        route = self._resolve_v1_route(_route)
        previous = self.__dict__.get("_coupon_cash_rows")
        self._coupon_cash_rows = self._resolve_coupon_cash_rows(product, pricing_env, route=route)
        try:
            return super()._compute_event_stats(product, pricing_env, streams=streams, pv=pv, _route=route)
        finally:
            # restore the caller's layout (a nested statistics call must not clear an outer one's)
            self._coupon_cash_rows = previous

    def _resolve_coupon_cash_rows(self, product, pricing_env, *, route: str = "stacked"):
        """The coupon-cash column layout of the route (stacked: ``_CouponCashRows``; forward: ``ForwardCouponRows``),
        or ``None`` when the recursion does not run (expired note)."""
        maturity = product.get_maturity(pricing_env)
        if maturity <= 0.0:
            return None
        ko_records = product.resolve_ko_observations(pricing_env)
        if not ko_records:
            return None
        n_ko = len(ko_records)
        fractions = product.get_coupon_period_year_fractions(pricing_env)[:n_ko]
        amounts = np.array(
            [product.get_coupon_payoff(i, year_fraction=fractions[i]) for i in range(n_ko)],
            dtype=float,
        )
        timings = resolve_structured_payment_timings(product, pricing_env, ko_records)
        memory = bool(product.has_memory_coupon)
        # The valuation instant, decided as the price decides it: a coupon paid now (no knock-out
        # here; a knock-out now never reaches the recursion) and the arrears it leaves behind.
        decision = product.decide_observations_at_valuation(
            float(pricing_env.spot),
            pricing_env,
            knocked_in=bool(getattr(product, "_otc_lifecycle_knocked_in", False)),
        )
        decided_index = None if decision.knocked_out else decision.coupon_index
        layout = ForwardCouponRows if route == "forward" else _CouponCashRows
        return layout(
            n_ko=n_ko,
            memory=memory,
            expiry=product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY,
            amounts=amounts,
            delay_dfs=np.asarray(timings.event_delay_dfs, dtype=float),
            terminal_delay_df=float(timings.terminal.delay_df),
            carried_arrears=float(decision.arrears) if memory else 0.0,
            decided_index=decided_index,
            decided_coupon=float(decision.coupon) if decided_index is not None else 0.0,
        )

    def _event_stats_product_type(self) -> type:
        return PhoenixOption

    _statistics_family = "phoenix"

    def _terminal_cash_surfaces(self, product, pricing_env, spot_grid):
        """``(knocked-in, not knocked-in)`` maturity redemptions as ``price`` seeds them (no coupon accumulated:
        coupons are paid by the observation jumps)."""
        v_in_cash = np.array(
            [product.get_maturity_payoff_v1(s, pricing_env=pricing_env) for s in spot_grid], dtype=float
        )
        v_out_cash = np.array(
            [product.get_maturity_payoff_v0(s, accumulated_coupons=0.0, pricing_env=pricing_env)
             for s in spot_grid],
            dtype=float,
        )
        return v_in_cash, v_out_cash

    def _event_stats_alignment_log(
        self, spot, product, ki_barrier_override=None, ko_candidates_override=None,
        pricing_env=None,
    ):
        records = product.resolve_ko_observations(pricing_env)
        candidates = ko_candidates_override
        reachable = [r for r in records if candidates is None or r.barrier in candidates]
        coupons = [self._coupon_barrier_for_obs(product, i) for i in range(len(records))]
        ki_records = (product.resolve_ki_observations(pricing_env)
                      if product.has_ki_barrier and not product.barrier_config.ki_continuous
                      else [])
        return self._select_alignment_log(
            spot, reachable, coupon_barriers=coupons, ki_records=ki_records, product=product,
        )

    def _event_stats_barrier_logs(self, spot, product, ko_records, ki_records, continuous_barrier):
        return self._barrier_logs(
            spot, [r.barrier for r in ko_records],
            [self._coupon_barrier_for_obs(product, i) for i in range(len(ko_records))],
            [r.barrier for r in ki_records], product.barrier_config.ki_barrier,
        )

    def _apply_observation(
        self, product, grid, spot_grid, spot, width, v_out, v_in, *,
        ko_record=None, ko_value=0.0, ko_weight=None, ki_record=None,
        ki_barrier_continuous=None, coupon_index=None, n_ko=0,
        coupon_record=None, observation_index=None, ko_group=None,
    ):
        # These are indicator columns, independent of coupon memory amounts.
        # Nevertheless every column must see the price's joint coupon/KI/KO
        # projection: even a non-jumping continuation is averaged at its breaks.
        record = coupon_record
        if ko_group is not None:
            return self._apply_group_observation(
                product, grid, spot_grid, spot, width, v_out, v_in, ko_group=ko_group, ki_record=ki_record,
                ki_barrier_continuous=ki_barrier_continuous, write_coupons=coupon_index is not None,
            )
        if record is None:
            return super()._apply_observation(
                product, grid, spot_grid, spot, width, v_out, v_in,
                ki_record=ki_record, ki_barrier_continuous=ki_barrier_continuous,
            )
        index = observation_index
        pay_out, pay_in = v_out.copy(), v_in.copy()
        miss_out, miss_in = v_out, v_in
        breach_miss = np.zeros_like(v_out) + ko_value
        breach_pay = breach_miss.copy()
        cash_rows = self._coupon_cash_rows
        if coupon_index is not None and cash_rows is not None:
            # the call's coupon layout writes its rows' branches (stacked: indicator and cash columns; forward: W)
            miss_out, miss_in = v_out.copy(), v_in.copy()
            cash_rows.write_observation(
                index, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss
            )
        return self._project_observation_branches(
            product, grid, spot_grid, spot, width, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss,
            ko_record=ko_record, ko_weight=ko_weight, ki_record=ki_record,
            ki_barrier_continuous=ki_barrier_continuous, record=record, index=index,
        )

    def _project_observation_branches(
        self, product, grid, spot_grid, spot, width, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss, *,
        ko_record, ko_weight, ki_record, ki_barrier_continuous, record, index,
    ):
        """Observation ``index`` on its branch values (stacked rows x points): the continuation after a paid
        (``pay_*``) and a missed (``miss_*``) coupon, the knock-out values with (``breach_pay``) and without
        (``breach_miss``) it -- the price's joint coupon/KO/KI projection (or the nodal blend), then the knock-in."""
        barrier = self._coupon_barrier_for_obs(product, index)
        if self._use_cell_average_events():
            v_out, v_in = self._project_joint_phoenix_event(
                grid=grid, spot=spot, is_reverse=product.is_reverse,
                coupon_barrier=barrier,
                ko_barrier=record.barrier if ko_record is not None else None,
                ki_barrier=ki_record.barrier if ki_record is not None else None,
                val_miss_out=miss_out, val_pay_out=pay_out,
                val_miss_in=miss_in, val_pay_in=pay_in,
                ko_value_without_coupon=breach_miss, ko_value_with_coupon=breach_pay,
                disable_ko_after_ki=product.barrier_config.disable_ko_after_ki,
                pointwise=self._pointwise_events,
            )
            ki_record = None
        else:
            pay_weight = self._event_weight(
                grid, barrier, spot, width, trigger_is_down=product.is_reverse,
            )
            v_out = pay_weight * pay_out + (1.0 - pay_weight) * miss_out
            v_in = pay_weight * pay_in + (1.0 - pay_weight) * miss_in
            if ko_record is not None:
                breach = pay_weight * breach_pay + (1.0 - pay_weight) * breach_miss
                v_out = ko_weight * breach + (1.0 - ko_weight) * v_out
                if not product.barrier_config.disable_ko_after_ki:
                    v_in = ko_weight * breach + (1.0 - ko_weight) * v_in
            if ki_record is not None:
                v_out = self._blend_ki_transition(
                    v_out, v_in, grid, spot_grid, ki_record.barrier, spot,
                    width, product.is_reverse, ko_weight=ko_weight,
                )
                ki_record = None
        return super()._apply_observation(
            product, grid, spot_grid, spot, width, v_out, v_in,
            ki_record=ki_record, ki_barrier_continuous=ki_barrier_continuous,
        )

    def _instant_ko_records(self, ko_record, shared: dict, reachable: dict):
        """Every record of the instant (D14: each is its own coupon date; an unreachable one keeps its place with its
        knock-out test inert)."""
        records = shared.get(id(ko_record), ())
        if len(records) < 2:
            return ko_record, None
        return records[0], tuple(records)

    def _valuation_ko_record(self, product, pricing_env, ko_records, ko_record_0, shared: dict):
        """The record a knock-out decided at the valuation instant pays: the product's zero-gap decision, by
        contract index (D14)."""
        decision = product.decide_observations_at_valuation(
            float(pricing_env.spot), pricing_env,
            knocked_in=bool(getattr(product, "_otc_lifecycle_knocked_in", False)),
        )
        if decision.knocked_out and decision.ko_index is not None:
            return ko_records[decision.ko_index]
        return ko_record_0

    def _group_branches(self, product, columns, group, *, write_coupons: bool):
        """Per record of a same-instant group (contract order), the function giving its branch values -- the
        continuation after a paid (``pay``) and a missed (``miss``) coupon, the knock-out values with (``breach_pay``)
        and without (``breach_miss``) it -- from the columns after it (stacked rows x points), as the single record's
        observation builds them (``_CouponCashRows.write_observation`` on the stacked route)."""
        cash_rows = self._coupon_cash_rows

        def branches(j, value, after):
            pay_out, pay_in = after.copy(), after.copy()
            miss_out, miss_in = after.copy(), after.copy()
            breach_miss = np.zeros_like(after) + value
            breach_pay = breach_miss.copy()
            if write_coupons and cash_rows is not None:
                cash_rows.write_observation(j, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss)
            return pay_out, miss_out, breach_pay, breach_miss

        return branches

    def _apply_group_observation(self, product, grid, spot_grid, spot, width, v_out, v_in, *, ko_group, ki_record,
                                 ki_barrier_continuous, write_coupons: bool):
        """The statistics' columns through the records of one env instant (0.16.0 D14 zero-gap rule, the price's
        ``_zero_gap_group`` on the stacked rows): record k, still alive, takes its coupon branch and then its knock-out
        branch; the knock-out wins a tie with a knock-in, which switches the regime of a note no record knocked out.
        Cell-average events: one pass of the piecewise projection whose branches are the records' chain on each
        region's indicators; nodal events: the records in reverse contract order, then the knock-in.
        ``ko_group``: ``(index, record, knock-out value)`` per record (barrier ``None``: its test is inert)."""
        reverse = bool(product.is_reverse)
        disable = product.barrier_config.disable_ko_after_ki
        branches = self._group_branches(product, v_out, ko_group, write_coupons=write_coupons)
        v_out = np.asarray(v_out, dtype=float)
        v_in = np.asarray(v_in, dtype=float)

        if self._use_cell_average_events():
            breaks, regions = self._group_regions(product, spot, ko_group, ki_record)

            def chain(after, coupon_hits, ko_hits, ko_active):
                for pos in range(len(ko_group) - 1, -1, -1):
                    j, _, value = ko_group[pos]
                    pay, miss, breach_pay, breach_miss = branches(j, value, after)
                    if ko_active and ko_hits[pos]:
                        after = breach_pay if coupon_hits[pos] else breach_miss
                    else:
                        after = pay if coupon_hits[pos] else miss
                return after

            out_branches, in_branches = [], []
            for coupon_hits, ko_hits, knocked_in in regions:
                out_branches.append(chain(v_in if knocked_in else v_out, coupon_hits, ko_hits, True))
                in_branches.append(chain(v_in, coupon_hits, ko_hits, not disable))
            project = pointwise_piecewise_event if self._pointwise_events else project_piecewise_event
            stacked = v_out.ndim > 1

            def projected(branch_list):
                arrays = [np.moveaxis(np.asarray(b), -1, 0) for b in branch_list] if stacked else branch_list
                result = project(grid, breaks, arrays)
                return np.moveaxis(result, 0, -1) if stacked else result

            v_out, v_in = projected(out_branches), projected(in_branches)
            ki_record = None
        else:
            survive = np.ones_like(np.asarray(grid, dtype=float))
            for pos in range(len(ko_group) - 1, -1, -1):
                j, rec, value = ko_group[pos]
                pay_out, miss_out, breach_pay, breach_miss = branches(j, value, v_out)
                pay_in, miss_in, _, _ = branches(j, value, v_in)
                pay_weight = self._event_weight(grid, self._coupon_barrier_for_obs(product, j), spot, width,
                                                trigger_is_down=reverse)
                ko_weight = (self._event_weight(grid, rec.barrier, spot, width, trigger_is_down=reverse)
                             if rec.barrier is not None else np.zeros_like(survive))
                v_out = pay_weight * pay_out + (1.0 - pay_weight) * miss_out
                v_in = pay_weight * pay_in + (1.0 - pay_weight) * miss_in
                breach = pay_weight * breach_pay + (1.0 - pay_weight) * breach_miss
                v_out = ko_weight * breach + (1.0 - ko_weight) * v_out
                if not disable:
                    v_in = ko_weight * breach + (1.0 - ko_weight) * v_in
                survive = survive * (1.0 - ko_weight)
            if ki_record is not None:
                v_out = self._blend_ki_transition(
                    v_out, v_in, grid, spot_grid, ki_record.barrier, spot, width, reverse, ko_weight=1.0 - survive,
                )
                ki_record = None
        return SnowballQuadEngine._apply_observation(
            self, product, grid, spot_grid, spot, width, v_out, v_in,
            ki_record=ki_record, ki_barrier_continuous=ki_barrier_continuous,
        )

    def _group_regions(self, product, spot, ko_group, ki_record):
        """``(breaks, regions)`` of a same-instant group's piecewise event: the log breaks (every record's coupon
        barrier and active knock-out barrier, a discrete knock-in) and per region the records' coupon and knock-out
        indicators and whether the knock-in hits there."""
        reverse = bool(product.is_reverse)

        def log_of(level):
            return safe_log(float(level) / float(spot))

        levels = [self._coupon_barrier_for_obs(product, j) for j, _, _ in ko_group]
        levels += [float(rec.barrier) for _, rec, _ in ko_group if rec.barrier is not None]
        if ki_record is not None:
            levels.append(float(ki_record.barrier))
        breaks = sorted({log_of(level) for level in levels if level > 0.0})
        samples = ([breaks[0] - 1.0] + [0.5 * (a + b) for a, b in zip(breaks[:-1], breaks[1:])]
                   + [breaks[-1] + 1.0])

        def hit(sample, level):
            return sample <= log_of(level) if reverse else sample >= log_of(level)

        regions = []
        for sample in samples:
            coupon_hits = tuple(hit(sample, self._coupon_barrier_for_obs(product, j)) for j, _, _ in ko_group)
            ko_hits = tuple(rec.barrier is not None and hit(sample, rec.barrier) for _, rec, _ in ko_group)
            knocked_in = ki_record is not None and (
                sample >= log_of(ki_record.barrier) if reverse else sample <= log_of(ki_record.barrier))
            regions.append((coupon_hits, ko_hits, bool(knocked_in)))
        return breaks, regions

    def _group_coupon_step(self, product, grid, spot_grid, spot, group, ki_record, ki_barrier_continuous, pre,
                           pointwise: bool):
        """Forward route, a same-instant group of coupon observations (``CouponGroupStep``): the regions' projections
        and indicators of ``_apply_group_observation``'s event, the continuous knock-in's nodes and the termination
        row after the instant."""
        if not self._use_cell_average_events():
            # the nodal events blend the records with fractional weights: the region decomposition the forward
            # coupon extension transposes does not exist; the statistics take the stacked route (AdjointUnavailable)
            return NodalGroupStep()
        breaks, regions = self._group_regions(product, spot, group, ki_record)
        project_fn = pointwise_piecewise_event if pointwise else project_piecewise_event
        n_regions = len(regions)

        def region_projection(r):
            def project(values):
                values = np.asarray(values, dtype=float)
                stacked = values.ndim > 1
                v = np.moveaxis(values, -1, 0) if stacked else values
                result = project_fn(grid, breaks, [v if i == r else 0.0 for i in range(n_regions)])
                return np.moveaxis(result, 0, -1) if stacked else result
            return project

        ki_mask = None
        if ki_barrier_continuous is not None:
            ki_mask = (spot_grid >= ki_barrier_continuous if product.is_reverse
                       else spot_grid <= ki_barrier_continuous).astype(float)
        cash_rows = self._coupon_cash_rows
        w_out = w_in = None
        if cash_rows.expiry:
            w = cash_rows.termination_row
            w_out, w_in = np.array(pre[0][w], dtype=float), np.array(pre[1][w], dtype=float)
        return CouponGroupStep(
            indices=tuple(int(j) for j, _, _ in group),
            regions=tuple((region_projection(r), coupon_hits, ko_hits, knocked_in)
                          for r, (coupon_hits, ko_hits, knocked_in) in enumerate(regions)),
            disable=bool(product.barrier_config.disable_ko_after_ki), ki_mask=ki_mask, w_out=w_out, w_in=w_in,
        )

    #: the coupon layout of the statistics call in progress (``_compute_event_stats``)
    _coupon_cash_rows = None

    def _n_extra_quad_rows(self, n_ko: int) -> int:
        # stacked: one coupon-trigger indicator row per observation, then the coupon-cash columns; forward: W (EXPIRY)
        cash_rows = self._coupon_cash_rows
        return 0 if cash_rows is None else cash_rows.n_extra_rows

    def _initialize_extra_quad_rows(self, v_out, v_in, n_ko: int) -> None:
        cash_rows = self._coupon_cash_rows
        if cash_rows is not None:
            cash_rows.initialize(v_out, v_in)

    def _statistics_march_extension(self):
        """Forward route: the coupon columns ride the adjoint forward pass (spec D7)."""
        cash_rows = self._coupon_cash_rows
        return cash_rows if isinstance(cash_rows, ForwardCouponRows) else None

    def _statistics_step_extension(self, product, grid, spot_grid, spot, width, *, ko_record, ko_weight, ki_record,
                                   ki_barrier_continuous, coupon_record, observation_index, pre, pointwise=False,
                                   group=None):
        """Forward route, a coupon observation: its branch-resolved event map and the ``W`` row before it."""
        cash_rows = self._coupon_cash_rows
        if not isinstance(cash_rows, ForwardCouponRows) or coupon_record is None:
            return None
        if group is not None:
            return self._group_coupon_step(product, grid, spot_grid, spot, group, ki_record, ki_barrier_continuous,
                                           pre, pointwise)

        def branches(pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss):
            arrays = [np.array(a, dtype=float) for a in (pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss)]
            return self._project_observation_branches(
                product, grid, spot_grid, spot, width, *arrays,
                ko_record=ko_record, ko_weight=ko_weight, ki_record=ki_record,
                ki_barrier_continuous=ki_barrier_continuous, record=coupon_record, index=observation_index,
            )

        return cash_rows.step(observation_index, self._pointwise_event(branches) if pointwise else branches, pre)

    def _observation_levels(self, product, ko_records, ki_records) -> list:
        return super()._observation_levels(product, ko_records, ki_records) + [
            (r.observation_time, self._coupon_barrier_for_obs(product, i)) for i, r in enumerate(ko_records)
        ]

    def _first_event_extra_levels(self, product, observation_index) -> list:
        return [] if observation_index is None else [self._coupon_barrier_for_obs(product, observation_index)]

    def _coupon_barrier_for_obs(self, product, obs_index: int) -> float:
        cb = product.coupon_config.coupon_barrier
        if isinstance(cb, list):
            return float(cb[obs_index])
        return float(cb)

    def _coupon_columns(self, values_at_spot, n_ko, ko_records, product, pricing_env, df_fn, *,
                        forward=None) -> dict:
        """The coupon fields (``coupon_table``'s keywords): from the forward pass's masses on the forward route, else
        from the coupon columns read at the spot."""
        cash_rows = self._coupon_cash_rows
        if isinstance(cash_rows, ForwardCouponRows):
            return cash_rows.coupon_fields(forward.extension, values_at_spot, ko_records, df_fn)
        coupon_probability = np.zeros(n_ko, dtype=float)
        for i, rec in enumerate(ko_records):
            # Coupon row was set to 1.0 (undiscounted indicator) at the observation,
            # so ed_coup = E[DF(0->obs) * 1{coupon, alive}].
            df_obs = float(df_fn(float(rec.observation_time)))
            if df_obs > 0.0:
                coupon_probability[i] = float(values_at_spot[cash_rows.indicator_row + i] / df_obs)
        termination_at_spot = (
            float(values_at_spot[cash_rows.termination_row]) if cash_rows.expiry else None
        )
        if cash_rows.decided_index is not None:
            # the coupon decided at the valuation instant is paid with certainty
            coupon_probability[cash_rows.decided_index] = 1.0
        # Each row's coupon paid as the price pays it: memory arrears (released by a coupon or by a
        # knock-out) included, EXPIRY coupons discounted at the note's own end.
        cash = cash_rows.readout(values_at_spot, termination_at_spot)
        earned = buckets = None
        if cash_rows.expiry:
            buckets = cash_rows.readout_buckets(values_at_spot)
            if cash_rows.memory:
                df_row = np.array([float(df_fn(float(r.observation_time))) for r in ko_records], dtype=float)
                earned = cash_rows.readout_earned(values_at_spot) / df_row
            else:
                # without memory a row pays exactly its own coupon when its barrier is met
                earned = np.asarray(cash_rows.amounts, dtype=float) * coupon_probability
        return dict(probability=coupon_probability, cash=cash, expiry=bool(cash_rows.expiry), earned=earned,
                    buckets=buckets)

    def __repr__(self):
        return "PhoenixQuadEngine()"

    @staticmethod
    def _project_joint_phoenix_event(
        *,
        grid: np.ndarray,
        spot: float,
        is_reverse: bool,
        coupon_barrier: float,
        ko_barrier: Optional[float],
        ki_barrier: Optional[float],
        val_miss_out: np.ndarray,
        val_pay_out: np.ndarray,
        val_miss_in: np.ndarray,
        val_pay_in: np.ndarray,
        ko_value_without_coupon: float,
        ko_value_with_coupon: float,
        disable_ko_after_ki: bool,
        pointwise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-pass dual-cell projection of a Phoenix observation.

        Coupon, KO, and an optional coincident discrete-KI transition are
        resolved as one piecewise function. This avoids sequentially averaging
        the same threshold cell and preserves the contractual precedence: a
        knock-out overrides a knock-in observed at the same instant, also when
        the knock-out is disabled after a knock-in (that flag suppresses the
        knock-outs AFTER the knock-in observation, not the one tied with it).
        """
        thresholds = [float(coupon_barrier)]
        if ko_barrier is not None:
            thresholds.append(float(ko_barrier))
        if ki_barrier is not None:
            thresholds.append(float(ki_barrier))
        breaks = sorted(
            {
                safe_log(level / float(spot))
                for level in thresholds
                if level > 0.0
            }
        )

        def region_samples() -> list[float]:
            if not breaks:
                return [0.0]
            samples = [breaks[0] - 1.0]
            samples.extend(
                0.5 * (left + right)
                for left, right in zip(breaks[:-1], breaks[1:])
            )
            samples.append(breaks[-1] + 1.0)
            return samples

        coupon_log = safe_log(float(coupon_barrier) / float(spot))
        ko_log = (
            safe_log(float(ko_barrier) / float(spot))
            if ko_barrier is not None
            else None
        )
        ki_log = (
            safe_log(float(ki_barrier) / float(spot))
            if ki_barrier is not None
            else None
        )

        out_branches = []
        in_branches = []
        for sample in region_samples():
            coupon_hit = sample <= coupon_log if is_reverse else sample >= coupon_log
            ko_hit = (
                (sample <= ko_log if is_reverse else sample >= ko_log)
                if ko_log is not None
                else False
            )
            ki_hit = (
                (sample >= ki_log if is_reverse else sample <= ki_log)
                if ki_log is not None
                else False
            )

            continue_out = val_pay_out if coupon_hit else val_miss_out
            continue_in = val_pay_in if coupon_hit else val_miss_in
            ko_value = (
                ko_value_with_coupon
                if coupon_hit
                else ko_value_without_coupon
            )

            if ko_hit and not disable_ko_after_ki:
                branch_in = ko_value
            else:
                branch_in = continue_in

            if ko_hit:
                # the knock-out wins a tie with a knock-in observed at the same instant, whether
                # or not the knocked-in regime keeps its knock-out
                branch_out = ko_value
            elif ki_hit:
                branch_out = continue_in
            else:
                branch_out = continue_out

            out_branches.append(branch_out)
            in_branches.append(branch_in)

        stacked = np.asarray(val_miss_out).ndim > 1

        def project(branches):
            if stacked:
                branches = [np.moveaxis(np.asarray(branch), -1, 0) for branch in branches]
            result = (pointwise_piecewise_event if pointwise else project_piecewise_event)(grid, breaks, branches)
            return np.moveaxis(result, 0, -1) if stacked else result

        return project(out_branches), project(in_branches)

    def _select_alignment_log(
        self,
        spot: float,
        ko_records: Sequence,
        coupon_barriers: Optional[np.ndarray],
        ki_records: Sequence,
        product: PhoenixOption,
    ) -> Optional[float]:
        ko_candidates: list[float] = []
        for rec in ko_records:
            if rec.barrier is not None and rec.barrier > 0:
                ko_candidates.append(float(rec.barrier))

        coupon_candidates: list[float] = []
        if coupon_barriers is not None:
            coupon_candidates.extend([float(b) for b in coupon_barriers if b > 0])

        ki_candidates: list[float] = []
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                ki_candidates.extend([float(b) for b in ki_barrier if b > 0])
            elif ki_barrier is not None and ki_barrier > 0:
                ki_candidates.append(float(ki_barrier))

        def to_logs(candidates: list[float]) -> list[float]:
            logs = []
            for b in candidates:
                try:
                    logs.append(safe_log(b / spot))
                except Exception:
                    continue
            return logs

        ko_logs = to_logs(ko_candidates)
        coupon_logs = to_logs(coupon_candidates)
        ki_logs = to_logs(ki_candidates)

        if not ko_logs and not coupon_logs and not ki_logs:
            return None

        def closest(logs: list[float]) -> Optional[float]:
            if not logs:
                return None
            idx = int(np.argmin(np.abs(np.asarray(logs))))
            return float(logs[idx])

        priority = self._resolve_align_priority()
        if priority == "ko":
            return closest(ko_logs) or closest(coupon_logs) or closest(ki_logs)
        if priority == "coupon":
            return closest(coupon_logs) or closest(ko_logs) or closest(ki_logs)
        if priority == "ki":
            return closest(ki_logs) or closest(ko_logs) or closest(coupon_logs)

        # auto: reverse prefers KO near spot, else coupon, else KI
        if product.is_reverse:
            ko_near = closest(ko_logs)
            if ko_near is not None and abs(ko_near) <= 0.05:
                return ko_near
            return closest(coupon_logs) or closest(ki_logs) or closest(ko_logs)

        return closest(ko_logs + coupon_logs + ki_logs)

    # _resolve_align_priority / _resolve_fft_padding_factor /
    # _resolve_fft_filter / _smooth_step_weight / _smooth_step_weight_log are
    # inherited from SnowballQuadEngine.

    def _resolve_event_smoothing_width(
        self, math_utils: QuadratureMath, product: PhoenixOption
    ) -> float:
        mode = getattr(self.params, "event_smoothing_mode", "fixed")
        cells = getattr(self.params, "event_smoothing_cells", 0)
        kernel_width = getattr(self.params, "event_smoothing_log_width", 0.002)

        try:
            cells = int(cells)
        except (TypeError, ValueError):
            cells = 0

        if str(mode).lower() == "reverse_aware" and product.is_reverse:
            cells = 0
        elif str(mode).lower() == "auto":
            h = float(math_utils.h)
            cells = max(1, int(0.5 + float(kernel_width) / h))

        if cells <= 0:
            return 0.0
        return float(cells) * float(math_utils.h)
