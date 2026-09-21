"""Compile QuantArk contractual rules into numerical state/event maps."""

import math
import numpy as np

from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.util.enum import (
    ObservationType,
    CouponPayType,
    ProtectionType,
    PostKOScheduleMode,
)
from quantark.util.exceptions import ValidationError, NumericalError
from .contract import Action, Event, CompiledContract


def terminal_breaks(product, env):
    """All payoff regions, including alternate airbag strikes and loss caps."""
    p, air = product.payoff_config, product.airbag_config
    levels = [product.strike, p.call_strike, air.airbag_barrier, air.airbag_strike]
    annual = 1.0
    if (
        not isinstance(product, PhoenixOption)
        and product.accrual_config.is_annualized_ki
    ):
        annual = (
            product._resolve_post_contract_tenor(env)
            if isinstance(product, KnockOutResetSnowballOption)
            else product.get_contract_tenor(env)
        )
    if p.protection_type == ProtectionType.PARTIAL:
        for strike, part in (
            (product.strike, p.participation_rate),
            (air.airbag_strike or product.strike, air.airbag_participation_rate),
        ):
            if part * annual != 0:
                loss_width = p.protection_rate * product.initial_price / (part * annual)
                levels.append(
                    strike + loss_width if product.is_reverse else strike - loss_width
                )
    return sorted(
        {float(x) for x in levels if x is not None and x > 0 and math.isfinite(x)}
    )


def _affine_payoff(product, env, ki, lo, hi):
    """Extract a declared piecewise-affine payoff and check it inside its region."""
    lower = 0.0 if not np.isfinite(lo) else np.exp(lo)
    upper = (
        max(2 * lower, 2 * product.initial_price) if not np.isfinite(hi) else np.exp(hi)
    )
    s1, s2 = lower + (upper - lower) * 0.25, lower + (upper - lower) * 0.75
    payoff = product.get_maturity_payoff_v1 if ki else product.get_maturity_payoff_v0
    f1, f2 = payoff(float(s1), pricing_env=env), payoff(float(s2), pricing_env=env)
    asset = (f2 - f1) / (s2 - s1)
    cash = f1 - asset * s1
    for s in (
        lower + (upper - lower) * 0.1,
        (s1 + s2) / 2,
        lower + (upper - lower) * 0.9,
    ):
        actual = payoff(float(s), pricing_env=env)
        if abs(actual - (cash + asset * s)) > 1e-9 * max(
            1.0, abs(actual), abs(f1), abs(f2)
        ):
            raise ValidationError(
                "QUAD V2 payoff is not affine within declared terminal regions"
            )
    return float(cash), float(asset)


def _reset_records(product, env, config):
    records, rates, originals = product._resolve_ko_schedule(config, env)
    result = []
    terminal_time = resolve_terminal_timing(product, env).payment_time
    for record, rate, original in zip(records, rates, originals):
        amount = (
            product.initial_price
            * product.contract_multiplier
            * float(rate)
            * product.compute_ko_accrual_factor(record.observation_time, original, env)
        )
        if product.payoff_config.include_principal:
            amount += product.initial_price * product.contract_multiplier
        payment = record.settlement_time
        if product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY:
            payment = terminal_time
        result.append(
            (
                float(record.observation_time),
                float(record.barrier),
                float(amount),
                float(payment),
            )
        )
    return result


def compile_contract(
    product,
    env,
    params,
    *,
    components=False,
    event_phase="before",
    lifecycle_state=None,
    extra_ko_cash=(),
    extra_terminal_cash=0.0,
):
    """Compile only known product classes; unknown subclasses fail closed.

    ``extra_ko_cash`` carries ``(observation_time, amount)`` pairs added to
    the matching KO record's cash (path-independent contingent amounts such
    as folded cash legs); a pair whose time matches no record raises.
    ``extra_terminal_cash`` is added to the terminal payoff's cash component
    at maturity in both the alive and knocked-in branches.
    """
    if type(product) not in (
        SnowballOption,
        PhoenixOption,
        KnockOutResetSnowballOption,
    ):
        raise ValidationError(
            f"QUAD V2 has no contractual adapter for {type(product).__name__}"
        )
    if event_phase not in {"before", "after"}:
        raise ValidationError("event_phase must be before or after")
    product.validate()
    phoenix = isinstance(product, PhoenixOption)
    reset = isinstance(product, KnockOutResetSnowballOption)
    if reset and product.post_ko_mode is not PostKOScheduleMode.ABSOLUTE:
        raise NotImplementedError(
            "QUAD V2 supports absolute post-KI schedules; relative-to-hit resets need additional state"
        )
    maturity = float(product.get_maturity(env))
    if not math.isfinite(maturity) or maturity < 0:
        raise ValidationError("maturity must be finite and nonnegative")
    if reset:
        # A contract past its first schedule and not knocked in has matured (rule 2 of
        # KnockOutResetSnowballOption); only a knock-in carried into the valuation keeps it alive.
        product.require_alive(
            env,
            bool(
                getattr(
                    lifecycle_state,
                    "knocked_in",
                    getattr(product, "_otc_lifecycle_knocked_in", False),
                )
            ),
        )
        pre = _reset_records(product, env, product.barrier_config)
        post = _reset_records(product, env, product.post_barrier_config)
        pre_maturity = float(product.get_pre_maturity_time(env))
    else:
        records = product.resolve_ko_observations(env)
        pre = [
            (
                float(r.observation_time),
                float(r.barrier),
                float(r.payoff or 0.0),
                float(r.settlement_time),
            )
            for r in records
        ]
        post, pre_maturity = pre, maturity

    def is_future(t):
        return 0 <= t <= maturity and (event_phase == "before" or t > 0)

    pre = [r for r in pre if is_future(r[0])]
    post = [r for r in post if is_future(r[0])]
    continuous = product.has_ki_barrier and (
        product.barrier_config.ki_continuous
        or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
    )
    ki = (
        []
        if continuous or not product.has_ki_barrier
        else [
            (float(r.observation_time), float(r.barrier))
            for r in product.resolve_ki_observations(env)
            if is_future(float(r.observation_time))
        ]
    )
    if continuous and isinstance(product.barrier_config.ki_barrier, list):
        raise NotImplementedError("QUAD V2 continuous KI requires a constant barrier")

    all_times = sorted({maturity, pre_maturity} | {r[0] for r in pre + post + ki})
    continuous_model = None
    if continuous:
        from .continuous import continuous_time_grid

        all_times, continuous_model = continuous_time_grid(env, all_times, params)
    if len(all_times) > params.max_events:
        raise NumericalError("QUAD V2 event count exceeds max_events")
    # Numeric dates merge only under the explicitly requested clock tolerance.
    canonical, mapping = [], {}
    for t in all_times:
        if not np.isfinite(t) or t < 0:
            raise ValidationError("event times must be finite and nonnegative")
        value = (
            canonical[-1]
            if canonical and t - canonical[-1] <= params.event_time_tolerance
            else t
        )
        if not canonical or value != canonical[-1]:
            canonical.append(value)
        mapping[t] = value

    def keyed(rows):
        result = {}
        for r in rows:
            t = mapping[r[0]]
            if t in result:
                raise ValidationError(
                    "multiple observations of one kind share a canonical time"
                )
            result[t] = r
        return result

    if extra_ko_cash:
        pending_extras = {}
        for raw_time, raw_amount in extra_ko_cash:
            t_extra = float(raw_time)
            if t_extra not in mapping:
                raise ValidationError(
                    f"extra KO cash time {t_extra} is not a contract event time"
                )
            key = mapping[t_extra]
            pending_extras[key] = pending_extras.get(key, 0.0) + float(raw_amount)

        def with_extras(rows):
            return [
                (t, b, a + pending_extras.get(mapping[t], 0.0), p)
                for (t, b, a, p) in rows
            ]

        pre = with_extras(pre)
        post = with_extras(post)

    pre_by, post_by, ki_by = keyed(pre), keyed(post), keyed(ki)
    maturity, pre_maturity = mapping[maturity], mapping[pre_maturity]
    terminal = resolve_terminal_timing(product, env)
    channels = ("total", "ko", "coupon", "terminal") if components else ("total",)
    statistics = components == "events"
    ko_metadata = []
    if statistics:
        for regime, rows in (
            (("pre", pre), ("post", post)) if reset else (("pre", pre),)
        ):
            for r in rows:
                key = f"{regime}:{mapping[r[0]]!r}"
                ko_metadata.append((key, mapping[r[0]], r[3], regime))
                channels += (f"ko_cash:{key}", f"ko_probability:{key}")
                if phoenix:
                    channels += (
                        f"coupon_cash:{key}",
                        f"coupon_probability:{key}",
                        f"coupon_payment:{key}",
                    )
        channels += (
            "terminal_alive_cash",
            "terminal_ki_cash",
            "terminal_alive_probability",
            "terminal_ki_probability",
            "ki_ever",
        )
        if phoenix:
            channels += ("terminal_alive_coupon", "terminal_ki_coupon")
    n_channels = len(channels)
    zeros = (0.0,) * n_channels
    coupon_cash_channels = tuple(
        i for i, name in enumerate(channels) if name.startswith("coupon_cash:")
    )
    coupon_payment_channels = tuple(
        i
        for i, name in enumerate(channels)
        if name.startswith("coupon_payment:")
        or name in {"terminal_alive_coupon", "terminal_ki_coupon"}
    )

    def cash_vector(value, channel):
        out = [0.0] * n_channels
        out[0] = value
        if components:
            out[channels.index(channel)] = value
        return tuple(out)

    initial_memory = (
        getattr(lifecycle_state, "coupon_memory_count", 0) if phoenix else 0
    )
    if (
        isinstance(initial_memory, bool)
        or not isinstance(initial_memory, (int, np.integer))
        or initial_memory < 0
        or (initial_memory and not product.has_memory_coupon)
    ):
        raise ValidationError("invalid Phoenix coupon memory state")
    n_memory = (
        len(pre) + initial_memory + 1 if phoenix and product.has_memory_coupon else 1
    )
    expiry = phoenix and product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
    n_states = 2 * n_memory + (2 if expiry else 0)
    if n_states > params.max_states:
        raise NumericalError(
            f"QUAD V2 requires {n_states} states; max_states={params.max_states}"
        )
    breaks_terminal = terminal_breaks(product, env)
    max_regions = len(breaks_terminal) + 5
    event_bytes = (
        len(canonical) * n_states * (2 * max_regions + 1) * (128 + 16 * n_channels)
    )
    if event_bytes > params.max_work_bytes:
        raise NumericalError(
            f"QUAD V2 event tables require about {event_bytes} bytes; max_work_bytes={params.max_work_bytes}"
        )
    names = [
        f"{'ki' if hit else 'alive'}:{memory}"
        for hit in (False, True)
        for memory in range(n_memory)
    ]
    w_offset = len(names)
    if expiry:
        # Unit-at-termination values carry accrued coupons without another
        # earned-amount state dimension. Dedicated coupon channels stay independent.
        names += ["termination_alive", "termination_ki"]
    coupon_amounts, coupon_prefix, coupon_barriers = [], np.zeros(1), []
    if phoenix:
        # Resolve original schedule indices when earlier observations expired.
        raw = product.barrier_config.ko_observation_schedule.records
        active = [
            i
            for i, r in enumerate(raw)
            if not (
                (
                    r.observation_time is not None
                    and (
                        r.observation_time < 0
                        or (r.observation_time == 0 and event_phase == "after")
                    )
                )
                or (
                    r.observation_date is not None
                    and (
                        r.observation_date < env.valuation_date
                        or (
                            r.observation_date == env.valuation_date
                            and event_phase == "after"
                        )
                    )
                )
            )
        ]
        coupon_barriers = [product.get_coupon_barrier_at(i) for i in active]
        if len(coupon_barriers) != len(pre):
            raise ValidationError(
                "Phoenix coupon schedule does not align with active KO records"
            )
        factors = product.accrual_config.accrual_factors
        if factors is not None and len(factors) == len(raw):
            fractions = [float(factors[i]) for i in active]
        elif (
            factors is None
            and product.coupon_config.fixed_coupon_year_fraction is None
            and product.initial_date is not None
            and all(r.observation_date is not None for r in raw)
        ):
            dates = [product.initial_date] + [r.observation_date for r in raw]
            fractions = [
                product.get_coupon_year_fraction(dates[i], dates[i + 1]) for i in active
            ]
        elif (
            factors is None
            and product.coupon_config.fixed_coupon_year_fraction is None
            and active
            and active[0] > 0
            and all(r.observation_time is not None for r in raw)
        ):
            fractions = [
                float(raw[i].observation_time - raw[i - 1].observation_time)
                for i in active
            ]
        else:
            fractions = product.get_coupon_period_year_fractions([r[0] for r in pre])
        coupon_amounts = [
            product.get_coupon_payoff(i, year_fraction=f)
            for i, f in enumerate(fractions)
        ]
        historic = []
        if initial_memory:
            fixed = product.coupon_config.fixed_coupon_year_fraction
            if fixed is not None:
                historic = [
                    product.get_coupon_payoff(0, year_fraction=fixed)
                ] * initial_memory
            elif (
                active
                and active[0] >= initial_memory
                and product.initial_date is not None
                and all(r.observation_date is not None for r in raw)
            ):
                dates = [product.initial_date] + [r.observation_date for r in raw]
                start = active[0] - initial_memory
                historic = [
                    product.get_coupon_payoff(
                        i, start_date=dates[i], end_date=dates[i + 1]
                    )
                    for i in range(start, active[0])
                ]
            else:
                raise NotImplementedError(
                    "aged Phoenix memory requires fixed_coupon_year_fraction or retained historical coupon dates"
                )
        coupon_prefix = np.r_[0.0, np.cumsum(historic + coupon_amounts)]
    obs_index = {mapping[r[0]]: i for i, r in enumerate(pre)}
    disable = product.barrier_config.disable_ko_after_ki
    initial_hit = bool(
        getattr(
            lifecycle_state,
            "knocked_in",
            getattr(product, "_otc_lifecycle_knocked_in", False),
        )
    )
    events = []
    for t in canonical:
        if t == 0 and event_phase == "after":
            continue
        pr, po, kr = pre_by.get(t), post_by.get(t), ki_by.get(t)
        if continuous:
            kr = (t, float(product.barrier_config.ki_barrier))
        levels = [r[1] for r in (pr, po, kr) if r is not None]
        index = obs_index.get(t)
        coupon_key = f"pre:{t!r}"
        if phoenix and index is not None:
            levels.append(coupon_barriers[index])
        if t in (maturity, pre_maturity):
            levels += breaks_terminal
        if any(not np.isfinite(b) or b <= 0 for b in levels):
            raise ValidationError(
                "QUAD V2 barriers and payoff breaks must be positive and finite"
            )
        breaks = tuple(sorted({float(np.log(b)) for b in levels}))
        edges = (-np.inf,) + breaks + (np.inf,)
        sample_logs = [
            (
                b - 1
                if not np.isfinite(a)
                else a + 1
                if not np.isfinite(b)
                else (a + b) / 2
            )
            for a, b in zip(edges[:-1], edges[1:])
        ]
        if not breaks:
            sample_logs = [np.log(product.initial_price)]
        point_spots = {float(np.log(level)): float(level) for level in levels}

        def action(state, x, lo, hi, point=False):
            is_w = state >= w_offset
            hit = bool(state - w_offset) if is_w else state >= n_memory
            mem = 0 if is_w else state % n_memory
            ki_hit = kr is not None and (
                x >= np.log(kr[1]) if product.is_reverse else x <= np.log(kr[1])
            )
            new_hit = hit or ki_hit
            record = po if hit and reset else pr
            ko_hit = record is not None and (
                x <= np.log(record[1]) if product.is_reverse else x >= np.log(record[1])
            )
            # ``disable`` suppresses the knock-outs AFTER a knock-in, read on the state carried INTO
            # this observation: a knock-out wins a tie with a knock-in observed at the same instant,
            # on every product (the Phoenix used to let the knock-in win under the flag).
            ko_hit = ko_hit and not (hit and disable)
            if reset and not hit and ki_hit and not ko_hit:
                record = po
                ko_hit = record is not None and (
                    x <= np.log(record[1])
                    if product.is_reverse
                    else x >= np.log(record[1])
                )
            unit_df = (
                env.get_discount_factor(record[3]) / env.get_discount_factor(t)
                if ko_hit
                else terminal.delay_df
            )
            if is_w:
                if ko_hit or t == maturity:
                    c = np.array(cash_vector(unit_df, "coupon"))
                    if statistics:
                        # Every origin shares the same termination discount.
                        # The earning action selects only its own origin row.
                        c[list(coupon_cash_channels)] = unit_df
                        payment_channel = (
                            f"coupon_payment:{coupon_key}"
                            if ko_hit
                            else f"terminal_{'ki' if new_hit else 'alive'}_coupon"
                        )
                        c[channels.index(payment_channel)] = unit_df
                    return Action(cash=tuple(c), asset=zeros)
                return Action(
                    terms=((w_offset + int(new_hit), 1.0),), cash=zeros, asset=zeros
                )
            coupon_hit = (
                phoenix
                and index is not None
                and (
                    x <= np.log(coupon_barriers[index])
                    if product.is_reverse
                    else x >= np.log(coupon_barriers[index])
                )
            )
            coupon_index = (index + initial_memory) if index is not None else None
            missed = (
                float(
                    coupon_prefix[coupon_index]
                    - coupon_prefix[max(0, coupon_index - mem)]
                )
                if phoenix and index is not None
                else 0.0
            )
            earned = (coupon_amounts[index] + missed) if coupon_hit else 0.0

            def coupon_cash(value, payment_channel):
                c = np.array(cash_vector(value, "coupon"))
                if statistics:
                    c[channels.index(f"coupon_cash:{coupon_key}")] = value
                    c[channels.index(payment_channel)] = value
                return c

            if ko_hit:
                c = np.array(cash_vector(record[2] * unit_df, "ko"))
                if statistics:
                    regime = (
                        "post"
                        if reset
                        and (hit or (ki_hit and record is po and record is not pr))
                        else "pre"
                    )
                    key = f"{regime}:{mapping[record[0]]!r}"
                    c[channels.index(f"ko_cash:{key}")] = record[2] * unit_df
                    c[channels.index(f"ko_probability:{key}")] = (
                        1 / env.get_discount_factor(t)
                    )
                    if not hit and ki_hit and (regime == "post" or phoenix):
                        c[channels.index("ki_ever")] = 1 / env.get_discount_factor(t)
                if phoenix:
                    c += coupon_cash(
                        (missed + (coupon_amounts[index] if coupon_hit else 0.0))
                        * unit_df,
                        f"coupon_payment:{coupon_key}",
                    )
                    if statistics and coupon_hit:
                        c[channels.index(f"coupon_probability:{coupon_key}")] = (
                            1 / env.get_discount_factor(t)
                        )
                return Action(cash=tuple(c), asset=zeros)
            next_mem = (
                (0 if coupon_hit else min(mem + 1, n_memory - 1))
                if phoenix and index is not None
                else mem
            )
            end = maturity if new_hit else pre_maturity
            c, a, terms = np.zeros(n_channels), np.zeros(n_channels), []
            channel_terms = []
            if t == end:
                if point:
                    payoff = (
                        product.get_maturity_payoff_v1
                        if new_hit
                        else product.get_maturity_payoff_v0
                    )
                    tc, ta = payoff(point_spots[x], pricing_env=env), 0.0
                else:
                    tc, ta = _affine_payoff(product, env, new_hit, lo, hi)
                if extra_terminal_cash and t == maturity:
                    tc += extra_terminal_cash
                # KO-reset's pre branch can end before the post-KI branch.
                delay = (
                    terminal.delay_df
                    if t == maturity
                    else env.get_discount_factor(
                        t + terminal.payment_time - terminal.determination_time
                    )
                    / env.get_discount_factor(t)
                )
                c += cash_vector(tc * delay, "terminal")
                a += cash_vector(ta * delay, "terminal")
                if statistics:
                    suffix = "ki" if new_hit else "alive"
                    c[channels.index(f"terminal_{suffix}_cash")] = tc * delay
                    a[channels.index(f"terminal_{suffix}_cash")] = ta * delay
                    c[channels.index(f"terminal_{suffix}_probability")] = (
                        1 / env.get_discount_factor(t)
                    )
            elif t < end:
                terms.append((int(new_hit) * n_memory + next_mem, 1.0))
            if statistics and not hit and ki_hit:
                c[channels.index("ki_ever")] = 1 / env.get_discount_factor(t)
            if statistics and coupon_hit:
                c[channels.index(f"coupon_probability:{coupon_key}")] = (
                    1 / env.get_discount_factor(t)
                )
            if earned:
                if expiry and t < end:
                    if statistics:
                        selected = (
                            0,
                            channels.index("coupon"),
                            channels.index(f"coupon_cash:{coupon_key}"),
                            *coupon_payment_channels,
                        )
                        channel_terms.append(
                            (w_offset + int(new_hit), earned, selected)
                        )
                    else:
                        terms.append((w_offset + int(new_hit), earned))
                else:
                    coupon_df = (
                        terminal.delay_df
                        if expiry
                        else env.get_discount_factor(pr[3]) / env.get_discount_factor(t)
                    )
                    payment_channel = (
                        f"terminal_{'ki' if new_hit else 'alive'}_coupon"
                        if expiry
                        else f"coupon_payment:{coupon_key}"
                    )
                    c += coupon_cash(earned * coupon_df, payment_channel)
            return Action(tuple(terms), tuple(c), tuple(a), tuple(channel_terms))

        rows, boundaries = [], []
        for state in range(len(names)):
            row = tuple(
                action(state, x, a, b)
                for x, a, b in zip(sample_logs, edges[:-1], edges[1:])
            )
            # At a payoff kink either affine side has the same PV; at an
            # airbag jump use the side chosen by the product at equality.
            boundary = []
            for j, b in enumerate(breaks):
                chosen = action(state, b, edges[j], edges[j + 1], point=True)
                boundary.append(chosen)
            rows.append(row)
            boundaries.append(tuple(boundary))
        events.append(Event(t, breaks, tuple(rows), tuple(boundaries)))
    if not events:
        events = [
            Event(0.0, (), tuple((Action(cash=zeros, asset=zeros),) for _ in names))
        ]
    return CompiledContract(
        tuple(events),
        tuple(names),
        int(initial_hit) * n_memory + initial_memory,
        channels,
        float(product.initial_price),
        float(np.log(product.barrier_config.ki_barrier)) if continuous else None,
        bool(product.is_reverse),
        tuple((i, n_memory + i) for i in range(n_memory))
        + (((w_offset, w_offset + 1),) if expiry else ()),
        {
            "product": type(product).__name__,
            "canonical_times": tuple(mapping.items()),
            "event_phase": event_phase,
            "continuous": continuous,
            "continuous_model": continuous_model,
            "ko_metadata": tuple(ko_metadata),
            "pre_maturity": pre_maturity,
            "maturity": maturity,
            "terminal_delay": terminal.payment_time - terminal.determination_time,
            "initial_ki": initial_hit,
            "expiry_coupons": expiry,
        },
    )
