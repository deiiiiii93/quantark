"""Compile QuantArk contractual rules into numerical state/event maps."""

import math
from copy import copy
import numpy as np

from quantark.asset.equity.product.option.accrual import leg_annualized
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.engine.settlement_support import (
    first_schedule_payment_time,
    first_schedule_settlement,
    knocked_in_row_timing,
    knocked_in_settlement,
    resolve_terminal_timing,
)
from quantark.util.enum import (
    ObservationType,
    CouponPayType,
    ProtectionType,
    PostKOScheduleMode,
)
from quantark.asset.equity.product.option.observation_schedule import same_instant_ki_groups, same_instant_ko_groups
from quantark.util.exceptions import ValidationError, NumericalError
from quantark.util.numerical import is_close
from .contract import Action, Event, CompiledContract

#: the refusal of two observations of one kind on one canonical time the contract does not order (a knock-in pair, a
#: KO-reset pair, or records a user ``event_time_tolerance`` merges although the contract clock keeps them apart)
_SHARED_TIME = "multiple observations of one kind share a canonical time"


def terminal_breaks(product, env):
    """All payoff regions, including alternate airbag strikes and loss caps."""
    p, air = product.payoff_config, product.airbag_config
    levels = [product.strike, p.call_strike, air.airbag_barrier, air.airbag_strike]
    annual = 1.0
    if (
        not isinstance(product, PhoenixOption)
        and leg_annualized(product, "knock_in")
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


def _expiry_payment(product, env, config):
    """``(time, date)`` an EXPIRY-pay knock-out of ``config``'s schedule is paid at: the end of the note it ends --
    schedule 1's maturity plus the terminal delay for a first-schedule knock-out (a path never knocked in ends there,
    spec D30), dated as the never-knocked-in maturity row dates it; for a second-schedule one the knocked-in note's
    end -- schedule 2's last fixing's settlement (``knocked_in_settlement``, D31), the product terminal when schedule 2
    ends with the product."""
    terminal = resolve_terminal_timing(product, env)
    if config is not product.barrier_config:
        knocked_in = knocked_in_settlement(product, env, terminal)
        if knocked_in is None:
            return terminal.payment_time, terminal.payment_date
        return knocked_in.payment_time, knocked_in_row_timing(product, env, terminal, knocked_in)["payment_date"]
    time = first_schedule_payment_time(product, env, terminal)
    if time == terminal.payment_time:
        return time, terminal.payment_date              # one maturity: the product terminal's own
    records = product._resolve_ko_schedule(config, env)[0]
    last = max(records, key=lambda r: float(r.observation_time)) if records else None
    delay = float(terminal.payment_time) - float(terminal.determination_time)
    date = (last.observation_date if last is not None and is_close(delay, 0.0) else None)
    return time, date


def _reset_records(product, env, config):
    records, rates, originals = product._resolve_ko_schedule(config, env)
    result = []
    terminal_time = (
        _expiry_payment(product, env, config)[0]
        if product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY
        else None
    )
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


def _reset_record_dates(product, env, config):
    """Raw observation time -> (observation date, payment date) of a KO-reset schedule's records (under EXPIRY pay
    the payment date of the note's end, as ``_reset_records`` pays it: ``_expiry_payment``)."""
    records = product._resolve_ko_schedule(config, env)[0]
    expiry = product.accrual_config.coupon_pay_type == CouponPayType.EXPIRY
    expiry_date = _expiry_payment(product, env, config)[1] if expiry else None
    return {
        float(record.observation_time): (
            record.observation_date,
            expiry_date if expiry else record.settlement_date,
        )
        for record in records
    }


def _bind_fold(fold, reset, pre, post, terminal, maturity, pre_maturity, no_ki_payment=None, ki_timing=None):
    """Add a fold overlay's KO cash to the records; return the per-branch terminal cash.

    Records are ``(observation_time, barrier, cash_at_settlement, payment_time)``.
    Times are the raw contractual ones (before canonical merging), the timing the
    overlay was built on. The alive branch of a KO-reset ends at its pre-maturity
    and settles after the same terminal delay as the knocked-in branch, or on the
    explicit settlement date when one governs (``no_ki_payment``, spec D33). The knocked-in branch ends at maturity
    and settles with the terminal, or at ``ki_timing`` ``(determination, payment)`` when a KO-reset's schedule 2 ends
    before the maturity (D31).
    """
    schedules = (("pre", pre), ("post", post)) if reset else (("main", pre),)
    records = [(name, r[0], r[3]) for name, rows in schedules for r in rows]
    amounts = iter(fold.bind_ko(records))
    folded = [
        [(t, b, a + float(next(amounts)), p) for (t, b, a, p) in rows]
        for _, rows in schedules
    ]
    new_pre = folded[0]
    new_post = folded[1] if reset else new_pre
    delay = terminal.payment_time - terminal.determination_time
    branch_terminal = {
        "no_ki": float(fold.bind_terminal(
            "no_ki", pre_maturity, pre_maturity + delay if no_ki_payment is None else no_ki_payment)),
        "ki": float(fold.bind_terminal("ki", maturity, terminal.payment_time) if ki_timing is None
                    else fold.bind_terminal("ki", *ki_timing)),
    }
    return new_pre, new_post, branch_terminal


def _ko_instants(records, mapping, is_reverse, phoenix):
    """The canonical times holding two or more knock-out records of a Snowball / Phoenix (0.16.0 DCN/dated run, D2/D7/
    D14): ``{canonical time: (positions, chain, effective)}``, indices into ``records`` (the future records, contract
    order) -- ``positions`` every record of the instant, ``chain`` the records its event applies at the same spot in
    contract order (Snowball: the effective records of ``same_instant_ko_groups``, a dominated record dropped; Phoenix:
    every record, each its own coupon date), ``effective`` the records whose knock-out can fire first.

    Raises:
        ValidationError: a canonical time holds records the contract clock keeps apart (D2: the helper's group is the
            unit; a user ``event_time_tolerance`` that merges two instants keeps V2's refusal).
    """
    by_time = {}
    for pos, record in enumerate(records):
        by_time.setdefault(mapping[float(record.observation_time)], []).append(pos)
    out = {}
    for t, positions in by_time.items():
        if len(positions) < 2:
            continue
        groups = same_instant_ko_groups([records[p] for p in positions], is_reverse)
        if len(groups) != 1:
            raise ValidationError(
                f"{_SHARED_TIME}: event_time_tolerance merges knock-out records the contract clock keeps apart"
            )
        kept = {id(r) for r in groups[0]}
        effective = tuple(p for p in positions if id(records[p]) in kept and records[p].barrier is not None)
        chain = tuple(positions) if phoenix else tuple(p for p in positions if id(records[p]) in kept)
        out[t] = (tuple(positions), chain, effective)
    return out


def _ki_instants(records, mapping, is_reverse):
    """The discrete knock-in rows keyed by canonical time (0.16.0 zero-gap rule): ``(rows, split)``. An instant holding
    several knock-in records knocks in where the spot breaches ANY of them (a knock-in is idempotent at one spot): one
    row at the widest effective level (``same_instant_ki_groups``); ``split[t]``: the narrower effective levels in
    contract order (two or more effective records), the breaks whose regions the statistics put on each record.

    Raises:
        ValidationError: a canonical time holds records the contract clock keeps apart (a user
            ``event_time_tolerance`` merging two instants keeps V2's refusal).
    """
    by_time = {}
    for record in records:
        by_time.setdefault(mapping[float(record.observation_time)], []).append(record)
    rows, split = {}, {}
    for t, members in by_time.items():
        if len(members) == 1:
            rows[t] = (float(members[0].observation_time), float(members[0].barrier))
            continue
        groups = same_instant_ki_groups(members, is_reverse)
        if len(groups) != 1:
            raise ValidationError(
                f"{_SHARED_TIME}: event_time_tolerance merges knock-in records the contract clock keeps apart"
            )
        effective = groups[0]
        rows[t] = (float(effective[-1].observation_time), float(effective[-1].barrier))
        if len(effective) >= 2:
            split[t] = tuple(float(r.barrier) for r in effective[:-1])
    return rows, split


def _refuse_fold_on_instants(records, instants):
    """Cash-leg folding onto an instant holding two or more effective knock-out records (0.16.0 D10, as QUAD V1 and the
    PDE refuse it): the binder is one to one by time while each record pays its own cash there. An instant whose later
    records are dominated binds every record and pays only the effective one's (the others never knock out)."""
    for _, _, effective in instants.values():
        if len(effective) >= 2:
            named = ", ".join(
                records[p].observation_date.date().isoformat() if records[p].observation_date is not None
                else f"t={float(records[p].observation_time)!r}" for p in effective)
            raise ValidationError(
                f"cash-leg folding cannot bind the knock-out records {named} sharing one env instant: each pays "
                "its own cash at that instant (an open item); price the legs on the event route")


def _zero_gap(step, state, count, k=0):
    """The action of the ``count`` records of one instant from record ``k`` on ``state`` (D14 zero-gap rule): record
    ``k``'s own action (``step(state, k)``) with each continuation it leads to replaced by the next record's action on
    that state -- consecutive observations with zero elapsed time, so no discounting between them."""
    action = step(state, k)
    if k == count - 1:
        return action
    return _compose(action, lambda s: _zero_gap(step, s, count, k + 1))


def _compose(action, rest):
    """``action`` with every continuation state ``s`` replaced by ``rest(s)`` (an action at the same instant); a
    channel-selected continuation keeps its selection on whatever it leads to."""
    cash, asset = np.array(action.cash, dtype=float), np.array(action.asset, dtype=float)
    terms, channel_terms = {}, []
    for s, w in action.terms:
        b = rest(s)
        cash = cash + w * np.asarray(b.cash, dtype=float)
        asset = asset + w * np.asarray(b.asset, dtype=float)
        for s2, w2 in b.terms:
            terms[s2] = terms.get(s2, 0.0) + w * w2
        channel_terms.extend((s2, w * w2, sel) for s2, w2, sel in b.channel_terms)
    for s, w, sel in action.channel_terms:
        b = rest(s)
        mask = np.zeros(len(cash))
        mask[list(sel)] = 1.0
        cash = cash + w * np.asarray(b.cash, dtype=float) * mask
        asset = asset + w * np.asarray(b.asset, dtype=float) * mask
        channel_terms.extend((s2, w * w2, sel) for s2, w2 in b.terms)
        for s2, w2, sel2 in b.channel_terms:
            both = tuple(c for c in sel if c in set(sel2))
            if both:
                channel_terms.append((s2, w * w2, both))
    return Action(tuple(terms.items()), tuple(cash), tuple(asset), tuple(channel_terms))


def _active_coupon_records(product, env, event_phase):
    """Source indices of the Phoenix KO/coupon records still to be observed (earlier ones expired)."""
    raw = product.barrier_config.ko_observation_schedule.records
    return [
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


def _count_memory_coupons(product, active, count):
    """The coupons of the ``count`` periods a lifecycle state says were missed before the first active record."""
    historic = []
    if count:
        fixed = product.coupon_config.fixed_coupon_year_fraction
        if fixed is not None:
            historic = [
                product.get_coupon_payoff(0, year_fraction=fixed)
            ] * count
        elif active and active[0] >= count:
            # the missed periods are still in the schedule: their coupons are their own periods'. A missed
            # period with no known start (a record before the contract's time origin) is missing contract
            # terms, not an unsupported feature: refuse as such, saying what needs it.
            try:
                historic = [
                    product.get_coupon_payoff(i)
                    for i in range(active[0] - count, active[0])
                ]
            except ValidationError as err:
                raise ValidationError(
                    "aged Phoenix memory needs the missed periods' starts to pay their arrears "
                    f"({err})"
                ) from err
        else:
            raise NotImplementedError(
                "aged Phoenix memory requires fixed_coupon_year_fraction or retained historical coupon dates"
            )
    return historic


def count_memory_arrears(product, env, count, event_phase="before"):
    """The memory arrears a lifecycle state's ``coupon_memory_count`` states for ``product`` at ``env``: the
    amount QUAD V2 rebuilds from the count (the missed periods' own coupons, or the fixed-fraction coupon each),
    summed in the order the compiler sums them. Theta uses it to move the debt onto the contract's amount."""
    historic = _count_memory_coupons(product, _active_coupon_records(product, env, event_phase), count)
    return float(np.cumsum(historic)[-1]) if historic else 0.0


def compile_contract(
    product,
    env,
    params,
    *,
    components=False,
    event_phase="before",
    lifecycle_state=None,
    fold=None,
):
    """Compile only known product classes; unknown subclasses fail closed.

    ``fold`` is a ``quantark.cashleg.folding.FoldOverlay`` of cash legs folded into
    the payoff (ship spec 2026-10-21, D4/D5). Each KO record is bound one to one
    through ``fold.bind_ko`` — schedule ``"main"`` for Snowball / Phoenix,
    ``"pre"`` / ``"post"`` for KO-reset — and its amount (cash per unit at the
    record's settlement) is added to the record's own cash. Each maturity branch is
    bound through ``fold.bind_terminal`` at its own end: ``"no_ki"`` at the alive
    branch's end (the KO-reset pre-maturity) and ``"ki"`` at maturity, both with
    the terminal settlement delay. Any unbound entry or timing mismatch raises.
    ``None`` or an empty overlay compiles exactly the unfolded contract.

    Knock-out records of a Snowball / Phoenix on one env instant (0.16.0 DCN/dated run, D7/D14) compile to ONE event
    whose action is the zero-gap chain of the instant's records in contract order (``_ko_instants``, ``_zero_gap``);
    a fold onto an instant with two or more effective knock-out records is refused (D10). Knock-in records, and
    KO-reset records, of one kind sharing a canonical time keep the refusal.
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
    # validate a copy: validate() rebuilds the schedules and terminal periods, and pricing must not rebuild the
    # caller's product. Shallow is enough: validate() rebinds the product's configs and periods (detached copies,
    # dataclasses.replace) and mutates none of the objects it shares with the caller. The product exists already,
    # so a numeric contract aged to exactly its maturity instant (maturity 0.0) is valid here: it prices its settled
    # value, as its dated twin on the exercise date does (user ruling 2026-10-10, Q1). A negative maturity refuses.
    product = copy(product)
    product.validate(allow_zero_maturity=True)
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
        terminal = resolve_terminal_timing(product, env)
        # the knocked-in note ends at schedule 2's last fixing (owner ruling D31, run 5 T7): before the maturity when
        # the product runs on past schedule 2 (``knocked_in_settlement``), else at the maturity as before
        knocked_in = knocked_in_settlement(product, env, terminal, post_schedule_times=[r[0] for r in post])
        post_maturity = maturity if knocked_in is None else float(knocked_in.determination_time)
        record_dates = (
            {"pre": _reset_record_dates(product, env, product.barrier_config),
             "post": _reset_record_dates(product, env, product.post_barrier_config)}
            if components == "events" else None
        )
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
        knocked_in, post_maturity = None, maturity
        terminal = resolve_terminal_timing(product, env)
        record_dates = (
            {"pre": {float(r.observation_time): (r.observation_date, r.settlement_date) for r in records}}
            if components == "events" else None
        )

    def is_future(t):
        return 0 <= t <= maturity and (event_phase == "before" or t > 0)

    pre = [r for r in pre if is_future(r[0])]
    post = [r for r in post if is_future(r[0])]
    # the Snowball / Phoenix knock-out records aligned with ``pre`` (contract order): same-instant records are
    # identified by their position, never by time (0.16.0 D4)
    ko_records = None if reset else [r for r in records if is_future(float(r.observation_time))]
    continuous = product.has_ki_barrier and (
        product.barrier_config.ki_continuous
        or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
    )
    ki_records = (
        []
        if continuous or not product.has_ki_barrier
        else [r for r in product.resolve_ki_observations(env) if is_future(float(r.observation_time))]
    )
    ki = [(float(r.observation_time), float(r.barrier)) for r in ki_records]
    if continuous and isinstance(product.barrier_config.ki_barrier, list):
        raise NotImplementedError("QUAD V2 continuous KI requires a constant barrier")

    all_times = sorted({maturity, pre_maturity, post_maturity} | {r[0] for r in pre + post + ki})
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
                raise ValidationError(_SHARED_TIME)
            result[t] = r
        return result

    # canonical times holding two or more knock-out records of a Snowball / Phoenix: one zero-gap event each (D7/D14);
    # empty for every schedule whose instants hold one record (the compile below is then exactly as before)
    instants = {} if reset else _ko_instants(ko_records, mapping, bool(product.is_reverse), phoenix)
    # a KO-reset note that ends with schedule 1 is paid on the explicit settlement date when booked (spec D33)
    first_schedule = first_schedule_settlement(product, env, terminal) if reset else None
    governed_payment = (
        first_schedule.payment_time
        if first_schedule is not None and first_schedule.settlement_date_governs
        else None
    )
    fold_terminal = {}
    if fold is not None and not fold.is_empty:
        if instants:
            _refuse_fold_on_instants(ko_records, instants)
        pre, post, fold_terminal = _bind_fold(
            fold, reset, pre, post, terminal, maturity, pre_maturity, governed_payment,
            None if knocked_in is None else (post_maturity, knocked_in.payment_time),
        )
    # a Snowball / Phoenix knock-out record's statistics key: ``pre:time`` (today's key), and per record of a shared
    # instant its position there (``#j``) so that every record has its own channels (D4); KO-reset keys stay by time
    row_keys = [] if reset else [
        f"pre:{mapping[r[0]]!r}" + (f"#{instants[mapping[r[0]]][0].index(pos)}" if mapping[r[0]] in instants else "")
        for pos, r in enumerate(pre)
    ]
    chains = {
        t: tuple((pos, pre[pos], row_keys[pos]) for pos in chain) for t, (_, chain, _) in instants.items()
    }
    if instants:
        alone = [r for r in pre if mapping[r[0]] not in instants]
        pre_by, post_by = keyed(alone), keyed(alone)
    else:
        pre_by, post_by = keyed(pre), keyed(post)
    # knock-in records on one instant (0.16.0 zero-gap rule): one row at the widest level; ``ki_split`` the narrower
    # levels of an instant whose records split its knock-in (empty for every schedule of one record per instant)
    ki_split = {}
    if len({mapping[r[0]] for r in ki}) == len(ki):
        ki_by = keyed(ki)
    else:
        ki_by, ki_split = _ki_instants(ki_records, mapping, bool(product.is_reverse))
    maturity, pre_maturity, post_maturity = mapping[maturity], mapping[pre_maturity], mapping[post_maturity]
    statistics = components == "events"
    # the price components' aggregate channels (``price_components``); a statistics context reads every outcome from
    # its own per-row channels instead
    aggregates = ("ko", "coupon", "terminal") if components and not statistics else ()
    channels = ("total",) + aggregates
    expiry = phoenix and product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
    # memory EXPIRY coupons: the coupon earned at a row (released arrears included) undiscounted, its own channel
    # (without memory it is the row's coupon amount x its coupon probability)
    earned_channels = statistics and expiry and product.has_memory_coupon
    ko_metadata = []
    if statistics:
        for regime, rows in (
            (("pre", pre), ("post", post)) if reset else (("pre", pre),)
        ):
            for pos, r in enumerate(rows):
                if reset:
                    key = f"{regime}:{mapping[r[0]]!r}"
                    dates = record_dates[regime].get(r[0], (None, None))
                else:   # by the record itself: two records of one instant have their own keys and dates (D4/D8)
                    key = row_keys[pos]
                    dates = (ko_records[pos].observation_date, ko_records[pos].settlement_date)
                ko_metadata.append((key, mapping[r[0]], r[3], regime, r[0], dates[0], dates[1]))
                channels += (f"ko_cash:{key}", f"ko_probability:{key}")
                if phoenix:
                    channels += (
                        f"coupon_cash:{key}",
                        f"coupon_probability:{key}",
                        f"coupon_payment:{key}",
                    )
                    if earned_channels:
                        # the coupon earned at the row, undiscounted (its payment time is path-dependent)
                        channels += (f"coupon_earned:{key}",)
        channels += (
            "terminal_alive_cash",
            "terminal_ki_cash",
            "terminal_alive_probability",
            "terminal_ki_probability",
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
        if channel in aggregates:
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
    # arrears owed at the pricing date and carried by the contract's amount (not a lifecycle count)
    initial_arrears = 0.0
    if phoenix:
        # Resolve original schedule indices when earlier observations expired.
        active = _active_coupon_records(product, env, event_phase)
        coupon_barriers = [product.get_coupon_barrier_at(i) for i in active]
        if len(coupon_barriers) != len(pre):
            raise ValidationError(
                "Phoenix coupon schedule does not align with active KO records"
            )
        # each active coupon's contractual period, by its source record (accrual time spec §6)
        fractions = [product.coupon_period_fraction(i) for i in active]
        coupon_amounts = [
            product.get_coupon_payoff(i, year_fraction=f)
            for i, f in enumerate(fractions)
        ]
        historic = _count_memory_coupons(product, active, initial_memory)
        coupon_prefix = np.r_[0.0, np.cumsum(historic + coupon_amounts)]
        # Memory arrears owed at the pricing date have two carriers (``CouponBarrierConfig.initial_coupon_arrears``):
        # a lifecycle state's COUNT of missed periods (rebuilt above from the periods' own coupons) and the contract's
        # AMOUNT, which every engine reads. A count, when present, carries them; an amount beside it must be
        # the same debt (the intraday twin hands this engine both), else the two disagree and nothing says which
        # holds. Without a count the amount is owed exactly as QUAD V1, the PDE solver and Monte Carlo owe it.
        # Known limitation: a lifecycle-tracker count beside a booked amount that covers periods before the
        # tracked schedule (separate debts, not one stated twice) is refused here too; no library path builds it.
        owed = (
            float(product.coupon_config.initial_coupon_arrears)
            if product.has_memory_coupon
            else 0.0
        )
        if initial_memory and owed:
            if not is_close(owed, float(coupon_prefix[initial_memory])):
                raise ValidationError(
                    f"Phoenix initial_coupon_arrears {owed!r} disagrees with the lifecycle state's "
                    f"{initial_memory} missed coupons ({float(coupon_prefix[initial_memory])!r}): "
                    "the two carriers of the memory arrears must state the same debt"
                )
        else:
            initial_arrears = owed
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
        # an instant holding two or more knock-out records: per record of its chain ``(row, position, key, knock-in,
        # last)`` -- the knock-in observed with the last record (the instant's knock-out wins the tie as one unit, as
        # QUAD V1 and the PDE stage it), the maturity payoff after the last record
        links = None
        if t in chains:
            entries = chains[t]
            links = tuple(
                (row, pos, key, kr if k == len(entries) - 1 else None, k == len(entries) - 1)
                for k, (pos, row, key) in enumerate(entries)
            )
            levels = [row[1] for _, row, _ in entries] + ([kr[1]] if kr is not None else [])
            if phoenix:
                levels += [coupon_barriers[pos] for pos, _, _ in entries]
            index, coupon_key = None, None
        else:
            levels = [r[1] for r in (pr, po, kr) if r is not None]
            index = obs_index.get(t)
            coupon_key = f"pre:{t!r}"
            if phoenix and index is not None:
                levels.append(coupon_barriers[index])
        if t in (maturity, pre_maturity, post_maturity):
            levels += breaks_terminal
        split = ki_split.get(t, ()) if kr is not None and not continuous else ()
        levels += list(split)       # each narrower knock-in level bounds a region (the statistics' per-record split)
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

        def note_action(state, x, lo, hi, point=False, link=None, pr=pr, index=index, coupon_key=coupon_key, kr=kr):
            # ``link``: one record of a shared instant's chain -- its row, position (coupon index), statistics key and
            # the knock-in observed with it; only the chain's ``last`` record ends the note at maturity
            last = True
            if link is not None:
                pr, index, coupon_key, kr, last = link
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
            # KO-reset: a knock-in at this observation brings schedule 2 into force here, so its record at this
            # instant applies -- unless ``disable`` is set: owner ruling 2026-11-27 (run 5 T6), that knock-in
            # disables schedule 2 at once, its record on the knock-in instant included (as MC, the PDE and V1 read it;
            # the schedule-1 knock-out above still wins its tie with the knock-in)
            if reset and not hit and ki_hit and not ko_hit and not disable:
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
                if ko_hit or (last and t == maturity):
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
            if initial_arrears and index is not None and mem == index:
                # Every period since the pricing date has missed (a hit resets the memory to 0, a miss adds one), so
                # the arrears owed then are still owed here: they join the memory exactly as QUAD V1's
                # ``accumulated_before`` and the PDE solver's ``_state_arrears`` carry them -- released with a coupon
                # hit, released at a knock-out whatever the coupon barrier says, carried on a miss. (The amount is
                # used only without a lifecycle count, so the memory starts at 0 and the state index equals ``mem``.)
                missed += initial_arrears
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
                    key = f"{regime}:{mapping[record[0]]!r}" if link is None else coupon_key
                    c[channels.index(f"ko_cash:{key}")] = record[2] * unit_df
                    c[channels.index(f"ko_probability:{key}")] = (
                        1 / env.get_discount_factor(t)
                    )
                if phoenix:
                    paid_at_ko = missed + (coupon_amounts[index] if coupon_hit else 0.0)
                    c += coupon_cash(
                        paid_at_ko
                        * unit_df,
                        f"coupon_payment:{coupon_key}",
                    )
                    if earned_channels and paid_at_ko:
                        c[channels.index(f"coupon_earned:{coupon_key}")] = (
                            paid_at_ko / env.get_discount_factor(t)
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
            end = post_maturity if new_hit else pre_maturity
            c, a, terms = np.zeros(n_channels), np.zeros(n_channels), []
            channel_terms = []
            if last and t == end:
                if point:
                    payoff = (
                        product.get_maturity_payoff_v1
                        if new_hit
                        else product.get_maturity_payoff_v0
                    )
                    tc, ta = payoff(point_spots[x], pricing_env=env), 0.0
                else:
                    tc, ta = _affine_payoff(product, env, new_hit, lo, hi)
                if fold_terminal:
                    tc += fold_terminal["ki" if new_hit else "no_ki"]
                # KO-reset's pre branch can end before the post-KI branch.
                delay = (
                    terminal.delay_df
                    if t == maturity
                    # the knocked-in note paid at schedule 2's end's settlement (D31), by its own exact ratio
                    else env.get_discount_factor(knocked_in.payment_time) / env.get_discount_factor(t)
                    if new_hit
                    else env.get_discount_factor(
                        t + terminal.payment_time - terminal.determination_time
                        if governed_payment is None
                        else governed_payment
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
            elif t < end or not last:
                terms.append((int(new_hit) * n_memory + next_mem, 1.0))
            if statistics and coupon_hit:
                c[channels.index(f"coupon_probability:{coupon_key}")] = (
                    1 / env.get_discount_factor(t)
                )
            if earned_channels and earned:
                c[channels.index(f"coupon_earned:{coupon_key}")] = earned / env.get_discount_factor(t)
            if earned:
                if expiry and (t < end or not last):
                    if statistics:
                        selected = (
                            0,
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

        if links is None:
            action = note_action
        else:
            def action(state, x, lo, hi, point=False, links=links, note_action=note_action):
                """The instant's records at one spot in contract order (D14 zero-gap rule)."""
                return _zero_gap(
                    lambda s, k: note_action(s, x, lo, hi, point, links[k]), state, len(links)
                )

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
        events.append(Event(t, breaks, tuple(rows), tuple(boundaries),
                            ki_split=tuple(float(np.log(level)) for level in split)))
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
            "post_maturity": post_maturity,
            "terminal_delay": terminal.payment_time - terminal.determination_time,
            "initial_ki": initial_hit,
            "expiry_coupons": expiry,
            "statistics": statistics,
            "n_memory": n_memory,
            "coupon_amounts": tuple(float(a) for a in coupon_amounts) if statistics else (),
            "ko_reset": reset,
            "ki_horizon": pre_maturity if reset else maturity,
            "ki_records": tuple(
                (mapping[float(r.observation_time)], float(r.observation_time), r.observation_date, r.barrier)
                for r in (product.resolve_ki_observations(env) if statistics and ki else ())
                if is_future(float(r.observation_time))
            ),
        },
    )
