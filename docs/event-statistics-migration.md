# Migrating to `EventStatistics` (0.11.0)

0.11.0 replaces the five event-statistics result types with one, `EventStatistics`
(`quantark.asset.equity.engine.event_stats`). There are no deprecated aliases: code that reads an
old type or field breaks at import or attribute access. This note maps every old type and field to
its replacement, lists the behaviour changes callers will see, and names the tracked study outputs
that still carry the old definitions.

The new type is documented in
[`autocallable_event_stats_api.md`](../quantark/asset/equity/engine/docs/autocallable_event_stats_api.md).

Notation: `obs = stats.observations`, `ko = obs.is_ko_obs`, `pre = (np.array(obs.schedule) == "pre")`,
`post = (np.array(obs.schedule) == "post")`.

## 1. Types and functions

| Old | New |
|---|---|
| `AutocallableEventStats` | `EventStatistics` (`product_family="snowball"`) |
| `PhoenixEventStats` | `EventStatistics` (`product_family="phoenix"`, `coupons` filled) |
| `KOResetEventStats` | `EventStatistics` (`product_family="ko_reset"`, `observations.schedule` tags; REBASED adds `post_ki_offsets` and `termination`) |
| `DCNEventStats` | `EventStatistics` (`product_family="dcn"`, `coupons` filled) |
| `RiskNeutralSnowballEventStats` (path analyzer) | `EventStatistics`; `AutocallablePathAnalyzer.analyze_snowball_risk_neutral` returns it |
| `EventDistribution.from_autocallable_stats(stats)` | `EventDistribution.from_statistics(stats)` |
| `EventDistribution.from_dcn_stats(stats)` | `EventDistribution.from_statistics(stats)` |
| `EventDistribution._from_ko_reset` (private) | removed; `from_statistics` handles KO-reset ABSOLUTE and REBASED |
| `_as_event_statistics` | never public (a transition helper inside the 0.11.0 development branch); removed |
| `payment_aware_cashflow_fields` | removed with the old types it served (it recovered undiscounted cash as discounted / DF, wrong on path-dependent rows). Producers use `build_cash_ledger` / `build_rebased_cash_ledger`. |
| `DCNEventStats.to_dict()` | removed. `DCNMCResult.to_dict()` serialises its `event_stats` (an `EventStatistics`) field by field. |
| `engine.calculate_event_stats(...)` | same name; returns `Optional[EventStatistics]`. DCN engines (all 8) now implement it, with `lifecycle_state=`. |
| `engine.price_with_events(...)` | same name and `PricingResult(npv, event_distribution)`; `npv == price()` bitwise on every engine |

## 2. Fields of `AutocallableEventStats` (and the Snowball part of every old autocallable type)

| Old field | New path | Notes |
|---|---|---|
| `pv` | `pv` | Now `price()` bitwise on every engine (MC: the value of `price()` on the same paths; ADI was 1–2 ULP off). |
| `ko_times` | `obs.times[ko]` | Rows are the union of KO and coupon observations; filter with `is_ko_obs`. |
| `ko_probability` | `obs.probability` | 0 on coupon-only rows. |
| `survival_probability` | `obs.survival` | P(alive after the row). KO-reset ABSOLUTE also drops the never-knocked-in maturity mass at its determination time. |
| `expected_discounted_ko_cashflow` | `obs.discounted_cash` | **Coupons excluded.** A Phoenix coupon (and released arrears) paid with a KO is now in `coupons.discounted_cash`. |
| `ki_probability` | removed | Its definition differed by engine (PDE/QUAD: settles knocked in; MC and legacy QUAD V1 KO-reset: ever breached). Use `ki_settlement.probability` for "settles knocked in", or `knock_in.probability` for P(KI before any KO). Neither is the old MC number. |
| `expected_discounted_maturity_cashflow` | `maturity.discounted_cash + ki_settlement.discounted_cash` | The old no-KO maturity total, now split into its two outcomes, each computed directly. |
| `reconciliation_error` | `reconciliation_error` | Computed by the type: `pv − Σ outcome cash − Σ coupon cash − Σ receivables`. Never passed in, never absorbed. |
| `ki_times` | `ki_table.times` | One row per product KI observation; continuous KI one row per trading day. |
| `ki_event_probability` | `ki_table.probability` | **Definition change:** P(first KI at the row, **before any KO**). |
| `ki_survival_probability` | `ki_table.survival` | **Definition change:** P(not knocked in and not knocked out after the row). |
| `ki_ever_probability` | removed → `knock_in.probability` | **Definition change.** The old field counted a breach *after* a knock-out ("including after a knock-out"). `knock_in.probability` is P(KI before any KO), smaller wherever paths knock out first. No field reports KI-ever. |
| `ki_survive_knocked_in_probability` | `ki_settlement.probability` (== `knock_in.probability_no_ko`) | Same meaning: KI and no KO. |
| `determination_times` | `ledger.determination_times` | |
| `payment_times` | `ledger.payment_times` | |
| `expected_undiscounted_cashflows` | `ledger.cash_to_pay` | Now computed directly. The old value was discounted / DF(payment), wrong on EXPIRY coupons, memory arrears and REBASED rows. |
| `expected_discounted_cashflows` | `ledger.discounted_cash` | |
| `determination_dates`, `payment_dates` | `ledger.determination_dates`, `ledger.payment_dates` | |
| `provenance` | `provenance` | Required (was `Optional`). |

New fields with no old counterpart: `maturity` and `ki_settlement` rows (time, payment time,
dates, `cash_to_pay`), `knock_in.expected_time`, `knock_in.expected_loss`, `expected_lifetime`,
`receivables`, `total_discounted_cash`, `total_cash_to_pay`, `probability_residual`,
`unavailable`, `standard_errors` (MC), per-row `payment_is_path_dependent`, the KI table's
`dates`, `payment_times`, `interpolated`, `interpolation_scheme`.

## 3. Fields of `PhoenixEventStats`

| Old field | New path | Notes |
|---|---|---|
| `coupon_probability` | `coupons.probability` | P(alive just before the row and barrier met, a simultaneous KO included). `None` when pruned. |
| `expected_discounted_coupon_cashflow` | `coupons.discounted_cash` | Now includes the coupon and arrears paid with a KO (moved out of KO cash). |
| `coupon_payment_is_path_dependent` | `coupons.payment_is_path_dependent` | Now a per-row array; True on every row of an EXPIRY coupon on every engine. |

## 4. Fields of `KOResetEventStats`

| Old field | New path | Notes |
|---|---|---|
| `ko_times`, `ko_probability`, `survival_probability`, `expected_discounted_ko_cashflow` | `obs.*` | Both schedules in one table, tagged; select with `pre` / `post`. |
| `pre_ko_times` | `obs.times[pre]` | |
| `pre_ko_probability` | `obs.probability[pre]` | |
| `post_ko_times` | ABSOLUTE: `obs.times[post]`; REBASED: `post_ki_offsets.offsets[kind == "ko_post"]` (offsets) or the `"ko_post"` rows of `termination` (absolute) | |
| `post_ko_probability` | ABSOLUTE: `obs.probability[post]`; REBASED: `post_ki_offsets.probability[kind == "ko_post"]` | |
| `pre_ko_probability_total` | `obs.probability[pre].sum()` | |
| `post_ko_probability_total` | ABSOLUTE: `obs.probability[post].sum()`; REBASED: the offset `ko_post` rows' sum | |
| `expected_discounted_post_ko_cashflow` | ABSOLUTE: `obs.discounted_cash[post].sum()`; REBASED: the offset `ko_post` rows' cash sum | |
| `pre_ko_payment_times` | `obs.payment_times[pre]` | |
| `post_ko_payment_times` | `obs.payment_times[post]` (ABSOLUTE); REBASED: `termination.payment_times` of the `ko_post` rows | |
| `expected_discounted_pre_maturity_cashflow` | `maturity.discounted_cash` | |
| `expected_discounted_post_maturity_cashflow` | `ki_settlement.discounted_cash` | |
| `pre_maturity_probability` | `maturity.probability` | |
| `post_maturity_probability` | `ki_settlement.probability` | |
| `pre_maturity_time`, `pre_maturity_payment_time` | `maturity.time`, `maturity.payment_time` | |
| `post_maturity_time`, `post_maturity_payment_time` | `ki_settlement.time`, `ki_settlement.payment_time` | REBASED: the totals row at the latest candidate knocked-in settlement; actual dates in the `termination` `ki_settlement` rows. |
| `post_ko_times_are_offsets` | `stats.rebased` | REBASED results carry `post_ki_offsets` and `termination`; `from_statistics` converts them instead of raising `NotImplementedError`. |

## 5. Fields of `DCNEventStats`

| Old field | New path | Notes |
|---|---|---|
| `ki_probability` | `knock_in.probability` | P(KI before any KO); the DCN already stopped KI monitoring at a KO, so the meaning is unchanged. |
| `ko_probability` | `obs.probability.sum()` | |
| `ko_timing_distribution` | `obs.probability` | Array on the shared monthly grid; rows inside `ko_lock_months` are 0. |
| `coupon_probability` | `coupons.probability` | P(the row's fixed coupon is paid), on the observation grid. |
| `expected_life_years` | `expected_lifetime` | Same quantity, on the DCN's ACT/365F axis. |
| `prob_survive_no_ki` | `maturity.probability` | |
| `prob_survive_ki` | `ki_settlement.probability` | |
| `expected_discounted_loss_leg` | `ki_settlement.discounted_cash` (signed by `direction_sign`) | Its magnitude is also `knock_in.expected_loss` (unsigned, ≥ 0). |
| `pv` | `pv` | `price()`, signed by `direction_sign`. |
| `determination_times`, `payment_times`, `expected_undiscounted_cashflows`, `expected_discounted_cashflows`, `determination_dates`, `payment_dates` | `ledger.*` | As in section 2. |
| `to_dict()` | removed | See section 1. |

## 6. Fields of `RiskNeutralSnowballEventStats` (path analyzer)

| Old field | New path | Notes |
|---|---|---|
| `pv_mc` | `total_discounted_cash` | The analyzer's cash sum. `pv` is now `SnowballMCEngine.price()`. |
| `std_error`, `num_paths` | removed | One path set is one batch: `standard_errors is None`, `unavailable["standard_errors"] == "undefined: single batch"`. |
| `ko_times`, `ko_prob`, `survive_prob`, `expected_discounted_ko_cf` | `obs.times`, `obs.probability`, `obs.survival`, `obs.discounted_cash` | |
| `ki_probability` | `knock_in.probability` (KI before any KO) or `ki_settlement.probability` | Was KI-ever. |
| `expected_discounted_maturity_cf` | `maturity.discounted_cash + ki_settlement.discounted_cash` | |
| `reconciliation_error` | `reconciliation_error` | **Sign flipped:** old `pv_mc − price`, new `pv − Σ cash`. |

## 7. `EventDistribution`

| Old | New | Notes |
|---|---|---|
| `event_times` | `stream_times[event_type]` | One axis per stream. KO and COUPON share the observation grid; KI has the KI table's axis. |
| `event_dates` | `stream_dates[event_type]` | |
| `mc_ko_times` | removed | Nothing read it. |
| `survival_probability` (n + 1 values with a leading 1) | `survival_times`, `survival_probability` | Own axis starting at 0. Values are unchanged when the first observation is after 0; an observation at 0 gives its post-event survival as the first value. |
| positional construction | keyword-only | |
| — | `probability_residual`, `rebased`, `terminated`, `determination_time(event)`, `ended()`, `ensure_live()` | New. |

## 8. Replay schema (product replay frames)

Column names and types are unchanged; the meanings follow the new type.

| Column | New source |
|---|---|
| `ki_probability_to_maturity` | `ki_settlement.probability` (was the engine-dependent `ki_probability`) |
| `expected_discounted_maturity_cashflow` | `maturity.discounted_cash + ki_settlement.discounted_cash` |
| `next_ko_probability`, `total_remaining_ko_probability`, `survival_probability`, `expected_discounted_ko_cashflow` | the KO rows (`is_ko_obs`) of `observations` |
| KI event rows | `ki_table` rows only. The old row invented at the maturity date with the scalar `ki_probability` is gone. |

A statistic listed in `unavailable` is recorded as NaN, never as a stand-in.

## 9. Behaviour changes callers will see

1. **KI is never triggered after a KO** on any engine (DCN already behaved so). P(KI) and the KI
   table shrink wherever paths knock out before the barrier is crossed; `FixedPayoffLeg` AT_KI
   values move accordingly.
2. **KO wins a same-instant tie** with a discrete KI, on every engine.
3. **Phoenix coupons paid with a KO** are coupon cash, not KO cash.
4. **Maturity outcomes are computed directly**, not as `1 − ΣKO − ki`. The difference is
   `probability_residual`, which `from_statistics` absorbs into MATURITY_NO_KO and carries on the
   distribution. Grid KO-reset MATURITY_NO_KO moves by up to about 3e-9.
5. **The KI stream has its own axis.** AT_KI legs pay on KI dates (KI observation time plus the
   leg's `payment_offset_days`) where they used to raise or the stream was dropped for having a
   different length from the KO grid.
6. **REBASED KO-reset on MC** converts to an `EventDistribution` on the absolute candidate axis; cash
   legs value it (an `AutocallableCashLeg` must list every candidate time that carries mass). Folding
   a REBASED note is still refused (`CapabilityError`).
7. **`price_with_events(...).npv == price()` bitwise** on every engine. MC trade PVs moved by MC
   noise to the `price()` value (statistics are now recorded on `price()`'s paths); Heston ADI moved
   by 1–2 ULP.
8. **MC statistics come from `price()`'s paths**, so MC event numbers differ from 0.5.5 by sampling
   noise. MC `calculate_event_stats` on an expired note returns `None` (was `ValueError`).
9. **Terminated notes** (QUAD V2, DCN with `lifecycle_state`): receivables-only statistics when cash
   is unpaid, `None` when settled; `price_with_events` returns `EventDistribution.ended()`, which
   event-dependent legs refuse. QUAD V2 used to raise; DCN returned no statistics.
10. **DCN:** every engine implements `calculate_event_stats`. COUPON-basis legs on a DCN now value
    (they raised). DCN PDE produces native statistics (it used to fall back to a maturity-only
    distribution).
11. **EXPIRY coupon rows are path-dependent on every engine**, not just QUAD V2 and PDE; asking the
    distribution for one fixed coupon payment time raises.
12. **KO-reset ABSOLUTE on MC:** the never-knocked-in maturity is determined at the first
    schedule's end with its own settlement, and the survival drops it there (was the one-maturity
    layout at the last KO record). Probabilities do not move.
13. **QUAD V1:** Phoenix KO-at-valuation now emits its coupon stream. `event_stats_mode=
    "forward_density"` reports no KI path statistics and raises `StatisticsRefused` for memory or
    EXPIRY Phoenix (it used to return empty coupon cash).
14. **Unavailable instead of wrong.** Routes that cannot compute the KI path statistics (ADI scalar
    / sparse / semi-Lagrangian transport, coarse ADI meshes with a negative first-KI row, QUAD V2
    non-default continuous routes, QUAD V1 BGK, the path analyzer on continuous KI) return `None`
    with a reason. Read such fields with `stats.require(path)`.
15. **QUAD V2 memory Phoenix with `initial_coupon_arrears > 0`** now pays the arrears (price
    moves; see the CHANGELOG). A lifecycle count plus a different non-zero amount raises
    `ValidationError`.
16. **Heston-SLV conditional-control MC** raises `PricingError` when a control-corrected outcome
    probability breaks its bound (seen below about 256 paths per batch).
17. **Risk report:** "P(KI before maturity)" is replaced by "P(KI before any KO)"
    (`knock_in.probability`) and "P(matured knocked in)" (`ki_settlement.probability`). The MC
    "PV ± SE, paths" line is replaced by "PV (event statistics; statistics: <model>)"; unavailable
    statistics are listed by path.

## 10. Known limitations in 0.11.0

- DCN statistics stay on the DCN's own ACT/365F axis, like DCN `price()`. Under `BUSINESS_DAYS` or
  `ACT_360` environments both are mis-discounted; the fix moves prices and is left to a separate
  release. DCN receivable rows mix axes until then (decision on the DCN axis, payment on the
  environment axis).
- REBASED KO-reset statistics exist on MC only.
- Local-vol MC `estimator="one_step_survival"` has no statistics.
- `PricingSession` `EVENT_STATS` is refused for `DCNPDEEngine` and `LocalVolDCNPDEEngine`.
- QUAD V2 continuous `knock_in.expected_time` sits about half a trading day below MC (day-midpoint
  convention vs MC step bucketing; QUAD V1 uses the same day-midpoint convention).
- QUAD V1 Phoenix EXPIRY `reconciliation_error` is about 7e-9 × pv at 401 nodes, shrinking under
  refinement (a pre-existing price-side projection difference).

## 11. Tracked study outputs still on the old definitions

These files were produced before 0.11.0 and were not re-run (each is a long study run). Their
prices stand; their KI numbers use the old KI-ever or the old maturity split. Re-run the study to
refresh them.

| File | Stale content |
|---|---|
| `example/case_studies/snowball_q_term_structure/data/*` (`static_risk.csv`, `static_summary.json`, `fleet_summary.json`, `ki_probe_roll_window.csv`, `q_term_structure_report.html`, ...) | `p_ki` was P(KI ever); the scripts now write P(KI before any KO) |
| `example/case_studies/snowball_q_term_structure/README.md` §2b | P(KI) tables and text on the KI-ever definition |
| `example/case_studies/snowball_2y_carry_curve/README.md` | "P(KI) 62.9% vs 61.6%" is KI-ever; `results/pricing.json` was regenerated (P(KI before KO) 0.354) |
| `example/case_studies/gaussian_quad_comparison/results/events_*.json`, `REPORT.md` | maturity numbers on the old `expected_discounted_maturity_cashflow` split |
| `example/case_studies/quad_v2_validation/README.md` | says continuous KI-ever is unavailable on V2; V2 now reports P(KI before any KO) and a continuous KI table |
| `docs/research/continuous-ki-consistency/`, `docs/research/vol-model-ki-consistency/` | measure the old KI-ever field; pricing results stand. Their scripts read the removed `ki_ever_probability` / `ki_survive_knocked_in_probability` fields and now raise `AttributeError` on 0.11.0 |
