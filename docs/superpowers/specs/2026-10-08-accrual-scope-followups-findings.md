# Accrual-time follow-ups (spec §3 items 6–10): findings

Status: measured 2026-10-08 on branch `accrual-scope-followups`, which starts at 8753894f. Not merged.
Parent spec: `2026-10-08-accrual-time-design.md` §3 (items 6–10) and §9.
Evidence scripts: `2026-10-08-accrual-scope-followups-evidence/`. Each one runs from the repo root with
`env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <script>`.

| item | verdict | change |
|---|---|---|
| 6. rebate/KI flags read raw | Working as coded, but the defaults are inconsistent. Following the effective flag would move cash on most default contracts | user decided 2026-10-09 (rebate follows, KI flat); branch `w2-rebate-flag`, see §6 status |
| 7. PDE V0 boundary rebate not tenor-scaled | **bug, proven and fixed**; the call-rebate and reverse rows left alone then are fixed by D7 (§7 status) | f4eeb5af; branch `w3-pde-rows` |
| 8. `get_tenor` and BUSINESS_DAYS | **bug, proven and fixed** (no production caller, no price moves) | ad0ef5ac |
| 9. range accrual annualizes by the remaining maturity | **bug, proven** (same class as accrual spec §3.1) | none: design proposal (§9) |
| 10. DCN / FX TARN / cashleg own accrual | DCN is correct. The TARN is correct when aged, wrong when rebuilt mid-life. The cashleg `day_count` field is never read | none: design note (§10) |

---

## 6. `is_annualized_rebate` / `is_annualized_ki`: `None` means "not annualized"

**Today.** Knock-out accrual reads `_effective_annualized_flag(is_annualized_ko)`, so `None` takes the value of
`is_annualized`, which defaults to True. The rebate (V0) and knock-in (V1) legs read their flags raw, so `None` is not
annualized. The raw readers are:
- `snowball_option.py` `get_maturity_payoff_v0` (×2) and `get_maturity_payoff_v1`;
- `ko_reset_snowball_option.py` (the same three);
- `phoenix_option.py` `get_maturity_payoff_v0` (×2);
- `accrual_build.py` (whether a terminal needs accrual terms);
- `quad/v2/adapters.py` `terminal_breaks`;
- `snowball_mc_engine.py` (REBASED V1 tenor);
- `snowball_pde_solver.py` V0 rows (×2);
- `snowball_option.py` `_validate_maturity_parameters`. This one refuses an annualized contract that has no
  `tenor`, `initial_date` or `settlement_date`.

Every factory (`create_standard_snowball`, `create_standard_phoenix`, `create_ko_reset_snowball`, …) leaves both
flags `None` with `is_annualized=True`. `create_standard_snowball` also sets `rebate_rate = ko_rate`.

**Measured.**

`item6_effective_flags_patch.py apply` rewrites the 13 raw cash readers (every site listed above but the validation) in the working tree to the effective flag. It
leaves the validation raw (see below), and `revert` restores the files. It was never committed.

`item6_cash_deltas.py` gives the factory-default contracts (QUAD, spot 100, vol 20%, r 2%). V0 is the rebate paid at
S=100 and V1 the knock-in payoff at S=60:

| contract | tenor | V0 today → effective | V1(60) today → effective | QUAD today → effective |
|---|---|---|---|---|
| snowball T=0.5 | 0.5 | 15.0 → 7.5 | −40 → −20 | 5.413997 → 3.589740 |
| snowball T=1 | 1.0 | 15.0 → 15.0 | −40 → −40 | 2.134278 → 2.134278 |
| snowball T=2 | 2.0 | 15.0 → 30.0 | −40 → −80 | 0.810672 → −2.744132 |
| snowball T=3 | 3.0 | 15.0 → 45.0 | −40 → −120 | 1.517037 → −6.495351 |
| phoenix T=1 | 1.0 | 0 → 0 | −40 → −40 | −2.382431 → −2.382431 |
| phoenix T=2 | 2.0 | 0 → 0 | −40 → −40 | −3.450003 → −3.450003 (Phoenix V1 never annualizes) |
| KO-reset pre 1 / post 2 | pre 1.0 | 100 → 100 | 60 → 20 | 98.721179 → 95.592108 |
| KO-reset, rebate 15% | pre 1.0 | 115 → 115 | 60 → 20 | 99.924758 → 96.795687 |

Nothing moves at a one-year tenor. Away from one year both legs scale with the tenor. The KI loss of a default
two-year snowball doubles, and its price falls by 3.55 per 100 of notional.

Test run with the patch applied (`-n 2`, the 59 paths listed in "How measured" below):
**22 failed, 2311 passed, 4 skipped** (37 min). Re-run on the clean tree (the patch reverted), the same 22 node ids
all pass (22 passed, 3.6 min), so the patch alone moves them.
- **Replay goldens** (`test/test_replay_goldens.py`), 15 tests: every frame of `scalar_bsm` and `book`, and three of
  `localvol`.
  - `scalar_bsm`: initial portfolio value −10131.703047 → −9999.301434, total PnL 130.607156 → −1.794457.
  - `localvol`: initial value −10156.459290 → −9999.596298, total PnL 155.363400 → −1.499593.
  - `book`: initial value −30394.824769 → −29998.131859, total PnL 2815.537098 → −5.155813, trades 2 → 0. The
    hedge never trades.
  - Largest column violation: 396.77 in absolute value, about 4.1% relative.
- **PDE event statistics**, 5 tests.
  - `pde_statistics/test_reference.py` and `test_statistics_vs_reference.py` (`sb_cont_3m`, a 3-month contract):
    PV 103.263208 → 101.215722.
  - `pde_statistics/test_ko_reset_native.py` (KO-reset, post tenor 2): PV 96.671472 → 92.088560 (spot 90),
    97.260676 → 93.239086, 88.658836 → 77.357696.
- **`test_snowball_option.py`**, 2 tests (`test_get_maturity_after_exercise_date_raises`,
  `test_missing_maturity_and_exercise_date_passes`). A dated contract without `initial_date` becomes refused at
  construction: "a dated contract whose annualized cash needs an accrual start needs initial_date (accrual time spec
  R5)". Its KI leg is now annualized.
- Every other autocallable product, engine, intraday, accrual and bitwise test in the set passed. Most of them build
  one-year contracts or set the flags explicitly.

Examples: 39 files in `example/` build an autocallable. 3 of them set either flag and 1 sets `is_annualized=False`.
The other 35 are an upper bound on the examples that could move. That count is static (grep), not a run.

The validation: if `_validate_maturity_parameters` also followed the effective flag, a default numeric contract
built without `tenor`, `initial_date` or `settlement_date` would be refused. That covers every factory default above.
`create_standard_snowball(..., is_annualized_rebate=True)` raises exactly that ("initial_date or settlement_date
required for annualized accruals when tenor is not provided") on the parent commit: see the first run of
`item7_pde_v0_boundary_rebate.py` before it passed `tenor`. Today the maturity fallback (`accrued_offset + maturity`)
would answer those contracts, so the check is stricter than the tenor rule it guards.

**Decision for the user.**
- **A. Keep today's semantics (recommended unless a desk term sheet says otherwise).** `None` on the rebate and KI
  legs means "a flat amount".
  - Make it explicit: document it on `AccrualConfig`. Optionally have the factories pass `False`, which leaves
    every value byte-identical.
  - Cost: `is_annualized=True` keeps meaning "knock-out (and Phoenix coupon) accrual only", which is not what its
    docstring says ("default for all").
- **B. Make `None` follow `is_annualized` on every leg, as the knock-out leg does.**
  - Cost: every default contract with a tenor other than one year changes its cash. The table above has the
    magnitude: a two-year snowball's KI loss doubles. Every test, golden and example on such a contract moves (the
    counts are above).
  - The construction validation must change in the same commit, either to the effective flag plus the maturity
    fallback or by deleting it. Left raw, it says nothing. Moved to the effective flag as it stands, it refuses
    every default numeric contract.
  - Each moved golden needs listing and re-pinning. That is a 0.x-breaking change.

Either way the flag resolution should live in one place, `accrual.py`, beside `ko_accrual_factor`. Fourteen call
sites (13 cash readers and the validation) read the raw flags today. Only the knock-out leg goes through `_effective_annualized_flag`.

**Status 2026-10-09 (branch `w2-rebate-flag`, from dcea77b6; not merged, awaiting sign-off on the moved values).**
The user chose neither A nor B as written: an unset **rebate** flag follows `is_annualized` (B for V0), an unset
**knock-in** flag stays flat (A for V1), the KO coupon is unchanged, an explicit flag always wins.
- One resolver, `accrual.leg_annualized(product, leg)`; all 13 cash readers above, the REBASED knock-in anchor in
  `accrual_build` and the two KO-reset lifecycle tenor gates (`lifecycle/ko_reset.py`, not in the list above)
  read it.
- `_validate_maturity_parameters`' annualization check is deleted; R5 in `accrual_build` (on the effective flags)
  is the one construction rule. A contract dated by `exercise_date`/`maturity_date` alone with a default rebate
  is now refused by R5.
- `item6_cash_deltas.py` after the change: snowball T=0.5 V0 15 -> 7.5, QUAD 5.413997 -> 3.107317; T=2 15 -> 30,
  0.810672 -> 1.217247; T=3 15 -> 45, 1.517037 -> 1.677312; V1 unchanged everywhere; T=1, Phoenix and KO-reset
  rows unchanged.
- The 59 paths above plus `test/lifecycle_ko_reset`: 17 failed (13 replay goldens, 2 `pde_statistics`
  sb_cont_3m, 2 `test_snowball_option` R5 refusals). A second sweep of 161 further files that build an
  autocallable: 6 failed (3 banked certificates, the first-passage near-expiry benchmark, the carry-identity
  QUAD pins, the simulation deliberate-disagreement test). All 6 pass with the rebate default reverted (a
  pytest plugin setting the rebate leg's unset default to flat), so the rebate rule alone moves them.
- Re-pinned with closed-form checks: the replay goldens (`test/replay_golden/README.md`), the carry-identity
  QUAD pins and the R5 tests; the simulation fixture states a flat rebate.
- The `sb_cont_3m` cases, the first-passage near-expiry benchmark and the three banked certificates
  (`adi2d-snowball-greeks`, `snowball-flat-bsm`, `snowball-localvol-1d`, 2026-10-06: 41 anchors, e.g.
  `equity.snowball.quad/near_expiry/pv` 103.262509 -> 99.547329 under the new default) all build their contract
  through the modelvalidation snowball builders. **User decision 2026-10-09: no re-certification; the certified
  contracts keep their terms.** The builders (`equity_snowball.make_snowball`, also used by the local-vol study,
  and `equity_snowball_vol.make_snowball`) state `is_annualized_rebate=False`: every anchor reproduces exactly
  (same machine, `==`), and those test cases keep their original (dcea77b6) banks.

---

## 7. The PDE V0 boundary rebate is never tenor-scaled: BUG, fixed in f4eeb5af

**Every V0 rebate site compared.**
- The terminal V0 payoff on every engine goes through the product's `get_maturity_payoff_v0`:
  - PDE: `snowball_pde_solver` 1293, `ko_reset_snowball_pde_solver` 326, `phoenix_pde_solver` 188, both vol PDE
    families;
  - QUAD: v1 snowball, Phoenix and KO-reset, and the v2 `_affine_payoff` and terminal;
  - MC: snowball, Phoenix, KO-reset and the vol MC engines.
- The only site that builds the rebate itself is `SnowballPDESolver._boundary_rows_v0`, the Dirichlet rows. Every
  PDE family inherits it: snowball, Phoenix, KO-reset, LocalVol, and the Heston and SLV ADI solvers through
  `_set_boundary_conditions_v0`.
  - Its fixed-rebate rows (`v0_lower_principal_rebate`, `v0_upper_principal_rebate`) paid `rebate_rate × S0`.
    They never multiplied by the contract tenor.
  - Its call-rebate row did scale.

**Proof** (`item7_pde_v0_boundary_rebate.py`). The control contract cannot knock out (KO at 5× spot) or knock in
(discrete KI at 1%). It matures in V0 for certain, so its price is (100 + 10 × tenor) × e^{−rT}. Rebate 10%,
r = 3%, vol 20%, `tenor = T`.

| T | rebate | closed form | PDE (parent) | PDE − cf | QUAD − cf | MC − cf | lower row (parent) |
|---|---|---|---|---|---|---|---|
| 1 | flat | 106.749009 | 106.749009 | 2.8e−7 | 5.7e−14 | −1.4e−14 | 106.7490 |
| 2 | flat | 103.594099 | 103.594100 | 8.3e−7 | 2.1e−7 | 0 | 103.5941 |
| 2 | annualized | 113.011744 | 113.011405 | **−3.39e−4** | 1.0e−7 | −1.4e−14 | **103.5941** (should be 113.0117) |
| 3 | flat | 100.532430 | 100.532560 | 1.3e−4 | 1.1e−4 | 0 | 100.5324 |
| 3 | annualized | 118.811054 | 118.810528 | **−5.26e−4** | 4.1e−5 | −1.4e−14 | **100.5324** (should be 118.8111) |

The annualized two-year case is 410 times further from the closed form than its flat twin. QUAD and MC, which read
the product, are exact. The T = 3 flat rows carry a +1.3e−4 knock-out leak on PDE and QUAD alike, so they are not a
rebate error. The error is small because the default grid puts the boundary far from spot. The row itself was wrong
by rebate_rate × S0 × (tenor − 1), 9.42 out of 113.01 (8.3%) at T = 2.

**Fix.** Two failing tests came first (`test/test_snowball_pde.py::TestSnowballPDEV0BoundaryRebate`): the lower row
equals the discounted V0 payoff to rel 1e−14, and the price equals the closed form to 1e−5. They failed with
103.594099 ≠ 113.011744 and 113.011405 ≠ 113.011744. The fix multiplies the fixed rebate by
`product.get_contract_tenor(env)` when `is_annualized_rebate` is set. That is the same tenor the product's V0 payoff
uses: the `pre` leg on a KO-reset.

**Moved values** (`item7_moved_values.py`, run on 8753894f and on f4eeb5af):

| engine | case | old → new |
|---|---|---|
| SnowballPDESolver | certain-V0, T=2, annualized | 113.011404700204 → 113.011744762260 (cf 113.011744030) |
| SnowballPDESolver | certain-V0, T=3, annualized | 118.810528135825 → 118.811105553031 (cf 118.811054, QUAD 118.811095) |
| Snowball PDE | KO103/KI75 discrete and continuous, T=2, annualized | unchanged, bit-identical |
| Phoenix PDE | KO103/KI75 discrete, T=2, annualized | unchanged, bit-identical |
| KO-reset PDE | ABSOLUTE, KI75 discrete | unchanged, bit-identical |
| all | the non-annualized twins | unchanged, bit-identical |

The desk-like contracts do not move. While knock-in observations remain, the lower row is replaced by the V1 row
(discrete KI) or copies V1 (continuous KI). A KO-reset's V0 tenor (`pre`) is 1.0 in the case measured.

The only repo contracts with `is_annualized_rebate=True` are in:
- three tests that never run a PDE (`test/accrual_time`, `test_aged_accrual.py`);
- two QUAD-only examples;
- `example/ko_reset_snowball_demo.py`. That demo fails on 8753894f before it reaches the PDE: `ValidationError:
  determination date and time are inconsistent: 0.233606557377 != 0.237704918033` in `observation_schedule.resolve`
  from its MC run. That is unrelated to this change and is a finding of its own.

**Left alone (separate findings, not fixed):**
- **Call-rebate lower row.** With `call_rebate_enabled`, the contract pays no fixed rebate, but the PDE lower row
  still adds `rebate_rate × S0`. The factory default sets `rebate_rate = ko_rate`.
  - `item7_call_rebate_lower_row.py`, PDE − QUAD on a certain-V0 call-rebate contract:
    - rebate_rate 0 (the row is correct): −4.32e−4 flat and −8.64e−4 annualized;
    - rebate_rate 0.10 (the row carries a phantom 10): −9.16e−5 and −1.84e−4.
  - The phantom rebate partly cancels another error on the call-rebate rows. Fixing the row alone would make PDE
    worse against QUAD. These rows are left byte-identical; the condition is spelled out in the code.
- **Reverse snowball V0 rows.** `s_vec[-1] >= max_ko_barrier` is always true when the KO barrier sits below spot, so
  the upper (S → ∞, knock-in side) row is the knock-out payoff and the lower (knock-out side) row is principal +
  rebate.
  - On the certain-V0 control, reversed (KO at 20, KI at 10000), PDE − QUAD is +1.1e−3 flat and +5.0e−4 annualized,
    against 1e−7 for the standard orientation.
  - Probe: scratch script, not kept. Reproduce it with `item7_pde_v0_boundary_rebate.product(T, annual,
    reverse=True)`.

**Status 2026-10-09 (user decision D7, "fix both PDE row bugs"): both FIXED on branch `w3-pde-rows`.**
- **The cancelling error was QUAD's, not the PDE's.** Against the closed form (100 e^{-rT} + 0.2 x tenor x BS call;
  `item7_v0_rows_by_edge.py`), with `rebate_rate` 0 (the correct row) the PDE is already +4.0e−6 flat / +7.2e−6
  annualized, while QUAD is +4.36e−4 / +8.71e−4. QUAD's gap is grid discretization of the call leg on its default
  1001 points: 2001 → 1.36e−4, 4001 → 1.43e−5, 8001 → 7.2e−6 (flat). The phantom rebate pushed the PDE up by
  +3.4e−4, towards QUAD. No library change on the QUAD side: its default grid is a resolution choice, and it
  converges to the closed form.
- **Fix** (`SnowballPDESolver._boundary_rows_v0`, shared by the 1D adapter, Phoenix, KO-reset, local vol and the 2D
  Heston/SLV boundary hook): each edge is the knock-out side or the knock-in side by orientation. The knock-out edge
  pays the knock-out record once it lies beyond every KO level (`v0_{edge}_ko`). Under continuous KI the knock-in
  edge copies the knocked-in row once it lies at or beyond the KI level (`v0_{edge}_copy_ki`; the statistics group
  copies it on either edge). Otherwise the edge carries the no-event V0 asymptote (`_v0_no_event_row`): principal +
  fixed rebate, or principal + the call rebate's forward intrinsic value (a put on a reverse Phoenix, as
  `PhoenixOption.get_maturity_payoff_v0` pays) and no fixed rebate.
- **A third row defect, fixed with them:** the continuous-KI copy was unconditional, so a KI barrier the grid never
  reaches (KI at 1% of spot) still knocked in the lower edge: certain-V0 PDE − cf −2.66e−3 flat / −3.00e−3
  annualized → 2.4e−6 / 2.5e−6. It is the same rule the reverse knock-in edge needed.
- **After** (PDE − closed form, flat / annualized; QUAD in brackets): fixed rebate standard 8.3e−7 / 7.3e−7
  (unchanged), reverse 6.7e−7 / 6.6e−7 (QUAD 9e−8 / 4e−8); call rebate standard 4.0e−6 / 7.2e−6 for both
  `rebate_rate` 0 and 0.10, reverse 6.7e−6 / 1.3e−5 (QUAD 4.4e−4 / 8.7e−4). Local vol 1D reproduces BSM. The call
  rebate does not reach 1e−7: its kink at the spot leaves a PDE floor that neither `accuracy="high"` nor 800 / 1600
  points removes (standard annualized 7.2e−6 → 1.6e−5), the European PDE misprices the same call by 1.7e−4.
- **Moved values: none.** Every live banked certificate anchor reproduces bit for bit (exact, banking machine),
  including `snowball-flat-bsm` pde/`reverse` and pde/`call_rebate` and `phoenix-flat-bsm` pde/`reverse`, whose rows
  changed: their edges lie too far from the spot for the change to reach the price. Replay goldens,
  `pde_statistics` references and the targeted PDE suites (2021 tests) pass unchanged. PDE byte book (713 rows,
  33201993 vs the branch, same machine): 0 diffs, the same 10 rows raising on both sides.
- **Left alone:** the 2D Heston certain-V0 control is off by +3.8e−3 standard / +8.7e−3 reverse at n_x=96, n_t=48
  (+2.1e−3 / +6.9e−3 at 192 x 192) with the rows right; the rows moved it by 3e−6. A 2D solver finding of its own.
  The 2D solvers' continuous-KI copy reads `_grid_v1`, which they never set, so it copies 0.0 (unchanged).

---

## 8. `get_tenor` passes no calendar to a BUSINESS_DAYS year fraction: BUG, fixed in ad0ef5ac

**Proof** (`item8_get_tenor_business_days.py`, on 8753894f):
- A dated European option with `annualization_day_count=BUSINESS_DAYS` raises in `get_tenor()`, and also in
  `get_tenor(env)` with an SSE calendar and 244 days a year: "BUSINESS_DAYS day count requires a business-day
  calendar". `get_tenor` took `pricing_env` and never read it. The business-day fraction the contract means is
  243/244 = 0.995901639344.
- A dated BUSINESS_DAYS snowball carrying its frozen terms (`accrual_calendar`, `accrual_bus_days_in_year=244`):
  - `get_contract_tenor(env)` = 0.9959016393442623;
  - the inherited `get_tenor(env)` raises. One contract had two tenors.
- Call sites of `.get_tenor(`: **0 in `quantark/`**, 6 in `test/`, 0 in `example/`. No price, Greek or golden reads it.

**Fix.** Three failing tests came first (`test/test_base_equity_option_contract.py`): two failed for the reason above
and the refusal test passed.
- `BaseEquityOption.get_tenor` passes the environment's `calendar` and `bus_days_in_year`, as the same class's
  `get_time_to_date` does for the maturity. Without a calendar it still refuses: no ACT/365 fallback.
- `SnowballOption.get_tenor` and `PhoenixOption.get_tenor` (the KO-reset inherits) return the frozen `contract`
  terminal, `contract_tenor(self, env, "contract")`. That is accrual spec §4.4's "today's `get_tenor` rule, frozen",
  so an autocallable's tenor reads no environment.

After the fix, the European answers 0.995901639344 and refuses without a calendar. The snowball's `get_tenor` equals
its `get_contract_tenor`, with any environment or none. Moved values: none (no production caller). The six existing
tests pass unchanged.

---

## 9. Range accrual annualizes by the remaining maturity: BUG (same class as accrual spec §3.1), FIXED on branch `range-accrual-tenor`

> **Status 2026-10-09: fixed** at the user's request ("fix this"; FX start field: "add start_date like the equity
> product", so it is `initial_date`, the equity name). What shipped differs from the proposal below in two places:
> the year fraction is resolved from the product's own terms on each call rather than frozen as an `AccrualPeriod`
> (points 2 and 5 dropped: the terms are constants of the contract, so the answer cannot drift as it ages; only a
> BUSINESS_DAYS count reads the environment's calendar, as `get_tenor` does), and FX theta fixes stepped-over fixings
> through a new `BaseFxProduct.record_passed_fixings` hook. The fix review added one refusal: an annualized
> numeric contract booked mid-life (a fixing before t = 0) without `tenor`, whose `maturity` is only what is left;
> eight existing fixtures of that shape now pass `tenor` (one payoff assertion 7.5 -> 9.375). Tests: `test/test_range_accrual_contract_period.py`. The
> table below measures the code BEFORE the fix; `item9_range_accrual_ageing.py` builds its FX contract without a
> start, which the fixed code refuses.


**Today.**
- Equity: `RangeAccrualOption.get_year_fraction` returns `self.maturity`, else `get_maturity(env)`, the time from
  valuation to exercise.
- FX: `FxRangeAccrualOption.get_year_fraction` does the same with `expiry_date`.
- Both multiply it by the WHOLE contract's in-range ratio (past plus future fixing weights over the total weight).
- The equity and FX analytical and MC engines all call `get_year_fraction`, including the INSTANT pay type and the
  analytic Greeks.

**Proof** (`item9_range_accrual_ageing.py`). The control: one year, 10% annualized, notional 100, twelve monthly dated
fixings, a band so wide every fixing is in range, every past fixing recorded in range. The contractual coupon is 10.0
on every date.

| days aged | equity year fraction | equity price | 10 × DF | paid / owed |
|---|---|---|---|---|
| 0 | 1.000000 | 9.704455 | 9.704455 | 1.0000 |
| 91 | 0.750685 | 7.339680 | 9.777311 | 0.7507 |
| 182 | 0.501370 | 4.938851 | 9.850715 | 0.5014 |
| 273 | 0.252055 | 2.501560 | 9.924669 | 0.2521 |
| 364 | 0.002740 | 0.027395 | 9.999178 | 0.0027 |

FX (`FxRangeAccrualOption`, the same control) gives the same five rows to 6 decimals. A contract one day from expiry
that is certain to pay 10.0 is priced at 0.027.

Also measured:
- `tenor=1.0` and the config's own `day_count_convention` (ACT/360) are not read: the year fraction stays 0.501370 at
  182 days.
- On the numeric contract, `time_shift(0.5)` halves the coupon (10.0 → 5.0). It also leaves the fixing times
  unmoved: the first fixings stay 0.0833, 0.1667, 0.25. `BaseEquityOption.time_shift` shifts only `maturity` and an
  `observation_schedule`, and the range accrual has none, so after ageing its past fixings still look future. That
  is a second defect on the theta path.

**Proposed design (not implemented):**
1. The coupon accrues over a contractual period, [`accrual_start`, `accrual_end`]:
   - dated: `initial_date` → the tenor-end date;
   - numeric: `0` → `accrued_offset + maturity`, with `accrued_offset` banked by `time_shift` as the autocallables do;
   - `tenor`, when set, wins.
   The year fraction is YF in `range_config.day_count_convention`, the field that is unread today.
2. Freeze it at construction as an `AccrualPeriod` with its `AccrualBasis`, calendar and denominator included for
   BUSINESS_DAYS (accrual spec §4.7). Read it with the resolver's cumulative leg (`contract_tenor`'s rule).
   `accrual_build._contract_terminal` is already product-agnostic: it reads `tenor`, `initial_date`,
   `get_tenor_end_date()`, `maturity` and an offset.
3. FX: `FxRangeAccrualOption` has no `initial_date` or `tenor` today. A contract-start term must be added. A dated
   FX range accrual without one is refused (R5's rule), never silently started at the valuation date.
4. Fix the numeric `time_shift` in the same change: shift the fixing times and bank the elapsed time.
5. The accrual module lives in `asset/equity/product/option/`. Moving `AccrualPeriod`, `AccrualBasis` and the
   resolver to a neutral package (for example `quantark/accrual/`) lets FX use it without importing equity product
   internals.

Compatibility: at inception, with the environment's day count equal to the config's, the dated contract's numbers do
not change. Every aged valuation moves: in the table above, from "paid / owed" < 1 to 1. Range-accrual theta and every
multi-day scenario, backtest and PnL explain on a range accrual move with it. That is a contract-cash change the user
signs off, the same as accrual spec R1.

---

## 10. DCN, FX TARN, cashleg: own accrual code (survey, no change)

Measured by `item10_own_accrual_survey.py`.

- **DCN** (`dcn_option.accrual_per_period`, `dcn_grid.build_dcn_grid_context`).
  - The accrual is a fixed per-month fraction `coupon_counted_days / coupon_days_denom` (30/360 by contract) times
    the number of eligible month indices merged into an observation. The KO coupon uses the same fraction.
  - Month indices count from `initial_date`, so the accrual is age-invariant. DCN_A has 22 observations at
    inception and 19 aged to 2023-07-03; the surviving rows that differ number **0** (each 0.083333 for coupon and KO).
  - Verdict: correct for aged contracts.
  - Onto the resolver: each monthly observation would carry an `AccrualPeriod` with `period_fraction = a × n` (the
    resolver's explicit-fraction leg), and the engines would read `coupon_period_fraction`. Little gain: one call
    site (the grid context), already exact. Recommend leaving it.
- **FX TARN** (`FxTargetRedemptionNote`).
  - Accrual factors default to fixing-time differences from 0 at construction, in the fixing-time clock (whatever
    clock produced the times; no contract basis).
  - `time_shift` keeps them, which is correct: 12 monthly fixings shifted 0.125 keep 0.083333 each.
  - The same contract rebuilt mid-life with its remaining fixing times accrues its first period from 0: **0.041667
    instead of 0.083333**, a coupon of 0.50 instead of 1.00 on notional 100 at 12%. That is the §3.1 class.
  - `time_shift` also keeps a past fixing (t = −0.041667) in the list (12 fixings).
  - The TARN has no lifecycle state (no realized cumulative coupon towards the target). A mid-life TARN cannot be
    booked correctly today whatever its accrual.
  - Onto the resolver means:
    - the accrual module in a neutral package (see §9.5);
    - fixings as records with `AccrualPeriod(TIME, start = previous quoted fixing, end = quoted fixing)`;
    - `accrued_offset` banking;
    - a realized-coupon / remaining-target state for mid-life booking.
- **cashleg**.
  - `AutocallableCashLeg` uses explicit workbook accrual factors, and its `time_shift` keeps each surviving
    observation's factor. Correct, and the same as R1's quoted fraction.
  - `AccrualLeg` computes `dcf = period_end − period_start` in the schedule's time units. Its `day_count` field is
    never read: ACT/365, ACT/360 and 30/360 US give the same PV, 38773.115547. `LegSchedule.period_start_dates` /
    `period_end_dates` exist and are never read.
  - It has no `time_shift`. A period that began before valuation is representable (negative start) and accrues in
    full, but only if the caller supplies it.
  - Onto the resolver: when the dates are present, build `AccrualPeriod(DATE, start_date, end_date,
    basis=day_count)` and read `coupon_period_fraction`; otherwise keep `TIME`. Alternatively delete the dead
    `day_count` field. Either is a user decision: the first moves every dated leg whose day count is not ACT/365 on
    times that came from ACT/365.
  - **Status (2026-10-09, user decision B1, branch `w4a-cashleg-daycount`): FIXED — `day_count` is real.** A dated
    period accrues `calculate_year_fraction(start_date, end_date, day_count)` directly (not via the equity
    resolver: `AccrualPeriod` lives in the equity product package); times-only keeps `end − start`. Dates come as
    a pair, one `datetime` per period, end strictly after start; partial or mixed dating is refused (no caller
    mixes). Dated `BUSINESS_DAYS` needs the leg's new `calendar` + `bus_days_in_year` (no guess); ACT/ACT
    ISMA/BOND/CHINABOND refused on a dated leg (TODO: no coupon frequency on the leg). Measured: no library, test
    or example caller builds a dated schedule, so all 14 existing caller configurations (demo, event-emission,
    required-streams, this survey's three day counts × trivial / QUAD-snowball distributions) are bit-identical
    to dcea77b6. A dated quarterly 2026 leg moves: ACT/360 38773.115547 → 39306.910836, ACT/365 → 38768.460003,
    30/360 US unchanged. Still no `time_shift`; dates are not reconciled against the times.

---

## How measured / what was run

- Item 7: `item7_pde_v0_boundary_rebate.py`, `item7_call_rebate_lower_row.py`, `item7_moved_values.py`. The parent
  values come from the same scripts with `snowball_pde_solver.py` checked out at 8753894f.
- Item 8: `item8_get_tenor_business_days.py`.
- Item 6: `item6_effective_flags_patch.py` (apply/revert harness) and `item6_cash_deltas.py`. The test-file list:
  - every `test/test_*.py` whose name contains snowball, phoenix, ko_reset, autocall, quad_v2, replay_golden,
    aged_accrual or base_equity_option (DCN excluded);
  - plus `test/accrual_time`, `test/intraday`, `test/pde_bitwise` and `test/pde_statistics`.
- Item 9: `item9_range_accrual_ageing.py`. Item 10: `item10_own_accrual_survey.py`.
