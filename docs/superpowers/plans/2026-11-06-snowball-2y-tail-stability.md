# 2Y Snowball Carry-Tail Stability Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on 2023–2026 CSI 1000 history, whether a carry tail that converges to a trailing far-IM anchor gives a more stable q(2Y) and a lower daily hedging variance than hold-forward, for a single-far-IM hedge and a pillar-lots hedge.

**Architecture:** A self-contained study folder with a uniquely named package `s2ts/` (chain loading, tails, note lifecycle, hedge books, measures, HTML) and four stage scripts. The converging tail is a study-local `CarryCurve` subclass, so no library file changes. Stage 03 runs 13 inceptions × 6 tails on 4 spawn workers with QUAD V2, writing one resumable JSON per run.

**Tech Stack:** Python 3.11 venv at `.venv/`, numpy, pandas, scipy, psutil, `quantark` (CarryCurve, SnowballQuadEngineV2, AutocallableLifecycleTracker, calculate_carry_curve_buckets).

**Spec:** `docs/superpowers/specs/2026-11-06-snowball-2y-tail-stability-design.md` (approved 2026-11-06). Section references below (§4.x) point there.

## Global Constraints

- No file under `quantark/` changes. Everything lives in `example/case_studies/snowball_2y_tail_stability/` and `test/test_snowball_2y_tail_stability_study.py`.
- Canonical `quantark.*` imports only; comparisons through `quantark.util.numerical` (`is_close`, `is_zero`, `safe_log`, `safe_exp`), no hardcoded float tolerances in study code (tests may use explicit `rel=`/`abs=`).
- Engine: `SnowballQuadEngineV2(params=QuadV2Params())`, defaults unchanged. V1 parity engine: `SnowballQuadEngine(params=QuadParams(readout="transition"))`.
- Rate `r = 0.014` flat; ACT/365F year fractions from the day being priced.
- Marks: spot = CSI 1000 close; IM = CFFEX **close** (col 8); MO = CFFEX **settlement** (col 9). The same marks for the curve and every P&L leg.
- IM pillars: listed IM with ≥ 7 calendar days to expiry. MO pillar: longest listed MO expiry. ATM pair = strike minimising `|C − P|` (both legs with a numeric settlement).
- Converging tails: half-lives `(0.25, 0.5, 1.0)` years; start level `c_n = B_n/T_n`; anchor = mean of daily `ln(F_far/S)/T_far` over the trailing **63** trading days ending at `d` (sensitivity only: **252**).
- Note: notional 10,000,000 CNY; strike = KO ref = inception spot; KO 100% monthly from month 3 to month 24; KI 75% discrete daily on SSE days; coupon = KO coupon = rebate; principal excluded; one coupon per inception solved to PV = 0 under `hold_forward`, shared by all tails; desk **short**; each tail books at its own day-0 PV.
- Hedge multipliers: IM 200 CNY/pt/lot; MO C−P pair `100·DF(T_n)`. Costs: IM 1 bp of traded notional; MO 0.2 pt per option leg per side (40 CNY per pair traded). Variance is measured gross of costs.
- V1 parity flag: on **day 0 (inception close, fresh product, no lifecycle state — V1 `price()` takes none)**, |PV_V2 − PV_V1| > 0.5 bp of notional, or any pillar hedge lot differing by > 0.01.
- **Sign convention (design review P1):** `DayGreeks` are **holder-side** (the engine prices the long note). The desk is short, so book targets are `+g.spot_delta·S/(200·F_far)` (Book A), `+δ_i/m_i` per pillar and `+g.pinned_delta·S/(200·F_far)` on the far IM (Book B). Short note daily P&L = `−(PV_t − PV_{t−1}) − cash_t` (cash = what the note pays the holder that day).
- **Cash account:** `cash_t = cash_{t−1}·(1 + r·Δt) − MO pair premium outlay_t (pairs bought × (C−P) × 100, signed) − note cash_t + IM variation margin_t − costs_t`, starting at `cash_0 = PV0_own` (the premium received for the short, at own day-0 PV); Δt = calendar days/365. Accrual enters daily hedged P&L.
- **"Pooled" std (design review P2):** `pooled_std_bp = sqrt(mean over inceptions of per-inception daily variance)` (equal weight per inception), never the std of concatenated days.
- **Terminal hedged P&L (verdict (c), controller ruling):** measured at the **traded price**, identical for every tail: `terminal = 0 (PV0_hf by the coupon solve) − realised note cash + Σ realised hedge P&L + cash accrual`, i.e. net of each tail's day-0 model gap, so tails differ only by their hedges. The own-PV version (`+ PV0_own`) is reported alongside, not used by the verdict.
- Verdict (§4.6): a converging tail beats hold-forward in a book iff (a) pooled daily hedged-P&L std is lower AND (b) its paired variance ratio < 1 in a strict majority of inceptions AND (c) over **completed** (non-truncated) inceptions its terminal hedged-P&L std is lower AND its |terminal| is lower in a strict majority; fewer than 5 completed inceptions ⇒ `"not_decidable"`. Checked separately for Book A and Book B; `beats_hold_forward` needs both. t-stats and Holm p-values reported, not required. Terminal as defined above (traded price), costs excluded.
- **The shared `mo_volmodels/data/history/` cache is READ-ONLY for this study** (design review P2: ~15 stages/tests and two other studies read it, some manifest/last-date sensitive). Days after its last file are downloaded to `OUTPUT_DIR/cffex_extension/` and spot rows to `OUTPUT_DIR/spot_extension.csv`; the loader reads the shared cache first, then the extension.
- Data stay untracked: `example/case_studies/mo_volmodels/data/history/` (input cache) and `example/output/snowball_2y_tail_stability/` (already gitignored via `example/output/`). Tracked: code, tests, `results/*.json`, `results/report.html`, `README.md`.
- Report follows `docs/UI_LANGUAGE.md` (Bloomberg × Swiss, frozen navigator, light/dark switch), self-contained, inline SVG, no external assets.
- Worker processes: `multiprocessing` **spawn** context launched from a script file (never stdin/heredoc), `max_workers=4`, abort if the process group's RSS exceeds 24 GB (psutil).
- New files under `docs/` need `git add -f` (public repo). Commit by pathspec only; never `git add example/` wholesale.
- In a worktree, every Python run is `env PYTHONPATH=<wt>:<wt>/test /Users/fuxinyao/quant-ark/.venv/bin/python …`.

## Review Focus

1. **A day missing from the CFFEX cache or the spot file** (download gap, holiday mismatch): the loader raises `StudyDataError` naming the date and the stage to run; it never silently skips a day, which would fuse two days of P&L into one. → test in Task 1.
2. **The ATM strike has one leg without a numeric settlement** (`--`/`null` on a thin strike): ATM selection only considers strikes where both legs parse; if no strike qualifies the day raises. → test in Task 1.
3. **A pillar expiring or the longest MO rolling while a position is held**: the dropped pillar is closed at that day's mark (its P&L counts), and the new pillar opens at zero; no position is ever marked against a different contract's price. → test in Task 4.
4. **The note knocks out (or matures) mid-run**: that day books the coupon cash against the short, unwinds both books at that close, and nothing is priced afterwards; a KI latches and stays latched. → test in Task 3.
5. **A run interrupted mid-write**: the per-run JSON is written to a temp name and renamed, so resume never reads a half-written file and re-runs only missing runs. → test in Task 5.

---

## File Structure

```
example/case_studies/snowball_2y_tail_stability/
├── README.md                 Task 7: question, setup, headline table, rerun
├── 01_data.py                Task 1: extend cache (CFFEX CSVs + spot), write daily chain
├── 02_stability.py           Task 2: q(2Y) series + stability measures -> results/stability.json
├── 03_backtest.py            Task 5: 13 x 6 runs on 4 workers -> example/output/.../runs/
├── 04_report.py              Task 7: aggregate + results/backtest.json + results/report.html
└── s2ts/
    ├── __init__.py           paths, constants (Global Constraints values), StudyDataError
    ├── chain.py              Task 1: CFFEX parsing, expiries, DayChain, load_chain_frame
    ├── tails.py              Task 2: ConvergingCarryCurve, build_curve, anchors
    ├── stability.py          Task 2: q2y series, day-type classification, measures
    ├── note.py               Task 3: terms per inception, coupon, NoteLife, daily_greeks
    ├── books.py              Task 4: BookA, BookB ledgers and P&L
    ├── runner.py             Task 5: run_one, run_all (pool, resume, RSS guard)
    ├── measures.py           Task 6: per-run stats, paired ratios, Holm, basis regression, verdict
    └── html.py               Task 7: report rendering
test/test_snowball_2y_tail_stability_study.py   all tests (one file, sections per task)
```

Stage scripts and the test file put the study folder on `sys.path` and `import s2ts.<module>`; the unique package name avoids the `_common` collision between studies under xdist.

---

### Task 1: Daily chain data layer

**Files:**
- Create: `example/case_studies/snowball_2y_tail_stability/s2ts/__init__.py`, `s2ts/chain.py`, `01_data.py`
- Create: `test/test_snowball_2y_tail_stability_study.py`
- Add (force): the spec and this plan

**Interfaces:**
- Consumes: `quantark.backtest.simulation.carry.im_expiry(year, month, calendar) -> pd.Timestamp` (MO expiries follow the same third-Friday rule); `quantark.param.vol.marketquotes.implied_vol_black(price, F, K, T, df, is_call)`; CSI 1000 SSE calendar via `quantark.util` `create_calendar(CalendarType.CHINA_SSE)`.
- Produces:
  - `s2ts.StudyDataError(MarketDataError)`; constants `RATE, NOTIONAL, IM_MULT, MO_MULT, MIN_TENOR_DAYS=7, HALF_LIVES=(0.25,0.5,1.0), ANCHOR_WINDOW=63, ANCHOR_WINDOW_LONG=252, HISTORY_DIR, CHAIN_PATH, OUTPUT_DIR, RESULTS_DIR`.
  - `chain.parse_cffex_csv(payload: bytes) -> dict[str, dict[str, float]]` (`{"close","settle"}` per code; non-numeric rows skipped; bad header raises).
  - `chain.contract_expiry(code: str, calendar: pd.DatetimeIndex) -> pd.Timestamp` for `IMyymm` and `MOyymm`.
  - `@dataclass(frozen=True) chain.ImQuote(code, expiry: date, tenor: float, close: float, settle: float)`
  - `@dataclass(frozen=True) chain.MoPillar(series, expiry: date, tenor: float, strike: float, call: float, put: float, forward: float, vol: float, strikes_near: tuple[tuple[float,float,float],...])` — `strikes_near` = the 5 strikes nearest ATM as `(K, C_settle, P_settle)` for the robustness fit and for marking held pairs.
  - `@dataclass(frozen=True) chain.DayChain(day: date, spot: float, im: tuple[ImQuote,...], mo: MoPillar, mo_all: dict[float, tuple[float, float]], far_carry: float)` — `im` filtered to ≥ 7 days, sorted by tenor; `mo_all` = every strike of the pillar series `{K: (C, P)}` (to mark a pair held at a non-ATM strike); `far_carry = ln(F_far/S)/T_far` over the longest listed IM.
  - `chain.build_day_chain(day: date, spot: float, rows: dict, calendar) -> DayChain`
  - `chain.load_chain(path=CHAIN_PATH) -> list[DayChain]` (pickle written by `01_data.py`; fails closed naming `01_data.py`).
- `01_data.py`: downloads to `OUTPUT_DIR/cffex_extension/YYYYMMDD_1.csv` every SSE trading day after the shared cache's last file up to the **latest published** day (CFFEX URL in the spec §3; GBK; sleep 0.5 s between requests; a 404 on a trading day before the last published one raises; a 404 on today before publication ends the range and is logged); writes spot rows from the dashboard DB (`latest_daily_basis`, `index_code='000852.SH'`, slot 1, read-only `?mode=ro`) after `csi1000_spot.csv`'s last date to `OUTPUT_DIR/spot_extension.csv`; never writes under `mo_volmodels/`; builds every `DayChain` and pickles the list to `OUTPUT_DIR/daily_chain.pkl`; prints first/last day, count, and the number of MO-roll days and ATM-switch days.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_cffex_skips_non_numeric_rows(): ...      # fixture bytes (GBK) with header + IM2611 + a "--" MO row -> only IM2611
def test_contract_expiry_third_friday():
    cal = sse_calendar(2026, 2028)
    assert contract_expiry("MO2709", cal) == pd.Timestamp("2027-09-17")
    assert contract_expiry("IM2706", cal) == pd.Timestamp("2027-06-18")
def test_day_chain_on_cached_2026_10_21():               # real cache file; skip if absent
    dc = build_day_chain(date(2026,10,21), spot, rows, cal)
    assert [q.code for q in dc.im] == ["IM2611","IM2612","IM2703","IM2706"]
    assert dc.mo.series == "MO2709"
    k = dc.mo.strike; c, p = dc.mo_all[k]
    assert abs(c - p) == min(abs(a - b) for a, b in dc.mo_all.values())
    assert dc.mo.forward == pytest.approx(k + (c - p) / math.exp(-0.014 * dc.mo.tenor), rel=1e-12)
def test_atm_ignores_strike_with_one_missing_leg(): ...  # Review Focus 2: min-|C-P| strike has P="--" -> next strike chosen; no strike -> StudyDataError
def test_im_under_7_days_excluded(): ...                 # synthetic rows on an expiry-week day
def test_missing_day_fails_closed(tmp_path): ...         # Review Focus 1: StudyDataError mentions the date and "01_data.py"
def test_loader_reads_shared_cache_then_extension(tmp_path): ...   # day present only in extension dir is loaded; shared dir never written (mtime/listing unchanged)
```

- [ ] **Step 2: Run and confirm they fail**
Run: `.venv/bin/python -m pytest -n0 test/test_snowball_2y_tail_stability_study.py -k "cffex or expiry or day_chain or atm or 7_days or missing_day" -v`
Expected: FAIL (`ModuleNotFoundError: s2ts`).

- [ ] **Step 3: Implement `s2ts/__init__.py`, `s2ts/chain.py`, `01_data.py`** to the interfaces above. Vol = mean of call and put Black-76 IV at the strike nearest `forward` (2Y study rule).

- [ ] **Step 4: Run the tests** — same command, expected PASS.

- [ ] **Step 5: Run the stage**
Run: `.venv/bin/python example/case_studies/snowball_2y_tail_stability/01_data.py`
Expected: first day 2023-05-04, last day = latest CFFEX-published trading day (logged), ~880 days, no exception, nothing written under `mo_volmodels/`.

- [ ] **Step 6: Commit**
```bash
git add -f docs/superpowers/specs/2026-11-06-snowball-2y-tail-stability-design.md docs/superpowers/plans/2026-11-06-snowball-2y-tail-stability.md
git add example/case_studies/snowball_2y_tail_stability/s2ts/__init__.py example/case_studies/snowball_2y_tail_stability/s2ts/chain.py example/case_studies/snowball_2y_tail_stability/01_data.py test/test_snowball_2y_tail_stability_study.py
git commit -m "feat(example): 2Y tail-stability study — daily CFFEX chain"
```

---

### Task 2: Tails, anchor, and the q(2Y) stability stage

**Files:**
- Create: `s2ts/tails.py`, `s2ts/stability.py`, `02_stability.py`
- Test: `test/test_snowball_2y_tail_stability_study.py`

**Interfaces:**
- Consumes: Task 1 `DayChain`, `load_chain`; `quantark.param.curve.carry_curve.CarryCurve`, `CarryExtrapolation`, `CurveInterpolation`.
- Produces:
  - `TAILS = ("hold_forward", "hold_yield", "zero_carry", "converge_hl025", "converge_hl050", "converge_hl100")`
  - `class ConvergingCarryCurve(CarryCurve)`: `__init__(nodes, *, b_inf: float, half_life: float, **kw)`; overrides `carry(T)`, `forward_carry(T)` and `interval_carry(t0, t1)` for `T > T_n` with the spec §4.2 formulas, computing `(1−e^{−κτ})/κ` as `-np.expm1(-κτ)/κ` (no cancellation at tiny κ); identical to the base inside `[0, T_n]`; `with_nodes(nodes)` returns a `ConvergingCarryCurve` with the same `b_inf`, `half_life`. **Construct the base with `extrapolation=ZERO_FORWARD_CARRY`** so `CarryImpliedDividendYield.parallel_shifted` takes the exact wrapper path rather than the node-shift path (which would be wrong for this tail).
  - `anchor_series(chains: list[DayChain], window: int) -> dict[date, float]` — trailing mean of `far_carry` over the last `window` days ending at `d`; days with fewer than `window` observations absent.
  - `build_curve(dc: DayChain, tail: str, b_inf: float | None, *, mo_forward: float | None = None) -> CarryCurve` — nodes `(tenor, ln(F/S))` from `dc.im` closes then `dc.mo`; `mo_forward` overrides the MO pillar (switch-floor and robustness variants); `b_inf` required for converging tails; `LOG_LINEAR_DF` interior.
  - `pillar_inputs(dc) -> tuple[list[str], list[float]]` — labels and lot multipliers (`200` per IM, `100·exp(−r·T_n)` for MO).
  - `stability.q2y(curve) -> float` = `r − B(2.0)/2.0`.
  - `stability.day_types(chains) -> dict[date, str]` ∈ `{"ordinary","mo_roll","atm_switch"}` (mo_roll wins when both).
  - `stability.switch_floor(prev: DayChain, dc: DayChain, tail, b_inf) -> float` = q2y with today's MO pillar priced at **yesterday's ATM strike** minus q2y at today's ATM (same day's settlements).
  - `stability.summarize(series: dict[date,float], types) -> dict` with `std, mad, p99_abs` of daily Δ (in bp of yield) overall and per day type.
  - `02_stability.py` writes `results/stability.json`: per tail × {base, multi_strike (F = mean over the 5 strikes of `K + (C−P)/DF`), all_settlement, anchor_252 (converging tails only, dates with a 252 anchor)} the summary, the switch-floor distribution (median, p90), and the daily q2y series.

- [ ] **Step 1: Write the failing tests**

```python
def test_converging_matches_base_inside_last_node(): ...   # B(t) equal to plain CarryCurve for t <= T_n, rel 1e-14
def test_converging_continuous_at_last_node(): ...          # B(T_n ± 1e-9) within 1e-12
def test_tiny_kappa_reproduces_hold_yield():
    # half_life = 1e9 -> B(2.0) equals the FLAT_CARRY_YIELD curve's B(2.0) within 1e-9
def test_huge_kappa_forward_carry_is_anchor(): ...          # half_life = 1e-6: forward_carry(T_n + 0.1) == b_inf within 1e-9
def test_closed_form_value():
    # nodes [(0.5,-0.03),(0.9,-0.06)], b_inf=-0.05, hl=0.5, t=2.0:
    # expected = -0.06 + (-0.05)*1.1 + ((-0.06/0.9) - (-0.05))*(1-exp(-k*1.1))/k, k=ln2/0.5
def test_with_nodes_keeps_type_and_params(): ...
def test_bucket_report_runs_and_delta_identity_holds():
    # 2Y study snapshot curve rebuilt as converge_hl050; calculate_carry_curve_buckets with V2:
    # spot_delta == pinned_forward_delta + sum(F_i/S * delta_i) within 0.5 CNY/pt, pinned != 0
def test_anchor_point_in_time(): ...                        # mutating far_carry after d leaves anchor[d] unchanged; first anchored day = index 62
def test_day_types_and_switch_floor_zero_when_no_switch(): ...
```

- [ ] **Step 2: Run, confirm FAIL** — `… -k "converging or kappa or closed_form or with_nodes or bucket_report or anchor or day_types" -v`.
- [ ] **Step 3: Implement `tails.py`, `stability.py`, `02_stability.py`.**
- [ ] **Step 4: Run the tests, expect PASS.**
- [ ] **Step 5: Run the stage** — `.venv/bin/python example/case_studies/snowball_2y_tail_stability/02_stability.py`; expected: `results/stability.json` with 6 tails × 4 variants; prints the std Δq2y table; runtime < 5 min.
- [ ] **Step 6: Commit** (pathspec: the three files, the test file, `results/stability.json`).

---

### Task 3: The note — terms, coupon, lifecycle, daily greeks

**Files:**
- Create: `s2ts/note.py`
- Test: `test/test_snowball_2y_tail_stability_study.py`

**Interfaces:**
- Consumes: Task 2 `build_curve`, `pillar_inputs`; `quantark.asset.equity.product.option.snowball_helpers.create_standard_snowball`; `quantark.asset.equity.lifecycle.autocallable.AutocallableLifecycleTracker`; V2 engine `price(product, env, *, lifecycle_state=None)`.
- Produces:
  - `inception_dates(chains, anchors) -> list[date]` — first trading day of each calendar quarter that has a 63-day anchor; expected first 2023-10-09, ~13 dates.
  - `build_terms(inception: date) -> Terms` — generalises the 2Y study's `build_terms` to any inception (same SSE rolling rules, times ACT/365F from inception).
  - `build_note(coupon: float, s0: float, terms) -> SnowballOption` — the 2Y study's `build_product` with `contract_multiplier = NOTIONAL/s0`.
  - `make_env(dc: DayChain, div_yield, spot: float | None = None) -> PricingEnvironment` — flat vol `dc.mo.vol`, flat `r`, `valuation_date = dc.day`.
  - `solve_coupon(dc, engine) -> float` under `hold_forward` (copy the 2Y study's affine two-probe solver; re-price check `1e-6·NOTIONAL`).
  - `class NoteLife`: `__init__(product, inception: date)` wraps `AutocallableLifecycleTracker(product=product, quantity=1.0, start_date=pd.Timestamp(inception))`; `step(dc: DayChain, env) -> NoteDay` does, in order (design review P1, mirroring `quantark/backtest/replay/engine.py:463-480`): `tracker.observe(day, tracker.product_for_lifecycle(), env, spot)`, then `tracker.settle_maturity_if_due(day, tracker.product_for_lifecycle(), env, spot)`, then `product = tracker.product_for_pricing(day, env)`. Returns `NoteDay(alive: bool, cash: float, knocked_in: bool, end_reason: "ko"|"maturity"|None, product)`; `cash` = increase in the ledger's realised cashflows that day (what the note pays the holder: KO coupon, rebate or KI loss, sign as the ledger books it, normalised so a payment TO the holder is positive). Day 0 uses `initial=True` and no observation.
  - `@dataclass DayGreeks(pv, spot_delta, pinned_delta, pillar_delta: tuple[float,...], pillar_lots: tuple[float,...])`
  - `daily_greeks(product, env, curve, engine, lifecycle_state, lot_mults) -> DayGreeks` — 15 prices **in the bucket report's call order** (base, spot ±1%, pinned-forward ±1% (every node `B_i − ln(S'/S)`), then each pillar `F_i ± 0.1%`) so V2's single-slot cache behaves identically; same formulas as `calculate_carry_curve_buckets` (spot bump 0.01, forward bump 1e-3); no rho. Holder-side values (see Global Constraints sign convention); `pillar_lots = −δ_i/m_i` are holder lots, the books negate.

- [ ] **Step 1: Write the failing tests**

```python
def test_daily_greeks_match_bucket_report():
    # 2Y study snapshot, hold_forward and converge_hl050, V2: pv, spot_delta, pinned_forward_delta,
    # each pillar delta equal to calculate_carry_curve_buckets(...) within rel 1e-12
def test_coupon_solves_par(): ...                 # synthetic DayChain; |pv(coupon)| <= 1e-6 * NOTIONAL
def test_note_knocks_out_on_first_ko_date():      # Review Focus 4
    # synthetic path at 105% of strike: alive until month-3 KO date, that day cash == coupon*NOTIONAL*accrual(>0), then alive False
def test_note_ki_latches(): ...                    # one close at 70%, then back to 100%: knocked_in stays True, pv lower than unknocked twin
def test_note_matures_without_ko_no_ki(): ...      # path in [76%, 99%] to maturity: end_reason "maturity", cash = rebate*NOTIONAL*accrual, alive False after
def test_note_matures_after_ki(): ...              # KI then below strike at maturity: end_reason "maturity", cash = -(1 - S_T/S0)*NOTIONAL (holder loses)
def test_day0_aged_pv_equals_fresh_pv(): ...       # NoteLife day 0 product priced == fresh product priced, rel 1e-12
def test_inception_dates(): ...                    # real chain if present: first == 2023-10-09, all quarter-firsts, count in [12, 14]
```

- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement `note.py`.** Mirror `quantark/backtest/replay/product_replay.py:225-260` and `:955` for how the replay engine calls `product_for_pricing` and `observe`; pass `tracker.lifecycle` as `lifecycle_state`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** (pathspec).

---

### Task 4: Hedge books

**Files:**
- Create: `s2ts/books.py`
- Test: `test/test_snowball_2y_tail_stability_study.py`

**Interfaces:**
- Consumes: Task 1 `DayChain`; Task 3 `DayGreeks`.
- Produces:
  - `@dataclass Trade(instrument: str, qty: float, price: float, cost: float)`
  - `class BookA`: holds one IM contract = the **longest listed** IM (rolls on the first day a longer contract appears in `dc.im`). `target(dc, g: DayGreeks) -> dict[str,float]` = `{far: +g.spot_delta·S/(200·F_far)}` (short desk, holder-side greeks).
  - `class BookB`: `target(dc, g)` = each IM pillar `+δ_i/200`, plus `+g.pinned_delta·S/(200·F_far)` **added to the far IM**, and the MO pair `+δ_MO/(100·DF)`. The MO position is keyed `MOyymm@K`; on open it takes `dc.mo.strike`; afterwards only its quantity changes; on an MO-roll day the old key is closed and a new one opened at the new ATM strike.
  - Shared base `Book`: `mark(dc) -> dict[str,float]` (IM close; MO pair `C−P` at the held strike from `dc.mo_all`, raising if the strike disappears); `pnl(prev: DayChain, dc: DayChain) -> float` (positions held overnight × price change × multiplier: IM 200, MO 100); `rebalance(dc, target) -> list[Trade]` (closes keys absent from target at today's mark; costs per Global Constraints); `unwind(dc) -> list[Trade]`.
  - A position whose contract drops out of the pillar set (IM < 7 days) is closed at today's mark by `rebalance`.

- [ ] **Step 1: Write the failing tests**

```python
def test_linear_note_is_perfectly_hedged():
    # synthetic: holder PV = a*(F_far - F_far0) with holder spot_delta = a*F_far/S (a pure forward);
    # desk SHORT note pnl = -(PV_t - PV_{t-1}); BookA target from that holder delta;
    # short note pnl + BookA pnl == 0 to 1e-9 over a 10-day path, AND a sign-flipped target makes it 2x (guards the sign)
def test_book_a_rolls_to_new_longest(): ...        # day when a new IM lists: one close + one open trade, cost = 1e-4*|qty|*F*200 each
def test_mo_pair_keeps_opened_strike(): ...        # ATM moves 6600->6400: key stays MO2709@6600, only qty changes, marked at 6600 C-P
def test_mo_roll_closes_old_pair():                # Review Focus 3: old pair closed at today's settle; its overnight P&L counted; new key opens qty from 0
def test_dropped_im_pillar_closed_at_mark(): ...   # IM front falls under 7 days -> closed, P&L that day uses its close
def test_book_b_spot_neutral():
    # with greeks satisfying spot_delta = pinned + sum(F_i/S * delta_i):
    # sum_IM lots*200*F_i/S + pairs*100*DF*F_MO/S == +spot_delta within 1e-6 CNY/pt (offsets the short's -spot_delta)
def test_mo_cost_40_cny_per_pair(): ...
```

- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement `books.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** (pathspec).

---

### Task 5: Runner and the backtest stage

**Files:**
- Create: `s2ts/runner.py`, `03_backtest.py`
- Test: `test/test_snowball_2y_tail_stability_study.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `run_one(inception: date, tail: str, chains: list[DayChain], anchors: dict, coupon: float, *, parity: bool = True) -> dict` — the daily loop:
    1. Day 0 (inception close): build curve for `tail`, env, `NoteLife`; `DayGreeks`; book PV at own day-0 PV; set both books to target (record day-0 trades).
    2. Each later day while alive and data remain: P&L of note `= −(PV_t − PV_{t−1}) − cash_t`, books' `pnl(prev, dc)`, cash account accrual `r·cash·Δt`; record per book `hedged_pnl = note_pnl + book_pnl`; then if alive recompute greeks and rebalance, else unwind.
    3. Day-0 V1 parity (when `parity`): price base and pillar lots with V1 on the fresh product; record `parity_flag`.
    - Returns `{"inception","tail","coupon","day0_pv","truncated": bool, "end_reason": "ko"|"maturity"|"data_end", "parity": {...}, "terminal_a","terminal_b" (traded-price), "terminal_own_a","terminal_own_b", "days": [{"date","note_pnl","pnl_a","pnl_b","cost_a","cost_b","turn_a","turn_b","roll_a","roll_b","d_far_basis","d_mo_basis","pv","spot_delta","pinned_delta"} ...]}`. `d_far_basis` and `d_mo_basis` each use the same contract both days (null on that leg's roll day); the basis OLS drops rows where either is null.
  - `run_all(inceptions, tails, out_dir, *, workers=4) -> list[Path]` — spawn pool; skips a run whose `<inception>__<tail>.json` exists; writes `*.json.tmp` then `os.replace`; RSS guard polls every 30 s and terminates the pool (process group) above 24 GB.
  - `03_backtest.py [--quick] [--workers N]` — `--quick` = first inception × (`hold_forward`, `converge_hl050`), 30 days; full = all inceptions × `TAILS`; out dir `OUTPUT_DIR/runs/`; prints progress and total wall time; coupon per inception solved once (hold_forward) and cached in `OUTPUT_DIR/coupons.json`.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_one_synthetic_short_path(): ...      # 8-day synthetic chain list: returns 7 P&L days, keys as specified, end_reason "data_end", truncated True
def test_same_coupon_and_contract_across_tails(): ...   # two tails, same inception: identical coupon, identical spot path; day0_pv differs
def test_resume_skips_complete_and_ignores_tmp(tmp_path):   # Review Focus 5
    # pre-create a.json (complete) and b.json.tmp (partial): run_all runs b only, never reads b.json.tmp, leaves no .tmp behind
def test_parity_flag_logic(): ...                 # pv diff 0.6 bp -> flagged; 0.4 bp and lots diff 0.005 -> not flagged
```

- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement `runner.py`, `03_backtest.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Quick run** — `.venv/bin/python example/case_studies/snowball_2y_tail_stability/03_backtest.py --quick`; expected: 2 run files, no parity flag, prints measured s/day. `03_backtest.py --plan` prints, from the spot path alone (KO/KI do not depend on the tail), the live-day count per inception and the budget `Σ days × 6 × s/day / 4`; it must fit one night (≤ 10 h) or the controller records a ruling (e.g. drop the 1.0Y half-life) before launch.
- [ ] **Step 6: Commit** (pathspec).

---

### Task 6: Measures and verdict

**Files:**
- Create: `s2ts/measures.py`
- Test: `test/test_snowball_2y_tail_stability_study.py`

**Interfaces:**
- Consumes: Task 5 run dicts.
- Produces:
  - `per_run(run: dict) -> dict` per book: `std_bp` (daily hedged P&L std, bp of notional), `r2 = 1 − var(hedged)/var(note_pnl)`, `turnover_lots_per_day`, `roll_day_abs_bp` (mean |hedged P&L| on roll days), `cost_bp`, `terminal_bp`, `max_dd_bp`, `basis` = OLS of hedged P&L on `(d_far_basis, d_mo_basis)` on non-roll days → `beta_far, beta_mo` (bp per 1% basis) and `basis_r2`.
  - `paired(runs_by_tail: dict[str, list[dict]], book: str) -> dict[str, dict]` vs `hold_forward`: per tail `pooled_std_bp`, `ratios` (per-inception var ratio), `n_lower`, `n`, `t_log_ratio`, `p_holm` (Holm across the three converging tails).
  - `terminal_paired(runs_by_tail, book) -> dict[str, dict]` — completed runs only: per tail `terminal_std_bp`, `n_abs_lower`, `n_complete`.
  - `verdict(paired_a, paired_b, terminal_a, terminal_b) -> dict[str, dict]` — per converging tail `{"book_a": True|False|"not_decidable", "book_b": …, "daily_ok_a", "terminal_ok_a", "daily_ok_b", "terminal_ok_b", "beats_hold_forward": bool}` by the Global Constraints rule; also a `complete_only` re-evaluation of the daily conditions excluding truncated runs.
  - `basis_mitigation(per_run_rows) -> dict` — per tail: mean `|beta_far|`, `|beta_mo|`, `basis_r2` for A and B and the B−A differences, plus B−A cost.

- [ ] **Step 1: Write the failing tests**

```python
def test_std_and_r2_on_known_series(): ...
def test_basis_regression_recovers_beta(): ...      # synthetic pnl = 3*d_far - 1*d_mo + noise(1e-6): betas within 1e-4
def test_holm_adjustment(): ...                     # p = [0.01, 0.04, 0.03] -> [0.03, 0.06, 0.06]
def test_pooled_std_equal_weight(): ...             # inception A 10 days var 1, inception B 100 days var 4 -> pooled = sqrt(2.5), not concatenated std
def test_terminal_traded_price_net_of_day0_gap(): ... # two tails, identical hedge P&L, day0_pv differs by 100 bp -> identical traded-price terminal
def test_verdict_requires_both_conditions():
    # lower pooled std but lower in only 6/13 -> False; lower in 7/13 but higher pooled std -> False; both -> True
def test_verdict_per_book_independent(): ...        # True in A, False in B -> beats_hold_forward False, book flags reported separately
def test_verdict_requires_terminal_agreement():
    # daily (a)+(b) pass but completed-run terminal std higher -> book False, terminal_ok False
def test_verdict_not_decidable_under_5_complete(): ...   # 4 completed inceptions -> "not_decidable", beats_hold_forward False
def test_terminal_excludes_truncated(): ...         # a truncated run's huge terminal value does not move terminal_std_bp
```

- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement `measures.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** (pathspec).

---

### Task 7: Report, README

**Files:**
- Create: `s2ts/html.py`, `04_report.py`, `README.md`
- Test: `test/test_snowball_2y_tail_stability_study.py`

**Interfaces:**
- Consumes: `results/stability.json`, `OUTPUT_DIR/runs/*.json`, Task 6.
- Produces: `results/backtest.json` (per-run summaries, paired tables, verdict, basis mitigation, parity flags, truncation list) and `results/report.html` with sections, in order: `#verdict` (decision table: tail × {Book A std, Book B std, n_lower/n per book, verdict}), `#stability` (std Δq2y per tail × variant, switch floor, q2y chart), `#hedging` (per-book tables incl. terminal measures on completed runs, paired ratio strip chart, inception-day model gap `day0_pv − day0_pv_hold_forward` per tail as a separate static table), `#basis` (calendar-spread question: betas and basis R², B−A), `#curve` (anchor series, one illustrative day's six tails), `#method` (setup, marks, costs, caveats incl. overlapping inceptions and truncation), `#parity`.
- Style: copy the token/CSS/nav/theme-switch scaffolding pattern from `example/case_studies/snowball_2y_carry_curve/04_report.py` (itself on `docs/UI_LANGUAGE.md`); inline SVG charts only.
- `README.md`: question, setup table, headline table filled from `results/*.json` by hand after Task 8, rerun commands, files table, open items — same shape as the 2Y study README.

- [ ] **Step 1: Write the failing test**

```python
def test_report_structure(tmp_path):
    # render from small synthetic results: every section id present, a theme toggle element and a fixed nav present,
    # no "http" in src/href attributes, verdict table has one row per converging tail
```

- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement `html.py`, `04_report.py`; write README skeleton.**
- [ ] **Step 4: Run, expect PASS.** Also run the whole test file: `.venv/bin/python -m pytest test/test_snowball_2y_tail_stability_study.py -v` — all PASS.
- [ ] **Step 5: Commit** (pathspec).

---

### Task 8: Overnight run, results, final review

**Files:**
- Modify: `README.md` (headline numbers), `results/backtest.json`, `results/report.html`

- [ ] **Step 1: Check the machine** — `pmset -g log | tail -5` and `uptime`; no other heavy pools running.
- [ ] **Step 2: Launch**
```bash
nohup caffeinate -i -m -s .venv/bin/python example/case_studies/snowball_2y_tail_stability/03_backtest.py --workers 4 \
  > example/output/snowball_2y_tail_stability/backtest.log 2>&1 &
```
Expected: 78 run files (13 × 6) by morning; log ends with total wall time.
- [ ] **Step 3: Verify the wall clock is real** — `pmset -g log | grep -i sleep` over the run window; if the host slept, note it and do not quote the wall time.
- [ ] **Step 4: Aggregate** — `.venv/bin/python example/case_studies/snowball_2y_tail_stability/04_report.py`; expected: `results/backtest.json`, `results/report.html`, zero or listed parity flags.
- [ ] **Step 5: Fill README headline table** from `results/*.json` (numbers copied, not retyped by memory).
- [ ] **Step 6: Full suite** — `.venv/bin/python -m pytest`; expected: no new failures vs main (known pre-existing mo_volmodels/otc failures excepted, listed by name).
- [ ] **Step 7: Commit** (pathspec: README, results/*.json, results/report.html).
- [ ] **Step 8: Whole-branch review** (fresh reviewer), then hand the merge to the user (ff-only main).
