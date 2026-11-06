# 2Y snowball carry tail: does converging to an anchor beat hold-forward?

Date: 2026-11-06 · Status: design, awaiting approval (builder check done) · Builds on
`example/case_studies/snowball_2y_carry_curve/` (landed dfdaa4b8) and
`example/case_studies/snowball_q_term_structure/`.

## 1. Intent

The CSI 1000 chain (IM futures + MO options) quotes the carry `B(T) = ln F(T)/S`
only out to the longest MO expiry (~0.6–1.0Y). A 2Y snowball lives past that, so
`q` beyond the last quote is set by a tail convention. The 2Y study showed the
default convention, **hold-forward** (`FLAT_FORWARD_CARRY`: continue the last
segment's slope), maps that unquoted risk into a levered calendar spread on the
last two pillars (on 2026-11-05: `∂B(T_mat)/∂B_IM2706 = −4.57`,
`∂B(T_mat)/∂B_MO2709 = +5.57`; hedge +11.25 IM2706 lots vs −31.26 MO2709 C−P
pairs). It also amplifies noise in the last two marks ~5× into q(2Y).

The literature (Smith–Wilson / EIOPA UFR; dividend term-structure work showing
mean-reverting implied yields) favours **converging the tail to an explicit
long-run anchor** instead of extending a local slope.

**The decision this study informs (user's words):** is a converging tail a more
stable estimate for long-tenor q? "Stable" means two things, both measured on
history:

1. **Day-to-day stability of q(2Y)** — the curve rebuilt each day from that
   day's chain.
2. **Hedging variance** — the variance of the daily hedged P&L of a 2Y snowball
   hedged daily under each tail.

**Adoption rule (user):** a converging tail replaces hold-forward as the desk
default if it gives **lower hedging variance**.

## 2. Goals, non-goals, anti-goals

Goals
- A daily historical series of q(2Y) under every tail, with its day-to-day change
  attributed to market moves vs. mechanical events (longest-MO-expiry roll, ATM
  strike switch).
- A paired hedging backtest, two hedge books × six tails, on quarterly inceptions.
- One decision table that ranks tails by hedging variance, per book, with the
  q(2Y) stability numbers alongside.

Non-goals
- No library change. The converging tail lives in study code and is applied at
  runtime (standalone-demo-first); promotion into `CarryCurve` is a separate,
  later decision made from this study's table.
- No flat-q model (the 2Y study already covered flat vs curve).
- No smile / local / stochastic vol, no stochastic basis, no rate curve.
- No monthly inceptions (budget: one overnight run).

Anti-goals (would make the result worthless even if it runs)
- **Look-ahead.** The anchor, the vol, the pillar set and every mark on date `d`
  must use only data available at the close of `d`.
- **Unpaired comparisons.** Every tail of one inception sells the *same* note
  on the *same* path; differences are read paired.
- **Strike-switch noise mistaken for tail instability.** The MO pillar's ATM
  strike changes day to day; that noise must be measured and shown next to the
  stability numbers, not silently folded in.
- **A tail ranking that is an artifact of an engine error.** QUAD V2 is used
  (see §4.7) with a V1 parity check.

## 3. Constraints

- Data: local, untracked cache `example/case_studies/mo_volmodels/data/history/`
  (`csi1000_spot.csv`, `im_futures.csv`, `settlement_csv/YYYYMMDD_1.csv` =
  CFFEX daily files incl. every MO strike), 2023-05-04 → 2026-10-21, 840 days.
  Extend to the latest available trading day (≥ 2026-11-06) by downloading CFFEX
  daily CSVs (`http://www.cffex.com.cn/sj/hqsj/rtj/YYYYMM/DD/YYYYMMDD_1.csv`, GBK,
  col 8 close / col 9 settle; reachable, verified for 2024–2025 dates) and spot
  from the dashboard DB (`/Users/fuxinyao/index-dashboard-codex/data/basis.sqlite3`,
  read-only) or akshare. The study's own extracted inputs stay untracked
  (public repo; see memory "no desk data in public repo" — the CFFEX data is
  public, but follow the existing cache convention).
- Every loader fails closed naming the stage to run when an input is missing.
- Engine: `SnowballQuadEngineV2` with default `QuadV2Params`. Measured on the 2Y
  study note (2026-11-06, same process, interleaved): cold price 0.36 s vs V1
  0.91–1.01 s; full bucket report 9.1–9.3 s vs 21.9–30.4 s; PV within 0.04 bp;
  pillar hedge lots within 0.002. Pillar bumps change the carry and always
  re-prepare (no cache hit); only spot bumps reuse a context.
- Budget: one unattended overnight run on 4 workers (~6 h estimated:
  84 runs × ~220 live days × ~13 prices × 0.36 s). Use
  `nohup caffeinate -i -m -s`, resumable per-run outputs, process-group kill,
  and a group-RSS guard (memory notes on wide pools).
- House conventions: canonical `quantark.*` imports, `quantark.util.numerical`
  comparisons, report in the Bloomberg × Swiss UI language
  (`docs/UI_LANGUAGE.md`, frozen navigator + light/dark switch; reference
  `example/snowball_ftp_study/05_report.py`), tests in `test/`.

## 4. Decided design

### 4.1 Daily curve (shared by every tail)

On each trading day `d`:
- Marks (builder call, = the 2Y study's base marks): spot = CSI 1000 close,
  IM = **close**, MO = **settlement**. The same marks are used for the curve AND
  for every P&L leg, so the note and its hedge are marked on one convention.
  The full IM chain comes from the CFFEX daily CSVs (col 8 close), not
  `im_futures.csv` (that file has only 2–3 contracts on 181 of 840 days).
  The all-settlement chain is run as a sensitivity in the stability test only
  (§4.3), because a common close-vs-settle shift in every `B_i` cancels in
  hold-forward's slope but is absorbed by every level-anchored tail.
- IM pillars: every listed IM contract with ≥ 7 calendar days to expiry
  (min-tenor rule from the q-term-structure study).
- MO pillar: the **longest listed MO expiry**, forward from put–call parity on
  the **ATM pair** (the user's convention, = Cboe's min-|C−P| rule): strike
  nearest the forward, `F = K + (C − P)/DF(T)`.
- Curve: `CarryCurve` through these pillars, `LOG_LINEAR_DF` interior, same as
  the 2Y study. Rate `r` flat 1.4% (drift is `B/T`, independent of `r`; `r`
  only discounts).
- Vol: flat, the Black-76 IV of that day's MO ATM pair (2Y study rule),
  deterministic so V2 accepts it.

### 4.2 The six tails (beyond the last pillar `T_n`)

| id | rule |
|---|---|
| `hold_forward` | existing `FLAT_FORWARD_CARRY` |
| `hold_yield` | existing `FLAT_CARRY_YIELD` |
| `zero_carry` | existing `ZERO_FORWARD_CARRY` |
| `converge_hl025` / `_hl050` / `_hl100` | converge to anchor, half-life 0.25 / 0.5 / 1.0Y |

Converging tail, with `c_n = B_n/T_n` (start from the last zero carry **level**,
not the last slope, so no calendar-spread term), `κ = ln 2 / half-life`, `τ = t − T_n`:

```
b(t) = b_∞ + (c_n − b_∞)·e^{−κτ}
B(t) = B_n + b_∞·τ + (c_n − b_∞)·(1 − e^{−κτ})/κ
```

Anchor `b_∞` on date `d`: the mean, over the trailing **3 months (63 trading
days ending at `d`)**, of the daily far-IM carry `ln(F_far/S)/T_far`, where
`F_far` is the longest listed IM contract. Point-in-time; no inception before 63
days of history exist (first ≈ 2023-08). `b_∞` is not hedged; its daily drift is
part of what the hedging variance measures.

For the q(2Y) stability series only (cheap, curve-only), also compute the three
converging tails with a **1Y** trailing anchor on dates where it exists
(≥ ~2024-05), as a sensitivity to the anchor window.

### 4.3 Stability test (all days, curve only)

- `q_d(2Y) = r − B_d(2.0)/2.0` at a **constant 2.0Y tenor**, per tail.
- Metrics per tail: std and MAD of `Δq(2Y)`, 99th-percentile |Δ|, on all days
  and split into: ordinary days / longest-MO-expiry roll days / ATM strike
  switch days.
- **Strike-switch noise floor**: on every switch day, the MO forward from both
  the previous and the new ATM strike (same day's settlements); the difference,
  pushed through each tail's `∂q(2Y)/∂B_n`, is the q(2Y) noise attributable to
  the switch alone.
- **Robustness**: rebuild the MO pillar by a weighted fit of `C − P = DF·(F − K)`
  over the near-ATM strikes; and rebuild the whole chain on all-settlement
  marks. Report whether the tail ranking on `std Δq(2Y)` changes under either.

### 4.4 Hedging backtest

- Inceptions: quarterly, first trading day of each quarter from the first date
  with a 3M anchor (≈ 2023-08/10) to the last quarter with data — ≈ 13.
- Note: the 2Y study's term sheet re-struck at each inception (strike = KO ref
  = spot; KO 100% monthly from month 3; KI 75% discrete daily on SSE days;
  coupon = KO coupon = rebate; notional 10 mio CNY; principal excluded).
  **One coupon per inception**, solved to PV = 0 under `hold_forward`, shared by
  all six tails so every tail sells the identical contract.
- Position: the desk is **short** the note. Builder call: each tail books the
  note at **its own day-0 model PV**, so the inception-day model gap (e.g.
  −125 / +158 bp vs hold-forward on 2026-11-05) is not P&L; it is reported
  separately as a static number per inception. Daily and terminal P&L then
  measure hedge error plus re-marking under that tail only.
- Life: until KO, maturity, or the last data day (truncated; flagged).
  Lifecycle (KI latched, KO observed) from the realised path, priced with V2's
  lifecycle state.
- Daily per (inception, tail): note PV, spot delta, pillar deltas
  `∂PV/∂F_i` (central bumps, `calculate_carry_curve_buckets` settings of the 2Y
  study), computed once and shared by both books.
- **Book A — single far IM**: the longest listed IM contract, lots
  `= −Δ_spot·S/(200·F_far)` (spot-neutral `S/F` sizing). Builder call:
  **follow the longest** — roll into a newly listed longest IM on its first
  listed day (quarterly), so the hedge stays far-dated. This is NOT the
  q-term-structure study's `FarContractRollPolicy`, which holds the contract
  until 5 days before expiry; a new policy is written in study code.
- **Book B — pillar lots**: `−(∂PV/∂F_i)/m_i` in every pillar instrument
  (IM `m = 200`; MO C−P pair `m = 100·DF(T_n)`), **plus the pinned-forward
  delta** (`∂PV/∂S` with every pillar forward fixed, reported by
  `calculate_carry_curve_buckets`) hedged in the Book A far IM contract at
  `−Δ_pinned·S/(200·F_far)`. Builder call, so Book B is spot-neutral under
  every tail: the residual is ≈ 0 under hold-forward (its tail does not depend
  on `S`) but not under level-anchored tails (`∂ln F(t)/∂ln S =
  −(1−e^{−κτ})/(κT_n)`, ≈ −0.66 at 2Y for hl050, ≈ −1.3 for hold-yield; the 2Y
  study measured −620 CNY/pt for hold-yield), so leaving it open would bias the
  verdict against every converging tail. The MO pair is held **at the
  strike it was opened at** (a C−P pair at any strike is the same synthetic
  forward plus cash), only its quantity rebalances; when the longest MO expiry
  changes, the old pair is closed and the new pillar opened. An IM contract
  dropping below 7 days leaves the pillar set and is closed.
- Daily hedged P&L = ΔPV(note, short) + Σ lots·ΔF·m (IM) + Σ pairs·Δ(C−P)·100
  (MO), marked at settlement, plus carry on cash at `r`.
- Costs: IM 1 bp of traded notional (q-term-structure study); MO 1 tick
  (0.2 pt) per option leg per side. Variance is measured **gross of costs**;
  costs and turnover are reported separately.

### 4.5 Measures and decision table

Per tail × book, paired by inception against `hold_forward`:
- **primary:** daily hedged-P&L variance (std in bp of notional), paired
  variance ratio vs hold-forward with an inception-level paired test (overlapping
  windows caveat stated);
- secondary: hedge R², lots/day and turnover, roll-day |ΔMTM|, cost drag,
  terminal P&L (mean, std), max drawdown;
- alongside: std Δq(2Y) and its strike-switch floor from §4.3.

Aggregation (builder calls): per-inception std of daily hedged P&L; paired
log variance ratio vs hold-forward per inception; inceptions weighted
equally; truncated (still-live) inceptions included and flagged, with a
complete-life-only row as a check; the three half-lives get a Holm correction
on the paired test. Daily variance (marked under each tail's own model) is the
primary measure, and the terminal measure must agree with it (§4.6 (c));
any disagreement is called out in the verdict text.

The table marks which converging tails beat hold-forward on the primary
measure in each book. **Verdict rule: see §4.6.**

### 4.6 Verdict rule

(User, 2026-11-06.) **Both books count, and a consistent lower point estimate
is enough.** A converging tail beats hold-forward when, in **Book A and in
Book B separately**: (a) its pooled daily hedged-P&L std is lower than
hold-forward's, and (b) its paired variance ratio is below 1 in a majority of
inceptions, **and (c) the terminal measure agrees** (user, 2026-11-06: "the 2
measures need to agree"): across completed inceptions (KO or maturity, not
truncated), its terminal hedged-P&L std is lower than hold-forward's AND its
|terminal hedged P&L| is lower in a strict majority of them. Reason: a tail
anchored to a slow 3M average smooths its own daily marks, which lowers daily
variance without reducing real risk; terminal P&L, measured at the traded
price (the same for every tail: 0 by the hold-forward coupon solve) as
− realised note cash + realised hedge P&L + cash accrual, depends on neither
the daily marks nor each tail's day-0 model gap (controller ruling after
design review; the own-PV version is reported alongside). Truncated runs are
excluded from (c) because their end value is itself a model mark. If too few
inceptions complete for (c) to be read (fewer than 5), the verdict is "not
decidable on this history", not a pass. Paired t-statistics and the
Holm-adjusted p-values are reported, not required to pass. If several
half-lives qualify, the table names all of them, ordered by Book A std.

**Second question (user):** can the calendar spread mitigate basis
oscillation? Measured per tail × book: regress daily hedged P&L on the daily
change in far-IM basis (`ln F_far/S`) and in the MO pillar basis; report the
basis betas, the share of hedged-P&L variance they explain, and Book B minus
Book A on both. A calendar spread that helps shows up as a smaller basis
beta and a smaller basis-explained variance in Book B than in Book A under the
same tail, net of its extra turnover cost.

### 4.7 Engine parity

On the first day of every inception and every tail, also price with the V1
study engine (`SnowballQuadEngine`, readout `transition`); flag any run whose
PV differs by > 0.5 bp or any pillar hedge lot by > 0.01 lots, and list them in
the report.

### 4.8 Deliverables

New folder `example/case_studies/snowball_2y_tail_stability/`:
`_common.py` (tails incl. the converging tail, anchor, daily chain loader),
`01_data.py` (extend cache + extract daily chain, fail closed), `02_stability.py`
(§4.3), `03_backtest.py` (§4.4, 4 workers, resumable), `04_report.py`
(Bloomberg × Swiss HTML), `README.md` with headline results; tests
`test/test_snowball_2y_tail_stability_study.py` (converging-tail formula limits:
κ→0 reproduces `hold_yield` exactly, since `B_n + c_n·τ = B_n·t/T_n`; κ→∞ gives
forward carry `b_∞` immediately past `T_n`;
anchor is point-in-time; pairing; P&L identity on a synthetic path).

## 5. Left to the builder

- How to apply the converging tail at runtime (a `CarryCurve` subclass, a
  `DividendYield` wrapper, or a patched extrapolation), provided
  `calculate_carry_curve_buckets` can bump its pillars and no library file
  changes.
- Whether to drive the daily loop from the replay engine's pieces or a
  study-local loop.
- Worker partitioning, output file formats, chart choices within the UI language.
- Exact holiday/expiry calendar handling, reusing existing helpers where they
  exist.
