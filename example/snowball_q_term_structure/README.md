# Does a dividend/carry term structure help hedge a snowball?

A desk that sells a CSI 1000 (000852.SH) snowball and delta-hedges it with IM
index futures has to tell its pricer what the index's dividend / carry yield
`q` is. The replay backtest engine's historical default reads **one flat
yield off the hedge contract it happens to hold** (simple compounding, floored
at zero). The alternative reads the **whole listed IM chain as a term
structure `q(T)`**. Same contract, same spot path, same vol: does the term
structure hedge better, and by which measure?

This folder is the study: two backtest-free diagnostics, a paired hedging
backtest fleet, and an aggregator that writes a self-contained HTML report.

## Data (local, untracked)

`example/mo_volmodels/data/history/` — CSI 1000 spot, the IM futures chain
(2–4 listed contracts a day), and the admitted MO IV-surface artifacts,
2023-05-04 to 2026-09-03. The caches are written by the `mo_volmodels` fetch
stages and are not tracked; every loader here fails closed with the stage to
run when a file is missing.

## The carry models (`_common.Q_MODELS`)

| model | what the pricer receives |
|-------|--------------------------|
| `flat_from_hedge` | the engine default: flat `q = max(0, r − basis)` from **whichever contract the hedge currently holds**, simple compounding |
| `flat_from_far` | the same flat channel, always inverted from the **longest listed contract**, whatever the hedge holds |
| `term_flat_q` | every listed contract with ≥ 7 days to expiry inverted to `q(T_i) = r − ln(F_i/S)/T_i`, linear in `q` between nodes, endpoint **zero yield held** beyond the last tenor |
| `term_flat_fwd` | same nodes, the last segment's **forward carry held** beyond the last tenor (`ForwardCarryCurve`) |
| `surface_fwd` | the MO option-implied parity forwards of the admitted IV surface (cross-market control) |
| `term_opt_tail` | every listed contract with ≥ 1 day to expiry as `ln(F_i/S)` (bounded, so no annualisation blow-up), carry **piecewise-linear in log-forward** between contracts (the calendar-spread slope), and beyond the last contract the MO surface's **option-implied forward carry** continues the curve, level-matched at the join, then flat |

`term_opt_tail` is the literature's answer to the near-expiry problem the
other models patch with a minimum tenor: interpolate total carry, not
annualised yield (the dividend-points convention; Bühler 2010), build the
interior from calendar spreads, and take the tail from the option market
(Binsbergen, Brandt and Koijen 2012; Golez 2014).

The four term models ride the engine's opt-in `dividend_source` channel on
`AutocallableEngineConfig` (see *Engine hook* below); `flat_from_hedge` is the
untouched default. The study's `dividend_for` builds the identical object
outside the engine for the static stage and is pinned against
`ProductReplay.build_env` by test.

Hedge contract policies: **front** (front month, rolled five days before
expiry — the engine default) and **far** (longest listed contract, same roll
rule; `FarContractRollPolicy`).

**Cell names.** A cell is `<q model>__<hedge policy>`. For the four term
models the two fields are independent: they read the whole chain, so
`__front` and `__far` differ only in which contract carries the delta. They
are **not** independent for `flat_from_hedge`. The replay engine passes one
selected contract row to both the hedge trade and `build_env`
(`quantark/backtest/replay/engine.py`), so that model's `q` is inverted from
the hedged contract: `flat_from_hedge__far` prices off the longest listed
contract, `flat_from_hedge__front` off the front month. The name says so, and
the two flat rows of the stage-01 table below are the measured consequence
(mean 12.1% vs 13.6%, std 5.6% vs 9.2%). One corollary for reading section 3:
`flat_from_hedge__far − flat_from_hedge__front` moves the carry model and the
hedge leg together, so only the `term_flat_q` pair isolates the hedge
contract.

`flat_from_far` cuts the other way. It is the same flat channel pinned to the
longest listed contract whatever the hedge does, so `flat_from_far__front`
shares its hedge leg with the baseline and differs from it only in which
contract the carry was read from. Hedged with the far contract it would be
`flat_from_hedge__far` by construction, which is why that pair stays out of
the grid and is asserted as an identity in the tests instead.

## The product and the fleet

1Y standard snowball, seller short, notional 50 mio, KO 103% observed monthly
from month 3, KI 75% observed daily, `r = 2%` flat, ATM 1Y implied vol off the
day's admitted MO surface as a single scalar vol channel (identical across
models, so vol never confounds carry). The fair coupon is solved **once per
inception under `term_flat_q`** and shared by every cell, and every cell
starts from the traded price (zero PV), so the paired terminal P&L difference
between two cells is exactly their hedge P&L difference.

Monthly inceptions from 2023-05; a run replays until knock-out, maturity
settlement, or the data end (censored runs are excluded by default). Pricing
is the QUAD engine (401 nodes), daily delta hedge in whole IM contracts, 1 bp
proportional cost per side.

## Measures

Static (stage 01):

- implied `q` per listed contract every day; the flat yield each hedge policy
  would hand the pricer; how often the zero floor binds; how much it jumps on
  roll days;
- **forward-pricing error** of each model at the listed tenors (RMS bp): the
  term models reprice every contract by construction, the flat model reprices
  only its own;
- on a fresh 1Y snowball each month: PV gap vs the reference model (bp of
  notional), delta gap (hands), rhoq, the share of the product's life and of
  its KO dates beyond the last listed tenor, and the futures-tenor carry
  buckets (carry risk by listed contract, analysis only);
- **knock-in / knock-out probability** under each carry model on the same
  contract (pricing measure, from the QUAD event recursion): the P(KI) gap
  vs the reference and its correlation with the q(T) gap;
- the **q-only knock-in probe**: one product frozen at its inception (spot,
  vol, coupon) and repriced with nothing but each day's carry input across
  the window around the largest front-contract basis jump, so the
  day-to-day movement of P(KI) is caused by the carry model alone.

**Two deltas, and only one of them trades.** The hedge is sized off the
**spot delta** `dPV/dS` at fixed `q(T)`, divided by the 200 index-point
multiplier and rounded to whole contracts. That is the only delta the fleet
ever trades. The **futures-tenor buckets** are a different partial,
`dPV/dF_i` at fixed spot: bump one IM mark by an index point, rebuild `q(T)`
from the chain, reprice. Spot is held, so the only channel is the carry
node, and

```
dPV/dF_i = -(1 / (T_i * F_i)) * dPV/dq(T_i)
```

which makes a bucket carry risk written in futures-price units. The two
partials are tied by an exact identity (a spot bump at fixed `q(T)` moves
every forward by the same percentage):

```
dPV/dS |q(T) fixed  =  dPV/dS |F_i fixed  +  Σ_i (F_i / S) · dPV/dF_i |S fixed
```

The middle term, spot bumped with every listed IM price pinned, is what
makes the buckets not add up to the hedging delta: on the static grid the
buckets sum to -50 hedge hands on a contract whose spot delta is +24 hands.
Section 5 (stage 04) computes all three terms directly and shows that the
middle term is entirely the flat-q tail convention. The buckets say where
the carry sensitivity sits along the chain. They are reported for analysis
and nothing in the study hedges them.

Paired backtest (stages 02–03), one row per inception × cell, differences on
matched inceptions:

- terminal hedged P&L (bp of notional) — path-dependent and noisy by nature;
- **daily hedged P&L standard deviation** — the hedge error a desk lives with;
- hedge variance-reduction `R² = 1 − var(hedged daily P&L)/var(product daily P&L)`;
- max drawdown of hedged P&L;
- turnover (rebalance and roll legs) and transaction cost;
- **|ΔMTM| on hedge-roll days vs other days** — a model tied to the active
  contract re-marks the whole book every time the hedge rolls, for no
  economic reason;
- contracts traded per day (delta churn).

A lifecycle consistency check asserts that every cell of an inception
terminated identically (the lifecycle depends on the spot path only).

## Running it

```bash
# 1) curve anatomy + static risk (≈ 6 min; --every-months 3 for a glance)
.venv/bin/python example/snowball_q_term_structure/01_curve_and_static_risk.py

# 2) the fleet — quick look first (4 inceptions x 3 cells, ≈ 8 min on 3 workers)
.venv/bin/python example/snowball_q_term_structure/02_backtest_fleet.py --quick --workers 3

#    full study grid (29 inceptions x 6 cells = 174 cells; 60–110 s per cell,
#    49 min on 4 workers; keep the Mac awake)
nohup caffeinate -i -m -s .venv/bin/python \
    example/snowball_q_term_structure/02_backtest_fleet.py --workers 4 --resume > fleet.log 2>&1 &

# 3) aggregate + report
.venv/bin/python example/snowball_q_term_structure/03_aggregate_and_report.py

# 4) the delta identity on one grid date under both tail conventions (≈ 10 s)
.venv/bin/python example/snowball_q_term_structure/04_tail_convention_probe.py [--date 2025-03-03]
```

Persisted outputs (small) live in `data/`: `curve_anatomy.csv`,
`static_risk.csv`, `static_buckets.csv`, `ki_probe_roll_window.csv`,
`static_summary.json`,
`fleet_per_run.csv`, `fleet_paired.csv`, `fleet_cells.json`,
`fleet_summary.json`, `tail_convention_probe.json`, and the report
`q_term_structure_report.html`,
whose closing appendix defines every column of every table it shows.
Per-run frames go under `output/snowball_q_term_structure/` (locally
excluded from git). Tests: `test/test_snowball_q_term_structure_study.py`
and `test/test_replay_dividend_source.py`.

## The bucket futures hedge (revised study)

Everything above hedges with ONE futures contract. The revised study adds a
second question: does holding a position in *every* listed contract, sized
from the book's sensitivity to each one, hedge the carry curve better than a
single leg — and does it pay for itself?

Design: [`docs/superpowers/specs/2026-09-09-bucket-futures-hedge-design-revised.md`]

### The 14-cell primary grid

Both supported term models crossed with seven hedge policies:

| Hedge | What it does |
|---|---|
| `front` | existing single-contract control, `-D/m`, unchanged |
| `far` | the same sizing on the longest listed contract |
| `front_scaled` | front month at `-D S/(m F)`, i.e. spot neutral |
| `far_scaled` | longest listed at `-D S/(m F)` |
| `buckets_nodes` | every modelled node neutral; residual spot delta `D_F` |
| `buckets_far` | spot neutral; the far node keeps `D_F S T_n` |
| `buckets_spot_parallel` | spot AND parallel rhoq neutral; `(+K, -K)` shape risk |

The two scaled controls exist to separate two effects that a naive
comparison conflates: correcting the `S/F` scaling, and holding a calendar
spread. Only the four models with actual futures coordinates
(`term_flat_q`, `term_flat_fwd`) can carry a bucket policy; the flat and
option-forward models have no nodes to hedge and are refused at config time.

### Running it

```bash
# 0) numerical validation, offline, no vendor history needed (≈ 2 s)
.venv/bin/python example/snowball_q_term_structure/05_bucket_hedge_validation.py \
    --synthetic --out-dir output/bucket_hedge_v2/validation

# 1) one historical date, serialised so it can be replayed exactly
.venv/bin/python example/snowball_q_term_structure/05_bucket_hedge_validation.py \
    --historical-dates 2025-03-03 \
    --out-dir example/snowball_q_term_structure/data/bucket_hedge_v2/validation

# 2) a one-inception subset of the primary grid, with daily audits
.venv/bin/python example/snowball_q_term_structure/02_backtest_fleet.py \
    --study-grid buckets --max-inceptions 1 --workers 2 \
    --carry-audit-mode daily --record-carry-exposure \
    --out-dir example/snowball_q_term_structure/data/bucket_hedge_v2/subset --resume

# 3) the full primary grid (14 cells per eligible inception)
nohup caffeinate -i -m -s .venv/bin/python \
    example/snowball_q_term_structure/02_backtest_fleet.py \
    --study-grid buckets --workers 4 --record-carry-exposure \
    --carry-audit-mode daily \
    --out-dir example/snowball_q_term_structure/data/bucket_hedge_v2/full \
    --resume > bucket_fleet.log 2>&1 &

# 4) aggregate
.venv/bin/python example/snowball_q_term_structure/03_aggregate_and_report.py \
    --run-dir example/snowball_q_term_structure/data/bucket_hedge_v2/full \
    --data-dir example/snowball_q_term_structure/data/bucket_hedge_v2/full_report
```

`--study-grid` defaults to `legacy`, so every command in the section above
runs exactly the cells it always did. An explicit `--cells` overrides the
grid and labels the output a subset.

### Artifacts

A revised run writes eight frames per cell instead of five:
`states`, `greeks`, `trades`, `rebalances`, `actions`, plus `hedge_legs`,
`hedge_attribution` and `hedge_stresses`; and two JSON files,
`run_config.json` (fully resolved strategy, numerical, source, notional,
schedule and stress settings, plus a content digest of the pricing modules)
and `audit_summary.json` (measured / pass / fail / inconclusive counts).

They land under a **new** versioned directory, `data/bucket_hedge_v2/`. The
original study's artifacts are never overwritten and remain readable: a
legacy run loads with `audit_coverage="not_available"`, which is a distinct
state from "audits ran and failed".

The validation stage writes `validation_manifest.json`,
`input_snapshots.json`, `price_ladder.csv`, `greek_ladder.csv`,
`policy_holdings.csv`, `direct_audits.csv`, `stress_results.csv` and
`validation_summary.md`.

### Resume and failure

Resume needs a matching fingerprint, `status="completed"`, the right format
version, every required file, and measured audit coverage when the task
asked for audits. A legacy artifact therefore cannot stand in for an
audited cell, and a run that completed with a failed audit stays available
for diagnosis but can never be reused as a passing result.

A worker failure writes `failure.json` *beside* the completed run, never
over it, with the date, objective, input fingerprint, an error category and
the traceback. The categories are distinct on purpose:

| Category | Retry? |
|---|---|
| `missing_price` | only after fixing the data |
| `infeasible_hedge` | no — investigate the objective |
| `numeric_audit` | no — convergence analysis |

### Reading the output

Four questions are answered separately, and none is inferred from another:

1. **Numerical validity** — did the independent repricing agree, and over
   how much of the run? A sampled run cannot report daily coverage.
2. **Objective achieved** — were the policy's own targets met, ideally and
   then actually after rounding and skipped trades?
3. **Joint mitigation** — did RMS spot-shock *and* RMS parallel-rhoq
   exposure both fall?
4. **Broader carry mitigation** — does gross nodal exposure support it too,
   and do the unquoted tail and shape scenarios?
5. **Economic comparison** — paired realised P&L variability, tail loss and
   costs against the controls, with an interval.

Zero parallel rhoq alone never earns the fourth. A policy can achieve its
two-factor objective and still lose on shape risk or turnover; that is a
valid study conclusion, not a failure of the implementation.

### Limitations

- The primary policy needs two distinct eligible tenors, always.
- Every risk coordinate must be a contract the hedge can actually trade;
  option-implied forwards are a different instrument.
- Tail and interpolation-shape risk is *unhedgeable* with listed futures
  and is reported separately, unchanged, however neutral the nodes are.
- Uncertainty uses paired calendar-block resampling over overlapping
  inceptions. Do not read the old per-run t-statistic as independent
  evidence.

## Engine hook

`AutocallableEngineConfig.dividend_source` (default `None` = the historical
behaviour, byte-identical, replay goldens unchanged):

- `"active_contract"` — the scalar-mode flat yield, explicitly;
- `"futures_curve"` — the chain through `IndexFuturesCurve`, with
  `futures_curve_extrapolation` (`"flat_q"` | `"flat_forward_carry"` |
  `"surface_forward_carry"`, the last one holding the chain in log-forward
  space and continuing it past the last contract with the admitted IV
  artifact's option-implied forward carry, level-matched, via
  `ForwardCarryCurve.extended_with`; needs `market_data.surface_history`) and
  `futures_curve_min_tenor_days` (default 7, at least 1: a contract inside
  its delivery week has no measurable annualised carry — a 1% basis two days
  out reads as a 180% yield; one eligible contract is priced as a flat
  continuous signed yield, the exact one-node limit of both conventions;
  none fails closed);
- `"surface_forwards"` — the IV artifact's parity forwards with the scalar
  vol channel.

The term sources reject `fixed_dividend_yield` and the flat-q surface grid
(`calculate_surfaces`) at config time. Under a term source the `pricing_q`
state column records the zero yield to the product's remaining maturity.

## Results (data window 2023-05-04 to 2026-09-07, 814 trading days)

The full report with charts is `data/q_term_structure_report.html`; every
number below is read from the persisted `data/` files.

### 1. What the market says about carry (stage 01, every day)

| series | mean | std | min | max |
|--------|-----:|----:|----:|----:|
| front contract implied q (continuous) | 13.2% | 10.5% | −72.8% | 93.3% |
| longest listed contract implied q | 10.9% | 4.0% | −0.8% | 22.8% |
| flat q the engine uses, front-month hedge | 13.6% | 9.2% | 0.0% | 92.0% |
| flat q the engine uses, longest-contract hedge | 12.1% | 5.6% | 0.0% | 39.4% |
| term q(1Y), flat-q tail | 10.9% | — | — | — |
| term q(1Y), flat-forward-carry tail | 10.7% | — | — | — |
| MO option-implied q(1Y) | 8.9% | — | — | — |
| term q(1Y), log-forward chain + option-forward tail (`term_opt_tail`) | 9.0% | 3.2% | 1.6% | 15.4% |

- The chain ends at a median 0.51y (0.08y–0.66y): a 1Y snowball spends **51%
  of its life and 6.6 of its 10 KO observations beyond the last listed
  tenor**. The tail convention is a modelling assumption, not a market read.
- The front-month flat yield is floored at zero on 50 days (6%) and rolls 40
  times; its mean daily move is 4.7% of yield, 6.6% on roll days. The term
  q(1Y) moves 0.8% a day.
- Forward-pricing error at the listed tenors (RMS log error vs the IM
  marks): front-hedge flat q **146 bp** (max 260 bp), longest-contract flat q
  59 bp, term q 0 bp by construction. The MO option forwards sit 25 bp from
  the futures, a genuine cross-market basis.

### 2. Static risk of a fresh 1Y snowball (stage 01, 41 monthly dates)

Fair coupon solved under `term_flat_q` (median 27.6%, range 2.6%–62.0%; the
futures discount is priced as expected drift, see caveats), then the same
contract priced under every model. PV is in bp of notional for the long
holder; delta is the hedging delta `dPV/dS` at fixed `q(T)` and hands =
delta / 200.

| model | PV gap vs reference | delta | delta gap | rhoq per +1% q | q at maturity |
|-------|--------------------:|------:|----------:|---------------:|--------------:|
| `flat_from_hedge` (front hedge) | **−368 bp** (median −179, min −1795, max +601) | 29.2 hands | **+5.3 hands** | −63 bp | 15.5% |
| `term_flat_q` (reference) | 0 | 23.9 hands | 0 | −55 bp | 10.9% |
| `term_flat_fwd` | +17 bp (median +10) | 23.5 hands | −0.4 hands | −55 bp | 10.6% |
| `surface_fwd` | +119 bp (median +105) | 21.5 hands | −2.4 hands | −52 bp | 8.8% |
| `term_opt_tail` | +107 bp (median +89) | 21.8 hands | −2.0 hands | −52 bp | 8.9% |

- The flat active-contract model misprices the fresh contract by hundreds of
  bp and **over-hedges by 5.3 of 24 hands (22%)** on average, because the
  front month's annualised basis is a noisy, upward-biased read of the
  chain's carry.
- The two tail conventions differ by 17 bp of PV and 0.4 hands: that is the
  honest width of the unquoted tail.
- Futures-tenor carry buckets under the term model (`dPV/dF_i` at fixed
  spot, in hands per contract, nearest eligible first): **0.0, −3.5, −13.6,
  −41.8**. These are not slices of the 23.9-hand hedging delta: they are the
  other partial, they sum to −50 hands, and no cell trades them. Read as
  carry, 95% of the total |bucket| sits in the last contract, the one
  carrying the extrapolated tail. The carry risk of a 1Y snowball lives in
  the longest listed contract, not in the front month a default hedge rolls
  every month. Section 5 shows that the size of that last bucket, like the
  gap between the two deltas, is set by the tail convention.

#### 2b. The knock-in probability each carry model implies

Under the pricing measure P(KI) is fixed by the forward curve. A flat q
extrapolates one contract's forward to the whole life, so its error scales
with 1/tenor of that contract: a 1% basis three weeks from delivery is 17%
annualised, and the zero floor then snaps it to nothing. The term curve pins
every listed forward and extrapolates only past the last tenor. On the same
41 contracts (same fair coupon, spot and vol), from the QUAD event recursion:

| model | P(KI) mean | P(KO) mean | P(KI) gap vs ref, mean | gap range | corr(gap, q(T) gap) |
|-------|-----------:|-----------:|-----------------------:|----------:|--------------------:|
| `flat_from_hedge` | **0.408** | 0.483 | **+0.072** | −0.160 .. +0.267 | 0.98 |
| `term_flat_q` (reference) | 0.336 | 0.542 | 0 | | |
| `term_flat_fwd` | 0.333 | 0.545 | −0.003 | −0.032 .. +0.021 | 0.99 |
| `surface_fwd` | 0.311 | 0.563 | −0.025 | −0.081 .. +0.009 | 0.97 |
| `term_opt_tail` | 0.316 | 0.555 | −0.020 | −0.078 .. +0.009 | 0.97 |

The flat active-contract model's knock-in probability is off by up to 27
points on a single day, and the error is the carry error and nothing else
(correlation 0.98 with the q(T) gap). On the worst dates the front month's
annualised discount is two to three times the chain's, P(KI) reads 0.73
instead of 0.46, and the mark is 1,600 bp too low.

**The q-only probe** freezes the 2023-05-04 product (spot 6,734, vol 15.5%,
coupon 4.53%) and reprices it with nothing but the carry read off each of
the 12 trading days from 2024-01-25 to 2024-02-19, the window around the
largest front-contract basis jump (the February-2024 small-cap sell-off).
Every move of P(KI) is caused by the carry input alone
(`data/ki_probe_roll_window.csv`, chart in the report):

| model | q(T) range | P(KI) range | mean \|ΔP(KI)\| per day |
|-------|-----------:|------------:|-----------------------:|
| `flat_from_hedge` | 0.0% .. 92.0% | **0.051 .. 1.000** | **0.234** |
| `term_flat_q` | 7.5% .. 19.4% | 0.121 .. 0.341 | 0.045 |
| `term_flat_fwd` | 8.0% .. 15.7% | 0.125 .. 0.278 | 0.032 |
| `surface_fwd` | 6.3% .. 9.8% | 0.109 .. 0.161 | 0.012 |
| `term_opt_tail` | 5.1% .. 12.5% | 0.100 .. 0.237 | 0.030 |

On 2024-02-05 the front contract, 11 calendar days from delivery, implied
92% annualised carry and the flat model priced a certain knock-in; the next
day the floor bound and it priced a 5% knock-in. The term curve moved from
0.34 to 0.15 over the same two days, the option-implied forwards from 0.16
to 0.15. This is the mechanism behind the 384 bp daily hedge noise and the
2,508 bp drawdowns of section 3: the engine's default q is not information
about the year-long drift, it is a small basis divided by a one-to-five-week
tenor.

### 3. Hedging backtest (stages 02–03)

29 monthly inceptions, 2023-05-04 to 2025-09-01 (every one uncensored),
8 cells, 232 runs, no failures; 22 inceptions knocked out and 7 knocked in and
matured, identically in every cell (lifecycle check passed). Means over
inceptions, bp of notional; turnover in multiples of notional.

| cell | terminal P&L | daily P&L std | R² | max DD | turnover | cost | \|ΔMTM\| roll days | \|ΔMTM\| other days | hands/day |
|------|-------------:|--------------:|---:|-------:|---------:|-----:|------------------:|-------------------:|----------:|
| `flat_from_hedge__front` (engine default) | 489 | **384** | **0.26** | **2508** | 24.2 | 24.2 | 317 | 256 | 5.6 |
| `term_flat_q__front` | 546 | **57** | **0.81** | 439 | 16.2 | 16.2 | 65 | 107 | 2.9 |
| `term_flat_fwd__front` | 564 | 54 | 0.80 | 386 | 16.1 | 16.1 | 65 | 104 | 2.8 |
| `surface_fwd__front` | 565 | 64 | 0.70 | 370 | 15.7 | 15.7 | 62 | 96 | 2.8 |
| `term_opt_tail__front` | 567 | 56 | 0.76 | 371 | 15.8 | 15.8 | 62 | 101 | 2.8 |
| `flat_from_hedge__far` | 426 | 86 | 0.74 | 697 | 8.7 | 8.7 | **334** | 115 | 2.9 |
| `term_flat_q__far` | 442 | 53 | 0.83 | 418 | 8.4 | 8.4 | 68 | 105 | 2.9 |
| `flat_from_far__front` | 535 | 90 | 0.72 | 705 | 16.7 | 16.7 | 88 | 117 | 2.9 |

Paired differences on matched inceptions (n = 29; *t* is the paired
t-statistic, overstated because inception windows overlap):

| variant − base | Δ terminal P&L | Δ daily std | Δ R² | Δ max DD | Δ turnover | Δ \|ΔMTM\| roll days | Δ hands/day |
|----------------|---------------:|------------:|-----:|---------:|-----------:|--------------------:|------------:|
| `term_flat_q__front` − `flat_from_hedge__front` | +57 (median +115, 55% > 0, t 0.9) | **−327** (0% > 0, t −11.9) | **+0.56** (100% > 0) | −2069 | −7.9× | **−251** (0% > 0) | −2.7 |
| `term_flat_fwd__front` − `flat_from_hedge__front` | +75 (t 1.0) | −330 | +0.54 | −2122 | −8.0× | −252 | −2.7 |
| `surface_fwd__front` − `flat_from_hedge__front` | +76 (t 1.0) | −321 | +0.44 | −2138 | −8.5× | −255 | −2.8 |
| `term_opt_tail__front` − `flat_from_hedge__front` | +78 (t 1.0) | −328 | +0.50 | −2137 | −8.3× | −255 | −2.8 |
| `term_opt_tail__front` − `term_flat_q__front` | +21 (62% > 0, t 1.4) | −1 (t −0.7) | −0.05 (34% > 0, t −3.5) | −68 (17% > 0, t −5.3) | −0.4× (t −9.3) | −4 (t −1.7) | −0.1 (t −3.9) |
| `flat_from_hedge__far` − `flat_from_hedge__front` | −63 (t −0.8) | −299 | +0.49 | −1811 | −15.5× | −38 (50% > 0, t −0.6) | −2.6 |
| `term_flat_q__far` − `term_flat_q__front` | **−103** (24% > 0, t −4.7) | −4 (t −4.2) | +0.02 | −21 | −7.9× | −9 (t −0.5) | 0.0 |
| `flat_from_far__front` − `flat_from_hedge__front` | +46 (55% > 0, t 0.7) | **−294** (0% > 0, t −10.1) | **+0.47** (100% > 0) | −1803 | −7.6× | −229 (0% > 0, t −12.9) | −2.6 |

What the fleet says:

- **The engine's default hedge is dominated by its own carry noise.** Under
  the flat active-contract yield the daily hedged P&L std is 384 bp of
  notional and the futures hedge removes only 26% of the product's daily
  variance; the term structure cuts the std to 57 bp (a 6.7× reduction, on
  every one of the 29 inceptions) and removes 81%. The 2508 bp mean
  drawdown of the default is mark-to-model noise that reverses, not
  economic loss, but a desk would have to fund and explain it.
- **The roll-day re-mark is the fingerprint.** With a flat active-contract
  yield the book re-marks by 317 bp on the days the hedge rolls (334 bp
  with the far contract, 3× an ordinary day), because the pricer's `q`
  jumps with the contract. Under the term models roll days are quieter than
  ordinary days (65 vs 107 bp): the curve does not know which contract the
  hedge holds.
- **The term structure halves hedge churn and cost.** 2.9 vs 5.6 contracts
  traded per day, 16× vs 24× notional turnover, 8 bp of notional less in
  costs per run at 1 bp per side.
- **A flat yield read off the far contract recovers most of it.** Its std
  (86 bp) and R² (0.74) are close to the term models' because the far
  yield is stable (std 4% vs 10.5% for the front), but its roll-day re-mark
  (334 bp) is the worst in the study: one flat number cannot be right for
  every tenor, and the error shows up whenever the number changes.
- **That recovery is the carry contract, not the hedge contract.**
  `flat_from_far__front` reads the same far yield while keeping the
  baseline's front-month hedge, and it captures −294 bp of daily std against
  the −299 bp that moving both together captures. The hedge leg is worth
  about 4 bp of the 299, roughly one part in seventy. A desk that dislikes
  holding the far contract can have almost the whole improvement by leaving
  the hedge in the front month and only changing which contract it reads its
  carry from. The far yield also fixes the roll-day re-mark that
  `flat_from_hedge__far` suffers, 88 bp against 334, because the carry no
  longer jumps when the hedge rolls. What it does not fix is the tenor
  mismatch: at 90 bp of daily std it is still 33 bp worse than `term_flat_q`,
  which is the cost of one flat number standing in for a curve.
- **The tail convention does not matter for hedging.** `term_flat_fwd` and
  `term_flat_q` are within 3 bp of daily std and 0.05 hands per day of each
  other; the MO option forwards (`surface_fwd`) hedge nearly as well (64 bp,
  R² 0.70) despite sitting 25 bp from the futures marks.
- **Terminal P&L is not where a carry model shows.** The +57 bp mean
  advantage of the term model has a t-statistic below 1 and a 623 bp
  standard deviation across inceptions; knock-out timing dominates it. The
  one significant terminal effect is the hedge *contract*, not the model:
  rolling the front month earned 103 bp more per run than holding the far
  contract under the same term model (t −4.7), because the IM discount is
  steepest at the front and a long front-month hedge collects it every
  month. That is a carry-harvesting choice, and it comes with 7.9× more
  turnover.

### 4. Verdict

Priced and hedged off the IM chain's implied `q(T)`, the same snowball on
the same path has a daily hedge error one sixth of the engine default's,
removes 81% instead of 26% of the product's daily variance, trades half the
contracts, and stops re-marking the book every time the hedge rolls. The
term structure is not a refinement here; the flat active-contract yield is
the wrong object (its knock-in probability swings between 5% and 100% on a
frozen product inside one roll window), and its error is largest exactly
where a front-month hedge lives, inside the roll window. Which contract carries the hedge is a
separate, terminal-P&L question the term model makes visible (its carry
buckets put 95% of the carry risk in the longest listed contract) but does
not answer on its own.

**The fifth model.** `term_opt_tail`, the log-forward chain with the
option-forward tail, was added after the knock-in-probability finding to
test the literature's fix for the annualisation problem. On the fleet it is
indistinguishable from `term_flat_q` on the measure that matters most
(daily hedge error 56 vs 57 bp, t −0.7) and modestly better on drawdown
(−68 bp, 83% of inceptions), turnover (−0.4×) and roll-day re-marks (62 bp,
the lowest of all cells), at the cost of a lower R² (0.76 vs 0.81, because
the option-forward tail is less correlated with the IM-marked product
P&L). Its P(KI) sits 0.02 below the reference on the static grid and its
q-only probe moves 0.030 a day, between the two IM tail conventions and the
pure option surface. The lesson is that once the near-expiry contract is
handled in the right coordinate (log-forward, or excluded), the remaining
choices, interpolation rule and tail source, are second-order for hedging a
1Y product: every term model lands within 3 bp of daily hedge error of every
other. The engine hook keeps the log-forward route (`"surface_forward_carry"`)
because it needs no minimum-tenor judgment and never floors.

### 5. Could the futures-tenor buckets size the hedge instead? (stage 04)

The natural follow-up is to hold `-dPV/dF_i / 200` contracts in each listed
tenor instead of `dPV/dS / 200` in the front month. Stage 04 answers it by
computing every term of the delta identity directly on one grid date,
2025-03-03 (spot 6273.67, ATM vol 23.5%, fair coupon 28.4%, chain IM2503 /
IM2504 / IM2506 / IM2509 ending at 0.548y), under both tail conventions.
Sensitivities below are the long holder's, in hands (index points per unit
divided by the 200 multiplier); central bumps of 0.5% of spot and 1 index
point per contract. Every number is in `data/tail_convention_probe.json`.

**The identity, computed.**

| term | flat-q tail | flat-forward-carry tail |
|------|------------:|------------------------:|
| `dPV/dS` at fixed `q(T)` (the hedging delta) | +28.73 | +28.03 |
| `dPV/dS` with the four IM prices pinned, direct | **−18.13** | **+0.01** |
| `Σ (F_i/S) · dPV/dF_i` | +46.86 | +28.02 |
| identity residual | 0.001 | 0.002 |
| 1Y forward move for spot +1% with the chain pinned | −0.82% | 0.00% |

**Where the −18 hands comes from.** Pin the four IM prices and lift spot
1%. The pricer re-inverts `q(T_i) = r − ln(F_i/S)/T_i`, so the node yields
jump (front 20.9% → 41.1%: a 1% spot move over 18 days is a 20-point
annualised carry) but every *listed* forward is unchanged by construction.
Inside the chain nothing economic moves: the terminal distribution at each
listed date is the same lognormal around the same forward, and a 75% KI
barrier is too far for 18 days of a 1% higher path to matter. The
difference is the tail. The flat-q convention holds the *last node's
yield* beyond 0.548y, and that yield just rose from 13.0% to 14.8%; applied
over the remaining 0.45y it lowers the 1Y forward from 5618.9 to 5573.0
(−0.82%). Six of the ten KO observations and the KI settlement live there,
so lower tail forwards mean fewer knock-outs and more matured knock-in
losses. The −18 hands is the product's sensitivity to that 0.82% drop in
the unquoted tail forwards, and nothing else: under the forward-carry
tail, which extrapolates the last *segment's* forward carry
(`F(1Y) = 5906.0 · exp(−0.0960 · (1 − 0.548)) = 5655.3`, a formula with no
spot in it), the same bump leaves every forward untouched and the term is
+0.01 hands.

The chain under the pinned-futures spot bump (identical listed rows under
both conventions; only the tail rows differ):

| tenor | F | q before → after | forward, flat-q tail | forward, flat-fwd tail |
|-------|--:|-----------------:|---------------------:|-----------------------:|
| IM2503, 0.049y | 6215.4 | 20.9% → 41.1% | 6215.4 → 6215.4 | 6215.4 → 6215.4 |
| IM2504, 0.126y | 6168.8 | 15.4% → 23.3% | 6168.8 → 6168.8 | 6168.8 → 6168.8 |
| IM2506, 0.299y | 6049.0 | 14.2% → 17.5% | 6049.0 → 6049.0 | 6049.0 → 6049.0 |
| IM2509, 0.548y | 5906.0 | 13.0% → 14.8% | 5906.0 → 5906.0 | 5906.0 → 5906.0 |
| 0.75y (tail) | — | 13.0% → 14.8% / 12.6% → 14.0% | 5775.9 → 5754.8 (−0.37%) | 5792.6 → 5792.6 |
| 1.00y (tail) | — | 13.0% → 14.8% / 12.4% → 13.4% | 5618.9 → 5573.0 (−0.82%) | 5655.3 → 5655.3 |

**The same lever inflates the far bucket.** The buckets, one IM price
bumped with spot held:

| contract | flat-q: hands | flat-q: (F/S)·hands | flat-fwd: hands | flat-fwd: (F/S)·hands | tail? |
|----------|--------------:|--------------------:|----------------:|----------------------:|-------|
| IM2503 | +0.01 | +0.01 | +0.00 | +0.00 | no |
| IM2504 | −0.19 | −0.19 | −0.11 | −0.11 | no |
| IM2506 | +0.71 | +0.68 | −40.68 | −39.22 | no |
| IM2509 | +49.25 | +46.34 | +71.54 | +67.35 | yes |
| sum | +49.75 | +46.84 | +30.75 | +28.02 | |

Under flat q a one-point move in IM2509 changes the last node's yield by
`1/0.548` in log terms and applies it to the whole tail, so the 1Y forward
moves 1.8× as far as the contract: that is why the IM2509 bucket (49
hands) exceeds the contract's entire delta (29). The far bucket exceeding
the delta and the negative pinned-futures term are one fact seen twice.
Under the forward-carry tail the buckets do add up to the delta with no
residual, but the tail is now extrapolated off the *slope* between the
last two contracts: lifting IM2509 flattens it and the tail forwards rise
by more than a point, lifting IM2506 steepens it and the tail forwards fall
even though a listed forward went up. Hence IM2506 −41 / IM2509 +72.

**What a bucket hedge would be.** The buckets alone cannot size the hedge
under flat q: a seller holding `-B_i/200` per tenor is long 50 contracts
against a 29-contract exposure and makes or loses ~46 bp of notional on
an ordinary 1% parallel day. A consistent multi-contract hedge has to
carry the pinned-futures term too, assigned to a contract by a basis
assumption (`dF_i ≈ (F_i/S) dS`); putting it in the front month gives, for
the seller:

| convention | IM2503 | IM2506 | IM2509 | net |
|------------|-------:|-------:|-------:|----:|
| study (spot delta, front month) | +29 | | | +29 |
| flat-q buckets + pinned-futures term in front | −18 | +1 | +49 | +32 |
| flat-forward-carry buckets | 0 | −41 | +72 | +31 |

Both bucket hedges net to the delta hedge (the 31–32 vs 29 is the `S/F`
scaling the flat `/200` conversion skips). What they add is a calendar
spread of tens of contracts whose size and even sign depend on how the
half of the life beyond the chain is filled in. On the data, the far
contract's basis moves 0.53 pt of spot a day (same-contract std, corr 0.21
with spot), which under the model re-marks this contract by ~60 bp, the
same order as the 57 bp residual daily hedge error of the term cells. So
a bucket hedge might reduce that residual, but it would be hedging the
tail extrapolation formula as much as a market price. The honest next step
is not to trade it but to regress each cell's hedged daily P&L on the
daily far-basis change and see whether the residual is basis-driven at
all; if it is, the replay engine needs one hedge position per contract
(out of scope, see the caveats).

### 6. Caveats

- One product, one underlying, one 3.3-year window of one regime (a deep,
  volatile IM discount); inception windows overlap, so the paired samples
  are not independent and every t-statistic overstates significance.
- The futures-implied carry treats the whole IM discount as expected drift.
  That is what "hedge with futures" implies, and it is why fair coupons
  reach 27% median (62% max) here; a desk that prices only part of the
  discount would see smaller coupons and a smaller flat-vs-term gap in
  levels, but the same roll-day and churn mechanics.
- The vol channel is one ATM 1Y IV for every model; this study isolates
  carry and says nothing about vol models (see `example/mo_volmodels`).
- Beyond the last listed tenor nobody quotes carry; the two tail conventions
  bracket that uncertainty and, for the spot-delta hedge, agree. They do
  not agree on the futures-tenor buckets (section 5): the far bucket and
  the pinned-futures spot term are both set by the tail convention.
- The hedge is one contract sized off the spot delta with a 1 bp
  proportional cost; margin funding, roll slippage and a multi-contract
  hedge across the carry buckets are out of scope. The `/200` conversion
  also ignores `F_i/S`, a 5% under-hedge for a far contract at a 5%
  discount and negligible for the front month.
