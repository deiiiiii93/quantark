# Intraday daily-KI Greek certification and the near-KI example

Status: **Design sections approved in discussion 2026-09-17; written spec under review.** Worktree branch
`worktree-intraday-plan1`, based on `e0ba6ba7`.

## Goal

Show how a snowball's PV and Greeks move intraday while the spot streams around its knock-in level, on the contract
desks actually book: **KI observed at every daily close**. Deliver an example script and a final report published as
an HTML artifact.

## Why certification comes first

A numerical intraday Greek is published only inside a Gate C certificate (`quantark/intraday/README.md`, "What a
status does and does not claim"). Since `e0ba6ba7` a certificate matches the conditional economics
(`capability.economic_identity`): payoff terms, remaining event times/barriers/cash, KI state, pending ledger,
curves and levels, and the session calendar. The only swept snowball has monthly KI, so a daily-KI snowball
reports every QUAD V2 Greek `unqualified` with no value, under both conventions. Measured 2026-09-17 on a
daily-KI contract at 2026-09-10 14:00 and 14:59:59, spots 75.3 and 74.7: all five Greeks `unqualified`, reason
"Gate C demonstrated point_delta only for other conditional contract/market economics".

The example must not difference published PVs itself: those are exactly the numbers the certificate layer declines
to vouch for.

## Confirmed decisions

1. Contract: the Gate C monthly fixture (`dated_snowball`: initial 100, strike 100, KO 103 monthly, 12% KO and
   rebate rate, no principal) with **KI 75 at every SSE close** from 2026-03-17 to maturity. Market: flat 20% vol,
   r 3%, q 1%. SSE sessions 09:30–11:30, 13:00–15:00.
2. One certified day: **Thu 2026-09-10**, an ordinary KI-only day whose previous close is exactly 24 h earlier.
   The economic identity contains the remaining event timestamps, so a certificate for this day does not cover any
   other day. The example streams on this day.
3. Profile: **desk only** (`VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))`). Sessions-only daily KI
   stays `unqualified`.
4. Route: QUAD V2 at the certified settings (`order=8, cells_per_sd=2.0`). No PDE, MC or price Gate C cells.
5. Every point and desk measure the QUAD V2 sweep already records: point delta/gamma/vega/rho/dividend rho/theta,
   desk delta/gamma/vega/rho/dividend rho/theta (default one-hour roll).
6. A pilot runs first with a go/no-go gate. A failed pilot stops the work for a user decision; no budget is relaxed.
7. The final report is an HTML artifact built from the example's run record and the packaged evidence.

## Measured inputs (2026-09-17 probes, this machine)

| Quantity | Value |
|---|---|
| Daily-KI remaining events at 2026-09-10 14:00 | 129 |
| QUAD V2 price only | ~1.35 s |
| QUAD V2 price + 5 point / desk Greeks | ~4.7 s / ~3.6 s |
| `spot_curve`, 49 spots | ~2.0 s |
| Reference `_solve_snowball`, 2001 / 4001 points (cold) | 5.5 s / 14.9 s; cached re-solve 0.02–0.04 s |
| Price at 14:00, spot 75.3: QUAD V2 vs reference (4001) | −22.12222 vs −22.12230 |

The reference sweep is cached across spots, horizons and theta rolls of one market; only vol/rate/dividend moves
trigger a new solve. Estimated sweep: 77 groups, ~1–1.5 h at 4 workers.

## Component 1 — the `snowball_daily_ki` fixture (`test/intraday/gate_c/cells.py`)

- `product("snowball_daily_ki")`: `dated_snowball(sse().calendar, T0)` with `ki_observation_schedule.records`
  replaced by one `ObservationRecord(observation_date=d, barrier=75.0)` per SSE business day `T0 < d <= maturity`.
- `BARRIERS["snowball_daily_ki"] = ("ko", "ki")`. Both are required: the runtime spot envelope spans
  `[min barrier × e^(−2σ√W), max barrier × e^(2σ√W)]` over every remaining barrier, so sweeping KI offsets alone
  would let the envelope claim spots near 103 without evidence.
- `MONITORING` discrete, `notional` 100, `market` (0.03, 0.01).
- `fixing_and_history("snowball_daily_ki")`: the fixing is the KI event at 2026-09-10 15:00 +08:00; the history
  is `Fixing(t, 100.0)` for every distinct event instant before it (monthly KO instants coincide with KI instants).
  Resolved state: alive, not knocked in, empty ledger.
- Horizon ladder `DAILY_KI_HORIZONS = (6 h, 1 h, 15 min, 5 min, 1 min, 10 s, 1 s)`. A one-day rung would land on
  the previous close, itself an event instant. Six hours before the close is 09:00, so the window covers the whole
  session.

## Component 2 — harness (`test/intraday/gate_c/greek_harness.py`)

- `GREEK_ENGINES["snowball_daily_ki"] = ("quad_v2",)`; `ROUTE_NAMES[("snowball_daily_ki", "quad_v2")] = "QuadV2Route"`;
  `PRODUCT_NAMES["snowball_daily_ki"] = "SnowballOption"`.
- Profiles per product: `snowball_daily_ki` sweeps `("desk",)`; the existing fixtures keep `GREEK_PROFILES`.
- `greek_groups()` uses `DAILY_KI_HORIZONS` for this fixture.
- `demonstrated()` names this fixture's ladder explicitly in its expected-horizon map, so an absent interior rung is
  a gap and the monthly ladder is never inferred for it.
- `build_context` already treats every `snowball_*` product's first event as its horizon anchor.

## Component 3 — pilot gate

Six desk groups through `test/intraday/gate_c/test_greek_ladders.py` (selected by id), under `tmp/rss_guard.py`:

| Horizon | Offset | Barrier |
|---|---|---|
| 1 s | bp−1 | ki |
| 1 s | eq | ki |
| 1 min | sd−1 | ki |
| 1 h | sd+0.5 | ki |
| 6 h | sd−2 | ki |
| 1 h | sd+1 | ko |

- **Go**: every measure `passed` (or `undefined` agreeing with the reference). Run the full sweep.
- **No-go**: stop and report each failing measure's route value, reference, reference uncertainty, budget and
  ladder. Remedies are user decisions: a refined certified setting (for example `cells_per_sd=4`, which the example
  would then use) when the QUAD ladder converges above budget, or more reference points when the reference's own
  uncertainty makes a cell `inconclusive`. The frozen budgets in `test/intraday/reference/budgets.py` are not edited.

## Component 4 — sweep, packaging, matrix

- Runner in `tmp/` (untracked), modelled on `tmp/gate_c_greek_run_v2.py`: resumable, 4 workers, own JSONL, under
  `rss_guard` and `nohup caffeinate -i -m -s`, selecting only `snowball_daily_ki` groups.
- Packaging combines the packaged `gate_c_greeks.json` cells with the new rows and re-aggregates with
  `aggregate_greeks`. It asserts that the `demonstrated` rows of every pre-existing family are identical before and
  after, and that the new fixture has all 77 groups (7 horizons × 11 offsets × 2 barriers). It refuses to write
  otherwise.
- `python -m quantark.intraday.publish` regenerates `docs/execution/intraday-capability-matrix.md`.
- Commits: (1) fixture + harness, with catalogue tests that need no evidence (group count, ids, ladder, history);
  (2) evidence + matrix, whose `git_sha` names commit (1); (3) example + its test + docs, after the evidence exists.

## Component 5 — the example (`example/intraday_snowball_near_ki_demo.py`)

Builds its own contract, market, calendar and profile (no `test/` imports) with terms identical to the fixture.

- **History**: deterministic synthetic closes 2026-03-17 … 2026-09-09 drifting from 100 to about 75.9, every close
  strictly between 75 and 103, supplied as confirmed `Fixing`s.
- **Tick stream** on 2026-09-10, hand-shaped keyframes, no randomness:

  | Time | Spot | | Time | Spot |
  |---|---|---|---|---|
  | 09:30 | 75.90 | | 14:45 | 75.15 |
  | 10:30 | 74.85 | | 14:50 | 74.90 |
  | 11:30 | 75.10 | | 14:55 | 75.05 |
  | 13:00 | 75.30 | | 14:58 | 74.95 |
  | 13:30 | 74.90 | | 14:59:00 | 75.02 |
  | 14:00 | 75.20 | | 14:59:30 | 74.97 |
  | 14:30 | 74.80 | | 14:59:50 | 74.90 |
  | | | | 14:59:59 | 75.01 |

  14:59:50 is deliberately below the ten-second envelope floor (75 × 0.999 = 74.925): its Greeks print
  `unqualified`, never a number.
- **Per tick** (point convention): time, spot, KI distance `ln(S/75)/σ√W` in remaining standard deviations, PV,
  delta, gamma, vega, theta per minute. A non-`ok` Greek prints `—` with a short tag; each distinct reason is printed
  once below the table.
- **Point vs desk** at 14:00, 14:55 and 14:59:59: both conventions side by side, desk theta on the certified
  default one-hour step (clamped to the close).
- **`spot_curve` snapshots** at 10:00, 14:00, 14:55 and 14:59:59 over 74.5 … 75.5 in 0.1 steps: PV and delta, with
  points outside the envelope reported by their status.
- **The close**: 15:00:00 under `before` with spot exactly 75.00 (inclusive KI decided at spot, delta and gamma
  `undefined`); `after` with fixings 74.98 and 75.02 (knocked in vs alive, PV only — the next day's economics are
  not certified); 15:00:30 with no fixing (provisional, the assumed knock-in and its `depends_on`).
- `--json PATH` writes the run record: every row above with timestamp, spot, σ distance, price, and each Greek's
  value, status, unit and reason; plus contract/market/profile identities and the evidence `git_sha`.
- Runtime ~2–3 min.

## Component 6 — tests and docs

- `test/intraday/test_near_ki_example.py` (fast, default suite): imports the example's builders and asserts
  (a) the example context's `economic_identity` at a sample tick equals the `snowball_daily_ki` fixture's;
  (b) at 14:00 / 75.20 point delta and gamma are `ok`; (c) at 14:59:50 / 74.90 they are `unqualified` with a spot
  envelope reason. The example itself does not run in CI.
- `quantark/intraday/README.md`: the Greeks section names the daily-KI certificate (one day, desk, 1 s – 6 h) and
  the example.
- `docs/superpowers/plans/2026-09-15-intraday-baseline.md`: dated note with the sweep's counts and resources.

## Component 7 — the report artifact

Built after the example runs, by a generator in the session scratchpad (not the repo), following the artifact
page contract. Charts are drawn from the run JSON embedded inline; no number is retyped.

1. The day: spot with the KI line and the shaded Greek envelope; PV, delta, gamma, theta through the session, with
   gaps where a Greek is not `ok`.
2. Crossing the barrier: the four `spot_curve` snapshots overlaid.
3. Point vs desk at the three instants, with why a 1% bump across the barrier differs from the derivative.
4. The close: knocked-in vs alive PV and the provisional row.
5. What the Greeks rest on: pilot outcome, groups and measures passed, horizon window, engine settings, and the
   statement that every pre-existing certificate reproduced.

Published private; the link is reported.

## Risks

- **QUAD V2 accuracy over 129 events.** The price at 14:00 already sits near the 1e-4 budget. The pilot measures it.
- **Reference uncertainty.** The reference delta moved 3.4e-3 between 2001 and 4001 points; at 4001/8001/16001 its
  Richardson uncertainty × 3 may approach the delta budget (~3e-4 at spot 75). An `inconclusive` cell blocks a
  certificate; the pilot measures it.
- **Envelope arithmetic.** Tick spots assume the desk profile's remaining variance; test (c) pins the one tick meant
  to fall outside, and the example reports every other status as computed.
- **Memory.** Worker RSS on a 129-event QUAD V2 is unmeasured; the guard kills the process group past its limit.

## Out of scope

Price Gate C cells for daily KI; PDE and MC routes; the sessions-only and uniform profiles; any day other than
2026-09-10; continuous KI; a generalised certificate covering every day of a daily-KI contract (it would certify
event sets no sweep ran).
