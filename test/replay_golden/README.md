# Replay golden data — provenance notes

Goldens are captured with `test/replay_golden/capture.py::write_goldens()`
(spec §9.1). Regenerating them is a deliberate act — a diff in
`test_replay_goldens.py` is a behavior change to investigate, never a
tolerance to widen. This file records why the frames were re-baselined and
what changed, so a future diff isn't mistaken for silent drift.

## 2026-08-01 — `PDEEngine` starts forwarding its solver's event stats

**What changed in the library.** `PDEEngine` is a dispatch facade: `price()`
routes to a product-specific solver (`SnowballPDESolver`, `PhoenixPDESolver`,
...) via `_get_solver`. `SnowballPDESolver` has always implemented
`calculate_event_stats`, returning exact PDE-derived event statistics — but
the facade never overrode `calculate_event_stats`, so it inherited
`BaseEngine`'s default (`return None`, meaning "not supported"), silently
discarding the solver's working result. `PDEEngine.calculate_event_stats` now
delegates to `self._get_solver(product).calculate_event_stats(...)`, the same
pattern `price()` already uses.

**Which goldens moved and why.** `fixtures.make_scalar_bsm_config()` and
`fixtures.make_localvol_config()` both set `pricing_engine_type=EngineType.PDE`
with `event_stats_fallback="mc"`. Before the fix, the primary engine
(`PDEEngine`) returned `None`, so every event-stats row in these two goldens
was actually produced by the **MC fallback** — the `event_stats_engine`
column read `mc_fallback` throughout. After the fix, the primary PDE path
returns real data, so the same column now reads `primary`. The four frames
that changed:

- `scalar_bsm_daily_event_summary.csv`
- `scalar_bsm_event_probabilities.csv`
- `localvol_daily_event_summary.csv`
- `localvol_event_probabilities.csv`

No other frame moved: `states`, `greeks`, `rebalances`, `trades`, `actions`,
`surfaces`, and both `summary.json` files are byte-identical before and after
(verified via `git diff --stat test/replay_golden/data/` after regeneration) —
the fix changes event reporting, not pricing or hedging.

**`book` did not move, and that is expected, not an oversight.**
`fixtures.make_book_config()` uses `pricing_engine_type=EngineType.QUADRATURE`,
not PDE (`SnowballQuadEngine`/`PhoenixQuadEngine` already implement
`calculate_event_stats` natively). Its `event_stats_engine` column already
read `primary` before this fix — the book golden was never exercising the
defect, so regenerating it reproduced the same bytes.

**KO probabilities cross-validate the fix.** The old (MC, 10,000-path) and
new (exact PDE) KO probabilities agree to MC's own quantization — every old
value is a multiple of `1e-4` (10,000-path counting noise); the new value
sits inside that same bucket:

| row (date -> event_date) | old (`mc_fallback`) | new (`primary`, exact PDE) |
|---|---|---|
| 2024-01-02 -> 2024-01-04 KO | 0.0338 | 0.034795 |
| 2024-01-02 -> 2024-01-07 KO | 0.1056 | 0.102031 |
| 2024-01-03 -> 2024-01-04 KO | 0.0 | 0.0000039 |
| 2024-01-03 -> 2024-01-07 KO | 0.0016 | 0.001682 |

The third row is the clearest tell: MC's 10,000 paths never sampled that
1-in-250,000-ish knockout and reported a flat `0.0`; the PDE resolves it.

**KI rows differ in date attribution and count — this is the documented
`ki_probability` definition split between engine families, not a new bug.**
The MC engine emits two dated KI rows per scalar_bsm/localvol case (`0.004`
at 2024-01-03 and `0.0646` at 2024-01-05, summing to `0.0686`); the PDE
emits one row (`0.065123` at 2024-01-08).

`ki_probability` is documented in `event_stats.py` as a **legacy** field
whose definition differs by engine, kept only for backward compatibility:
MC reports `P(KI ever)` — every path that ever breaches the KI barrier,
counted regardless of what happens to it afterward — while QUAD/PDE report
`P(KI ever AND never KO)`, with the KI indicator absorbed to 0 on **any**
KO the path experiences, order-agnostic (see the `KI_indicator` /
`KI_ever_indicator` surface-column comment in
`SnowballPDESolver._compute_event_stats`, and the `ki_probability`
docstring in `event_stats.py`). `AutocallableEventStats` also carries two
unambiguous, cross-engine-consistent alternatives —
`ki_ever_probability` and `ki_survive_knocked_in_probability` — that the
docstring says to prefer over `ki_probability`. The KI column in these
goldens is the legacy field; a reader comparing this frame across engine
families should not treat it as directly comparable without accounting
for that.

Concretely: MC's `P(KI ever)` = 0.0686 counts every path that ever breached
the KI barrier; the PDE's `P(KI ever AND never KO)` = 0.065123 excludes
those that breached KI and subsequently knocked out. The 0.0035 delta is
paths that knocked in and then knocked out — predominantly KI first, KO
second, given this barrier geometry (KO barely above spot, KI far below:
see the `_ref_phoenix` comment in `test_ki_probability_definitions.py`,
where the dominant contributor to this same gap shape is paths that knock
in at the lower barrier, later recover, and autocall). This gap predates
this fix and is explicitly out of scope here (see the
KI-probability-definitions project notes); it is not to be "fixed" as part
of a golden re-baseline.

Row counts: `scalar_bsm_event_probabilities.csv` and
`localvol_event_probabilities.csv` went from 7 data rows to 6 (the two dated
KI rows collapsed into one); `daily_event_summary.csv` frames kept their row
count (one row per alive day) with only the per-row values changing.
