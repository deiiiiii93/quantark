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
`quantark/asset/equity/engine/pde/core/statistics.py`, and the `ki_probability`
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

## 2026-09-28 — the PDE reads delta and gamma off a local cubic

**What changed in the library.** `BasePDESolver._calculate_delta_gamma`
(e0ba6ba7) fits a cubic through the four log-spot nodes around the query and
evaluates both derivatives at the query. The nearest-node quadratic it replaced
has a constant curvature, so it carried the node's gamma to the query: an
O(dx) gamma error whose sign flipped as refinement moved the nearest node from
one side to the other. A three-node grid keeps the quadratic. The change is a
readout repair in code, so the banked PDE certificates that anchor delta and
gamma were re-certified alongside it (release procedure, section 1).

**Which goldens moved and why.** Both PDE-priced goldens report PDE Greeks:

- `scalar_bsm_greeks.csv`, `localvol_greeks.csv`: the delta and gamma columns
  and every column derived from them (product/position, pre/post hedge, cash
  1%). Delta moved by at most 2.6e-5 of its column scale, gamma by 1.2%.
- `scalar_bsm_surfaces.csv`: `delta`, `gamma` and their cash columns; delta by
  1.2e-4 of scale, gamma by up to 3.6% on the surface's off-spot nodes, where the
  query sits farthest from a grid node.

No other frame moved. PV, states, trades, rebalances, actions, event statistics
and all three summaries are byte-identical: the hedge rounds the new delta to
the same futures quantity on every day, and `book` prices on QUAD.

## 2026-10-02 — 0.8.0, the PDE 1D numerical policy (spec 3A)

Spec: `docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-design.md`.
Seven flips, one commit each, every one checked against the PDE byte book
(`test/pde_bitwise/`) with its declared movers:

- (a) the grid day is `PricingEnvironment.bus_days_in_year`; (b) `max_steps`
  is 12,000 and an over-cap grid raises; (c) the barrier/touch family projects
  discrete observations on the dual cell; (d) Snowball `NODAL` is nodal on
  interior events; (e) Phoenix takes θ = 1 on zero-diffusion steps; (f)
  half-step damping (1, 1) on every core solve whose product has no continuous
  knock-in; (g) forward far-field rows.

**Which frames moved and why.** Only flip (f) moved these frames: the
`scalar_bsm` and `localvol` fixtures price a snowball with discrete knock-in
on `PDEEngine`, whose damping is now half-step (1, 1). The frames that moved:
`scalar_bsm_{states,greeks,surfaces,daily_event_summary,event_probabilities}`
and `scalar_bsm_summary.json`; `localvol_{states,greeks,daily_event_summary,
event_probabilities}` and `localvol_summary.json`. The size:

| fixture | initial portfolio value before | after | total PnL before | after |
|---|---|---|---|---|
| scalar_bsm | −10131.86225680955 | −10131.70307842902 | 130.76636639859134 | 130.6071880180607 |
| localvol | −10156.288723735766 | −10156.45932315723 | 155.192833324807 | 155.36343274627143 |

About 1.6e-5 of the notional, the size of the time-error change spec 3A §4.2
measures for discrete knock-in at the standard density. Flips (a)–(e) and (g)
moved no frame: the fixtures' environments have 252 days, their grids are
under the cap, they hold no barrier/touch product, they price under the
default projection, they hold no Phoenix, and they hold no put.

`book` did not move: it prices on QUAD. `trades`, `rebalances`, `actions` and
both `localvol_surfaces` and `book_*` frames are byte-identical: the hedge
rounds the moved delta to the same futures quantity on every day.

**Execution goldens** (`test/execution/goldens/`, regenerated in the same
commit): see `test/execution/goldens/PROVENANCE.md`.

## 2026-10-04 regeneration: PDE 1D event statistics (0.9.0, spec 3B §8)

The event statistics now march as a value-policy column group (allocated rows, the value's θ and
stages, a marched maturity leg), and the lifecycle-knocked-in note reports its marched `ki_survive`
(stage 2 (d)) instead of 1.0. Prices did not move. Four frames moved, the rest are byte-identical
(every `book_*` frame included: `book` prices on QUAD):

| file | largest per-column delta |
|---|---|
| `scalar_bsm_daily_event_summary.csv` | `ki_probability_to_maturity` 1.18e-3 (after the knock-in: 1.0 -> 0.998817); KO / survival 4.39e-9; cash 4.39e-5 |
| `scalar_bsm_event_probabilities.csv` | KI `event_probability` 2.55e-6 (before the knock-in, stage (a)); KO 4.39e-9; cash 4.39e-5 |
| `localvol_daily_event_summary.csv` | `ki_probability_to_maturity` 1.12e-4 (after the knock-in: 1.0 -> 0.999888); KO / survival 2.01e-10; cash 2.01e-6 |
| `localvol_event_probabilities.csv` | KI `event_probability` 3.05e-6 (stage (a)); KO 2.03e-10; cash 2.01e-6 |

Per stage: (a) moved the probabilities before a knock-in (KI 2.6e-6 / 3.1e-6, KO <= 2e-10 here);
(d) moved `ki_survive` / `ki_probability_to_maturity` after one. `pv` is unchanged to the byte.

```
 test/replay_golden/data/localvol_daily_event_summary.csv   |  4 ++--
 test/replay_golden/data/localvol_event_probabilities.csv   | 10 +++++-----
 test/replay_golden/data/scalar_bsm_daily_event_summary.csv |  4 ++--
 test/replay_golden/data/scalar_bsm_event_probabilities.csv | 10 +++++-----
 4 files changed, 14 insertions(+), 14 deletions(-)
```
