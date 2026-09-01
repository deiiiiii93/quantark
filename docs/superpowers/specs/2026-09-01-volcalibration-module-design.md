# `quantark.volcalibration` — official vol-model calibration module

**Date:** 2026-09-01
**Status:** design approved, implementation plan not yet written

## 1. Purpose

Turn the calibration procedure demonstrated in `example/mo_volmodels/` and
`example/fx_volmodels/` into a supported `quantark` module, so that:

1. a user — human or agent — can operate a daily vol-model calibration with one
   config file and one command;
2. the backtest module can consume the resulting calibration artifacts directly,
   instead of the 2760-line `12_snowball_volmodel_backtest.py` glue that does it
   today;
3. model certification stays out of scope and is handled by
   `quantark.modelvalidation`.

## 2. Starting position

### 2.1 What is already in the library

- **Kernels** (`quantark/volmodels/`): `build_dupire_local_vol`,
  `calibrate_heston` / `calibrate_heston_from_quotes`,
  `calibrate_leverage_surface_fp`.
- **Artifact contract** (`quantark/param/vol/surface_history.py`):
  `IvSurfaceArtifact`, `VolSurfaceHistory`. These *consume* a surface artifact;
  nothing in the library *produces* one.
- **Model calibration** (`quantark/volmodels/calibration.py`, 772 lines):
  `VolModelCalibrator` fits `localvol` / `heston` / `heston_slv` from an
  `IvSurfaceArtifact`, with a sha-keyed disk cache, a config fingerprint and a
  fail-closed contract.

### 2.2 What is only in `example/`

- **The whole market-data half**: put-call-parity recovery of `(DF, F, r, q)`,
  the OTM wing filter, call-equivalent Black-IV inversion, rectangular grid
  assembly, SABR + calendar-projection smoothing, static-arbitrage admission
  (`_mo_common.py`, `03_build_iv_surface_history.py`).
- **The daily operating layer**: resumable run, lock, three manifests, freshness
  status, exit codes (`14_daily_calibration_pipeline.py`, 1370 lines).
- **The backtest handover**: no module under `quantark/backtest/` imports
  `VolSurfaceHistory` or `VolModelCalibrator`. Today the handover *is*
  `12_snowball_volmodel_backtest.py`.

### 2.3 Two defects this design corrects

- **Layering inversion.** `VolModelCalibrator.__init__` takes `config: Any`,
  duck-typed, because `VolModelCalibrationConfig` lives in
  `quantark/backtest/replay/config.py`. A library calibration engine depends on
  the backtest package's config shape.
- **Duplicated procedure.** `example/mo_volmodels/` and `example/fx_volmodels/`
  implement the same stages 02–05 twice. They differ only in stage 02: MO
  recovers IVs from strike-quoted call/put prices via parity; FX reads
  delta-quoted CFETS pillars. Everything downstream is already common.

## 3. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Input boundary: normalized snapshot in.** Three normalizers ship — listed-live (bid/ask/last), listed-settlement (exchange EOD marks), FX delta-quoted. Vendor fetching stays in `example/` and the akshare / wind skills. | Matches the existing convention (`quantark/util/marketdata/adapter/` ships only a base protocol and a mock). Keeps the library network-free and deterministic under test. The live/settlement split is not a refinement of one rule: see §4.4. |
| D2 | **Run interface: YAML config + CLI verbs.** `python -m quantark.volcalibration run\|status\|show\|list`, JSON on `--json`, exit codes as state. | Mirrors `quantark.modelvalidation`, the one agent-operable module already in the tree. One versionable, diffable file reproduces a run. |
| D3 | **Backtest handover: publish a `CalibrationSet` loader and move the config.** `VolModelCalibrationConfig` moves into this module; `backtest.replay` re-exports it; backtest gains one config field accepting a `CalibrationSet`. | Fixes D-2.3's inversion and gives backtest a supported entry point. Fleet/cohort orchestration stays in `example/`. |
| D4 | **Home: new top-level `quantark/volcalibration/`.** | `quantark/volmodels/` promises asset-neutrality and no `PricingEnvironment`; this layer needs both. House precedent: `modelvalidation`, `execution`, `stresstest`, `dynamicscenario`. |
| D5 | **Structure: functional stages + thin orchestrator.** The normalizer is the only pluggable point. | Two quote conventions differ in exactly one stage; a general stage registry would be speculative and could express numerically invalid orderings. Keeps the existing per-stage tests usable. |
| D6 | **Agent surface: CLI only, documented.** No skill, no scheduler promotion in this effort. | The four CLI guarantees (§6.3) serve any agent. A skill can be added later without touching the package. |

## 4. Architecture

### 4.1 Layout

```
quantark/volcalibration/
  __init__.py        public API re-exports
  __main__.py        python -m quantark.volcalibration
  cli.py             verbs: run | status | show | list
  config.py          RunConfig, SurfaceBuildConfig, VolModelCalibrationConfig (moved)
  yaml_loader.py     YAML -> RunConfig, fail-closed
  snapshot.py        QuoteSnapshot schema + validated load
  normalize/
    __init__.py      QuoteNormalizer protocol
    listed.py        live books: parity (DF, F, r, q); OTM wing filter; IV inversion
    settlement.py    exchange EOD marks: expiry-calendar check, volume+OI filter,
                     normalized IV inversion, auditable node universe
    fxdelta.py       tenor selection; pillar mid extraction; delta<->strike round-trip check
  surface.py         grid assembly, SABR slice fit, calendar projection of total variance
  admission.py       static-arb admission (butterfly + calendar; reduced form at 2 expiries)
  calibrate.py       VolModelCalibrator, HESTON_PRESETS, sha-keyed cache   <- moved
  store.py           on-disk layout, manifests, atomic writes, CalibrationSet loader
  runner.py          orchestrator: per-date loop, resume, lock, freshness/status
```

### 4.2 Dependency direction

`volcalibration` depends on `volmodels` (neutral kernels), `param`
(`GridVolSurface`, SABR, curves, `surface_history`), `priceenv` and `util`.
Nothing in `volcalibration` imports `backtest` or `asset`. After the config
move, `backtest.replay` imports `VolModelCalibrationConfig` from
`volcalibration`, reversing the current inversion.

### 4.3 What stays where it is

- `IvSurfaceArtifact` / `VolSurfaceHistory` remain in
  `quantark/param/vol/surface_history.py`. They are the contract between the
  producer (`volcalibration`) and consumers (`backtest`, `modelvalidation`,
  examples); keeping them in the shared lower layer lets both sides depend on
  them without depending on each other. `volcalibration.__init__` re-exports
  them for discoverability.
- Certification is out of scope: stage 11 (PDE convergence gate), stages 16/17
  (ADI Greek certification) and stage 13's G1 cohort verifier are study-level
  and belong to `quantark.modelvalidation`. What `volcalibration` owns is the
  builder's own admission — does this date's smoothed surface pass
  butterfly/calendar and yield a valid Dupire surface.

### 4.4 The convergence point

`listed.py`, `settlement.py` and `fxdelta.py` all emit the same `QuoteSet`.
Everything downstream — smoothing, admission, Dupire/Heston/SLV calibration —
is convention-blind. This is what collapses the duplicated stages 02–05 into
one code path.

**Why listed-live and listed-settlement are separate normalizers, not one with
a flag.** They are both "listed strike-quoted", but they disagree on nearly
every rule that matters, because settlement marks are official EOD prices
rather than executable quotes:

| | listed-live | listed-settlement |
|---|---|---|
| Price | bid/ask mid, else last | the `settlement` field |
| Liquidity | `volume >= 1` | `volume > 0` **and** `oi > 0` |
| Maturity | the payload's `T_years` | derived `calendar_days / 365` |
| Expiry date | trusted | verified against the third-Friday rule plus a holiday-shift table; a mismatch raises |
| IV inversion | raw units at `(S, K, r, q)` | normalized units at `(1, K/F, C/(DF·F), 0, 0)` |
| Rejections | dropped | counted by named reason into a `node_universe` |

Collapsing these into one parameterized function would mean a caller could
select a combination no real venue produces. They share the `QuoteSet` output
and nothing else, which is exactly the boundary the protocol draws.

## 5. Data contracts

### 5.1 `QuoteSnapshot` (new; the module's input)

```json
{
  "schema_version": 1,
  "convention": "listed_strike" | "fx_delta",
  "trade_date": "2026-09-01",
  "underlying": {"symbol": "000852.SH", "spot": 5432.1},
  "source": {"vendor": "cffex", "artifact": "20260901_1.csv",
             "sha256": "...", "price_field": "settlement"},
  "expiries": [ ... convention-specific ... ]
}
```

- `listed_strike` expiry:
  `{expiry_date, T_years, quotes: [{strike, type, settlement?, last?, bid?, ask?, volume, oi}]}`
  — the union of today's two MO snapshot shapes (see the price-field rule below).
  Open interest is `oi` in both existing shapes; the canonical schema keeps that name.
- `fx_delta` expiry:
  `{tenor, expiry_date?, T_years, forward, domestic_rate, quotes: [{pillar, strike, bid_iv, mid_iv, ask_iv, delta}]}`
  — today's CFETS snapshot shape, plus an optional `expiry_date` (§5.2).

`snapshot.py` validates both fail-closed. `01_fetch_*` scripts stay in
`example/` and emit this schema.

**Lifting today's files.** Neither existing MO snapshot shape is already this
envelope: the live snapshot (`fetched_at`, `market_open`, `underlying`,
`expiries`) has no `trade_date`, `schema_version` or `source`, and the
settlement snapshot (`schema_version`, `source_class`, `price_field`,
`expiries`, `expiry_calendar`, …) carries **no spot at all** — spot comes from a
separate CSV via `load_spot_map`, and the trade date from the filename. So
`snapshot.py` ships two explicit lifters,
`QuoteSnapshot.from_legacy_live(payload, *, trade_date)` and
`QuoteSnapshot.from_legacy_settlement(payload, *, trade_date, spot, source_sha256)`,
which supply the missing fields from their arguments and fail closed when a
caller omits one. Nothing infers a spot or a date.

**Price-field rule (listed conventions).** The MO suite has *two* live quote
shapes, not one: `01_fetch_mo_snapshot.py` writes `last`/`bid`/`ask`, while
`01_fetch_mo_settlement_history.py` writes a `settlement` key, and
`10_calibration_diagnostics.py:392` reads prices as `quote.get(PRICE_FIELD)`
with `PRICE_FIELD = "settlement"`. All 766 historical surfaces come from the
settlement path. `source.price_field` is therefore the **canonical, required
selector**, with exactly two allowed values:

| `price_field` | Price used per quote |
|---|---|
| `settlement` | the quote's `settlement` value |
| `mid_or_last` | `(bid + ask) / 2` when both are present and positive, else `last` (today's `_mo_common._quote_price`) |

The normalizer resolves the price *only* through this selector and rejects a
snapshot whose declared `price_field` is absent from its quotes. This preserves
both existing behaviours exactly while making the choice explicit rather than
inferred from which keys happen to be present.

**No mixed cohorts.** `10_calibration_diagnostics.py:129` already enforces that
"midpoint and settlement cohorts cannot mix". That invariant is promoted: the
run config declares the expected `price_field`, the normalizer rejects any
snapshot that disagrees, and each manifest record carries the `price_field`
actually used (§5.4) so a mixed history is detectable after the fact.

### 5.2 `QuoteSet` (new; the normalizer output)

```python
@dataclass(frozen=True)
class IvNode:
    strike: float
    iv: float
    weight_hint: float

@dataclass(frozen=True)
class ExpiryQuotes:
    expiry_label: str          # listed: the expiry date; FX: the tenor ("3M")
    expiry_date: str | None    # None when the snapshot carries no calendar date
    T: float
    forward: float
    discount_factor: float
    r: float
    q: float
    nodes: tuple[IvNode, ...]
    diagnostics: Mapping[str, float]   # parity RMSE/forward, implied rate, n_pairs

@dataclass(frozen=True)
class QuoteSet:
    trade_date: date
    spot: float
    convention: str
    expiries: tuple[ExpiryQuotes, ...]   # strictly increasing T
```

`listed.py` fills this via OLS parity, the OTM wing filter and call-equivalent
inversion, applying the parity quality gates (`|implied rate| <= 10%`,
`RMSE/forward <= 1%`). `fxdelta.py` fills it by tenor selection and pillar mid
extraction. `weight_hint` carries each convention's SABR fit weighting (MO:
Gaussian in log-moneyness; FX: per-pillar mode), so `surface.py` never branches
on convention.

`expiry_label` is separate from `expiry_date` because CFETS snapshots identify
a slice by tenor (`1M`, `3M`) and carry no calendar date; deriving one would
require an FX expiry calendar and a spot-lag/adjustment convention this module
does not own. Listed snapshots populate both. The artifact's `per_expiry` and
`atm_pillars` blocks emit `expiry_date` only when it is known — `IvSurfaceArtifact`
validates `T` and `forward`/`atm_vol`, and treats `expiry_date` as optional, so
this is compatible with existing artifacts unchanged.

### 5.3 Frozen contracts (unchanged)

- **Artifact JSON** (verified against a stored artifact, 2026-09-01):
  `{schema_version, trade_date, source_class, price_field, source_url,
  source_sha256, s0, strikes, maturities, iv_grid, per_expiry, node_universe,
  atm_pillars, extrapolation_policy, admission, target_smoothing}`.
- **Calibration cache entry**: `{schema_version, variant, surface_sha,
  surface_date, config_fingerprint, params, record}` at
  `cache_dir/{variant}-{sha256(surface_sha|variant|fingerprint)}.json`.

**Hard constraint — artifact bytes must not change.** The artifact sha256 is
taken over raw file bytes, and that sha is an input to the calibration cache
key. 766 admitted MO surfaces and a warm calibration cache are keyed on those
shas. Writing even one extra key into the artifact body changes every sha and
invalidates the cache and every cohort pin. Therefore: **no field may be added
to or removed from the artifact body.** Anything new goes in the manifest
record.

**The body already carries its own source provenance.** An earlier draft of
this section claimed provenance had to live in the manifest because the body
was frozen; that conflated "frozen" with "empty". Stored artifacts already
record `source_sha256`, `source_url`, `source_class`, `price_field`,
`trade_date` and `schema_version`, plus a `node_universe` block accounting for
every quote the normalizer filtered and every expiry it excluded. The freeze
means those fields must stay exactly as they are — it does not mean the
snapshot sha is unavailable per date. §5.4's migration depends on this.

### 5.4 Store layout

Formalizes stage 14's `PipelinePaths`; unchanged on disk.

```
<root>/
  snapshots/{YYYYMMDD}.json
  iv_surface/mo_iv_surface_{YYYYMMDD}.json
  surface_manifest.json          {date, status, reason, detail, n_expiries, artifact_sha256,
                                  snapshot_sha256, price_field, builder_fingerprint,
                                  builder_schema_version, provenance}
  calibration_cache/{variant}-{key}.json
  calibration_manifest.json      per-date per-variant records + resolved config
  status.json                    freshness: latest refreshed / admitted / calibrated, lag, state
  .lock
```

Three manifests, because they answer three questions with three lifetimes:
which dates produced a surface (rebuilt only when surfaces are); which surfaces
have models under which config (rebuilt per calibration run); is the pipeline
current (rewritten every run). Merging them would couple those lifetimes.

**Resume invalidation.** The fields beyond today's record exist so that
"already built" is a *checkable* claim rather than a filename's existence:

| Field | Invalidates on |
|---|---|
| `snapshot_sha256` | a re-published or corrected source snapshot for that date |
| `price_field` | `settlement` ↔ `mid_or_last` change |
| `builder_fingerprint` | canonical JSON of the resolved `surface:` config block |
| `builder_schema_version` | a change to builder *code* that alters output |

Before skipping a date, the runner compares all four; any mismatch rebuilds.
Without this, a re-published settlement CSV or a changed `sabr_beta` leaves a
stale surface that a resume run reports as complete — today the only remedy is
a blanket `--force`.

`builder_schema_version` covers what a config fingerprint structurally cannot:
a change to normalization, smoothing or admission code with inputs and config
unchanged. This is the same hazard `quantark/volmodels/calibration.py` already
handles with `_CACHE_SCHEMA_VERSION`, whose comment records the obligation to
bump it "when any kernel default that affects calibration output changes …
the version bump is the only mechanism that invalidates warm entries after a
kernel upgrade". The surface builder inherits that obligation verbatim, and it
is stated in the module `CLAUDE.md`.

Both the calibration cache (keyed on surface sha + config fingerprint) and this
comparison are pure metadata: neither adds anything to the artifact body, so
the §5.3 byte constraint holds.

**Legacy record migration (one-time, no rebuild).** Every existing record
predates these fields, so a naive "any mismatch rebuilds" would rebuild all 766
artifacts on the first resumed run — changing their bytes on a different
architecture and destroying the warm cache and cohort pins §5.3 exists to
protect. That is not acceptable, so the migration is defined explicitly:

1. Most of the missing metadata is already in the manifest, just at the wrong
   level. `save_manifest` writes top-level `price_field` and a `config` block
   (`sabr_beta`, `min_expiries`, `min_strikes_per_expiry`, `min_common_strikes`,
   `artifact_schema_version`) that *is* the resolved surface config those
   artifacts were built with. The migration copies it down to each record and
   derives `builder_fingerprint` from it. This is lossless and touches no
   artifact.
2. `snapshot_sha256` is read from the **artifact's own `source_sha256` field**
   (§5.3). Every admitted date has one, so this is a per-date recovery, not a
   guess and not a recomputation — it does not require the source CSV to still
   be on disk.
3. Each record gets `provenance: "verified"` when every field was recovered, or
   `"grandfathered"` when any is `null`.

**Almost every admitted record migrates to `verified`.** An earlier draft
assumed `snapshot_sha256` would usually be unrecoverable and designed
grandfathering as the common path; because the artifact body carries
`source_sha256`, that is wrong. `grandfathered` is now the narrow residue: a
record whose artifact is missing or unreadable, or an `excluded` date that
never had an artifact and so has no sha to recover.

**A `grandfathered` record is never treated as a mismatch and never triggers a
rebuild.** It is trusted as-is, because the artifact bytes are the pinned
object and rebuilding would destroy them to prove a property nobody doubts.
This is a deliberate, bounded exception — not a silent fallback: the count of
grandfathered dates appears in `status --json` and per-date in `show`, and
`--force` remains the way to demand a verified rebuild. New records are always
`verified`.

### 5.5 `CalibrationSet` (new; the backtest handover)

```python
cs = CalibrationSet.open(root)                    # validates manifests, fail-closed
cs.dates()                                        # admitted trading dates
cs.surface_for(date)      -> IvSurfaceArtifact    # manifest carry-forward applied
cs.model_for(date, variant) -> CalibratedVolModel
cs.environment_for(date)  -> PricingEnvironment   # spot + parity r/q term structures + surface
cs.status()               -> dict                 # same payload as status.json
```

`environment_for` removes real duplication: `_mo_common.build_env`,
`_fx_common.build_fx_environment` and stage 12's per-date environment
construction are the same function written three times.

`backtest.replay` gains one config field accepting a `CalibrationSet`; nothing
in backtest reaches into directories.

The provenance chain closes end to end: snapshot sha (recorded in the manifest)
-> artifact sha -> calibration cache key.

## 6. Operating interface

### 6.1 Run config

```yaml
schema_version: 1
name: mo-daily
underlying:
  symbol: "000852.SH"
  convention: listed_strike        # selects the normalizer
  price_field: settlement          # settlement | mid_or_last; snapshots must agree
paths:
  root: example/mo_volmodels/data/history
surface:
  sabr_beta: 1.0
  min_expiries: 2
  min_strikes_per_expiry: 5
  min_common_strikes: 5
  extrapolation: flat_beyond_last_listed_expiry
  parity_gate: {max_abs_implied_rate: 0.10, max_rmse_over_forward: 0.01}
calibration:
  variants: [localvol, heston, heston_slv]
  heston_preset: mo_frozen
  heston_max_nfev: 200
  slv: {eta: 1.0, n_steps: 40, n_x: 161, n_z: 81}
run:
  workers: 2
```

Every value currently hard-coded as a module constant in
`03_build_iv_surface_history.py` (`SABR_BETA`, `MIN_EXPIRIES`,
`MIN_STRIKES_PER_EXPIRY`, `MIN_COMMON_STRIKES`, `EXTRAPOLATION_POLICY`) becomes
config with that value as its default, so the config above is what is run
today, written down. The resolved config is echoed into both manifests.

### 6.2 Verbs

```
python -m quantark.volcalibration run    <config.yaml> [--as-of DATE]
                                         [--backfill [--from D] [--to D]]
                                         [--variants ...] [--workers N]
                                         [--force] [--plan] [--json]
python -m quantark.volcalibration status <config.yaml> [--json]
python -m quantark.volcalibration show   <config.yaml> --date D [--variant V] [--json]
python -m quantark.volcalibration list   [--config <config.yaml>]
```

- `show` reports one date's admission verdict, per-expiry parity diagnostics,
  fit RMSE in vol points, Heston vector with Feller ratio and bound hits, and
  SLV leverage range. Today that requires reading three JSON files by hand.
- `--plan` resolves the config, scans state and reports what a run would do —
  dates to build, dates to calibrate, expected cache hits — writing nothing.

### 6.3 Agent contract (enforced by tests)

1. **stdout is data, stderr is narrative.** With `--json`, every verb emits
   exactly one JSON object on stdout; all progress logging goes to stderr.
2. **Exit codes carry state** (inherited from stage 14 unchanged): `0` current;
   `2` non-current but fail-closed (source pending, surface excluded, stale);
   `1` pipeline or calibration failure; `75` another process holds the lock.
   `2` means the system is working correctly and the answer is "not yet" — an
   agent must not retry-loop on it. A date can be legitimately uncalibratable
   forever: the two thin MO surfaces (2024-09-30, 2025-04-08) are excluded in
   the manifest and will never become admissible.
3. **Idempotent and resumable.** Re-running a successful config is a no-op
   reporting `current`. An interrupted run resumes from the manifests; every
   write is atomic. "Already done" is verified, not assumed: a date is skipped
   only when its recorded `snapshot_sha256`, `price_field` and
   `builder_fingerprint` still match the current inputs and config (§5.4).
4. **Failures are machine-readable.** The `AdmissionError` reason vocabulary
   already in the builder — `missing_spot`, `invalid_spot`, `missing_csv`,
   `parse_failed`, `sabr_smoothing_failed`, `static_arbitrage`,
   `invalid_atm_pillar` — is promoted to a documented, stable enum, extended
   with `price_field_mismatch` (§5.1), and
   `status.json` names the blocking date, stage and reason code. An agent's
   decision procedure is `exit code -> reason code -> act`, never prose parsing.

Fail-closed with no fallback remains the rule: a failed calibration raises,
naming the surface date and sha. There is no flat-vol fallback anywhere.

## 7. Migration

### 7.1 Inventory

| From | Functions | To |
|---|---|---|
| `_mo_common.py` | `load_snapshot` | `snapshot.py` |
| | `ExpirySlice`/`iter_expiries`, `imply_forward_and_rate`, `select_otm`, `otm_implied_vol` | `normalize/listed.py` |
| | `sabr_smoothed_surface`, `prepare_model_surface` | `surface.py` |
| | `build_env` | `store.py` (`environment_for`) |
| `03_build_iv_surface_history.py` | `_surface_base` — including the quoted-range-overlap domain iterated to its fixed point, `off_grid_node_count`, and the `node_universe` block | `surface.py` |
| | `_validate_static_arbitrage`, `AdmissionError` + reason codes | `admission.py` |
| | `serialize_artifact`, `save_manifest`, `load_manifest_records`, `_record_from_existing_artifact` | `store.py` |
| | `_build_one` worker pool | `runner.py` |
| `10_calibration_diagnostics.py` | `build_calibration_nodes`, `_third_friday` + `EXPIRY_DATE_OVERRIDES`, `_near_atm_parity_sensitivity`, parity quality-gate constants | `normalize/settlement.py` |
| `14_daily_calibration_pipeline.py` | `PipelinePaths`, `persist_calibration_manifest`, `load_calibration_records` | `store.py` |
| | `acquire_lock`, `select_calibration_dates`, `calibrate_one_surface`, `build_freshness_status`, `status_exit_code`, EWMA temporal smoothing | `runner.py` |
| `_fx_common.py` | `selected_slices`, `iter_nodes`, `strike_from_spot_delta`, `spot_delta_from_strike` | `normalize/fxdelta.py` |
| | `build_fx_environment` | `store.py` |
| `volmodels/calibration.py` | whole file | `calibrate.py` (shim at old path) |
| `backtest/replay/config.py` | `VolModelCalibrationConfig` | `config.py` (re-exported from backtest) |

**Stays in `example/`:** all `01_fetch_*` vendor IO and CSV parsing, spot-cache
refresh, plotting, the lecture/explainer, exotics demos (07–09), fleet/cohort
orchestration (12–13), and Heston identification diagnostics (Jacobian/SVD,
multistart, bootstrap).

**Deleted:** stage 14's `default_stage_runner`, which shells out to sibling
scripts via `subprocess`. The runner calls functions.

### 7.2 `_mo_common.py` becomes a re-export shim

14 Python modules import `_mo_common` (9 example scripts, 5 test modules);
stages 11/12/13/16/17 transitively depend on
stage-module behaviour beyond that, and one of them is 6128 lines. Rewriting
them in the same change would make a library-promotion PR indistinguishable
from a suite refactor.

With the shim, promotion is verifiable by a property that needs no judgement:
**the existing 35 `test/mo_volmodels/` test modules pass unchanged against library
implementations.** Simplifying stages 02/03/05 into thin library demos becomes
an optional follow-up, not a prerequisite.

`quantark/volmodels/calibration.py` and `backtest.replay`'s
`VolModelCalibrationConfig` follow the same shim pattern already used for
`backtest/otc` -> `backtest/replay` (deprecation shims until 0.5.0).

## 8. Testing

- **Ported unit tests.** `test_parity`, `test_otm_filter`, `test_iv_inversion`,
  `test_snapshot_io`, `test_stage02_surface`, `test_surface_admission_dupire`
  are re-pointed at library imports in `test/volcalibration/`. The originals
  stay, now exercising the shim.
- **Artifact reproduction.** Rebuild existing dates and compare. **Byte
  comparison is not CI-safe**: CI is x86_64 Linux, the artifacts were frozen on
  ARM64, and SABR calibration runs through `scipy.optimize`, so bitwise
  equality across architectures is not a property we have. The test therefore
  compares with tolerance always (via `test/golden_compare.py`), and byte-
  compares only when pointed at a local baseline directory — which is where the
  §5.3 constraint actually matters, since it is the local warm cache and cohort
  pins at risk.
- **Relocation invariant.** Mirrors `test/test_calibration_relocation.py`: an
  entry written through `quantark.volmodels.calibration` must be a cache *hit*
  for `quantark.volcalibration.calibrate`, proving no module path leaked into
  the key.
- **CLI contract.** Exit codes 0/1/2/75; stdout is pure JSON under `--json`;
  `run` idempotence; `--plan` writes nothing; an excluded-date fixture returns
  2 rather than 1.
- **Resume invalidation.** Four cases must each trigger a rebuild rather than a
  skip: a re-published snapshot for an existing date (new `snapshot_sha256`), a
  changed `surface:` config block (new `builder_fingerprint`), a changed
  `price_field`, and a bumped `builder_schema_version`. A fifth case asserts the
  converse — unchanged inputs skip and leave the artifact byte-identical.
- **Legacy migration.** Run the migration against a copy of the real
  `surface_manifest.json` and assert: zero artifact files modified (compare
  mtimes and bytes before/after), every `ok` record ends `verified` or
  `grandfathered`, and a subsequent resume run rebuilds nothing. This is the
  test that protects §5.3 from the invalidation rule added above.
- **Price-field discipline.** A snapshot declaring `price_field: settlement`
  whose quotes carry only `last`/`bid`/`ask` is rejected, and vice versa; a
  snapshot whose `price_field` disagrees with the run config is rejected. Both
  fail-closed with a named reason code, never by silently picking another field.
- **FX parity.** `fxdelta` produces a structurally identical `QuoteSet` and
  builds one end-to-end surface from the committed CFETS sample.

Tests must not rewrite tracked sample data under `example/`; the known churn in
the mo suite's sample files stays confined to the existing suite.

## 9. Phasing

Each phase is independently mergeable and leaves the tree green.

1. **Contracts + normalizers** — `snapshot.py`, `QuoteSet`,
   `normalize/listed.py`; `_mo_common` shim; ported tests green.
2. **Surface + admission** — `surface.py`, `admission.py`, `store.py` writers;
   the legacy manifest migration (§5.4) and its no-rebuild test; artifact
   reproduction test.
3. **Calibration relocation** — move `calibration.py` and
   `VolModelCalibrationConfig`; shims both sides; relocation invariant.
4. **Runner + CLI** — `runner.py`, `cli.py`, `__main__.py`, status and exit
   codes; CLI contract tests.
5. **`CalibrationSet` + backtest field** — loader, `environment_for`, one
   backtest config field.
6. **FX normalizer** — `normalize/fxdelta.py`, FX parity test.
7. **Docs** — module `CLAUDE.md`, worked MO config, snapshot schema reference,
   root `CLAUDE.md` table row.

## 10. Out of scope

- Model certification (stages 11, 16, 17) and the G1 cohort verifier — these go
  to `quantark.modelvalidation`.
- Vendor market-data fetching; the akshare and wind skills and the `01_fetch_*`
  scripts keep that job.
- Stage 12's fleet/cohort orchestration; only the date -> (surface, models,
  environment) resolution is promoted.
- Heston identification diagnostics (Jacobian/SVD, multiplier bootstrap,
  multistart) and the cross-date stability report.
- An agent skill and scheduler installation (D6).
