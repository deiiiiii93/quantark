# `quantark.volcalibration`

Market quotes → admitted IV-surface artifact → calibrated Local Vol / Heston /
Heston-SLV. This is the layer between `quantark.volmodels` (asset-neutral
kernels, no `PricingEnvironment`) and the consumers that need a *dated,
market-calibrated* model: `backtest`, `modelvalidation`, and the example suites.

```
quotes  →  QuoteSnapshot  →  normalizer  →  QuoteSet  →  surface  →  artifact  →  model
           (envelope)        (convention)   (neutral)    (SABR +      (frozen     (cached
                                                         admission)    bytes)      by sha)
```

## Run it

```bash
# one-time: make the CFFEX CSV history readable by the library
.venv/bin/python example/mo_volmodels/export_snapshots.py

# what would happen, writing nothing
.venv/bin/python -m quantark.volcalibration run example/mo_volmodels/mo_calibration.yaml --plan --json

# do it
.venv/bin/python -m quantark.volcalibration run example/mo_volmodels/mo_calibration.yaml

# where are we
.venv/bin/python -m quantark.volcalibration status example/mo_volmodels/mo_calibration.yaml --json
.venv/bin/python -m quantark.volcalibration show   example/mo_volmodels/mo_calibration.yaml --date 2026-07-20 --json
.venv/bin/python -m quantark.volcalibration list   --config example/mo_volmodels/mo_calibration.yaml
```

## The agent contract

1. **stdout is data, stderr is narrative.** Under `--json`, stdout carries
   exactly one JSON object; every progress line goes to stderr.
2. **Exit codes carry state.**

   | code | meaning | what an agent should do |
   |---|---|---|
   | 0 | current | nothing |
   | 2 | non-current, fail-closed | look at `reason`; **do not retry-loop** |
   | 1 | pipeline or config failure | read `reason`, fix the cause |
   | 75 | another process holds the lock | wait, or come back later |

   A `2` means the system is working correctly and the answer is "not yet".
   Some dates are uncalibratable *forever*: two MO surfaces (2024-09-30,
   2025-04-08) are excluded in the manifest and will never become admissible.
3. **Idempotent and resumable.** Re-running a successful config is a no-op
   reporting `current`; `--plan` writes nothing; every write is atomic and the
   calibration manifest is persisted after each date, so an interrupted
   backfill resumes where it stopped.
4. **Failures are machine-readable.** Every `--json` exit carries a stable
   `reason` code. The decision procedure is `exit code → reason code → act`,
   never prose parsing. Surface-level reasons come from `AdmissionReason`.

`--as-of` defaults to the system's local date. A scheduler should pin it.

## Two obligations you must not skip

**`BUILDER_SCHEMA_VERSION` (`store.py`).** Bump it whenever a change to
normalization, smoothing or admission **code** alters builder output. A config
change is covered by the builder fingerprint; a code change is not, and the
version bump is the only mechanism that invalidates artifacts after a builder
upgrade. This is the same obligation `_CACHE_SCHEMA_VERSION` carries in
`calibrate.py` for kernel changes.

**Artifact bytes must not change.** The artifact's sha256 is taken over raw
file bytes and feeds the calibration cache key; 787 admitted MO surfaces, a
warm calibration cache and every cohort pin are keyed on those shas. **No field
may be added to or removed from the artifact body.** Anything new goes in a
manifest record. (The body already carries `source_sha256`, `source_url`,
`source_class`, `price_field`, `trade_date`, `schema_version` and
`node_universe` — frozen means unchangeable, not empty.)

## Resume: "already built" is checked, not assumed

Before skipping a date the runner compares four recorded fields against the
current inputs and config; any mismatch rebuilds.

| field | invalidates on |
|---|---|
| `snapshot_sha256` | a re-published or corrected source snapshot |
| `price_field` | `settlement` ↔ `mid_or_last` change |
| `builder_fingerprint` | canonical JSON of the resolved `surface:` config |
| `builder_schema_version` | a builder **code** change (see above) |

**The fingerprint payload is legacy-shaped on purpose.** It carries the five
keys the frozen manifest records (`sabr_beta`, `min_expiries`,
`min_strikes_per_expiry`, `min_common_strikes`, `artifact_schema_version`) plus
any knob moved off its default. Hashing all seven `SurfaceBuildConfig` fields
would mismatch every migrated record on the first resumed run and rebuild 787
artifacts — destroying exactly the bytes the rule exists to protect. A moved
knob still enters the payload, so nothing is silently ignored.

**Two bounded exceptions, both discriminated by data, both overridable with
`--force`:**

- A record marked `provenance: "grandfathered"` is trusted as-is. Its artifact
  bytes are the pinned object; rebuilding would destroy them to prove a
  property nobody doubts. The count appears in `status --json`.
- A record whose `reason` is **outside** `AdmissionReason` was written by
  something other than the builder — `example/mo_volmodels/exclude_thin_surfaces.py`
  writes `insufficient_expiries_for_dupire` and keeps the artifact on disk.
  Rebuilding it would silently re-admit a surface a human decided to exclude.

## Three normalizers, one `QuoteSet`

`listed.py` (live books), `settlement.py` (exchange EOD marks) and
`fxdelta.py` (CFETS delta-quoted) all emit the same `QuoteSet`, so everything
downstream is convention-blind. The two listed normalizers are **not** one
function with a flag — they disagree on the price field, the liquidity rule,
the maturity derivation, the expiry-date check and the IV-inversion units, and
a parameterized version would let a caller select combinations no real venue
produces.

**The strike-grid rule is declared, not inferred.** Listed ladders share
observed strikes, so their grid is the strikes the market quoted (inside the
quoted-range overlap, iterated to a fixed point). Delta-quoted books share
none — every tenor's 25-delta strike is its own — so their grid is laid
uniformly over the interval all tenors cover, and `node_universe` records
`grid_values_are_model_values: true` so nobody mistakes grid width for
liquidity. The normalizer states which rule its book obeys in
`QuoteSet.universe`; auto-detecting would hide an empty intersection as a
design choice.

## Price-field discipline

A snapshot declares one `price_field` and `quote_price` resolves that field
only — there is no cross-field fallback. A snapshot declaring `settlement`
whose quotes carry only `last` is rejected as `price_field_mismatch`, never
silently priced off another key. `mid_iv` (delta-quoted books) has no price at
all, and `quote_price` says so rather than inventing one from a vol.

## Deliberate divergences from the ported example code

Do not "fix" these back.

- **`CalibrationSet.environment_for` dates the environment to the day being
  priced.** `_mo_common.build_env` hardcoded `datetime(2026, 7, 6)` for every
  date — harmless for a one-date demo, wrong for a history, and wrong again
  when the manifest carries a surface forward.
- **`HESTON_PRESETS["mo_frozen"]` sets `enforce_feller=True`** where the
  diagnostics scripts use only the soft penalty. See the comment in
  `config.py`: feasibility is bought with smile accuracy, so runs must report
  per-date fit RMSE.
- **The runner calls functions.** Stage 14's `default_stage_runner` shelled out
  to sibling scripts via `subprocess`; its `stages` block is gone with it.

## Before a production run

The legacy manifest migration (`store.migrate_manifest`) is a one-time,
no-rebuild step. Run it against a **copy** of the real manifest first and
check that zero artifact files changed:

```python
from quantark.volcalibration.store import migrate_manifest
migrated = migrate_manifest(json.loads(path.read_text()), iv_surface_dir=iv_dir)
```

Every admitted date recovers its `snapshot_sha256` from the artifact's own
`source_sha256`, so almost all records reach `verified` without needing the
source CSV. This check is not in CI: it needs the git-excluded history.

## Out of scope

- **Model certification** — stages 11/16/17 and the G1 cohort verifier belong
  to `quantark.modelvalidation`. What this module owns is the *builder's*
  admission: does this date's smoothed surface pass butterfly/calendar and
  yield a valid Dupire surface.
- **Vendor fetching** — the `01_fetch_*` scripts and the akshare/wind skills
  keep that job. `export_snapshots.py` is the one-way bridge into the library's
  canonical envelope.
- **Fleet/cohort orchestration** — stage 12/13 stay in `example/`.
