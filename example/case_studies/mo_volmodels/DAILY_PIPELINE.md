# MO Daily Calibration Operations

## Contract

`14_daily_calibration_pipeline.py` is the operational entry point for the MO
EOD calibration chain. One invocation:

1. refreshes CSI 1000 spot and IM futures caches through the Shanghai as-of
   date using the AKShare interpreter;
2. downloads newly observed official CFFEX settlement CSVs with retry,
   provenance hashes, and atomic writes;
3. builds new SABR-smoothed IV surfaces and applies the existing calendar and
   butterfly-arbitrage admission gates;
4. calibrates Local Vol, hard-Feller Heston, and Heston-SLV once per admitted
   surface;
5. writes calibration and health artifacts atomically.

The workflow is fail-closed. It never substitutes flat volatility after a
surface or calibration failure, and an excluded current surface is reported as
`surface_excluded`, not silently labelled current.

## Commands

```bash
# Run through the current Shanghai date.
.venv/bin/python example/mo_volmodels/14_daily_calibration_pipeline.py run

# Deterministic historical/as-of operation.
.venv/bin/python example/mo_volmodels/14_daily_calibration_pipeline.py run \
  --as-of 2026-07-31

# Read-only health check.
.venv/bin/python example/mo_volmodels/14_daily_calibration_pipeline.py status --json
```

Surface construction defaults to one worker. This is intentional: the earlier
fleet attempt demonstrated that aggressive process-pool sizing can terminate
the pool and invalidate many otherwise independent tasks.

The first operational invocation uses `latest_admitted_surface_only` bootstrap
policy. It does not launch an accidental multi-year Heston-SLV backfill. To
request such a backfill explicitly:

```bash
.venv/bin/python example/mo_volmodels/14_daily_calibration_pipeline.py run \
  --backfill-calibrations \
  --calibration-start 2025-07-31 \
  --calibration-end 2026-07-30 \
  --max-calibration-dates 5
```

Repeat the command until no backlog remains. Persistent SHA/config-keyed cache
entries make retries resumable.

## Temporal calibration opt-in

Independent daily calibration remains the default. Enable the stability scheme
explicitly with:

```bash
.venv/bin/python example/mo_volmodels/14_daily_calibration_pipeline.py run \
  --temporal-smoothing
```

The opt-in performs three distinct, auditable operations:

1. Calibrate today's independent hard-Feller Heston vector. Its `v0` is always
   retained without averaging.
2. Update a recursive structural EWMA over `kappa`, `theta`, `sigma`, and
   `rho`. The default span is five admitted calibration dates, so
   `alpha = 2 / (5 + 1) = 1/3`. SLV leverage is recalibrated from today's raw
   `v0` plus this updated structural state.
3. Calibrate pure Heston with a structural temporal penalty toward the
   *prior-date* EWMA. The penalty is normalized by each frozen parameter-bound
   span and does not apply to `v0`. Its default weight is `0.1`.

The governed objective adds

```text
0.5 * lambda * sum_j ((parameter_j - prior_j) / bound_span_j)^2
```

for `j in {kappa, theta, sigma, rho}`. The hard Feller constraint remains
active; temporal regularization does not replace it.

Every temporal manifest record preserves the raw Heston fit, prior and updated
EWMA state, source-date count/range, regularized-Heston reference and penalty
cost, exact Heston vector used to recalibrate SLV leverage, and its Feller
ratio/verdict. Historical raw seeds and the resulting SLV vector fail closed
if they violate Feller. A first opt-in run bootstraps only the latest admitted
surface and can seed its EWMA from existing independent daily records.
Historical conversion still requires `--backfill-calibrations`.

Controls:

```bash
--structural-ewma-span 5
--heston-temporal-regularization 0.1
```

This scheme reduces parameter churn at the cost of a controlled fit penalty.
It does not repair sparse or arbitrage-invalid input surfaces; those remain
excluded by the existing admission gates.

Create a stability artifact from a completed bounded backfill:

```bash
.venv/bin/python example/mo_volmodels/15_calibration_stability_report.py \
  --start 2025-07-31 \
  --end 2026-07-30
```

The report writes a self-contained HTML dashboard, machine-readable JSON
evidence, and one-row-per-surface CSV under
`output/mo_calibration_stability/`. Excluded surfaces and missing/failed
calibrations remain explicit rather than being removed from the denominator.

## Runtime artifacts

Default runtime root: `output/mo_daily_calibration/`

| Artifact | Meaning |
|---|---|
| `status.json` | latest atomic pipeline/freshness state |
| `calibration_manifest.json` | one auditable record per calibrated surface date |
| `calibration_cache/` | SHA + configuration keyed LV/Heston/SLV artifacts |
| `pipeline.lock` | advisory single-run lock |
| `logs/daily.stdout.log` | launchd stdout |
| `logs/daily.stderr.log` | launchd stderr |

Historical source and surface artifacts remain under
`example/mo_volmodels/data/history/`.

## Status and exit codes

| State | Exit | Interpretation |
|---|---:|---|
| `current` | 0 | settlement, admitted surface, and all three calibrations match the latest refreshed trading date |
| `source_pending` | 2 | latest CFFEX file is not available yet; the 20:30 job retries |
| `surface_pending` | 2 | settlement exists but no surface decision is persisted |
| `surface_excluded` | 2 | latest surface failed static-arbitrage/admission checks |
| `calibration_pending` | 2 | surface exists but governed calibrations are incomplete |
| `market_cache_stale` | 2 | read-only status sees a spot cache more than four calendar days old |
| `calibration_failed` / `failed` | 1 | calibration or pipeline stage failed |
| `locked` | 75 | another invocation owns the advisory lock |

`status.json` also reports settlement/surface/calibration lag in trading dates,
the exact expected-date manifest records, stage commands and output tails, and
the last exception when present.

## Scheduler

Install the per-user launchd service:

```bash
.venv/bin/python example/mo_volmodels/install_daily_scheduler.py install
.venv/bin/python example/mo_volmodels/install_daily_scheduler.py status
```

Install the same schedule with temporal calibration enabled:

```bash
.venv/bin/python example/mo_volmodels/install_daily_scheduler.py install \
  --temporal-smoothing
```

The job runs at 18:30 and 20:30 Asia/Shanghai, Monday through Friday. The
second run is a source-publication retry. Repeated runs are safe because the
pipeline has a non-blocking lock, atomic manifests, and persistent caches.

Force an immediate run after installation:

```bash
.venv/bin/python example/mo_volmodels/install_daily_scheduler.py install --kickstart
```

Uninstall only when intentionally retiring the service:

```bash
.venv/bin/python example/mo_volmodels/install_daily_scheduler.py uninstall
```

## Recovery

- `source_pending`: wait for the scheduled retry or rerun the pipeline.
- `surface_excluded`: inspect the exact `reason`/`detail` in
  `surface_manifest.json`; do not fabricate or carry-forward a calibration and
  call it current.
- `calibration_failed`: inspect the date and per-variant error in
  `calibration_manifest.json`, correct the input/numerical issue, and rerun.
  Failed records are retried.
- `market_cache_stale`: run the full command without
  `--skip-market-refresh`.
- `locked`: inspect `pipeline.lock`; do not delete it while the recorded PID is
  alive. Advisory lock release is automatic when the process exits.

## Verification

```bash
PYTHONPATH=. .venv/bin/python -m pytest -o addopts='' -p no:cacheprovider \
  test/mo_volmodels/test_daily_calibration_pipeline.py \
  test/mo_volmodels/test_daily_scheduler.py \
  test/mo_volmodels/test_build_iv_surface_history.py \
  test/mo_volmodels/test_stage12_backtest_runner.py \
  test/test_mo_frozen_feller.py -q
```
