# Vol-model calibration

Worked examples for `quantark.volcalibration`: market quotes → an admitted
IV-surface artifact → calibrated Local Vol / Heston / Heston-SLV, driven by a
YAML config and four CLI verbs.

```
quotes  →  QuoteSnapshot  →  normalizer  →  QuoteSet  →  surface  →  artifact  →  model
           (envelope)        (convention)   (neutral)    (SABR +      (frozen     (cached
                                                         admission)    bytes)      by sha)
```

## Start here

```bash
.venv/bin/python example/volcalibration/01_end_to_end.py
```

Runs in seconds from a clean clone — no network, no private data. It walks the
whole module against the six committed CFETS USD/CNY snapshots: stages them
into a store, shows what a run *would* do, runs it, shows that re-running is a
no-op, inspects one date, then consumes the result through `CalibrationSet`.

Everything it writes lands under `output/volcalibration_demo/` (gitignored).

## The two configs

| File | Underlying | Runs from a clean clone? |
|---|---|---|
| `usdcny_calibration.yaml` | USD/CNY, delta-quoted (CFETS) | **Yes** — the snapshots carry their own spot |
| `../mo_volmodels/mo_calibration.yaml` | CSI 1000 MO options, strike-quoted settlement | No — needs the git-excluded settlement history |

The MO config is the real daily workflow this module was built for. Its
snapshots come from the CFFEX settlement CSVs, which carry no spot, so it also
needs `data/history/csi1000_spot.csv`. Once that history exists:

```bash
# one-time: make the CSV history readable by the library
.venv/bin/python example/mo_volmodels/export_snapshots.py

.venv/bin/python -m quantark.volcalibration run    example/mo_volmodels/mo_calibration.yaml --plan
.venv/bin/python -m quantark.volcalibration run    example/mo_volmodels/mo_calibration.yaml
.venv/bin/python -m quantark.volcalibration status example/mo_volmodels/mo_calibration.yaml --json
```

Running the shipped MO config rebuilds nothing: its `surface` block
fingerprints identically to the 787 admitted artifacts already on disk.

## The four verbs

```bash
python -m quantark.volcalibration run    <config.yaml> [--as-of D] [--backfill]
                                         [--from D] [--to D] [--variants ...]
                                         [--workers N] [--force] [--plan] [--json]
python -m quantark.volcalibration status <config.yaml> [--as-of D] [--json]
python -m quantark.volcalibration show   <config.yaml> --date D [--json]
python -m quantark.volcalibration list   [--config <config.yaml>] [--json]
```

An agent's decision procedure is `exit code → reason code → act`:

| Exit | Meaning | What to do |
|---|---|---|
| 0 | current | nothing |
| 2 | not current, fail-closed | read `reason`; **do not retry-loop** |
| 1 | pipeline or config failure | read `reason`, fix the cause |
| 75 | another process holds the lock | wait |

`2` means the system is working and the answer is "not yet". Some dates are
uncalibratable *forever* — two MO surfaces are excluded in the manifest and
never become admissible — so a retry loop on `2` never terminates.

Under `--json` stdout carries exactly one JSON object and every progress line
goes to stderr, on every exit path including failures.

## Consuming the result

```python
from quantark.volcalibration import CalibrationSet

cs = CalibrationSet.open("example/mo_volmodels/data/history",
                         runtime="output/mo_daily_calibration")

cs.dates()                          # admitted trading dates
cs.surface_for(d)                   # IvSurfaceArtifact, carry-forward applied
cs.model_for(d, "heston_slv")       # the calibrated model that date's record names
cs.environment_for(d)               # (PricingEnvironment, GridVolSurface, spot)
cs.status()                         # the same payload `status --json` prints
```

A backtest takes the object, not a directory:

```python
AutocallableMarketDataSet(..., calibration_set=cs)
```

`model_for` refuses rather than quietly re-fitting when the record does not
describe the surface in force, when the store was built under a different
calibration config, or when the date used temporal smoothing (whose EWMA
reference cannot be rebuilt from config alone). An unaudited fit returned under
an audited record's cover is the failure this prevents.

## Known limitation

`environment_for` returns a generic `PricingEnvironment` for every convention.
For an FX surface the carry sits in the dividend-yield slot, so forwards and
discounting are correct — `01_end_to_end.py` checks them against the published
CFETS forwards and they agree exactly — but an FX consumer wanting a typed
`FxPricingEnvironment` does not get one yet.

## Further reading

- `quantark/volcalibration/README.md` — the module guide: the
  `BUILDER_SCHEMA_VERSION` obligation, the artifact-bytes constraint, the
  resume rule and its two bounded exceptions.
- `docs/superpowers/specs/2026-09-01-volcalibration-module-design.md` — the
  design and the decisions behind it.
