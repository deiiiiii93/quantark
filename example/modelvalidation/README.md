# Certification studies

Study files for `quantark.modelvalidation`. A study names *builders* and their
parameters; the builders construct the real pricing objects. That keeps a study
declarative and diffable, and lets a certificate carry its own definition
verbatim — so a banked certificate is re-runnable anywhere the builders exist.

## Running

```bash
# Wiring check (seconds; explicitly not bankable evidence)
python -m quantark.modelvalidation run example/modelvalidation/european_selftest.yaml --quick

# Real runs
python -m quantark.modelvalidation run example/modelvalidation/european_selftest.yaml
python -m quantark.modelvalidation run example/modelvalidation/snowball_flat_bsm.yaml

# Resume an interrupted run (reuses checkpoints whose configuration still matches)
python -m quantark.modelvalidation run example/modelvalidation/snowball_flat_bsm.yaml --resume

# What builders and studies exist?
python -m quantark.modelvalidation list
```

Output lands under `output/modelvalidation/<study>/`: `certificate.json`,
`report.md`, and a `checkpoints/` directory that is scratch, not evidence.
That path is gitignored — banking evidence is a deliberate act, described in
[the release procedure](../../docs/modelvalidation/RELEASE_PROCEDURE.md).

## The studies

| File | What it certifies | Cost |
|---|---|---|
| `european_selftest.yaml` | Closed-form Black-Scholes against a small RQMC benchmark. The candidate is exact by construction, so the framework **must** admit it — this is the machinery's own calibration check, and it runs in CI. | ~3 s |
| `snowball_flat_bsm.yaml` | Snowball PDE and quadrature engines against one paired-RQMC benchmark, on PV and both spot Greeks, across five scenarios including near-KO, near-KI, low-vol, and near-expiry. | minutes |

## Writing a new study

Read the [release procedure](../../docs/modelvalidation/RELEASE_PROCEDURE.md),
section 7. In short: add builders for your engine family under
`quantark/modelvalidation/builders/`, write the YAML here, measure the
achievable standard error before choosing bounds, then run, bank, and extract
anchors.

One rule worth repeating: bounds encode what a desk can tolerate. Never widen
one to turn a REJECTED into an ADMITTED.
