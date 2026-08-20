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
python -m quantark.modelvalidation run example/modelvalidation/phoenix_flat_bsm.yaml
python -m quantark.modelvalidation run example/modelvalidation/ko_reset_flat_bsm.yaml

# Resume an interrupted run (reuses checkpoints whose configuration still matches)
python -m quantark.modelvalidation run example/modelvalidation/snowball_flat_bsm.yaml --resume

# What builders and studies exist?
python -m quantark.modelvalidation list
```

Output lands under `output/modelvalidation/<study>/`:

| File | What it is |
|---|---|
| `certificate.json` | The machine record, with its projected SHA-256. |
| `report.md` | Terminal- and diff-friendly report. |
| `report.html` | Review copy: one self-contained file, no external requests. Adds a margin gauge on every cell so "passed with room to spare" and "passed by a hair" are distinguishable at a glance, plus an Engine configuration section showing the grid each engine actually ran on. |
| `checkpoints/` | Resume state. Scratch, not evidence — never banked. |

That path is gitignored — banking evidence is a deliberate act, described in
[the release procedure](../../docs/modelvalidation/RELEASE_PROCEDURE.md).

## The studies

| File | What it certifies | Cost |
|---|---|---|
| `european_selftest.yaml` | Closed-form Black-Scholes against a small RQMC benchmark. The candidate is exact by construction, so the framework **must** admit it — this is the machinery's own calibration check, and it runs in CI. | ~3 s |
| `snowball_flat_bsm.yaml` | Snowball PDE and quadrature engines against one paired-RQMC benchmark, on PV and both spot Greeks, across five scenarios including near-KO, near-KI, low-vol, and near-expiry. | minutes |
| `phoenix_flat_bsm.yaml` | Phoenix PDE and quadrature engines against one paired-RQMC benchmark, across seven scenarios. Adds what the snowball has not: a coupon barrier, so every observation date carries a digital — `near_coupon` sits right on it — plus a memory-coupon case. | minutes |
| `ko_reset_flat_bsm.yaml` | KO-reset snowball PDE and quadrature engines against one paired-RQMC benchmark, across seven scenarios. The payoff switches regime at knock-in — a pre-KI schedule to `maturity_pre`, a second lower KO schedule on to `maturity_post` — so the two value surfaces live on different horizons. | minutes |
| `adi2d_snowball_greeks.yaml` | **Imported, not runnable end to end.** The 2D ADI Heston and Heston-SLV snowball solvers on spot delta and gamma, across seven variance regimes (low Feller, collapsed vol-of-vol, near-KO, near-KI, …). Its benchmark is a multilevel control-variate telescope that cost 28.6 h of held-out sampling, so `run` refuses; the candidate arm is live and anchored. See [the banked certificate](../../docs/modelvalidation/certificates/adi2d-snowball-greeks/2026-08-19/README.md) and release procedure §10. | anchors ~6 min |

## Writing a new study

Read the [release procedure](../../docs/modelvalidation/RELEASE_PROCEDURE.md),
section 7. In short: add builders for your engine family under
`quantark/modelvalidation/builders/`, write the YAML here, measure the
achievable standard error before choosing bounds, then run, bank, and extract
anchors.

One rule worth repeating: bounds encode what a desk can tolerate. Never widen
one to turn a REJECTED into an ADMITTED.
