# Model validation: example study

`quantark.modelvalidation` certifies a deterministic pricing engine against a
stochastic benchmark. A *study* is a YAML file that names the builders (the
benchmark and the candidate), the quantities to compare, the error bounds and
the sampling budget; the builders construct the real pricing objects.

## Try it

```bash
# Wiring check (seconds; not bankable evidence)
python -m quantark.modelvalidation run example/modelvalidation/european_selftest.yaml --quick

# The same study at full sampling
python -m quantark.modelvalidation run example/modelvalidation/european_selftest.yaml
```

`european_selftest.yaml` is the smallest complete study: a closed-form
Black-Scholes candidate against Monte Carlo, so any difference is benchmark noise.
Copy it as the starting point for your own study.

## Where the real studies live

The certified desk studies (snowball, phoenix, KO-reset, intraday, local-vol and
2-D ADI) are CI fixtures, not examples. They sit with the golden tests in
`test/golden/modelvalidation/`, next to the committed surfaces they read, and
their banked certificates are under `docs/modelvalidation/`. Read them for the
full YAML vocabulary; follow `docs/modelvalidation/RELEASE_PROCEDURE.md` to
certify a new engine.
