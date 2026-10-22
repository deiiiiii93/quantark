# Examples

Two kinds of material, kept apart.

**How-to examples** show how to use `quantark`. The folders mirror the package tree, so the demo for a module
is where you would look for the module: `asset/` (`bond`, `credit`, `equity`, `fx`, `rate`), `var/`, `stresstest/`,
`backtest/`, `dynamicscenario/`, `execution/`, `intraday/`, `param/`, `volmodels/`, `volcalibration/`,
`modelvalidation/`, `pnlexplain/`, `portfolio/`, `saccr/`, `sacva/`, `simm/`, `cashleg/`. Under `asset/equity/` the
demos are grouped by product family (`snowball`, `phoenix`, `options`, `engines`, ...). Run from the repository root:

```bash
python example/asset/equity/options/european_option_demo.py
```

**Case studies** live in [`case_studies/`](case_studies/README.md), one self-contained folder per study.
[`case_studies/mo_volmodels/`](case_studies/mo_volmodels/README.md) is the reference pattern for calibrating
vol models on a new market.

Generated output (`*_sample.*`, plots, `__pycache__`) is not tracked.
