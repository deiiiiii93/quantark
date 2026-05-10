# 交叉验证 / MC Comparison Report

## Result

**Status**: PASS_WITH_NOTES

## Summary

The analytical and MC engines are cross-checked through shared tests and
behavioral constraints:

- expiry analytical formulas match direct truncated-lognormal expectations
- MC expiry pricing stays inside a reasonable payoff range and reports standard error
- multiplier scaling is exact under identical MC seeds
- discrete schedule generation and execution are covered

## Notes

The test suite favors deterministic assertions over high-path stochastic
acceptance tests to keep CI runtime stable. Manual high-path comparisons can be
added as benchmark scripts if tighter numerical acceptance bands are required.
