# Gate Report: Range Accrual Analytical Engine

## SR 11-7 Model Validation - Developer B Independent Verification

| Field              | Value                                      |
|--------------------|--------------------------------------------|
| Model              | Range Accrual Analytical Engine (Digital Decomposition) |
| Developer A        | Engine at `asset/equity/engine/analytical/range_accrual_analytical_engine.py` |
| Developer B        | Independent implementation at `model-validation-output/range-accrual-analytical/validation/independent-impl/verify_range_accrual_analytical.py` |
| Date               | 2026-02-10                                 |
| Gate Decision      | **PASS**                                   |
| Tolerance          | 1e-10 (relative)                           |

---

## 1. Methodology

Developer B independently derived and implemented the Range Accrual pricing formula from the research specification only (`research/research-report.md`), without reading Developer A's engine source code.

### Core Formula (from research report)

```
Price = exp(-r*T) * S_0 * M * c * tau * (1/W) * [past_in_range + sum_i w_i * P_i]

P_i = N(d2_L) - N(d2_U)           (standard mode)
P_i = 1 - [N(d2_L) - N(d2_U)]    (reverse mode)

d2(K, t_i) = [ln(S/K) + (r - q - sigma^2/2) * t_i] / (sigma * sqrt(t_i))
```

The independent implementation uses only `scipy.stats.norm` and `math` -- no shared code with Developer A's engine.

---

## 2. Test Cases

| # | Description | Parameters | Feature Tested |
|---|-------------|------------|----------------|
| 1 | Standard 12M monthly | S=100, L=90, U=110, sigma=0.2, r=0.05, q=0.02, T=1.0, rate=5% ann. | Baseline pricing |
| 2 | Reverse mode | Same as Case 1 with `is_reverse=True` | Reverse mode decomposition |
| 3 | Narrow range, low vol | S=100, L=95, U=105, sigma=0.1, T=0.5, 4 obs, rate=8% non-ann. | Non-annualized rate, lower vol |
| 4 | Time-varying barriers | L=[85,88,90,92], U=[115,112,110,108], 4 quarterly obs | Per-observation barrier lookup |
| 5 | Past + future obs | 2 past (1 in-range, 1 out) + 2 future | Historical observation handling |

---

## 3. Results

| Case | Independent Price | Developer A Price | Abs Diff | Rel Diff | Status |
|------|------------------:|------------------:|---------:|---------:|--------|
| 1 - Standard 12M monthly  | 2.63864899 | 2.63864899 | 1.33e-15 | 5.05e-16 | PASS |
| 2 - Reverse mode           | 2.11749814 | 2.11749814 | 0.00e+00 | 0.00e+00 | PASS |
| 3 - Narrow range, low vol  | 5.08983659 | 5.08983659 | 0.00e+00 | 0.00e+00 | PASS |
| 4 - Time-varying barriers  | 2.63656266 | 2.63656266 | 0.00e+00 | 0.00e+00 | PASS |
| 5 - Past + future obs      | 2.62040895 | 2.62040895 | 0.00e+00 | 0.00e+00 | PASS |

### Supplementary Checks

| Check | Result | Status |
|-------|--------|--------|
| Case 1 ratio + Case 2 ratio = 1.0 | sum = 1.000000000000000 (\|err\| = 2.22e-16) | PASS |
| Research benchmark (E[ratio]=0.5548, Price=2.6386) | Matches to 4 decimal places | PASS |

---

## 4. Per-Observation Probability Comparison (Case 1)

All 12 monthly observation probabilities match exactly between implementations:

| Obs | Time (yr) | Independent P_i | Developer A P_i | Match |
|-----|-----------|----------------:|----------------:|-------|
| 1   | 0.0833    | 0.9161874618    | 0.9161874618    | Exact |
| 2   | 0.1667    | 0.7793263153    | 0.7793263153    | Exact |
| 3   | 0.2500    | 0.6829407257    | 0.6829407257    | Exact |
| 4   | 0.3333    | 0.6138703272    | 0.6138703272    | Exact |
| 5   | 0.4167    | 0.5617949399    | 0.5617949399    | Exact |
| 6   | 0.5000    | 0.5208674281    | 0.5208674281    | Exact |
| 7   | 0.5833    | 0.4876559525    | 0.4876559525    | Exact |
| 8   | 0.6667    | 0.4600269173    | 0.4600269173    | Exact |
| 9   | 0.7500    | 0.4365834106    | 0.4365834106    | Exact |
| 10  | 0.8333    | 0.4163703499    | 0.4163703499    | Exact |
| 11  | 0.9167    | 0.3987108091    | 0.3987108091    | Exact |
| 12  | 1.0000    | 0.3831103524    | 0.3831103524    | Exact |

---

## 5. Conclusion

**GATE DECISION: PASS**

Developer A's Range Accrual Analytical Engine implementation matches the independently derived formula to machine precision (relative difference < 1e-15) across all five test cases covering:

- Standard pricing with monthly observations
- Reverse mode (pay-outside-range) decomposition
- Non-annualized rate with narrower barriers and lower volatility
- Time-varying (step-down) barriers
- Partially observed instruments with historical and future observations

The digital decomposition approach (linearity of expectation for independent observation probabilities under GBM) is correctly implemented. The standard-plus-reverse identity (ratios summing to exactly 1.0) provides additional confirmation of internal consistency.

No discrepancies were found. The model is validated for production use.
