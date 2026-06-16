# SA-CVA — Basel standardised approach for CVA risk (MAR50.27–50.77)

Regulatory reference for `quantark/sacva`. Source: Basel Committee on Banking
Supervision, MAR50 (consolidated framework), paragraphs 50.27–50.77. This file is
the module-local copy of the methodology; the verbatim standard text the module
was built from is in `docs/sacva.md`.

## Nature of SA-CVA

SA-CVA is an adaptation of the FRTB standardised approach for market risk (MAR21)
to CVA risk. It is a **sensitivity-based aggregation (SBA)** method. Differences
from the market-risk SA (MAR50.27): reduced risk-factor granularity; no default
risk and no curvature risk.

The bank supplies, per risk factor `k`, the sensitivity of the aggregate
regulatory CVA (`S_k^CVA`) and of the eligible hedges (`S_k^Hdg`). This module
**consumes** those sensitivities (MAR50.29) — it does not compute CVA or run an
exposure model.

## Aggregation (MAR50.51–50.53)

Weighted sensitivities (MAR50.51–50.52):

```
WS_k^CVA = RW_k · S_k^CVA
WS_k^Hdg = RW_k · S_k^Hdg
WS_k     = WS_k^CVA − WS_k^Hdg
```

Bucket capital, with hedging-disallowance `R = 0.01` (MAR50.53(1)):

```
K_b = sqrt( max(0, Σ WS_k² + Σ_{k≠l} ρ_kl WS_k WS_l) + R · Σ (WS_k^Hdg)² )
```

Risk-class capital, with `S_b = max(−K_b, min(Σ WS_k, K_b))` and `m_CVA = 1`
(MAR50.41, 50.53(2)–(3)):

```
K = m_CVA · sqrt( max(0, Σ K_b² + Σ_{b≠c} γ_bc S_b S_c) )
```

Delta capital = simple sum of `K` over the 6 delta risk classes (MAR50.43); vega
capital = simple sum over the 5 vega risk classes (MAR50.45). **SA-CVA =
delta + vega** (MAR50.42). Counterparty credit spread has **no vega** (MAR50.63).

## Risk classes and factor structure

| Risk class | Vega | Buckets | Factors per bucket |
|---|---|---|---|
| Interest rate | yes | per currency | multi (specified: 1/2/5/10/30y + inflation; other: parallel-yield + inflation; vega: rate-vol + inflation-vol) |
| FX | yes | per non-reporting currency | single |
| Counterparty credit | no | 1–8 (8 = qualified index) | multi (entity × tenor 0.5/1/3/5/10y) |
| Reference credit | yes | 1–17 (16/17 = qualified index) | single |
| Equity | yes | 1–13 (12/13 = qualified index) | single |
| Commodity | yes | 1–11 | single |

## Tables (decimals)

### Interest rate (MAR50.54–50.58)
- Cross-bucket γ = 0.5.
- Specified-ccy = reporting ccy ∪ {USD, EUR, GBP, AUD, CAD, SEK, JPY}.
- Specified delta RW: 1y 0.0111, 2y 0.0093, 5y 0.0074, 10y 0.0074, 30y 0.0074,
  inflation 0.0111.
- Specified tenor ρ (Table 4): 1–2 0.91, 1–5 0.72, 1–10 0.55, 1–30 0.31,
  2–5 0.87, 2–10 0.72, 2–30 0.45, 5–10 0.91, 5–30 0.68, 10–30 0.83; any
  tenor↔inflation 0.40.
- Other-ccy delta RW 0.0158 (yield & inflation); ρ(yield, inflation) 0.40.
- Vega RW 1.00; ρ(rate-vol, inflation-vol) 0.40.

### FX (MAR50.59–50.62)
Cross-bucket γ = 0.6. Delta RW 0.11. Vega RW 1.00.

### Counterparty credit spread (MAR50.63–50.65) — delta only
Cross-bucket γ (Table 6), symmetric, unit diagonal:

```
       1     2     3     4     5     6     7     8
1    1.00  0.10  0.20  0.25  0.20  0.15  0.00  0.45
2          1.00  0.05  0.15  0.20  0.05  0.00  0.45
3                1.00  0.20  0.25  0.05  0.00  0.45
4                      1.00  0.25  0.05  0.00  0.45
5                            1.00  0.05  0.00  0.45
6                                  1.00  0.00  0.45
7                                        1.00  0.00
8                                              1.00
```

Delta RW (Table 7): bucket 1a IG 0.005 / HY-NR 0.020; 1b 0.010 / 0.040;
2 0.050 / 0.120; 3 0.030 / 0.070; 4 0.030 / 0.085; 5 0.020 / 0.055;
6 0.015 / 0.050; 7 0.050 / 0.120; 8 0.015 / 0.050.

Intra-bucket ρ = ρ_tenor · ρ_name · ρ_quality (MAR50.65(4)–(5)):
- ρ_tenor: 1.00 same tenor, else 0.90.
- ρ_name (buckets 1–7): 1.00 same name, 0.90 distinct but legally related
  (non-null equal legal-entity group), 0.50 otherwise. (bucket 8): 1.00 same
  index & series, 0.90 same index distinct series, 0.80 otherwise.
- ρ_quality: 1.00 same quality (IG/IG or HY-NR/HY-NR), else 0.80.

### Reference credit spread (MAR50.66–50.69) — delta + vega
Buckets 1–7 IG sectors, 8–14 HY/NR sectors, 15 other, 16 IG index, 17 HY index.
Single delta/vega factor per bucket.

Sector cross-bucket γ (Table 9), sector s∈1..7:

```
     s1    s2    s3    s4    s5    s6    s7
s1  1.00  0.75  0.10  0.20  0.25  0.20  0.15
s2  0.75  1.00  0.05  0.15  0.20  0.15  0.10
s3  0.10  0.05  1.00  0.05  0.15  0.20  0.05
s4  0.20  0.15  0.05  1.00  0.20  0.25  0.05
s5  0.25  0.20  0.15  0.20  1.00  0.25  0.05
s6  0.20  0.15  0.20  0.25  0.25  1.00  0.05
s7  0.15  0.10  0.05  0.05  0.05  0.05  1.00
```
Composite γ(b,c): buckets 1–14 use the sector value; if the two buckets differ in
credit quality (IG vs HY/NR) the value is **halved** (MAR50.67(2)). Bucket 15
correlates 0 with everything else; γ(16,17)=0.75; γ({16,17}, 1–14)=0.45;
γ(15, 16/17)=0.

Delta RW (Table 10) by bucket 1–17: 0.005, 0.010, 0.050, 0.030, 0.030, 0.020,
0.015, 0.020, 0.040, 0.120, 0.070, 0.085, 0.055, 0.050, 0.120, 0.015, 0.050.
Vega RW 1.00.

### Equity (MAR50.70–50.73) — delta + vega
Single factor per bucket. Cross-bucket γ: 0.15 between any two distinct buckets
1–10; 0.75 between 12 and 13; 0.45 between {12,13} and 1–10; 0.00 for any pair
involving bucket 11.

Delta RW (Table 12) by bucket 1–13: 0.55, 0.60, 0.45, 0.55, 0.30, 0.35, 0.40,
0.50, 0.70, 0.50, 0.70, 0.15, 0.25. Vega RW 0.78 for large-cap buckets
{1–8, 12}, else 1.00.

### Commodity (MAR50.74–50.77) — delta + vega
Single factor per bucket. Cross-bucket γ: 0.20 between any two distinct buckets
1–10; 0.00 for any pair involving bucket 11.

Delta RW (Table 14) by bucket 1–11: 0.30, 0.35, 0.60, 0.80, 0.40, 0.45, 0.20,
0.35, 0.25, 0.35, 0.50. Vega RW 1.00.

## Numerical convention

The `max(0, ·)` floor on the bucket and cross-bucket radicands is the prescribed
SBA numerical treatment: the supervisory correlation matrices (ρ, γ) are not
guaranteed positive semi-definite, so the radicand can be negative and is floored
at zero. This is the regulator's own SBA semantics, not an approximation.
