# Autocallable Risk Profile Report (Snowball)

**Generated**: 2026-01-19T17:36:48

## Executive Dashboard
- Status: Knock-Out Likely
- PV: 0.918595; Delta: 0.156070; Gamma: -0.038382; Vega: -46.094068; Theta: 0.705699
- Next KO: level=103.000000, T=0.083y, dist=3.00%, sigma=0.445
- Next KI: level=75.000000, T=1.000y, dist=-25.00%, sigma=-1.370

## Snapshot
- Engine used for PV surfaces: `quad`
- Surface mode: finite-difference
- PV (base): 0.918595
- Spot S0: 100.000000
- Dividend q0: 0.028000
- Vol σ0: 0.210000

## Barrier Watch
- Next KO: level=103.000000, T=0.083y, dist=3.00%, sigma=0.445
- Next KI: level=75.000000, T=1.000y, dist=-25.00%, sigma=-1.370

## Barrier Risk (Zoom)
- Barrier zoom grid: ±2% around KO/KI level, spot nodes=21
- KO barrier: `plots/barrier_ko_gamma.png`, `plots/barrier_ko_vega.png`
- KI barrier: `plots/barrier_ki_gamma.png`, `plots/barrier_ki_vega.png`

## Lifecycle Context
Pre-KI state. Focus on KO vs KI probabilities.
- P(KO before maturity): 0.696469
- P(KI before maturity): 0.156680

## Grids
- Spot: 0.60–1.20 × S0 (11 nodes)
    - Dividend: [0.000000, 0.078000] (11 nodes, parallel term-structure shifts; lower bound clipped at 0 if needed)
    - Vol: σ0 × (1±5.00%) (11 nodes, term-structure scaling)

## Dividend / Basis Risk
Basis mapping (China index futures): define carry/basis `b = r - q`, so `RhoB = ∂V/∂b = -∂V/∂q = -RhoQ`.

### DividendRho surfaces
- `plots/rhoq_spot_div.png`
- `plots/rhoq_spot_vol.png`

### Delta and Spot–Dividend cross
- `plots/delta_spot_div.png`
- `plots/cross_s_q.png`

## Advanced Volatility Risk
- Vanna/Volga surfaces: `plots/vanna_spot_vol.png`, `plots/volga_spot_vol.png`
- Skew/smile shock model: `vol = base + skew * ln(K/S) + smile * ln(K/S)^2`
- skew: -0.100000
- smile: 0.050000
- PV impact: 0.000000

## Higher-Order Time Greeks
- `plots/charm_spot_div.png`, `plots/color_spot_div.png`

## Scenario Ladder (Spot × Vol)
- Spot shocks: -20%, -10%, -5%, 0%, +5%, +10%
- Vol shocks: -5%, 0%, +5%
- Worst cell (PnL): -20% spot × +5% vol = -18.329304

|        -5% |         0% |        +5% |
|-----------:|-----------:|-----------:|
| -17.399    | -17.91     | -18.3293   |
|  -4.61139  |  -5.55757  |  -6.45313  |
|  -0.781637 |  -2.02397  |  -2.87592  |
|   0.698067 |   0        |  -0.670642 |
|   1.06878  |   0.660686 |   0.253839 |
|   0.441881 |   0.354342 |   0.258205 |

## Stress Scenarios
| scenario   | spot_stress   | vol_stress   |   div_yield_stress |       pnl |
|:-----------|:--------------|:-------------|-------------------:|----------:|
| Black Swan | -30%          | +50%         |               0.01 | -31.3611  |
| Slow Bear  | -10%          | +5%          |               0    |  -6.45313 |

## Bucketed Greeks (Term Structure)
- Bucketed Vega is per +1 vol point (0.01).
- Bucketed Dividend Rho is per +1% dividend yield (0.01); Basis Rho = -Dividend Rho.

| bucket            |   bucket_vega |   bucket_rho_q |   bucket_rho_b |
|:------------------|--------------:|---------------:|---------------:|
| 1M (0-0.0833y)    |        0      |       0        |      -0        |
| 3M (0.0833-0.25y) |        0      |       0        |      -0        |
| 6M (0.25-0.5y)    |        0      |       0        |      -0        |
| 1Y (0.5-1y)       |      -63.9154 |      -0.311102 |       0.311102 |

## Risk-neutral event stats & cashflow attribution
- PV (engine event stats): 0.918595
- KI probability: 0.156680
- PV reconciliation error (PV - sum(expected discounted cashflows)): 0

|   ko_time |      p_ko |   p_survive |   ed_ko_cf |
|----------:|----------:|------------:|-----------:|
| 0.0833333 | 0.268297  |    0.731703 |   0.334812 |
| 0.166667  | 0.134247  |    0.597456 |   0.334502 |
| 0.25      | 0.0782812 |    0.519175 |   0.29209  |
| 0.333333  | 0.0523549 |    0.46682  |   0.260035 |
| 0.416667  | 0.0380751 |    0.428745 |   0.235995 |
| 0.5       | 0.0292783 |    0.399466 |   0.217402 |
| 0.583333  | 0.0234287 |    0.376038 |   0.202624 |
| 0.666667  | 0.0193144 |    0.356723 |   0.190586 |
| 0.75      | 0.0162907 |    0.340432 |   0.180542 |
| 0.833333  | 0.013988  |    0.326444 |   0.17196  |
| 0.916667  | 0.0121822 |    0.314262 |   0.164462 |
| 1         | 0.0107314 |    0.303531 |   0.157784 |

## Conditional Cashflow Projection
|   ko_time |      p_ko |   ed_ko_cf |   conditional_ed_ko_cf |
|----------:|----------:|-----------:|-----------------------:|
| 0.0833333 | 0.268297  |   0.334812 |                1.24792 |
| 0.166667  | 0.134247  |   0.334502 |                2.49168 |
| 0.25      | 0.0782812 |   0.29209  |                3.7313  |
| 0.333333  | 0.0523549 |   0.260035 |                4.96678 |
| 0.416667  | 0.0380751 |   0.235995 |                6.19813 |
| 0.5       | 0.0292783 |   0.217402 |                7.42537 |
| 0.583333  | 0.0234287 |   0.202624 |                8.64851 |
| 0.666667  | 0.0193144 |   0.190586 |                9.86755 |
| 0.75      | 0.0162907 |   0.180542 |               11.0825  |
| 0.833333  | 0.013988  |   0.17196  |               12.2934  |
| 0.916667  | 0.0121822 |   0.164462 |               13.5002  |
| 1         | 0.0107314 |   0.157784 |               14.703   |

## Historical / Real-world analysis
### Historical Shock PnL (1-step)

- count: 7
- mean: -0.312543
- std: 0.531988
- p01/p05/p50/p95/p99: -1.172913, -1.103097, -0.163853, 0.200413, 0.269403

