# W13 — EuropeanMCEngine below its own lower bound: the check, not the price (2026-10-10)

Diagnosis only; no library change. Branch `w13-mc-lower-bound-diagnosis` from main 0d1e4133. Scripts and raw outputs
in this folder; all patches are runtime patches inside the scripts. (Report text by the W13 agent, committed by the
controller because the agent could not write files here.)

## Contract and reproduction (`01_reproduce.py`)

`test/test_expiry_carried_knock_in.py`, `_BEFORE_REFUSED` (~line 202): down-in put K 100, barrier 80, rebate 5, carried
as knocked in, valued 2026-01-04, T = 1/365, flat vol 20%, r 3%, q 1%, no settlement lag;
`BarrierOptionMCEngine(MCParams(seed=7, num_paths=256, use_antithetic=True))`, 100 steps, priced through
`EuropeanMCEngine`. Both forms refused: S = 90 MC 9.99345301 vs bound 9.99424688; S = 75 MC 24.99317437 vs bound
24.99383593; S = 110 OTM prices 0. BS − bound = 0.000e+00 in double precision at both refused spots. The numbers in the
brief (9.99399 / 9.99430 / 9.99415) were not reproduced with seeds 7 / 42 and 1 / 100 / 252 steps (`01b`); same noise
range, bound 9.99424688 every time.

## Root cause

`quantark/asset/equity/engine/mc/euro_mc_engine.py:216`: `if price < lower_bound - 1e-6: raise` — the deterministic
Black-Scholes check (`black_scholes_engine.py:132`) applied to a noisy estimate. One day out and deep ITM the time value
is ~0, the true price sits on the bound, and an unbiased estimator lands below it about half the time; 1e-6 is 100 to
60,000x smaller than the MC error. Refusal rate 38–62% at every path count and method: path-independent because the
true value is ON the bound (what looked like "the gap does not shrink"; the error itself shrinks: QMC sd 1.7e-3 at 256
paths, 4.4e-6 at 65k). The bound (`_european_lower_bound`, lines 224–241) and the estimator are both correct.

## Decision matrix (`02_bias_grid*.out`, `03*.out`; bias = mean(MC − BS) over R seeds, t = bias / s.e.)

| Knob | Measured | Reads |
|---|---|---|
| Paths 256 → 64k (W13 put) | plain abs(t) ≤ 1.37; antithetic ≤ 1.70; QMC ≤ 0.76; RQMC ≤ 1.64; refused 25–62% in every row | no bias; refusals are a coin flip |
| Method (plain / antithetic / QMC / QMC+anti / RQMC) | empirical sd at 4k: 1.6e-2 / 1.6e-4 / 9.5e-5 / = QMC / 5.2e-5 | none biased; antithetic silently dropped under QMC (`euro_mc_engine.py:299`) |
| T = 1d / 5d / 30d / 1y | abs(t) ≤ 1.74; BS − bound 0 / 1.6e-6 / 0.076 / 4.1; refused 53% / 53% / 0–19% / 0% | refusals only where time value << MC error |
| Settlement lag 0 vs 2d | bias changes ~1e-7 (same draws); bound − parity −1.8e-14 | lag discounting consistent: bound = payment_df·(K − F_T) |
| Term curve vs flat | engine fwd / curve fwd − 1 ≤ 4.4e-16; last-node DF = determination_df = exp(−r(T)T) to 13 digits | forwards telescope exactly; r(T) pillar matches |
| (a) simulated forward | exact log-Euler; 256 seeds abs(t) ≤ 2.06, signs flip | E[S_T] = F |
| (b) discount factors | payment_df = determination_df x delay_df; bound and payoff agree | consistent |
| (c) multiplier / std error | multiplier 1; reported s.e. is the iid formula (`euro_mc_engine.py:383-385`), overstating the real error 90x (antithetic 256), 155x (QMC 4k), 834x (QMC 64k) | secondary finding |
| (d) closed form vs bound | never below by more than 1.8e-15, with or without lag | bound tight and correct |
| Deep ITM call, ATM controls | call abs(t) ≤ 0.42, refused 44–62%; ATM refused 0 | same mechanism; fires only where time value ≈ 0 |

Size of any bias: abs(bias) < ~3e-6 (~0.003 bp) in the tightest cells (QMC 64k −7.2e-7 ± 1.1e-6; RQMC 262k
+5.6e-7 ± 7.9e-7). All 60 knob-grid cells within 1.74 s.e. of the closed form. No European MC price needs to move.

## Recommended fix (`04_fix_candidates.py`)

Statistical check: refuse only if `price < bound − max(1e-6, z·std_error)`, z = 4. Refusals: 0 of 1,848 replications in
7 cells at z = 3, 4, 5 (current rule: 38–62%). Moved values: none — the estimator is unchanged, every price passing
today is bit-identical; only today's refused cells now price (S = 90: 9.99345301, BS − 7.9e-4; S = 75: 24.99317437).
With the rule patched in, the day-before continuity assertion passes in all 4 cells (diff 6.5e-3 / 6.8e-3 vs one-day sd
0.94 / 0.79), so `("mc", "down_in_put")` can leave `_BEFORE_REFUSED` with the fix. Caveat: the overstated reported s.e.
makes the rule loose under antithetic / QMC (QMC 4096: accepts down to bound − 0.059 vs real error 9.5e-5).

Alternatives: clamp to the bound (silent upward bias — a fallback, rejected); parity estimator (simulate the OTM side
plus the forward leg: unbiased and bound-respecting by construction, but moves every ITM price — mean 5.4e-4 at 256
antithetic, up to 0.98 for the 1y put at 256 plain — and can raise variance, sd 0.539 vs 0.305: opt-in only); drop the
check (no moves, loses gross-error detection); relative tolerance (arbitrary; still refuses at 256 plain).

## Open questions (recommendations)

1. Fix the reported std error: antithetic over pair means (matched the empirical sd within ~12%); plain QMC keep the
   conservative iid figure and point users to RQMC. Changes `get_last_std_error()`, not prices — separate workstream.
2. z = 4 (false refusal ~3e-5 when the error is reported honestly).
3. `EuropeanQuadEngine` has the same pattern with a 1e-4 tolerance (`european_quad_engine.py:186`); deterministic, but
   its near-expiry quadrature error was not measured — a separate one-day deep-ITM check.
4. RQMC seed overlap: batch b is scrambled with `base_seed + batch_id` (`montecarlo/qmc_sobol.py:220`), so seeds s and
   s+1 share 31 of 32 scrambles; a first replication run with consecutive seeds gave a spurious t = +6.7
   (`02_bias_grid_A_overlapping_rqmc_seeds.out`). Biases no price; document it or hash (seed, batch_id).
5. Other copies of the hard check: only the Black-Scholes, QUAD and European MC engines; the vol-model MC Europeans have
   none.
