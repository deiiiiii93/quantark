# QUAD V2: implementation and qualification

QUAD V2 is an opt-in Gaussian autocallable engine for QuantArk. It prices Snowball, Phoenix and absolute-schedule KO-reset products, and computes point price, delta and gamma through one evaluator. The existing QUAD engines and default selection remain available.

The strongest measured result is for **price plus Greeks and scenario curves**, the selected primary workload. V2 passes the original 25-case Snowball comparison, an independent 404-point curve comparison, matched desk-risk repricing, analytical controls and seven independent family QMC comparisons. These are results for the cases below, not a universal error certificate or a claim of complete V1 API parity.

The [V1 versus V2 mathematical comparison](THEORY_COMPARISON.md) explains their common pricing model and differences in barrier integration, Greeks and continuous monitoring.

## Use V2

For existing QuantArk `product` and `env` objects:

```python
import numpy as np
from quantark.asset.equity.engine.quad import (
    SnowballQuadEngineV2,
    PhoenixQuadEngineV2,
    KOResetSnowballQuadEngineV2,
)
from quantark.asset.equity.param import QuadV2Params
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator

params = QuadV2Params()
engine = SnowballQuadEngineV2(params)  # choose the wrapper matching the product

pv = engine.price(product, env)
point = engine.calculate_point_greeks(product, env)  # price, dV/dS, d²V/dS²
desk = engine.calculate_greeks(product, env)         # existing finite spot bumps

spots = np.linspace(0.8 * env.spot, 1.2 * env.spot, 101)
context = engine.prepare(product, env, spots)
curve = context.evaluate(spots)                     # arrays: price, delta, gamma
rows = engine.calculate_spot_greeks_curve(product, env, spots)

risks = GreeksCalculator(params).calculate(
    product, env, engine,
    greeks=["delta", "gamma", "vega", "rho", "dividend_rho", "theta"],
)
components = engine.price_components(product, env)
```

`calculate_greeks` preserves the existing desk spot-bump convention, including the default 1% delta/gamma bumps. `calculate_point_greeks`, `PreparedQuad.evaluate` and `calculate_spot_greeks_curve` use analytical derivatives of the Gaussian integral. Curve rows label this as `calculation_mode="quad_v2_point"`. These two Greek conventions need not agree at finite bump size, especially near a barrier.

The risk facade retains its existing units: vega is the PV change for the configured upward volatility bump; rho and dividend rho are scaled to a one-percentage-point rate change; theta is the configured time-bump PV change. The example uses the same bump settings for independent reference repricing.

The prepared context freezes the contract and market snapshot. Changing the original product or market cannot change its answers. Queries outside its declared spot range raise `ValidationError`. Normal engine repricing recompiles the fingerprint and rebuilds as needed; spot bumps inside the range reuse the continuation. Volatility, rate, carry and aged-contract scenarios require their own affected sweeps:

```python
scenario_results = engine.calculate_scenarios([
    (product, base_env),
    (product, shocked_env),
    (aged_product, future_env, future_lifecycle),
])
```

Time scenarios must supply the correctly aged contract and lifecycle state. This API does not infer how a path reached the future state. The ordinary Greeks facade supplies its existing contract-aging convention for theta.

## Contract and market coverage

| Feature | Implementation and qualification |
| --- | --- |
| Snowball, normal/reverse; alive/already KI | Implemented; analytical controls, original 25-case reference matrix and independent curve/risk checks |
| Discrete KO/KI, irregular and step schedules | Resolved contractual events, with no monitoring-frequency substitution |
| Phoenix coupons, memory, deferred coupons | Implemented; deterministic cashflow cases and independent QMC for nonmemory, memory, deferred and reverse variants |
| KO reset | Absolute post-KI schedules, including pre/post maturity ownership; deterministic and discrete/continuous QMC checks |
| Payoff caps, protection and airbag strikes | All declared piecewise-affine payoff breaks compiled separately; independent multi-kink terminal integration |
| Zero forward variance | Exact deterministic transport and state maps; includes mixed zero/positive variance intervals |
| Discrete deterministic term structures | Exact interval discount, carry and variance moments; negative forward variance rejected |
| Continuous KI, constant coefficients | Full survival/hit kernel and derivatives, with analytical survival controls and independent bridge-QMC comparisons |
| Continuous KI, piecewise constant coefficients | Split at known coefficient knots; exact model law between knots, with spatial quadrature error still present |
| General continuous term curves | Explicit `piecewise_constant` approximation only; no broad tight-tolerance qualification yet |
| Lifecycle | Explicit KI and Phoenix memory state, pre/post valuation phase, terminated positions and pending receivable PV |
| Cashflow components | Independently propagated KO, coupon and terminal channels, plus realized pending receivables |
| Event statistics | Snowball, KO-reset and Phoenix KO/terminal cashflows and probabilities; Phoenix coupon-trigger probabilities and per-observation coupon PVs, including memory and deferred payments; continuous KI-ever probability is unavailable |
| Execution framework | Serial PRICE/EVENT_STATS adapter, explicit lifecycle and phase; family-specific capability checks |

Known limits are explicit. Relative-to-hit reset schedules, stored backward exposure/CVA grids, stochastic volatility, local volatility and multiple underlyings are unsupported. Unknown product subclasses are rejected rather than assigned the Snowball adapter through inheritance.

### Phoenix coupon event statistics

```python
stats = PhoenixQuadEngineV2().calculate_event_stats(product, pricing_env)
stats.ko_times                              # Active observation times
stats.coupon_probability                    # P(alive before observation AND coupon trigger)
stats.expected_discounted_coupon_cashflow   # Coupon PV attributed to each observation
stats.payment_times                         # Actual payment-event buckets
stats.expected_discounted_cashflows          # Additive payment ledger, including pending cash
assert abs(stats.pv - stats.expected_discounted_cashflows.sum()) < 1e-8
```

Coupon probabilities include a simultaneous KO and coupons earned after KI when KO is disabled. Memory catch-up belongs to the observation that releases it. The existing V2 contractual rule also releases missed memory on KO even if the current coupon does not trigger; that observation can therefore have positive coupon PV and zero coupon-trigger probability. KO redemption and terminal payoff diagnostics exclude these coupon amounts.

For `EXPIRY`, earned coupons are paid at the actual KO settlement or terminal settlement. Per-observation coupon PVs integrate that random payment date. The separate additive ledger groups payments at their actual settlement events: KO redemption rows, coupon payment rows, pending receivables, then the alive/KI terminal rows. Terminal rows include deferred coupons paid at maturity. The two coupon representations are alternative decompositions and must not be added together.

`price_with_events()` and framework `EVENT_STATS` requests support Phoenix. Deferred stats set `coupon_payment_is_path_dependent=True`; their `EventDistribution` includes coupon-trigger probabilities but cannot supply a single fixed payment time for each trigger. `payment_times_for(EventType.COUPON)` raises for this case; use the payment ledger for expected coupon cashflows. Lifecycle memory, pending receivables and `event_phase="before"/"after"` are preserved.

Coupon-specific controls are in `test/test_quad_v2_coupon_event_stats.py`: analytical lognormal probabilities and discounted payments, deterministic memory/KO cases, non-flat curves and settlement lags, direct/FFT agreement, and four independent QMC cases using eight scrambles of 65,536 paths each. The QMC tests compare three spatial refinements `(order, cells_per_sd) = (6, 1.5), (8, 2), (10, 3)` and use Student-t uncertainty across scrambles; these cases do not certify the entire parameter envelope.

```bash
.venv/bin/python -m pytest -n0 test/test_quad_v2_coupon_event_stats.py -q
```

Known deterministic volatility classes and their supported wrappers are accepted. A custom deterministic variance object must declare `quad_v2_deterministic_variance=True`; this declaration is a model assertion by its author. A sampled smile is not a local-volatility transition model.

For continuous KI, flat rates/carry and supported piecewise constant coefficient curves use `continuous_term_structure="exact"`. General curves require an explicit setting and a separate refinement study:

```python
approximation = QuadV2Params(
    continuous_term_structure="piecewise_constant",
    continuous_steps_per_year=96,
)
```

Diagnostics distinguish `piecewise_constant_exact` from `piecewise_constant_approximation`. A parallel shift of flat volatility remains exact; a wrapper that shifts an interpolated term volatility need not preserve piecewise constant forward variance.

## Lifecycle and event ownership

All public valuation APIs accept `event_phase="before"` or `"after"`, with `"before"` the default. This determines whether observations exactly at valuation are still due. Supply the incoming state before the event, or the resolved state after it:

```python
value_after = engine.price(
    product, env, lifecycle_state=resolved_state, event_phase="after"
)
```

For the execution framework, pass `operation_options=(("event_phase", "after"),)` on `PricingRequest`, together with the lifecycle state. Observed-index sets are not used to guess event phase: indices can refer to the original or an aged schedule. The caller must declare phase explicitly when valuation follows today's observation. Partially processed coincident events need a fully resolved state/schedule before valuation.

Discrete KI never follows merely from being below the barrier between observations. At a true deterministic jump/kink or a continuous barrier boundary, point Greeks may be undefined and return NaN. PV still follows the contract's equality ownership. Desk finite bumps remain available and retain their different meaning.

An aged Phoenix memory state needs fixed coupon year fractions or retained historical coupon dates sufficient to recover the unpaid amounts. Ambiguous missing coupon history raises. Already determined payments are discounted through the authoritative lifecycle ledger; paid cash is excluded from remaining PV. Initial prepayment, margin and funding legs are separate cashflows outside this option engine.

## Numerical design and controls

The compiler produces piecewise affine state/event maps. Smooth continuation branches live on uniform Gauss cells; event and terminal boundaries are integrated explicitly. Direct and linear-FFT backends apply the same block kernels, including partial-cell corrections. Cash/asset Gaussian moments are analytical, and local affine subtraction stabilizes tiny-variance derivatives. Gamma uses `(V_xx - V_x) / S²` for log spot `x`.

The domain follows cumulative drift as well as variance and the requested query range. A short first event gets its own final integration resolution; it does not force the whole remaining trade onto that first interval's mesh. Genuine short interior intervals can still exhaust the declared node budget and fail with a required-resolution diagnostic.

Defaults are order 8, 2 cells per standard deviation of the narrowest positive interior transition, 11 domain standard deviations, a 10-standard-deviation integration tail, and readout order 32. A research refinement used here is `QuadV2Params(order=10, cells_per_sd=3.0, domain_sd=12.0)`. `backend="direct"` and `"fft"` permit explicit equivalence checks; `"auto"` uses a deterministic size threshold.

Budgets limit nodes, events, states, predicted work arrays and kernel-cache bytes. General continuous subdivisions and state/event tables are checked before large allocation. These are allocation controls, not an exact process-RSS limit. Floating event times remain distinct by default; `event_time_tolerance` is an explicit clock-equivalence declaration, not an automatic short-step workaround.

Tail cutoffs, edge continuation, interpolation and finite integration orders remain numerical approximations. Mesh agreement and the domain-standard-deviation setting do not prove a global error bound. Refine spatial/basis resolution, readout and domain independently for a new parameter envelope; continuous approximations need time refinement too. No claim of universal 0.01 bp accuracy is made.

## Accuracy evidence

The independent reference is the frozen [original Gaussian quadrature implementation](../gaussian_quad_comparison/quad_reference_snapshot.py), SHA256 `70938c07768f0512765bed99ca9d2c50dd561bb649fe595ebda90bbe3d090ac7`. Its original domain and short-interval limitations remain documented in the [comparison report](../gaussian_quad_comparison/REPORT.md). It is used only within its separately qualified envelope.

| Check | Result |
| --- | --- |
| 25 saved Snowball reference cases | All pass; maximum observed PV gap about 2.85e-12 and delta gap 3.23e-13 |
| Four independent 101-spot curves | All 404 price/delta/gamma points pass; maximum gamma gap about 3.52e-11 |
| Full desk risk, monthly and aged five-day | Matched independent bump repricing; maximum output gap about 1.98e-11 |
| Vanilla and continuous-survival controls | Analytical PV/derivatives; tiny positive variance, zero-variance semigroup and complete barrier-kernel derivatives |
| Direct versus FFT; scalar versus curve | Kernel algebra and common-context values agree within tested roundoff tolerances |
| Low-volatility/high-carry domain failure | V2 -30.01254875, matching discounted stock-minus-strike control |
| Short first KO `[0.0001, 0.5, 1]`, S=103.001 | V2 3.2277326782; resolved without a huge global mesh |
| Regression suite | 204 tests pass, including V2 and existing QUAD, Greeks, execution and settlement checks |

Curve gamma is independently estimated from reference deltas with three relative spot bumps: `1e-4`, `5e-5`, `2.5e-5`, followed by Richardson extrapolation. Both medium/fine reference settings and adjacent extrapolations are checked. Their observed variation is recorded separately from V2 error; it is not a rigorous reference uncertainty bound.

The benchmark contracts use initial notional 100 and reference spot 100. Gates are absolute PV error ≤1e-4 (0.01 bp of notional), delta error ≤`1e-5 + 1e-4*abs(reference_delta)`, and gamma error ≤`1e-6 + 1e-3*abs(reference_gamma)`. The latter two equal the design's normalized-risk gates for these contracts. Vega/rate/carry/theta outputs use `1e-4 + 1e-4*abs(reference_output)` in their stated desk units. These constants must be rescaled for different notionals/reference spots.

Workload grid selection uses refined V2 values after the independent curve/reference checks above. Full-risk independent repricing is presently recorded for monthly and aged-five-day contracts; daily/three-year full risk has spatial-refinement evidence. No reference failure is turned into a passing speed comparison.

Seven family cases use 8 independent randomized Sobol scrambles ×65,536 paths, or 524,288 paths per case. GBM endpoint transitions follow the contractual calendar; continuous cases additionally sample conditional barrier hits. Confidence intervals are Student-t intervals across scramble means. Path transitions, coupon memory and termination are implemented separately from the V2 compiler; contractual payoff amounts are obtained from QuantArk product APIs.

| Family case | V2 fine PV | QMC 95% interval |
| --- | ---: | ---: |
| Phoenix, no memory | 94.31138829 | [94.30986996, 94.31704768] |
| Phoenix, memory | 94.83152564 | [94.82951499, 94.83701818] |
| Phoenix, memory and deferred coupons | 94.81243831 | [94.81043586, 94.81793318] |
| Reverse Phoenix, memory | 92.97418031 | [92.96554206, 92.97742633] |
| Continuous Phoenix, memory and deferred coupons | 94.20430262 | [94.20277476, 94.20742811] |
| Discrete KO reset | 94.00975815 | [94.00492302, 94.02616482] |
| Continuous KO reset | 93.48331373 | [93.46514818, 93.49269702] |

All seven fine values fall inside their intervals; standard/fine PV differences are below 7e-13. The intervals are wider than the deterministic target and do not, by themselves, certify 0.01 bp family pricing or Greek accuracy.

## Performance

The final timing table is generated from [all_results.json](all_results.json). Each workload selects its own coarsest passing V1 grid from 1001, 2001, 4001, 8001 and 16001 points, using transition readout. V2 uses default settings. A V1 workload that fails at every tested level is labelled unqualified; its latency is not an equal-accuracy speedup.

| Scenario | Workload | V1 grid | V1 ms | V2 ms | Qualified speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Monthly KI | PV + desk delta/gamma | 8001 | 314.42 | 10.37 | 30.3x |
| Monthly KI | 101-point PV/delta/gamma | 16001 | 240.22 | 23.43 | 10.3x |
| Monthly KI | Six-Greek batch | 8001 | 763.94 | 56.67 | 13.5x |
| Daily KI | PV + desk delta/gamma | 8001 | 5564.59 | 128.38 | 43.3x |
| Daily KI | 101-point PV/delta/gamma | 16001 | 4311.67 | 135.84 | 31.7x |
| Daily KI | Six-Greek batch | 8001 | 12304.34 | 649.61 | 18.9x |
| Aged, five days remaining | PV + desk delta/gamma | 1001 | 10.29 | 4.98 | 2.1x |
| Aged, five days remaining | 101-point PV/delta/gamma | 16001 | 101.66 | 17.12 | V1 unqualified |
| Aged, five days remaining | Six-Greek batch | 2001 | 79.67 | 21.19 | 3.8x |
| Three years | PV + desk delta/gamma | 8001 | 1343.53 | 40.62 | 33.1x |
| Three years | 101-point PV/delta/gamma | 16001 | 1098.40 | 52.42 | V1 unqualified |
| Three years | Six-Greek batch | 4001 | 865.33 | 198.85 | 4.4x |

The aged and three-year V1 curves fail the point-gamma gate at some spots even at 16001 points. Their V2 curves pass; no comparative speedup is claimed for those rows. The six-Greek batch returns risk outputs, not an additional PV; its V1 grid is selected against those outputs.

| Scenario | V2 cold point PV/delta/gamma ms | V2 cold prepare ms | V2 warm point ms | V2 warm 101-point curve ms |
| --- | ---: | ---: | ---: | ---: |
| Monthly KI | 8.943 | 8.357 | 0.364 | 13.944 |
| Daily KI | 124.866 | 121.896 | 0.293 | 13.406 |
| Aged, five days remaining | 4.005 | 3.267 | 0.559 | 13.907 |
| Three years | 37.603 | 34.306 | 0.397 | 25.663 |

| Scenario | V2 curve nodes | V1 traced peak MiB | V2 traced peak MiB | V2 kernel cache MiB at preparation |
| --- | ---: | ---: | ---: | ---: |
| Monthly KI | 1368 | 11.37 | 24.53 | 0.44 |
| Daily KI | 6136 | 11.30 | 27.50 | 26.06 |
| Aged, five days remaining | 1224 | 6.48 | 24.51 | 0.20 |
| Three years | 2264 | 13.82 | 24.59 | 4.89 |

V2 trades higher temporary allocation for faster batched evaluation in these cases. The 24–28 MiB traced peaks are below the default work budgets; memory use still depends on state count, requested channels and event calendar.

Cold timings include a new engine/context, contract compilation and the requested output; imports and process-wide small Gauss-rule caches are already warm. Each measurement has a warmup and three recorded repetitions; the table uses medians, with all samples/ranges in JSON. Warm point/curve timings retain a prepared market/contract context. Risk batches include the six facade Greeks and all required repricing; they are not warm spot queries. Peak traced allocations are measured separately, outside latency samples; they exclude some native-library memory and are not process RSS.

The machine runs macOS on ARM64, Python 3.11.8, with BLAS/OMP/Accelerate thread counts requested as one. Hardware load can change timings. These speed comparisons are Snowball workloads; they do not establish equivalent speedups for Phoenix, KO reset or continuous monitoring.

## Reproduce

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python example/quad_v2_validation/validate.py --phase all --repeats 3

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python example/quad_v2_validation/families.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest -n0 \
  test/test_quad_v2_core.py test/test_quad_v2_engine.py \
  test/test_quad_v2_continuous.py test/test_quad_v2_terms_lifecycle.py \
  test/test_snowball_quad_engine.py test/test_phoenix_quad.py \
  test/test_ko_reset_snowball_quad_engine.py test/test_quad_term_structure_engines.py \
  test/test_quad_zero_vol_step.py test/test_spot_greeks_curve.py \
  test/test_greeks_mode_and_engine_type.py test/execution/test_registry.py \
  test/execution/test_capability_matrix.py test/test_engine_settlement_capabilities.py \
  test/test_structured_numerical_settlement.py \
  test/test_settlement_event_stats_reconciliation.py \
  test/test_cashleg/test_phoenix_quad_event_stats.py -q
```

`--phase matrix`, `risk`, `curves` and `benchmark` run individual parts. [all_results.json](all_results.json) records inputs, settings, errors, reference refinement, diagnostics, timings and source hashes; [family_results.json](family_results.json) records family PVs, scramble means and provenance. The frozen original comparison and feasibility artifacts are preserved.

Implementation entry points are [engine.py](../../quantark/asset/equity/engine/quad/v2/engine.py), [QuadV2Params](../../quantark/asset/equity/param/quad_v2_params.py), the [design](../../docs/superpowers/plans/2026-09-11-quad-v2-design.md) and the [implementation plan](../../docs/superpowers/plans/2026-09-11-quad-v2-implementation.md).
