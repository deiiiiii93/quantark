# Riskmeasures refactor + higher-order greeks — design

**Date:** 2026-09-01
**Branch:** `worktree-riskmeasures-greeks` (off `main` @ d348d6a)
**Status:** approved design, pending implementation plan

## 1. Motivation

`quantark/asset/equity/riskmeasures/greeks_calculator.py` is 2,497 lines / one
class / ~55 methods and has already produced one silent defect: `charm` and
`color` pass request validation (they are in `EquityGreek` and the
`_normalize_greeks` allowed set) but nothing computes them — a request comes
back with those keys missing, no error. The validation set and the dispatch
if-chain are two hand-maintained lists that agree only by discipline.

Separately, the desk needs commonly-used higher-order greeks (charm, color,
speed, zomma, vega decay, q-convexity) and a full theta decomposition that
supports both the natural-calendar clock (1D) and the trading clock (1TD).

## 2. Current state (inventory)

| Section of the file | ~lines | Content |
|---|---|---|
| Core dispatch + normalization | 230 | `calculate()`, aliases, engine-mode resolution |
| Bucketed greeks | 1,240 | rate keyrate, futures delta, carry rhoq, vol tenor vega, vol-model points, futures bucket helpers |
| Analytical BS greeks | 130 | price/delta/gamma/vega/theta(+components)/rho/dividend_rho |
| Scalar numerical greeks | 580 | delta, gamma, vega, volga, vanna, theta (3 clock modes), rho, dividend_rho, delta_q |
| Theta components + env builders | 215 | estimate + exact(private) decomposition, vol/div bumped envs |
| Linear/expiry edge cases + compare | 95 | |

Existing scaling conventions (all preserved verbatim):

- `one_sided` sensitivity = raw price difference per bump (vega per 1 vol pt,
  theta per day); `rho`/`dividend_rho` rescaled to per-1% via `0.01/bump`.
- `vanna` = (Δ⁺ − Δ⁻)/(2·Δσ) raw per unit vol; `volga` central second-order raw.
- Theta decomposition in the numerical path is an **estimate**
  (`estimate_theta_components`): `r_theta ≈ −r/T · rho/0.01 / 365`, `q_theta`
  likewise, `convexity_theta` = residual. An exact zeroed-r/q repricing path
  exists (`_calculate_numerical_theta_components`) but is private and
  unreachable from the public API.

Downstream importers (~40 files): SIMM, backtest, dynamicscenario, stresstest,
execution, reports, and the FX/credit/bond riskmeasures packages which
re-export `GreeksCalculator`.

## 3. Compatibility contract (hard invariants)

1. `GreeksCalculator` and every current public method keep their exact
   signatures and module path (`quantark.asset.equity.riskmeasures
   .greeks_calculator`). Downstream sees no import or call-site change.
2. Every currently-working request returns **numerically identical** results:
   - Default greek set stays exactly `{price, delta, gamma, vega, theta, rho,
     dividend_rho, convexity_theta, r_theta, q_theta}`.
   - Auto-routing for **existing** names is unchanged: `{price, delta, gamma,
     vega, theta, rho}` route analytical for vanillas; `vanna`/`volga`
     /`delta_q`/`dividend_rho`/theta components keep routing numerical under
     `method="auto"` (flipping them would shift values by FD error). New
     names, which never routed anywhere before, may join the analytical
     auto-set (§9).
   - Bare `theta`/`r_theta`/`q_theta`/`convexity_theta` keep today's semantics
     bit-for-bit, including the estimate path's `/365` scaling even when the
     resolved theta clock is business days (legacy quirk, preserved).
3. `calculate_analytical_greeks()` returns its **current key set** by default;
   extended greeks appear only when requested via a new optional `greeks=`
   parameter (unconditional new keys would silently change dict-iterating
   callers).
4. New greeks are opt-in by request only — no cost or key-set change for any
   existing caller.

## 4. Package structure

```
quantark/asset/equity/riskmeasures/
├── greeks_calculator.py       # slim facade; public API unchanged (~300 ln)
├── greeks/
│   ├── __init__.py
│   ├── registry.py            # GreekDef table + normalizer (aliases, clock
│   │                          #   qualifiers, default set) — single source
│   ├── analytical.py          # BS closed forms incl. new greeks
│   ├── numerical.py           # bump-based scalar greeks + NumericalGreekContext
│   ├── bump_envs.py           # vol/div bumped envs, theta date advance
│   └── theta_decomposition.py # estimate + exact components, gamma_theta
├── bucketed_coordinates/
│   ├── __init__.py
│   ├── rate_keyrate.py
│   ├── futures_delta.py
│   ├── carry_rhoq.py
│   ├── vol_tenor_vega.py
│   └── vol_model.py
├── bucketed_greeks.py         # existing dataclasses, unchanged
├── greek_conventions_report.py  # unchanged
├── surface_shock_*.py           # unchanged
└── vol_model_risk.py            # unchanged
```

The bucketed coordinate families move as **pure code motion** (module-level
functions taking the calculator/context as first argument); the facade's
`calculate_bucketed_greeks` keeps the dispatch. The two standalone public
bucket methods (`calculate_futures_delta_buckets`,
`calculate_futures_rhoq_buckets`) stay on the facade, delegating.

## 5. Registry

```python
@dataclass(frozen=True)
class GreekDef:
    name: str                    # canonical, e.g. "zomma"
    aliases: tuple[str, ...]     # e.g. ("veta",) on vega_theta
    numerical: Callable          # (ctx: NumericalGreekContext) -> float  (REQUIRED)
    analytical_auto: bool        # member of the analytical auto-routing set (§3.2, §9)
    default: bool                # member of the greeks=None default set
    linear_value: float          # value for delta-one products
    supports_clock: bool         # accepts _1d/_1td qualifiers (§8)
    requires: tuple[str, ...]    # derived-greek dependencies, e.g. ("gamma",)
```

- `_normalize_greeks` validation, alias resolution, the default set, and
  dispatch all derive from this one table. A greek can no longer be
  requestable-but-uncomputed; a registry-completeness test enforces it
  (§12.2).
- Clock qualifiers: the normalizer parses `<name>_1d` / `<name>_1td` into
  (canonical name, clock override) for entries with `supports_clock=True`, and
  rejects the suffix otherwise (e.g. `vanna_1td` → `ValidationError`). Result
  dict keys echo the requested qualified name, so `["theta_1d", "theta_1td"]`
  returns both in one call.
- Existing aliases preserved: `deltaq`/`deltadq`/`d_delta_d_q`/`d_delta_dq` →
  `delta_q`; `rhoq`/`div_rho`/`dividendrho` → `dividend_rho`. New: `veta` →
  `vega_theta`.

### NumericalGreekContext

Carries (product, pricing_env, engine, bump_config, greeks_mode, clock
override) plus a **scenario memo** keyed by structural strings — `"spot_up"`,
`"spot_up2"`, `"vol_up"`, `"time_adv"`, `"time_adv:spot_up"`, … — never by
floats. Cross greeks requested together share engine prices deterministically:
charm + color + vega_theta price the time-advanced env's spot/vol bumps once
(~4 extra pricings total instead of ~9). The memo lives for one
`calculate_numerical_greeks` call; the standalone public per-greek methods
(`calculate_numerical_vanna`, …) construct a fresh context and keep their
exact current behavior.

## 6. New greeks

Conventions follow the file's existing scaling exactly. "Inner Δ/Γ" means the
evaluation goes through `_get_delta_gamma`, so `greeks_mode=ENGINE/AUTO` PDE
grid readout is honored.

| Greek | Definition | Numerical method | Units | Extra pricings* |
|---|---|---|---|---|
| `charm` | dΔ/dt | Δ(t+step) − Δ(t); inner Δ; reuses the theta advance machinery (`time_shift`, observation-drop → 0, at-expiry → 0) | per step (day) | 2 (shares `time_adv` spots with color) |
| `color` | dΓ/dt | Γ(t+step) − Γ(t); inner Γ; same advance | per step (day) | shared with charm (+1 base if Γ absent) |
| `vega_theta` (`veta`) | dVega/dt | vega(t+step) − vega(t); each vega one-sided vol bump | vega per step | 2 (1 shared with `time_adv`) |
| `zomma` | dΓ/dσ | (Γ(σ+Δσ) − Γ(σ−Δσ))/(2Δσ); inner Γ; one-sided fallback when σ−Δσ ≤ 0 (same guard as vanna/volga) | raw per unit vol | ≤6 |
| `speed` | d³V/dS³ | 4-point stencil (V₊₂ₕ − 2V₊ₕ + 2V₋ₕ − V₋₂ₕ)/(2(Sh)³), relative bumps on the original spot axis — no nested envs | raw | 2 (V±ₕ shared with delta/gamma when bumped; 4 under engine-greeks mode) |
| `dividend_volga` | d²V/dq² | (V(q+Δ) − 2V₀ + V(q−Δ))/Δ², `second_order` mode via `_build_div_bumped_env`; no fallback (negative q is legitimate) | raw per unit q² | 2 |
| `gamma_theta` | −½σ²S²Γ per day (BS-PDE identity) | from already-computed Γ + `pricing_env.get_vol(strike-or-spot, T)` — same vol lookup as vega | per day (clock-scaled, §8) | 0 when Γ requested |

\* beyond the base price, assuming the greek is requested alongside the
default set.

Rationale for `dividend_volga`: for CNY index underlyings q is a large,
uncertain implied carry (futures basis), and PV is convex in q via
e^((r−q)T); first-order `dividend_rho` misstates large carry moves.
`delta_q` (dΔ/dq) already exists and is unchanged.

`speed` deliberately uses the flat 4-point stencil rather than differencing
gamma at bumped spots: single-level bumps only, 2 extra pricings, and no
nested-env float-key ambiguity.

## 7. Theta suite

| Greek | Definition | Computation | Status |
|---|---|---|---|
| `theta` | dV/dt per step | repricing at advanced date | existing, unchanged |
| `r_theta` | funding r(V−SΔ) | estimate (default) or exact mode | existing keys/values unchanged for bare names |
| `q_theta` | carry qSΔ | estimate (default) or exact mode | existing keys/values unchanged for bare names |
| `convexity_theta` | θ − r_theta − q_theta (residual) | existing | unchanged; stays in default set |
| `gamma_theta` | −½σ²S²Γ — the hedgeable gamma bleed | PDE identity from measured Γ | **new**, opt-in |

`gamma_theta` is a separate greek, not an alias of `convexity_theta`: for
vanillas they coincide (analytical invariant test, §12.3), but for
autocallables the residual absorbs observation/coupon/barrier effects while
the identity term is only the smooth diffusive bleed. Their difference is the
"unexplained theta" of a P&L explain — derivable by subtraction, so no fourth
key is added.

### Exact decomposition mode

New parameter `theta_decomposition_mode` ∈ {`"estimate"` (default),
`"exact"`} on `calculate()` and `calculate_numerical_greeks()`. `"exact"`
promotes the currently-private zeroed-r/q repricing path: `convexity_theta =
θ|r=0,q=0`, `r_theta = θ|q=0 − convexity_theta`, `q_theta = θ|r=0 −
convexity_theta` (3 extra theta repricings). The estimate stays the default
because it is the incumbent behavior and flipping would both change numbers
and triple default-set theta cost across backtest fleets; the docstring labels
it clearly as an estimate.

## 8. Dual clock (1D / 1TD)

Request-level clock qualifiers on every `supports_clock` greek: `theta`,
`r_theta`, `q_theta`, `convexity_theta`, `gamma_theta`, `charm`, `color`,
`vega_theta`.

- `<name>_1d` — advance `valuation_date + 1 calendar day`.
- `<name>_1td` — advance `calendar.add_business_days(valuation_date, 1)`;
  requires `pricing_env.calendar` (same error as today's
  `business_days` mode when absent).
- Bare `<name>` — resolved by `BumpConfig.time_bump_mode`
  (`auto`/`calendar_days`/`business_days`) exactly as today, preserving all
  existing numbers.

Per-component semantics (split-clock principle: diffusion runs on trading
time, carry accrues on calendar time):

| Component | `_1d` | `_1td` |
|---|---|---|
| `theta` (repricing) | price(d+1cd) − price(d) | price(next TD) − price(d); carries the *actual* calendar gap (Fri→Mon = 3 days of carry) |
| `r_theta`, `q_theta` (estimate) | annual component × step year-fraction (1/365 on an ACT/365-style env — the incumbent per-day scaling) | annual component × step year-fraction under the env's own day count (Fri→Mon = 3/365 on a calendar env; 1/244 on a BUSINESS_DAYS env). Amended during implementation from "calendar days/365": on a BUSINESS_DAYS env a Fri→Sat 1D step prices zero time passing (theta_1d = 0), so scaling carry by the raw calendar gap would fabricate carry the repriced theta does not contain; the env's year fraction keeps theta and its components on the same clock. |
| `gamma_theta` | −½σ²S²Γ / 365 | −½σ²S²Γ / `pricing_env.bus_days_in_year` |
| `charm`, `color`, `vega_theta` | per their clock's advance step | same |
| exact mode | correct under either clock automatically (reprices through the same advance) | same |

Bare `gamma_theta` (new, no incumbent behavior) follows the same resolved
clock as bare `theta`: `/365` under calendar, `/bus_days_in_year` under
business days.

Only `_1d`/`_1td` are supported — no arbitrary horizons; multi-day steps
remain a `time_bump_days` config concern for the bare names.

Forward compatibility: qualified greeks advance `valuation_date` and reprice,
so when the trading-clock-vol environment (separate spec, pending) merges,
`*_1td` inherits correct vol-time handling with no change here.

## 9. Analytical module

`greeks/analytical.py`, European vanillas with continuous q. Closed forms for
the existing set **plus**: `vanna`, `volga`, `charm`, `color`, `speed`,
`zomma`, `vega_theta`, `gamma_theta` (≡ the existing convexity term),
`dividend_volga`, `delta_q`. Every numerical greek gains an oracle.

- `dividend_volga`'s closed form is not textbook but is a clean derivation
  from the call/put with q in both the discount factor and d₁/d₂; validated
  against FD in tests. If the derivation cannot be made exact, the analytical
  entry is omitted (TODO + numerical only) rather than approximated.
- Time-derivative closed forms are reported per calendar day (/365), matching
  the existing analytical theta convention; they are the oracle for the `_1d`
  variants. `_1td` variants are validated structurally (§12.4), not against a
  closed form.
- Auto-routing: existing names keep their incumbent routing (§3.2). The new
  canonical names (charm, color, speed, zomma, vega_theta, dividend_volga,
  gamma_theta) join the analytical auto-set, so a vanilla request routes
  analytical only when *all* requested names are in that set (the existing
  subset rule). Clock-qualified names (`*_1d`/`*_1td`) always route
  numerical — the analytical per-day forms are reachable explicitly via
  `calculate_analytical_greeks(greeks=[...])` and serve as the `_1d` oracle
  in tests.

## 10. Enum changes

`EquityGreek` gains `SPEED = "speed"`, `ZOMMA = "zomma"`, `VEGA_THETA =
"vega_theta"`, `DIVIDEND_VOLGA = "dividend_volga"`, `GAMMA_THETA =
"gamma_theta"` (CHARM/COLOR already exist). Clock-qualified names are
string-level request syntax, not enum members. `CommonGreek` unchanged.

## 11. Sequencing & gates

Phased commits on `worktree-riskmeasures-greeks`, each gated before the next:

1. **R1 — pure moves.** Bucketed families → `bucketed_coordinates/`;
   bump-env + theta-advance machinery → `greeks/bump_envs.py` +
   `greeks/theta_decomposition.py`. No logic edits.
   *Gate:* full greeks-related test suite + fingerprint script (§12.1)
   byte-identical.
2. **R2 — registry.** `GreekDef` table; `_normalize_greeks` and
   `calculate_numerical_greeks` dispatch rewired through it, preserving
   call order.
   *Gate:* same as R1.
3. **E1 — numerical new greeks** + context memo + clock qualifiers + exact
   theta mode exposure + enum additions.
   *Gate:* new-greek test battery (§12.2–12.5); fingerprint still identical
   for old requests.
4. **E2 — analytical closed forms** + oracle tests + docs.
   *Gate:* analytical-vs-numerical agreement battery; optional QuantLib
   cross-check.

## 12. Testing

1. **Fingerprint gate (R1/R2/E1):** a local script prices a small matrix —
   EuropeanVanillaOption × (analytical path, PDE quick, QUAD quick) and a
   snowball × (PDE quick, QUAD quick) — requesting the default set plus every
   currently-working optional greek, dumping the result dicts to JSON.
   Run on `main`, rerun per phase, byte-compare. Same-machine invariant only
   (not a committed golden — cross-arch float drift makes bitwise goldens
   CI-hostile; CI relies on the existing suite plus the new tolerance tests).
2. **Registry completeness:** for every canonical name in the registry (and
   every alias, and every clock-qualified form), request it individually on a
   vanilla + BSM engine and assert the key is present in the result — the
   test that makes the charm/color failure mode unrepresentable. Also:
   `vanna_1td` and other unsupported qualifiers raise `ValidationError`.
3. **Oracles & invariants:** analytical vs numerical within FD-scaled
   tolerances on a strike × maturity grid (calls and puts); analytical
   `gamma_theta == convexity_theta` to machine precision for vanillas;
   numerical `gamma_theta ≈ convexity_theta` within tolerance for vanillas;
   put-call relationships where applicable (charm/color equal up to the
   dividend term).
4. **Clock tests:** valuation date chosen so next trading day ≠ next calendar
   day (Friday with a weekend, and a CN-holiday bridge): `theta_1td` carries
   the multi-day calendar gap; `theta_1d == theta` when config resolves to
   calendar; `gamma_theta_1td/gamma_theta_1d == 365/bus_days_in_year` exactly
   (same Γ, different scaling); `r_theta_1td` uses the actual calendar gap.
5. **Edge cases:** linear products → `linear_value` (delta 1.0, all new
   greeks 0.0); at-expiry → 0.0; observation-drop on time advance → 0.0;
   low-vol one-sided fallbacks for zomma (mirroring vanna/volga tests);
   `theta_decomposition_mode="exact"` reconciles: components sum to theta
   within tolerance on a vanilla and are exact vs analytical.
6. **Refactor safety net:** existing tests (`test_point_greeks`,
   `test_greeks_mode_and_engine_type`, `test_greeks_bump_config`,
   `test_greeks_theta_schedule`, `test_rate_keyrate_buckets`,
   `test_equity_futures_delta_buckets`, SIMM/backtest/portfolio suites) run
   green at every phase, with worktree source shadowing the editable install
   (`PYTHONPATH=$PWD`).

## 13. Out of scope

- Any semantic change to bucketed coordinates, auto-routing for
  vanna/volga, or the QUAD KI-probability definition.
- Trading-clock-vol environment integration (separate pending spec; §8 is
  forward-compatible with it).
- FX/credit/bond calculator internals (they re-export the equity calculator
  and benefit automatically).
- Cash-greeks report extension for new greeks (possible follow-up).
- Arbitrary theta horizons (`_3d`, …) and a `theta_residual` output key
  (derivable as `convexity_theta − gamma_theta`).
