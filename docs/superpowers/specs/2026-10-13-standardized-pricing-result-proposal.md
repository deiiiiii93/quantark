# Standardized PricingResult: proposal (I12)

Status: proposal only, no production code. Follow-up queue item I12 (spec `.ship/2026-10-13-followup-queue/spec.md`).
It starts from where the 2026-10-10 brainstorm stopped, which was the PURPOSE question. The user had floated "values +
error codes" for edge cases such as a valuation after expiry, right after keeping the W8 Q4 same-instant rule. Every
file:line below was read in this worktree.

## 1. Problem

A caller that prices something today can get back any of six shapes:
- a `float`;
- a per-engine dataclass reached through `get_last_result()`;
- `price_detailed`;
- the cash-leg `PricingResult`;
- an execution `PricingOutcome` / `PricingFailure`;
- an intraday result with per-greek status.

Refusals are exceptions, and the exception type does not say what kind of refusal it is. Three groups hit this:

- **Run code** (replay, backtests, dynamic scenarios, PnL explain) has to decide per edge case whether to stop, skip,
  zero or carry. Today it decides by catching exception TYPES whose meaning is overloaded (2.3).
- **Desk / batch callers** pricing a book want every trade valued and the failures listed, instead of the first
  refusal aborting the book. The execution layer offers this only for its own entry points (2.2).
- **Report code** wants one shape for price + s.e. + components + diagnostics across engines. Each engine family
  exposes its own (2.1).

Concrete symptom: `time_to_exercise` refuses a valuation strictly after the exercise date with a plain
`ValidationError` (`quantark/asset/equity/product/option/base_equity_option.py:720-741`, raise at `:730`). That is the
same type as a negative strike or a malformed schedule. A replay that wants "expired ⇒ settle" cannot tell the two
apart without parsing the message. The repo has 4312 `raise ValidationError` sites, 232 `raise PricingError`, 170
`raise NumericalError`, 58 `raise MarketDataError` and 94 `raise CapabilityError` (grep over `quantark/`). No error
code exists anywhere (`grep error_code|ErrorCode|reason_code` finds nothing).

## 2. Current behaviour

### 2.1 Output shapes

- `BaseEngine.price(product, env, *, lifecycle_state=None) -> float`
  (`quantark/asset/equity/engine/base_engine.py:131-150`). This is the universal API.
- `BaseEngine.price_with_events(...) -> PricingResult` (`base_engine.py:182-215`). That `PricingResult` is the cash-leg
  one: `npv` + optional `EventDistribution` (`quantark/cashleg/event_distribution.py:523-528`).
- Per-engine results: there are 64 top-level `class *Result` definitions under `quantark/`, and 34 of them sit in the
  pricing packages (`asset/`, `montecarlo/`, `volmodels/`, `cashleg/`, `intraday/`, `execution/`). Examples:
  - `DCNMCResult` (`asset/equity/engine/mc/dcn_mc_engine.py:66`; `pv`, `std_error`, per-leg PVs, pending receivables);
  - `SnowballMCResult` (`snowball_mc_engine.py:81`);
  - `PDESolutionResult` (NamedTuple, `engine/pde/base_pde_solver.py:137`);
  - `RQMCResult` (`montecarlo/qmc_rqmc_driver.py:38`);
  - `FxTarnMCResult` (`asset/fx/engine/mc/fx_tarf_mc_engine.py:46`).
  They are reached through `get_last_result()` (12 definitions) or `price_detailed` (4), so they are stateful on the
  engine instance.
- Execution contracts (`quantark/execution/contracts.py`):
  - `PricingOutcome(value, normalized_economics, diagnostics, manifest)` (`:82-87`);
  - `PricingFailure(item_id, error: FrameworkErrorInfo, diagnostics, manifest)` (`:90-95`);
  - `FrameworkErrorInfo(error_type: str, message: str)` (`:76-79`);
  - `ScenarioOutcome` (`:123-129`) and `BatchOutcome` (`:183-189`).
- Intraday:
  - `IntradayValuationResult` (`quantark/intraday/result.py:98-121`): price split into contingent / pending /
    paid, provenance, identities.
  - `GreekValue` (`:24-55`): `status ∈ {ok, undefined, failed, not_requested}` (`:18`), with a value iff ok and a
    reason otherwise.
  - `EnginePriceOutcome.exact` and `PointGreeks.status/reason` (`intraday/engines/base.py:10-60`).
  - `IntradayFailure(item_id, error_type, message)` (`intraday/batch.py:28-32`).
  This is the one place in the repo that already reports status per output.

### 2.2 Failure handling that already exists

- `PricingSession.price_many(..., collect_errors=True)` (`quantark/execution/api.py:169-206`) turns ANY exception into
  `PricingFailure(FrameworkErrorInfo(type(exc).__name__, str(exc)))`. `run_scenarios` (`api.py:207-`),
  `execution/scenario/runner.py:100-107` and the dask backend (`execution/backends/dask_backend.py:168`) do the same.
  The error carries the exception class name and message only.
- `value_intraday_many(collect_errors=True)` (`intraday/batch.py:35-48`) does the same into `IntradayFailure`.
- Exception roots: `QuantArkException` → `ValidationError` / `NumericalError` / `MarketDataError` / `PricingError`
  (`quantark/util/exceptions.py:9-64`). Separately there are `CapabilityError`, `ResourceBudgetExceeded`, `PreparationError`, …
  (`quantark/execution/errors.py:19-40`), and `StatisticsRefused(PricingError)`
  (`asset/equity/engine/event_stats.py:167`).

### 2.3 How runs branch on edge cases today

- `ProductReplay`: `except ValidationError` around `get_maturity` means "on/after a date-based maturity"
  (`backtest/replay/product_replay.py:541-551`; the comment notes it no longer raises ON the date).
- `except ValidationError` on a term-dividend build re-raises unless the caller allowed flat carry (`:294-301`).
- A failed surface node is logged and written as a NaN row (`:662-692`).
- Event stats fall back to MC on `except Exception` only in `mode == "mc"`, and `StatisticsRefused` always re-raises
  (`:786-812`).
- Replay engine: `except ValidationError` records `_hedge_infeasibility` and re-raises, so the run stops with a
  recorded cause (`backtest/replay/engine.py:857-869`).
- Dynamic scenario: `except Exception: continue` while looking for a futures maturity (`dynamicscenario/engine.py:754-757`).
- Matured-before-run is a warning + zero: `MaturedBeforeRunWarning`
  (`asset/equity/lifecycle/autocallable.py:80`; W1 rule).

So runs already implement "do not stop on this edge case", once per site, by exception type plus local convention.
`ValidationError` is caught for at least three unrelated meanings.

## 3. Decision 0: the purpose

This is the question the brainstorm stopped at. The designs differ by purpose, so it comes first.

| | Purpose | Who benefits | What it requires |
|---|---|---|---|
| P1 | **Callers branch on codes**: a stable, machine-readable reason for every refusal | run code (2.3), batch callers | a code taxonomy; no change to return types |
| P2 | **Runs never stop on an edge case**: a run (replay, backtest, dynamic, explain, VaR) values every day and lists what it could not value | desk runs, overnight batches | P1's codes plus a per-run policy: code → action |
| P3 | **One output shape across engines**: price, s.e., components, events, diagnostics, status | reports, the execution layer, notebooks | a common result type returned by every engine |
| P4 | Several of the above, staged | | the union, in some order |

The dependency matters: P2 needs P1, because a policy keyed on exception class names inherits the overloading in
2.3. P3 is independent of P1 and P2, but if it carries a `status` it needs P1's codes too.

## 4. Designs per purpose

### P1: codes on the exceptions (no return-type change)

- One enum `RefusalCode` in `quantark/util/exceptions.py`, with a short initial list grounded in 2.3 and the
  W-series rulings:
  - `INVALID_INPUT`;
  - `CONTRACT_EXPIRED` (after the exercise/final date; the W8 Q4 same-instant rule);
  - `NOT_MODELLED` (the `TODO(...)` refusals: no closed form, a semantics not yet ruled);
  - `MISSING_MARKET_DATA`;
  - `NUMERICAL_FAILURE`;
  - `CAPABILITY` (an engine/route/output not supported);
  - `STATE_CONFLICT` (a lifecycle state contradicting the product, e.g. the W14 knocked-in conflict).
- `QuantArkException.__init__(msg, *, code=None)`. The default code is derived from the class
  (`ValidationError` → `INVALID_INPUT`, `CapabilityError` → `CAPABILITY`, …), so all 4900+ existing raise sites keep
  working unchanged. Sites are re-coded only where a caller branches: the expiry refusals, the `TODO` refusals, the
  sites in 2.3.
- `FrameworkErrorInfo` and `IntradayFailure` gain `code`. The field is additive with a default.
- Cost: small to start (a dozen sites), and it grows with each caller that wants to branch. No value moves. API change
  is additive. Existing `except ValidationError` keeps catching.
- Risk: a code drifts from its site. This is mitigated by a test that every `TODO(` refusal carries `NOT_MODELLED`
  (grep-driven).

### P2: run-boundary envelope + explicit policy (on top of P1)

- A helper `value_or_refusal(engine, product, env, **kw) -> Valued | Refused`, where `Valued(value)` and
  `Refused(code, message, error_type)` are frozen dataclasses. It is used ONLY at run boundaries. `price()` stays
  `float`.
- Each run takes a `RefusalPolicy`: `{code: action}`, where action ∈ {`stop`, `skip_day`, `zero`, `carry_previous`}.
  The default is `stop` for every code, so no run changes behaviour unless the caller opts in ("no stupid fallbacks":
  a non-stop action is a declared choice and never a silent default). Every non-stop action is recorded in the run
  output: date, position, code, message, action.
- Existing per-site conventions (2.3) become policy rows with today's behaviour as their default:
  - matured-before-run → `zero` + warning;
  - surface node → NaN.
  So the change is bitwise for current runs.
- Cost: medium. The five run families each route their pricing calls through the helper, and the run outputs gain a
  refusal table.
- Risk: a desk turns on `zero` for `NUMERICAL_FAILURE` and books wrong P&L silently. Mitigation: some codes
  (`NUMERICAL_FAILURE`, `STATE_CONFLICT`) cannot be mapped to `zero` or `carry_previous`, only to `stop` or
  `skip_day`, and a skipped day is visible in the output.

### P3: one result type (`EngineResult`)

- A frozen `EngineResult(price, std_error: Optional, components: Mapping[str, float], event_distribution: Optional,
  diagnostics: Mapping, status, code, reason)`, returned by a new non-abstract `BaseEngine.evaluate(...)`. Its
  default wraps `price()`. Engines with a detailed result override `evaluate` to map their dataclass into it.
- The existing `PricingResult` (cash-leg) becomes the `event_distribution` carrier. `PricingOutcome.value` can hold an
  `EngineResult`. The intraday result keeps its richer shape and exposes `to_engine_result()`.
- Cost: large. There are 34 pricing-side result classes to map. Typed per-leg fields (`DCNMCResult.pv_loss_leg`, …)
  flatten into `components` and lose static typing, or `EngineResult` keeps a `detail` field holding the original
  dataclass.
- Risk: a second API next to `price()` that drifts. `price()` must stay `== evaluate().price` bitwise, enforced by a
  parametrised test over the engine registry. No values move.

### P4: staged

P1 → P2 → P3, each its own item with its own approval. P1 alone already answers the user's example: a replay can
branch on `CONTRACT_EXPIRED`.

## 5. Recommendation

**P4 staged, starting with P1, then P2. P3 is deferred until a concrete consumer needs it.**

- P1 is the foundation both other purposes need. It is additive, moves no value, and touches only the sites someone
  branches on.
- P2 is what "runs never stop on an edge case" actually needs. The execution layer already proves the envelope
  pattern (`collect_errors=True` → `PricingFailure`), but without codes and without a policy. P2 adds both, with
  `stop` as the default, so it complies with "no stupid fallbacks" by construction.
- P3 has the highest cost and no consumer identified today. The execution layer and intraday already have their own
  envelopes, and reports read per-engine results deliberately. It is worth revisiting when a report or the
  certification pipeline needs a cross-engine s.e./components view.
- Valuation after expiry (the motivating example) stays a refusal at the engine, per W8 Q4. With P1 it carries
  `CONTRACT_EXPIRED`. With P2 a run can declare `CONTRACT_EXPIRED → zero` (or settle via the tracker), and the
  declaration is visible in the output.

## 6. Open questions for the user

1. **Purpose.** Which of P1 (codes), P2 (runs never stop), P3 (one shape), or P4 staged? (Recommended: P4 as P1 → P2,
   P3 deferred.)
2. **Codes on exceptions or on return values?** Recommended: on exceptions (P1). `price()` keeps returning `float`
   and raising, and envelopes exist only at run/batch boundaries. Agree?
3. **Initial code list.** Are `INVALID_INPUT`, `CONTRACT_EXPIRED`, `NOT_MODELLED`, `MISSING_MARKET_DATA`,
   `NUMERICAL_FAILURE`, `CAPABILITY` and `STATE_CONFLICT` the right granularity? Add or split any (e.g. separate
   "after expiry" from "later on the exercise day than the booked instant")?
4. **Run default policy.** Default `stop` for every code, with non-stop actions opt-in per run and listed in the
   output? Yes/no.
5. **Forbidden mappings.** May `NUMERICAL_FAILURE` / `STATE_CONFLICT` ever map to `zero` or `carry_previous`?
   (Recommended: no, only `stop` / `skip_day`.)
6. **Public execution API.** `FrameworkErrorInfo` / `PricingFailure` ship in the PyPI package (v0.3.0+). Is an
   additive `code` field with a default acceptable there, or is a version bump wanted?
7. **Greeks.** Should non-intraday greeks get per-greek status like intraday `GreekValue` (e.g. theta `undefined` at
   expiry; see the I11 proposal), or stay floats that raise? (Recommended: decide after P1/P2.)
