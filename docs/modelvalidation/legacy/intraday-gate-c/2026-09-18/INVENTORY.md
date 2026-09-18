# Gate C inventory and where each piece goes

Every fixture, route and measure the Gate C harness covered, what it recorded, and the
modelvalidation study that replaces it. "Plan 1" is
`docs/superpowers/plans/2026-09-18-intraday-modelvalidation-certification.md`. A row marked
Plan 2 or Plan 3 is **uncertified until that plan banks its study**; under the release rule
an uncertified intraday route is not shipped in a production release, and researchers use
it freely. Nothing here is a runtime check.

| Gate C fixture | Routes | What Gate C recorded | Target study | Where |
|---|---|---|---|---|
| `snowball_daily_ki` (KI 75 at every SSE close) | QUAD V2 | 2026-09-18 sweep, 154 groups, never packaged; gamma inconclusive at `bp-10-ko` on every horizon (reference floor) | `snowball-intraday-daily-ki-bsm`: PV, desk delta/gamma, desk theta certified; point delta/gamma uncertified (no RQMC point estimator) | Plan 1 |
| `snowball_discrete_ki` (monthly KI on the KO dates) | QUAD V2, PDE, MC RQMC | price cells passed 528/528 QUAD V2, 3/528 PDE, 5/528 MC; Greeks: all 12 measures for QUAD V2 under `desk` and `sessions_only`, `desk_delta` only for PDE, none for MC | `snowball-intraday-monthly-ki-bsm`: adds `{point,desk}_{vega,rho,dividend_rho}` and `point_theta`, and an MC candidate against an independently constructed reference | Plan 2 |
| `snowball_long_gap` (long first period) | QUAD V2 | windows to 90 days before the first fixing | long-gap cases of the monthly study | Plan 2 |
| `digital` (cash-or-nothing, terminal) | analytical, MC RQMC | price cells passed 264/264 analytical, 8/264 MC; all 12 Greek measures for the analytical route | `digital-intraday-bsm`: the analytical candidate is exact; the MC candidate is measured against the closed form as an analytical reference | Plan 2 |
| `barrier_uo_zero_carry` (continuous up-and-out call) | analytical, PDE, MC RQMC | price cells passed 264/264, 186/264, 148/264 | `barrier-intraday-zero-carry-bsm`: continuous monitoring needs a first-passage treatment in the reference, or the analytical reference kind | Plan 2 |
| `one_touch_zero_carry` (continuous up one-touch) | analytical, PDE | price cells passed 264/264, 144/264 | same study family | Plan 2 |
| analytical point theta (per-request stencil limit with a runtime budget gate) | analytical barrier | the gate is removed in Plan 1; the estimator and its error estimate stay | `point_theta` as a certified quantity of the barrier study; until then the killed-density control in `test/intraday/controls` is a runtime regression test | Plan 2 |
| Phoenix, KO-reset snowball | QUAD V2, PDE | never had Gate C evidence | the product suites of spec section 4.1 | Plan 3 |

Independent numerical controls kept as runtime regression tests (they license nothing):
the Gaussian-transition control and its self-checks, the single-period snowball closed form,
the digital closed form, the zero-carry barrier closed form with its exact-bridge MC check,
the killed-density theta control, and the manufactured non-uniform-grid PDE gamma control.
