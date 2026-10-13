# EuropeanQuadEngine near expiry vs its lower-bound check (W16, signed W13 follow-up 2)

2026-10-13 · ship run `followup-queue` · task T9

## Question

`EuropeanQuadEngine.price` refuses (`NumericalError`) a price below the discounted no-arbitrage lower bound minus an
ABSOLUTE `1e-4`. Does the quadrature near expiry produce false refusals or hidden errors against that check? Code
changes only if the measurement shows a problem.

## Measurements

`measure.py` → `outputs/measure.txt`. 1680 cells: Simpson and Gauss-Legendre × spot 100 / 10 000 × multiplier 1 / 300 ×
K/S 0.8 … 1.2 × vol 0.1 / 0.3 / 1.0 × maturity 1 day, 1 hour, 1 minute, 1 second, 1e-9 y × call/put; the Black-Scholes
closed form is the reference.

- 0 refusals, 0 false refusals.
- The smallest margin (price − (bound − 1e-4)) was 9.7e-5, on deep-ITM 1e-9 y cells at spot 1e4 × multiplier 300.
  The margin is an absolute number, while the quadrature error grows with the notional.

`probe_scale.py` → `outputs/probe_scale.txt` (before) and `outputs/probe_scale_after.txt` (after):

1. **Notional scale.** Deep ITM (K/S 1.2 put, 0.8 call), maturities 1e-9 y / 1 s / 1 day, spot up to 1e6,
   multiplier up to 1e6.
   - **20 false refusals** before the fix. The engine's relative error was at most 2e-11 (Gauss-Legendre) and
     7.7e-13 (Simpson), but an absolute error of 1e-4 is exceeded once the price reaches about 1e8.
   - Example: Simpson, spot 1e5, multiplier 1e4, 1e-9 y put. The engine priced 199 999 999.973896 against a bound of
     199 999 999.974000. The true price equals the bound, so the price was refused.
2. **Gauss-Legendre accuracy.** The relative ATM error is about 4e-3 at 1e-6 y, 1 day and 1 month, 5e-4 at 1 year and
   3e-3 at 5 years.
   - This error is NOT near-expiry specific, and the check does not catch it: it is time value, far above the bound.
   - Simpson, the engine default, is within 2e-5 at every maturity.

## Change

The check's tolerance is now `max(1e-4, Tolerance.RELATIVE · lower_bound)`, with `Tolerance.RELATIVE = 1e-9`, about 50×
the largest measured relative error.
- Any bound below 1e5 behaves exactly as before.
- After the fix: 0 refusals in the probe.
- A price 1% below the bound is still refused (test).
- Tests: `test/test_european_quad_engine.py::test_lower_bound_check_scales_with_the_notional` and
  `::test_lower_bound_check_still_refuses_a_price_far_below_the_bound`.

## Open (not changed here)

- Gauss-Legendre has a 1e-3-level ATM error at its default grid (all maturities). It needs its own accuracy ticket.
- The engine's expiry shortcut `if T < 1e-10` is a hard-coded tolerance. Per the repo rule it should be `is_zero(T)`.
  This is pre-existing, and switching would be bitwise (Tolerance.ZERO = 1e-10, `<` vs `<=` aside).
