# Code Review Report

## 📋 Review Summary
Files: 3 | Changes: 1 modified, 1 added, 0 deleted | Findings: 3 issues (0 critical, 1 high, 1 medium, 1 low)

---

## 🟠 High Priority

### Issue #1: Early return in theta path drops rho computation and omits 'rho' key
Location: asset/equity/riskmeasures/greeks_calculator.py:245-256
Category: Logic

Issue:
When observation_schedule exists but bumping removes all future records, the code sets theta to 0.0 and returns early:
- Lines 245-256:
  - if bumped_schedule is None and hasattr(product_theta, "observation_schedule"):
    ...
    if schedule_present:
        theta = 0.0
        greeks["theta"] = theta
        return greeks

This bypasses the subsequent rho bump and leaves greeks without the 'rho' entry, violating the method's contract (documented to return delta, gamma, vega, theta, rho).

Impact:
- Callers expecting greeks["rho"] will raise KeyError or operate on incomplete greeks, causing downstream failures or inconsistent behavior.
- Missing rho also harms risk reporting comparability across products.

Fix:
- Remove the early return. Set theta and proceed to compute rho as usual. Example:
  - Replace the return greeks with just setting theta and letting execution continue:
    - Set theta = 0.0; greeks["theta"] = theta; do not return here.

Related:
- The test/test_greeks_theta_schedule.py currently doesn't cover the "no future observation points remain" branch; consider adding a test where all observation entries drop to ensure rho remains present.

---

## 🟡 Medium Priority

### Issue #2: Incorrect fallback when observation_schedule is present but not ObservationSchedule instance
Location: asset/equity/riskmeasures/greeks_calculator.py:245-256
Category: Logic

Issue:
The early-return branch triggers based on the presence of product_theta.observation_schedule (schedule_present) regardless of its type. If a product has a non-None observation_schedule that is not an ObservationSchedule (e.g., a legacy or custom object), bumped_schedule stays None and this branch will set theta to 0.0 and return, skipping recomputation and rho.

Impact:
- Theta becomes 0.0 and rho is skipped for products with non-standard schedules, even when the schedule has future observations and should be priced normally.
- Silent functional degradation for products using legacy schedule representations.

Fix:
- Gate the "no future observation points remain" early path on both "schedule is an ObservationSchedule" and the bump result being None. For example:
  - Only consider schedule_present True if isinstance(schedule, ObservationSchedule).
  - If schedule is not an ObservationSchedule, fall back to pricing without altering schedule, and do not short-circuit.

Related:
- BarrierOption.validate normalizes schedule to ObservationSchedule, but other products may not. Keeping this guard avoids surprises outside barrier-like products.

---

## 🔵 Low Priority

### Issue #3: Updating legacy observation_dates with times for date-based schedules may be misleading
Location: asset/equity/riskmeasures/greeks_calculator.py:257-260
Category: Other

Issue:
After generating a bumped ObservationSchedule, the code sets product_theta.observation_dates = bumped_schedule.times. For date-based schedules (records use observation_date), ObservationSchedule.times returns an empty list. This overwrites any legacy observation_dates with [] for date-based schedules.

Impact:
- Engines or code paths that still consult legacy observation_dates could see an empty list even though a valid schedule with remaining observation_date records exists, potentially altering behavior in legacy paths.

Fix:
- Only synchronize legacy observation_dates when the schedule uses observation_time values (i.e., when bumped_schedule.times is non-empty or schedule uses times). Example:
  - if hasattr(product_theta, "observation_dates") and bumped_schedule.times: product_theta.observation_dates = bumped_schedule.times

Related:
- BarrierOption.validate prioritizes observation_schedule over legacy fields; this reduces practical risk, but the guard above will future-proof behavior.

---

## ✅ Recommendations
- Next commit: Fix Issue #1 (remove early return that skips rho), and implement the guard in Issue #2 to avoid incorrect short-circuits for non-ObservationSchedule types.
- Before merging: Add a unit test covering the "all observation entries dropped" case to ensure greeks still include 'rho' and other keys.
- Future refactor: Consider centralizing time-bump logic for schedules in a utility to keep consistency across products and engines.

---

## 📌 Context Notes
- Tech stack detected: Python, scipy, NumPy-style numerical routines; custom pricing framework with schedule-based path-dependent options.
- Tests added: test/test_greeks_theta_schedule.py validates time/dates record dropping and that the theta bump uses a filtered schedule in pricing; it does not cover the "no remaining observation points" branch or rho presence.
