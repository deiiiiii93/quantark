# KO-Reset Snowball PDE Engine

This document describes the KO-reset snowball PDE solver implemented in:
`asset/equity/engine/pde/ko_reset_snowball_pde_solver.py`.

## Overview
- Based on the two-surface Snowball PDE framework (V0/V1 surfaces).
- **V0** applies the **pre-KI** KO schedule.
- **V1** applies the **post-KI** KO schedule.
- V0 and V1 share the same time grid, aligned to all relevant event times.
- If no KI happens, **V0 settles at the end of the pre-KO schedule** (pre-maturity).

## Supported Features
- Discrete KO observations (pre and post schedules).
- Discrete or continuous KI monitoring (same as SnowballPDESolver).
- `disable_ko_after_ki=True` suppresses post-KI KO on V1.

## Limitations
- Only `PostKOScheduleMode.ABSOLUTE` is supported.
- `REBASED` post-KO schedules raise `ValidationError`.

## Notes
- Terminal conditions use `get_maturity_payoff_v0` / `get_maturity_payoff_v1`.
- KO payoffs are resolved from each schedule using the product accrual settings.
- KI is applied as a direct jump from V0 to V1 at KI times.
