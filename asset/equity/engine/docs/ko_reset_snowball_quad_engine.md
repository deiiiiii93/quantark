# KO-Reset Snowball Quadrature Engine

This document describes the KO-reset snowball quadrature engine implemented in:
`asset/equity/engine/quad/ko_reset_snowball_quad_engine.py`.

## Overview
- Extends the Snowball quadrature recursion with two schedules:
  - **V_out** uses **pre-KI** KO observations.
  - **V_in** uses **post-KI** KO observations.
- If no KI happens, **V_out settles at the end of the pre-KO schedule** (pre-maturity).

## Supported Features
- Discrete KO observations for both pre/post schedules.
- Discrete or continuous KI monitoring (Brownian-bridge).
- `disable_ko_after_ki=True` suppresses post-KI KO on V_in.

## Limitations
- Only `PostKOScheduleMode.ABSOLUTE` is supported.
- `REBASED` post-KO schedules raise `ValidationError`.
