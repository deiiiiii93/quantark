# Engine parameter guide

The `PDEParams` fields, the `GridConfig` presets and the `CrankNicolson` settings are
tabled in `README.md` ("PDE parameters"); the QUAD presets are in the same README
section ("Batch-Friendly Engine Params") and in the docstrings of
`quantark/asset/equity/param/`. This page gives the measurements behind three PDE choices
that 0.10.0 changed, when the local-vol vanilla and barrier solvers and the DCN moved onto
the shared core (spec `docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-design.md`,
sections 4.2, 4.3, 4.4, 4.6 and 9).

Units: the vanilla and barrier in price units at spot 100; the DCN in basis points of
notional; the autocallables per 100 of notional, so a difference is a number of hedge
contracts.

## Daily monitoring wants 8 steps per day

Every 1D route takes the shared time density: 4 steps per business day at `fast` and
`standard`, 8 at `high`. Under daily monitoring the error left at `standard` is time
error: it closes with the steps, not with the points. On a daily
monitored barrier under a constant local vol of 0.22, against an exact discrete-barrier
recursion:

| Steps per day | Error |
|---|---|
| 4 (`standard`) | +9.3e-4 |
| 8 | +2.1e-4 |
| 16 | +1.1e-5 |

At `standard` the largest error on the measured daily cells is 1.2e-3 / 1.1e-3 / 2.1e-3
on a constant surface / the byte book's smile / a steep synthetic skew (0.9.0's local-vol
kernel: 1.1e-2 / 8.3e-3 / 9.4e-2). Monthly monitoring at `standard` is within
2.5e-4 / 2.5e-4 / 6.2e-4 and weekly within 1.6e-4 (constant surface), so the advice is
for daily schedules. Where 1e-3 matters there, ask for 8 steps per day:

```python
from quantark.asset.equity.engine.pde import GridConfig
from quantark.asset.equity.param import PDEParams

daily = PDEParams(grid=GridConfig(steps_per_day=8.0))
```

It roughly doubles the step count. The DCN's knock-in is observed daily too; at 400
points, on one contract at the money, its error is 1.71 / 0.34 / 0.10 / 0.036 / 0.020 bp
at 1 / 2 / 4 / 8 / 16 steps per day, so it takes the shared presets unchanged.

## `GridConfig(time_steps=200)`: the old local-vol step budget

0.9.0's local-vol vanilla and barrier kernel marched a total step budget (200 at `fast`
and `standard`, 400 at `high`) on a uniform grid of its own. Its error was spatial
(2.2e-3 / 5.5e-4 / 1.4e-4 at 400 / 800 / 1,600 uniform nodes, unchanged from 200 to 4,032
steps). Since 0.10.0 those routes take the core's density and concentrated points, so
their accuracy does not depend on the maturity. `GridConfig(time_steps=200)` restores the
`standard` budget on the core, at a cost of:

| Product | Cost of the 200-step budget |
|---|---|
| Vanilla | at most 1.4e-5 |
| Continuously monitored barrier | at most 3.4e-5 |
| Monthly barrier | up to 5.4e-4 |
| Weekly barrier | up to 4.3e-3 |
| Daily barrier | up to 2.6e-2 |

A total budget spreads its steps over the horizon whatever the schedule, so it suits a
vanilla or a continuously monitored barrier and not a discretely monitored one.

```python
budget = PDEParams(grid=GridConfig(time_steps=200))   # time_steps replaces steps_per_day
```

## An explicit bound belongs well beyond a barrier

By default the domain comes from the shared rule (`num_std`, 4.0, with `bounds` unset).
The far-field rows are Dirichlet values, and two of them are the contract's certain
outcome beyond a barrier: the certain knock-out at the next observation beyond a
discretely monitored single barrier, and, under discretely observed knock-in, the
knocked-in row on an autocallable's knock-in side while a knock-in observation remains.
Far beyond a barrier that outcome is certain and the row does not reach the spot (on the
default domain the knock-in-side row moves monthly and daily knock-in prices at
round-off). Next to a barrier it is not certain: the right row there is a knock-in or
knock-out probability, which no route builds, and a constant row is wrong.

| Case | Error before 0.10.0 | Error in 0.10.0 |
|---|---|---|
| Snowball at `standard`, `bounds=(60, None)` | +1.255 | −1.2e-3 |
| Snowball at `standard`, `bounds=(70, None)` (15 below the knock-in barrier) | +4.295 | −0.129 |
| Monthly up-and-out call, `bounds=(None, 140)` | +1.70 / +9.18 | −8.8e-3 / −1.6e-2 |

With the edge close to the barrier the autocallables' knock-in-side row is still off by
−0.09 to −0.13, and the single barrier's row, with the edge two or three points beyond the
barrier or with one distant observation, by 0.1 to 3.3. The knocked-in upper row of an
autocallable, which 0.10.0 did not change, is off by −0.73 to −1.74 under a narrow upper
bound. The double-barrier and touch rows under discrete monitoring were not measured.

So leave `bounds` unset, or put the edge well beyond every barrier; to widen the domain
raise `num_std`.
