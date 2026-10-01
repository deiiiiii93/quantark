"""Bank the 0.7.0 base-family event semantics the Task 6 tests compare against (spec 3A §5.4).

Run on the 0.7.0 base tree (a detached worktree), with this file from the branch:

    ( cd $SCR/base && PYTHONPATH=$PWD python "$OLDPWD/test/pde_3a/fixtures/capture_v070_semantics.py" OUT.json )

Per case of ``pde_3a.semantics.cases``: the price, the grid delta and gamma, and the full t = 0
column as ``float.hex`` (a knock-in prices through its composite and has no single column, so it
banks the three numbers only). The accumulating no-touch cases also bank their single-observation
companion (the same contract observed at maturity only) and the tolerance the test uses, both
measured here. Not collected by pytest (no test_ prefix).
"""
import json
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parents[2]
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from pde_3a.semantics import DISC, cases  # noqa: E402
from term_structure_benchmarks import make_term_env  # noqa: E402

from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare, run  # noqa: E402
from quantark.asset.equity.param import PDEParams  # noqa: E402
from quantark.asset.equity.product.option import DoubleOneTouchOption, OneTouchOption  # noqa: E402
from quantark.util.enum.option_enums import BarrierDirection, TouchType  # noqa: E402


def _single_observation(name):
    if name.startswith("no_touch"):
        return OneTouchOption(barrier=115.0, barrier_direction=BarrierDirection.UP, rebate=1.0, payment_at_hit=False,
                              touch_type=TouchType.NO_TOUCH, observation_dates=[1.0], **DISC)
    return DoubleOneTouchOption(upper_barrier=115.0, lower_barrier=85.0, rebate=1.0, payment_at_hit=False,
                                touch_type=TouchType.DOUBLE_NO_TOUCH, observation_dates=[1.0], **DISC)


def main(out_path):
    env = make_term_env("flat")
    table = cases()
    out = {}
    for name, (cls, factory, _ops) in sorted(table.items()):
        g = cls(PDEParams()).calculate_greeks(factory(), env)
        row = {"price": float(g["price"]), "delta": float(g["delta"]), "gamma": float(g["gamma"])}
        if "_di_" not in name and not name.startswith("double_ki"):
            res = run(prepare(cls(PDEParams()), factory(), env, OutputRequest(Operation.SESSION_OUTPUTS, grid=True)))
            row["column"] = [float(v).hex() for v in res.solution.solution_vec]
        out[name] = row
    for name in ("no_touch|acc", "double_no_touch|acc"):
        cls = table[name][0]
        acc = out[name]["price"]
        stop = out[name.replace("acc", "stop")]["price"]
        single = float(cls(PDEParams()).price(_single_observation(name), env))
        tol = max(2.0 * abs(acc - single), 1e-4)
        assert tol <= 1e-3, f"{name}: accumulating vs single-observation gap {abs(acc - single)} is not small"
        assert abs(acc - stop) > 0.03, f"{name}: accumulate and stop-first-hit agree ({acc} vs {stop})"
        out[name].update(single_observation_price=single, single_observation_tolerance=tol)
        print(f"{name}: acc {acc:.6f} single {single:.6f} stop {stop:.6f} tol {tol:.1e}")
    Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{len(out)} cases -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
