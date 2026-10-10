"""W17a probe hit: test/pde_grid/test_anchor_certification.py::test_snowball_pde_vs_mc[False] builds a snowball MC
grid with a 3.3e-11 step: KO dates ``i / 12`` against discrete KI dates ``round(i / 48, 10)`` (the same instants,
the KI spelled to 10 decimals). The merge would make them one node. This measures the cell as the test computes it
(its own fixtures and ``_mc_price_stderr``), unpatched and under the runtime merge (patch_grid.install, the W17a
semantics), with the gate the test applies. Also the colliding pairs and the KO/KI tie order they decide."""
import sys

import numpy as np

sys.path.insert(0, "/Users/fuxinyao/quant-ark/.claude/worktrees/agent-ab1b5cb9e2b9eb8b4/test/pde_grid")

import patch_grid  # noqa: E402
import test_anchor_certification as t  # noqa: E402
from quantark.asset.equity.engine.mc import SnowballMCEngine  # noqa: E402
from quantark.asset.equity.engine.mc.term_inputs import make_df_fn  # noqa: E402
from quantark.asset.equity.engine.pde import SnowballPDESolver  # noqa: E402
from quantark.asset.equity.param import MCParams, PDEParams  # noqa: E402

if __name__ == "__main__":
    product, e = t.snowball(False), t.env()
    eng = SnowballMCEngine(params=MCParams(num_paths=16, seed=42, use_qmc=True))
    eng._term_ctx = (e, product.strike)
    eng._df = make_df_fn(e)
    T = product.get_maturity(e)
    at, dt, ko_idx, ki_idx = eng._build_time_grid(product, e, T)
    small = np.flatnonzero(dt < 1e-10)
    print(f"grid nodes {at.size}, steps < 1e-10: {small.size}")
    for i in small:
        print(f"   {at[i - 1]!r} / {at[i]!r}  step {dt[i]:.3e}")
    print(f"disable_ko_after_ki = {product.barrier_config.disable_ko_after_ki}")
    pde = float(SnowballPDESolver(params=PDEParams()).price(product, e))
    for label in ("unpatched", "patched"):
        undo = patch_grid.install() if label == "patched" else (lambda: None)
        try:
            mc, se = t._mc_price_stderr(SnowballMCEngine, t.snowball(False), t.env())
        finally:
            undo()
        gate = max(3.0 * se, 2e-4 * 100.0) if np.isfinite(se) else 2e-3 * 100.0
        print(f"{label:9s}: pde={pde:.6f} mc={mc:.6f} stderr={se!r} |pde-mc|={abs(pde - mc):.5f} gate={gate:.5f} "
              f"{'PASS' if abs(pde - mc) < gate else 'FAIL'}", flush=True)
