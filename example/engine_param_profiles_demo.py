"""
Demo: engine parameter presets and config loaders.
"""

from pathlib import Path

from quantark.asset.equity.param import PDEParams, QuadParams, make_pde_params, make_quad_params


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    quad_cfg = base_dir / "engine_params_quad.yaml"
    pde_cfg = base_dir / "engine_params_pde.json"

    quad_params = QuadParams.from_config(quad_cfg)
    pde_params = PDEParams.from_config(pde_cfg)

    print("Quad params (config):", quad_params)
    print("PDE params (config):", pde_params)

    fast_quad = make_quad_params(profile="fast")
    accurate_pde = make_pde_params(profile="accurate")

    print("Quad params (fast):", fast_quad)
    print("PDE params (accurate):", accurate_pde)


if __name__ == "__main__":
    main()

