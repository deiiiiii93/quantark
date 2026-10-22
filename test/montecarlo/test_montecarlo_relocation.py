"""The QMC primitives live in quantark.montecarlo and remain importable from
their legacy equity.process.bsm paths via shims."""


def test_new_package_exposes_primitives():
    from quantark.montecarlo import (
        RandomStream,
        PseudoRandomNormalGenerator,
        SobolNormalGenerator,
        BrownianBridge,
        apply_brownian_bridge,
        compute_step_crossing_probabilities,
        VarianceReductionConfig,
        apply_variance_reduction_to_normals,
        RQMCResult,
        run_rqmc,
    )

    assert PseudoRandomNormalGenerator is not None


def test_legacy_paths_still_work():
    from quantark.asset.equity.process.bsm.qmc_sobol import (
        SobolNormalGenerator as Legacy,
    )
    from quantark.montecarlo.qmc_sobol import SobolNormalGenerator as New

    assert Legacy is New  # shim re-exports the same object


def test_path_generator_still_imports():
    # qmc_path_generator stays in equity and imports the (now shimmed) primitives.
    from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator

    assert GBMPathGenerator is not None
