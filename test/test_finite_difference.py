import numpy as np

from quantark.util.numerical import fd1_nonuniform, fd2_nonuniform


def test_exact_for_quadratics_nonuniform():
    x = np.array([0.0, 0.1, 0.35, 0.5, 1.0, 1.7])
    y = 3.0 * x * x - 2.0 * x + 5.0
    np.testing.assert_allclose(fd1_nonuniform(y, x), 6.0 * x - 2.0, rtol=1e-12)
    np.testing.assert_allclose(fd2_nonuniform(y, x), np.full_like(x, 6.0), rtol=1e-12)


def test_batched_last_axis():
    x = np.array([0.0, 0.2, 0.7, 1.0])
    ys = np.vstack([x * x, 2.0 * x * x + x])
    d = fd1_nonuniform(ys, x)
    np.testing.assert_allclose(d[0], 2.0 * x, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(d[1], 4.0 * x + 1.0, rtol=1e-12, atol=1e-14)
