import numpy as np
from epiverse.models.IRLS import IterativelyReweightedLeastSquares


class TestIRLS():

    def test_IRLS(self):
        rng = np.random.default_rng()
        exposure = rng.uniform(low=-10, high=10, size=(10000, 3))
        outcome = exposure @ np.array([3, -1, 2])

        # Check if works under deterministic
        irls = IterativelyReweightedLeastSquares(
            outcome=outcome, exposure=exposure)

        assert np.allclose(np.array([3, -1, 2]), irls.beta)

        assert np.allclose(np.eye(10000), irls.weights)

        # Check when adding randomness
        outcome += rng.normal(0, 0.1, 10000)
        irls2 = IterativelyReweightedLeastSquares(
            outcome, exposure)

        assert np.allclose(np.array([3, -1, 2]), irls2.beta, atol=1e-3)

        assert np.allclose(np.eye(10000), irls.weights)
