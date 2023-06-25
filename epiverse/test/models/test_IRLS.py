import unittest
import numpy as np
from epiverse.models.IRLS import IterativelyReweightedLeastSquares


class TestIRLS(unittest.TestCase):

    def test_IRLS(self):
        rng = np.random.default_rng()
        exposure = rng.uniform(low=-10, high=10, size=(10000, 3))
        outcome = exposure @ np.array([3, -1, 2])

        # Check if works under deterministic
        irls = IterativelyReweightedLeastSquares()
        beta, var = irls.fit(outcome, exposure)

        self.assertTrue(np.allclose(np.array([3, -1, 2]), beta))
        self.assertTrue(np.allclose(np.eye(10000), var))

        # Check when adding randomness
        outcome += rng.normal(0, 0.1, 10000)
        beta, var = irls.fit(outcome, exposure)

        self.assertTrue(np.allclose(np.array([3, -1, 2]), beta, atol=1e-3))
        self.assertTrue(np.allclose(np.eye(10000), var))
