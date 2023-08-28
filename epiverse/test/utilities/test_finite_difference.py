import unittest
from epiverse.utilities.math.finite_differences import second_central_difference
import numpy as np


class TestFiniteDifference(unittest.TestCase):

    def test_linear(self):
        def linear_func(x): return x[0] + x[1] + 1

        linear_scd = second_central_difference(
            linear_func, np.array([0.5, 0.2]))
        print(linear_scd)
        self.assertTrue(
            np.all(np.isclose(
                np.array([[0, 0], [0, 0]]),
                linear_scd
            ))
        )

        def quadratic_func(x): return x[0]**2 + 2 * x[1]**2 + 1

        quadratic_scd = second_central_difference(
            quadratic_func, np.array([0.25, 0.5]))
        print(quadratic_scd)
        self.assertTrue(
            np.all(np.isclose(
                np.array([[2, 0], [0, 4]]),
                quadratic_scd
            ))
        )

        def dual_func(x): return 3 * x[0] * x[1] + x[0]**2 + 2 * x[1]**2 + 2

        dual_scd = second_central_difference(
            dual_func, np.array([0.25, 0.5]))
        print(dual_scd)
        self.assertTrue(
            np.all(np.isclose(
                np.array([[2, 3], [3, 4]]),
                dual_scd
            ))
        )
