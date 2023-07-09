import unittest
import numpy as np
from epiverse.survival.kaplan_meier import KaplanMeier
from epiverse.survival.nelson_aalen import NelsonAalen


class TestKaplanMeier(unittest.TestCase):

    def test_simple_example(self):
        time = np.array([2, 4, 6, 8, 10])
        events = np.array([1, 0, 1, 0, 1])

        expected_survival = np.array([[0, 1, 0],
                                      [2, (1 - 1/5), 0.032],
                                      [6, (1 - 1/5) * (1 - 1/3), 0.06162963],
                                      [10, (1 - 1/5) * (1 - 1/3) * (1 - 1/1), 0]])

        km = KaplanMeier()
        survival_fit = km.fit(
            time=time,
            delta=events
        )

        self.assertTrue(np.allclose(
            expected_survival, survival_fit, atol=1e-4))

        time2 = np.array([1, 3, 5, 7, 9])
        events2 = np.array([1, 1, 0, 1, 0])
        weights2 = np.array([1, 2, 1, 2, 1])

        expected_survival2 = np.array([
            [0, 1, 0],
            [1, (1 - 1/7), 0.01749271],
            [3, (1 - 1/7) * (1 - 2 / 6), 0.03498542],
            [7, (1 - 1/7) * (1 - 2 / 6) * (1 - 2 / 3), 0.02807472]
        ])

        survival_fit2 = km.fit(
            time=time2,
            delta=events2,
            weights=weights2
        )

        self.assertTrue(
            np.allclose(expected_survival2, survival_fit2)
        )

    def test_predict(self):
        time = np.array([2, 4, 6, 8, 10])
        events = np.array([1, 0, 1, 0, 1])

        expected_survival = np.array([[0, 1, 0],
                                      [2, (1 - 1/5), 0.032],
                                      [6, (1 - 1/5) * (1 - 1/3), 0.06162963],
                                      [10, (1 - 1/5) * (1 - 1/3) * (1 - 1/1), 0]])

        km = KaplanMeier()
        survival_fit = km.fit(
            time=time,
            delta=events
        )

        self.assertTrue(
            np.allclose(
                np.array([[1, 0], [0.8, 0.032], [0.8, 0.032]]),
                km.predict(np.array([1, 3, 5]))
            )
        )
