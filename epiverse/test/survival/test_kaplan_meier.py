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

        km = KaplanMeier(time=time, delta=events, weights=1, event_indicator=1)

        survival_predict = km.predict(expected_survival[:, 0])

        self.assertTrue(np.allclose(
            expected_survival, survival_predict, atol=1e-4))

        time2 = np.array([1, 3, 5, 7, 9])
        events2 = np.array([1, 1, 0, 1, 0])
        weights2 = np.array([1, 2, 1, 2, 1])

        expected_survival2 = np.array([
            [0, 1, 0],
            [1, (1 - 1/7), 0.01749271],
            [3, (1 - 1/7) * (1 - 2 / 6), 0.03498542],
            [7, (1 - 1/7) * (1 - 2 / 6) * (1 - 2 / 3), 0.02807472]
        ])

        km2 = KaplanMeier(time=time2, delta=events2,
                          weights=weights2, event_indicator=1)

        survival_predict2 = km2.predict(np.array([0, 1, 3, 7]))

        self.assertTrue(
            np.allclose(expected_survival2, survival_predict2)
        )

    def test_predict(self):
        time = np.array([2, 4, 6, 8, 10])
        events = np.array([1, 0, 1, 0, 1])

        expected_survival = np.array([[0, 1, 0],
                                      [2, (1 - 1/5), 0.032],
                                      [6, (1 - 1/5) * (1 - 1/3), 0.06162963],
                                      [10, (1 - 1/5) * (1 - 1/3) * (1 - 1/1), 0]])

        km = KaplanMeier(time=time, delta=events, weights=1, event_indicator=1)

        self.assertTrue(
            np.allclose(
                expected_survival[[0, 1, 1], :],
                km.predict(np.array([1, 3, 5]))
            )
        )

    def test_multiple_indicators(self):
        time = np.array([2, 4, 6, 8, 10])
        events = np.array([1, 0, 1, 2, 1])

        expected_survival = np.array([[0, 1, 0],
                                      [2, (1 - 1/5), 0.032],
                                      [6, (1 - 1/5) * (1 - 1/3), 0.06162963],
                                      [10, (1 - 1/5) * (1 - 1/3) * (1 - 1/1), 0]])

        km = KaplanMeier(time=time, delta=events, weights=1, event_indicator=1)

        self.assertTrue(
            np.allclose(
                expected_survival[[0, 1, 1], :],
                km.predict(np.array([1, 3, 5]))
            )
        )
